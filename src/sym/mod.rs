//! Symbolic microcode execution.

use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};

use crate::flow::{AbstractLocation, StorageLocation};
use crate::ir::{MicroOperation, Location, Temporary, MemoryMapped};
use crate::math::{SymExpr, SymCondition, Integer, DataType, Symbol, SharedSolver, Traversed};
use crate::x86_64::{Instruction, Mnemoic, Register};
use DataType::*;

mod mem;
pub use mem::*;


/// The symbolic execution state.
#[derive(Debug, Clone)]
pub struct SymState {
    /// The values of the temporaries (T0, T1, ...).
    pub temporaries: HashMap<usize, SymExpr>,
    /// The two memory spaces for main memory and registers.
    pub memory: [SymMemory; 2],
    /// A mapping from symbols to the abstract locations where they could be
    /// found in an actual execution.
    pub symbol_map: SymbolMap,
    /// The path of trace points which were set for this state. These are u sed
    /// for describing the context in symbol generation.
    pub trace: Vec<u64>,
    /// The current instruction pointer.
    pub ip: u64,
    /// The shared SMT solver.
    pub solver: SharedSolver,
    /// The number of used symbols.
    stdin_symbols: usize,
    stdout_symbols: usize,
}

/// When and where to find the symbolic values in memory in a real execution.
pub type SymbolMap = HashMap<Symbol, AbstractLocation>;

/// Events occuring during symbolic execution.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Event {
    Jump { target: SymExpr, condition: SymCondition, relative: bool },
    Stdio(StdioKind, Vec<(Symbol, TypedMemoryAccess)>),
    Exit,
}

/// Kinds of standard interfaces (stdin or stdout).
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum StdioKind {
    Stdin,
    Stdout,
}

impl SymState {
    /// Create a blank symbolic state that will use the given solver and strategy for
    /// main memory.
    pub fn new(mem_strategy: MemoryStrategy, solver: SharedSolver) -> SymState {
        SymState {
            temporaries: HashMap::new(),
            memory: [
                SymMemory::new("mem", mem_strategy, solver.clone()),
                SymMemory::new("reg", MemoryStrategy::PerfectMatches, solver.clone())
            ],
            symbol_map: SymbolMap::new(),
            trace: Vec::new(),
            ip: 0,
            stdin_symbols: 0,
            stdout_symbols: 0,
            solver
        }
    }

    /// Execute a micro operation.
    pub fn step(&mut self, addr: u64, operation: &MicroOperation) -> Option<Event> {
        use MicroOperation as Op;

        crate::timings::start("sym-step");

        self.set_reg(Register::RIP, SymExpr::from_ptr(addr));
        self.ip = addr;

        match operation {
            Op::Mov { dest, src } => self.do_move(*dest, *src),

            Op::Const { dest, constant } => self.set_temp(*dest, SymExpr::Int(*constant)),
            Op::Cast { target, new, signed } => {
                let new_value = self.get_temp(*target).cast(*new, *signed);
                self.set_temp(Temporary(*new, target.1), new_value);
            },

            Op::Add { sum, a, b } => self.do_binop(*sum, *a, *b, SymExpr::add),
            Op::Sub { diff, a, b } => self.do_binop(*diff, *a, *b, SymExpr::sub),
            Op::Mul { prod, a, b } => self.do_binop(*prod, *a, *b, SymExpr::mul),

            Op::And { and, a, b } => self.do_binop(*and, *a, *b, SymExpr::bitand),
            Op::Or { or, a, b } => self.do_binop(*or, *a, *b, SymExpr::bitor),
            Op::Not { not, a } => self.set_temp(*not, self.get_temp(*a).bitnot()),

            Op::Set { target, condition } => {
                self.set_temp(*target, self.evaluate_condition(&condition).as_expr(target.0));
            },
            Op::Jump { target, condition, relative } => {
                crate::timings::stop();
                return Some(Event::Jump {
                    target: self.get_temp(*target),
                    condition: condition.clone(),
                    relative: *relative,
                });
            },

            Op::Syscall => {
                if let SymExpr::Int(int) = self.get_reg(Register::RAX) {
                    if let Some(event) = self.do_syscall(int.1) {
                        crate::timings::stop();
                        return Some(event);
                    }
                } else {
                    panic!("step: unhandled symbolic syscall number");
                }
            },
        }

        crate::timings::stop();
        None
    }

    /// Adjust the trace based on the instruction.
    pub fn track(&mut self, instruction: &Instruction, addr: u64) {
        // Adjust the trace.
        match instruction.mnemoic {
            Mnemoic::Call => self.trace.push(addr),
            Mnemoic::Ret => { self.trace.pop(); },
            _ => {},
        };
    }

    /// Evaluate a symbolic expression with temporary symbols.
    pub fn evaluate_condition(&self, condition: &SymCondition) -> SymCondition {
        let mut evaluated = condition.clone();
        evaluated.replace_symbols(&|sym| match sym {
            Symbol(data_type, "T", index) => self.get_temp(Temporary(data_type, index)),
            sym => SymExpr::Sym(sym),
        });
        evaluated
    }

    /// Generate a symbol map with just the symbols needed for the condition.
    pub fn get_symbol_map_for(&self, condition: &SymCondition) -> SymbolMap {
        let mut symbols = HashMap::new();
        condition.traverse(&mut |node| {
            if let Traversed::Expr(&SymExpr::Sym(symbol)) = node {
                if let Some(loc) = self.symbol_map.get(&symbol) {
                    symbols.insert(symbol, loc.clone());
                }
            }
        });
        symbols
    }

    /// Return the address expression and data type of the storage location if
    /// it is a memory access.
    pub fn get_access_for_storage(&self, location: StorageLocation) -> Option<TypedMemoryAccess> {
        use StorageLocation::*;
        match location {
            Direct(_) => None,
            Indirect { data_type, base, scaled_offset, displacement } => Some({
                let mut addr = self.get_reg(base);

                if let Some((index, scale)) = scaled_offset {
                    let scale = SymExpr::from_ptr(scale as u64);
                    let offset = self.get_reg(index).mul(scale);
                    addr = addr.add(offset);
                }

                if let Some(disp) = displacement {
                    addr = addr.add(SymExpr::from_ptr(disp as u64));
                }

                TypedMemoryAccess(addr, data_type)
            }),
        }
    }

    /// Retrieve data from a location.
    pub fn read_location(&self, src: Location) -> SymExpr {
        match src {
            Location::Temp(temp) => self.get_temp(temp),
            Location::Direct(data_type, space, addr) => {
                self.memory[space].read_direct(addr, data_type)
            },
            Location::Indirect(data_type, space, temp) => {
                let addr = self.get_temp(temp);
                assert_eq!(addr.data_type(), N64, "read_location: address has to be 64-bit");
                self.memory[space].read_expr(addr, data_type)
            }
        }
    }

    /// Write data to a location.
    pub fn write_location(&mut self, dest: Location, value: SymExpr) {
        assert_eq!(dest.data_type(), value.data_type(),
            "write_location: incompatible data types for write");

        match dest {
            Location::Temp(temp) => self.set_temp(temp, value),
            Location::Direct(_, space, addr) => {
                self.memory[space].write_direct(addr, value);
            },
            Location::Indirect(_, space, temp) => {
                let addr = self.get_temp(temp);
                assert_eq!(addr.data_type(), N64, "write_location: address has to be 64-bit");
                self.memory[space].write_expr(addr, value);
            }
        }
    }

    /// Return the integer stored in the temporary.
    pub fn get_temp(&self, temp: Temporary) -> SymExpr {
        let expr = self.temporaries[&temp.1].clone();
        assert_eq!(temp.0, expr.data_type(), "get_temp: incompatible data types");
        expr
    }

    /// Set the temporary to a new value.
    pub fn set_temp(&mut self, temp: Temporary, value: SymExpr) {
        assert_eq!(temp.0, value.data_type(), "set_temp: incompatible data types");
        self.temporaries.insert(temp.1, value);
    }

    /// Get a value from a register.
    pub fn get_reg(&self, reg: Register) -> SymExpr {
        self.memory[1].read_direct(reg.address(), reg.data_type())
    }

    /// Set a register to a value.
    pub fn set_reg(&mut self, reg: Register, value: SymExpr) {
        self.memory[1].write_direct(reg.address(), value);
    }

    /// Do a binary operation.
    fn do_binop<F>(&mut self, target: Temporary, a: Temporary, b: Temporary, binop: F)
    where F: FnOnce(SymExpr, SymExpr) -> SymExpr {
        self.set_temp(target, binop(self.get_temp(a), self.get_temp(b)));
    }

    /// Move a value from a location to another location.
    fn do_move(&mut self, dest: Location, src: Location) {
        assert_eq!(dest.data_type(), src.data_type(), "do_move: incompatible data types for move");
        let value = self.read_location(src);
        self.write_location(dest, value);
    }

    /// Emulate a Linux syscall.
    fn do_syscall(&mut self, num: u64) -> Option<Event> {
        match num {
            // Read from or write to a file descriptor.
            // We generate one symbol per byte read / written.
            0 | 1 => {
                let read = num == 0;

                let buf = self.get_reg(Register::RSI);
                let count = self.get_reg(Register::RDX);
                let byte_count = match count {
                    SymExpr::Int(Integer(N64, bytes)) => bytes,
                    _ => panic!("do_syscall: read: unknown byte count"),
                };

                let mut locs = vec![];

                for i in 0 .. byte_count {
                    let symbol_ptr = if read {
                        &mut self.stdin_symbols
                    } else {
                        &mut self.stdout_symbols
                    };

                    let symbol = Symbol(N8, if read { "stdin" } else { "stdout" }, *symbol_ptr);
                    let value = SymExpr::Sym(symbol);
                    *symbol_ptr += 1;

                    let target = buf.clone().add(SymExpr::from_ptr(i));
                    if read {
                        self.memory[0].write_expr(target.clone(), value);
                    }

                    let location = AbstractLocation {
                        addr: self.ip,
                        trace: self.trace.clone(),
                        storage: StorageLocation::Indirect {
                            data_type: N8,
                            base: Register::RSI,
                            scaled_offset: None,
                            displacement: if i > 0 { Some(i as i64) } else { None },
                        },
                    };

                    self.symbol_map.insert(symbol, location);
                    locs.push((symbol, TypedMemoryAccess(target, N8)));
                }

                let kind = if read { StdioKind::Stdin } else { StdioKind::Stdout };
                Some(Event::Stdio(kind, locs))
            },

            // System exit
            60 => Some(Event::Exit),
            s => panic!("do_syscall: unimplemented syscall number {}", s),
        }
    }
}

/// A typed symbolic memory access.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct TypedMemoryAccess(pub SymExpr, pub DataType);

impl Display for TypedMemoryAccess {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "[{}]:{}", self.0, self.1)
    }
}
