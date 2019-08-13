//! Symbolic microcode execution.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};

use crate::x86_64::{Register, Operand};
use crate::ir::{MicroOperation, Location, Temporary, MemoryMapped};
use crate::ir::{JumpCondition, FlaggedOperation};
use crate::num::{Integer, DataType};
use crate::expr::{SymExpr, SymCondition, Symbol};
use crate::smt::SharedSolver;
use DataType::*;


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
    /// The path of trace points which were set for this state.
    pub path: Vec<u64>,
    /// The current instruction pointer.
    pub ip: u64,
    symbols: usize,
    solver: SharedSolver,
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
            path: Vec::new(),
            ip: 0,
            symbols: 0,
            solver
        }
    }

    /// Execute a micro operation.
    pub fn step(&mut self, addr: u64, operation: MicroOperation) -> Option<Event> {
        use MicroOperation as Op;

        self.set_reg(Register::RIP, SymExpr::from_ptr(addr));
        self.ip = addr;

        match operation {
            Op::Mov { dest, src } => self.do_move(dest, src),

            Op::Const { dest, constant } => self.set_temp(dest, SymExpr::Int(constant)),
            Op::Cast { target, new, signed } => {
                let new_value = self.get_temp(target).cast(new, signed);
                self.set_temp(Temporary(new, target.1), new_value);
            },

            Op::Add { sum, a, b } => self.do_binop(sum, a, b, SymExpr::add),
            Op::Sub { diff, a, b } => self.do_binop(diff, a, b, SymExpr::sub),
            Op::Mul { prod, a, b } => self.do_binop(prod, a, b, SymExpr::mul),

            Op::And { and, a, b } => self.do_binop(and, a, b, SymExpr::bitand),
            Op::Or { or, a, b } => self.do_binop(or, a, b, SymExpr::bitor),
            Op::Not { not, a } => self.set_temp(not, self.get_temp(a).bitnot()),

            Op::Set { target, condition } => {
                self.set_temp(target, self.evaluate(condition).as_expr(target.0));
            },
            Op::Jump { target, condition, relative } => {
                return Some(Event::Jump {
                    target: self.get_temp(target),
                    condition,
                    relative
                });
            },

            Op::Syscall => {
                if let SymExpr::Int(int) = self.get_reg(Register::RAX) {
                    if let Some(event) = self.do_syscall(int.1) {
                        return Some(event);
                    }
                } else {
                    panic!("step: unhandled symbolic syscall number");
                }
            },
        }

        None
    }

    /// Add an address to the path of this state.
    pub fn trace(&mut self, addr: u64) {
        self.path.push(addr);
    }

    /// Evaluate a jump condition.
    pub fn evaluate(&self, condition: JumpCondition) -> SymCondition {
        use JumpCondition::*;
        use FlaggedOperation as Op;

        match condition {
            True => SymCondition::TRUE,

            Equal(Op::Sub { a, b }) => self.get_temp(a).equal(self.get_temp(b)),

            Below(Op::Sub { a, b }) => self.get_temp(a).less_than(self.get_temp(b), false),
            BelowEqual(Op::Sub { a, b }) => self.get_temp(a).less_equal(self.get_temp(b), false),
            Above(Op::Sub { a, b }) => self.get_temp(a).greater_than(self.get_temp(b), false),
            AboveEqual(Op::Sub { a, b }) => self.get_temp(a).greater_equal(self.get_temp(b), false),

            Less(Op::Sub { a, b }) => self.get_temp(a).less_than(self.get_temp(b), true),
            LessEqual(Op::Sub { a, b }) => self.get_temp(a).less_equal(self.get_temp(b), true),
            Greater(Op::Sub { a, b }) => self.get_temp(a).greater_than(self.get_temp(b), true),
            GreaterEqual(Op::Sub { a, b }) => self.get_temp(a).greater_equal(self.get_temp(b), true),

            Equal(Op::And { a, b }) => self.get_temp(a).bitand(self.get_temp(b))
                .equal(SymExpr::int(a.0, 0)),

            _ => panic!("evaluate: unhandled condition/comparison pair"),
        }
    }

    /// Return the address expression and data type of the operand if it is a memory access.
    pub fn get_access_for_operand(&self, operand: Operand) -> Option<TypedMemoryAccess> {
        match operand {
            Operand::Indirect { data_type, base, scaled_offset, displacement } => Some({
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
            _ => None,
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
            // Read from a file descriptor.
            // We generate one symbol per byte read.
            0 => {
                let buf = self.get_reg(Register::RSI);
                let count = self.get_reg(Register::RDX);
                let byte_count = match count {
                    SymExpr::Int(Integer(N64, bytes)) => bytes,
                    _ => panic!("do_syscall: read: unknown byte count"),
                };

                for i in 0 .. byte_count {
                    let target = buf.clone().add(SymExpr::from_ptr(i));

                    let symbol = Symbol(N8, "stdin", self.symbols);
                    let value = SymExpr::Sym(symbol);
                    self.symbols += 1;

                    self.memory[0].write_expr(target, value);

                    let location = CpuLocation::IndirectMemory(Register::RSI, i as i64);
                    self.symbol_map.insert(symbol, AbstractLocation {
                        addr: self.ip,
                        path: self.path.clone(),
                        location,
                    });
                }
            },

            // Write to a file descriptor (has no effect, so we do nothing).
            1 => {},

            // System exit
            60 => return Some(Event::Exit),
            s => panic!("do_syscall: unimplemented syscall number {}", s),
        }
        None
    }
}

/// Events occuring during symbolic execution.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Event {
    Jump { target: SymExpr, condition: JumpCondition, relative: bool },
    Exit,
}

/// Symbolic memory handling writes and reads involving symbolic
/// values and addresses.
#[derive(Debug, Clone)]
pub struct SymMemory {
    data: RefCell<MemoryData>,
    solver: SharedSolver,
    strategy: MemoryStrategy,
}

/// How the memory handled complex symbolic queries.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum MemoryStrategy {
    /// Build an if-then-else tree of values that could possibly match.
    ConditionalTrees,
    /// Only return perfect matches (faster).
    PerfectMatches,
}

/// The actual memory data, which is wrapped in an interior mutability type
/// to make reads on immutable borrows possible while performing some extra work
/// requiring mutable access.
#[derive(Debug, Clone)]
struct MemoryData {
    name: &'static str,
    entries: Vec<MemoryEntry>,
    symbols: usize,
    epoch: u32,
}

/// A piece of data written to memory.
#[derive(Debug, Clone)]
struct MemoryEntry {
    addr: SymExpr,
    value: SymExpr,
    epoch: u32,
}

impl SymMemory {
    /// Create a new blank symbolic memory.
    pub fn new(name: &'static str, strategy: MemoryStrategy, solver: SharedSolver)
    -> SymMemory {
        SymMemory {
            data: RefCell::new(MemoryData {
                name,
                entries: Vec::new(),
                symbols: 0,
                epoch: 1,
            }),
            solver,
            strategy,
        }
    }

    /// Read from a direct address.
    pub fn read_direct(&self, addr: u64, data_type: DataType) -> SymExpr {
        self.read_expr(SymExpr::from_ptr(addr), data_type)
    }

    /// Write a value to a direct address.
    pub fn write_direct(&mut self, addr: u64, value: SymExpr) {
        self.write_expr(SymExpr::from_ptr(addr), value)
    }

    /// Read from a symbolic address.
    pub fn read_expr(&self, addr: SymExpr, data_type: DataType) -> SymExpr {
        let mut data = self.data.borrow_mut();

        let expr = if self.strategy == MemoryStrategy::ConditionalTrees {
            // Build a map of addresses that could be the one we look for and
            // add the conditions under which they match.
            let mut condition_map = Vec::new();

            for entry in &data.entries {
                let perfect_match = entry.addr == addr;

                if perfect_match {
                    condition_map.push((SymCondition::TRUE, entry));

                } else if self.solver.check_equal_sat(&entry.addr, &addr) {
                    let condition = entry.addr.clone().equal(addr.clone());
                    let simplified = self.solver.simplify_condition(condition);
                    condition_map.push((simplified, entry));
                }
            }

            // We sort the conditional entries by epoch inversed. This way, we can
            // traverse them from new to old and arrange them in an if-then-else tree
            // until we hit a perfect match (then we can stop as everything that
            // follows was written earlier and would thus have been overwritten).
            // If we traversed the whole map without hitting a perfect match, the
            // accessed memory is possibly still uninitialized. In this case, we
            // generate a new symbol and put it in as the last leaf of the tree.

            condition_map.sort_by_key(|(_, entry)| std::cmp::Reverse(entry.epoch));

            let mut end = 0;
            let mut perfect_match = None;

            for (index, (condition, entry)) in condition_map.iter().enumerate() {
                if *condition == SymCondition::TRUE {
                    // We have a perfect match.
                    perfect_match = Some(entry.value.clone());
                    break;
                }
                end = index + 1;
            }

            // Remove all entries after the perfect match.
            condition_map.truncate(end);

            // If we ended on a perfect match, we use it, otherwise we generate a default symbol.
            let used_default_symbol = perfect_match.is_none();
            let mut tree = perfect_match.unwrap_or_else(|| data.get_default_value(data_type));

            // Now we build the if-then-else tree bottom-up.
            for (condition, entry) in condition_map.into_iter().rev() {
                tree = condition.if_then_else(entry.value.clone(), tree);
            }

            // If we used a default symbol previously, generate it now. We did not
            // generate it earlier because `data` was still borrowed immutably.
            if used_default_symbol {
                data.generate_default_symbol(addr, data_type);
            }

            tree

        } else {
            // Look only for the newest perfect matches.
            let mut newest_match: Option<(&SymExpr, u32)> = None;

            for entry in &data.entries {
                let perfect_match = entry.addr == addr;
                let newer = newest_match.iter().all(|(_, epoch)| *epoch < entry.epoch);

                if perfect_match && newer {
                    newest_match = Some((&entry.value, entry.epoch));
                }
            }

            newest_match.map(|(expr, _)| expr.clone())
                .unwrap_or_else(|| data.generate_default_symbol(addr, data_type))
        };

        if expr.data_type() == data_type { expr } else { expr.cast(data_type, false) }
    }

    /// Write a value to a symbolic address.
    pub fn write_expr(&mut self, addr: SymExpr, value: SymExpr) {
        let mut data = self.data.borrow_mut();

        let new_entry = MemoryEntry {
            addr,
            value,
            epoch: data.epoch,
        };

        for entry in data.entries.iter_mut() {
            let perfect_match = entry.addr == new_entry.addr;
            if perfect_match {
                *entry = new_entry;
                return;
            }
        }

        data.epoch += 1;
        data.entries.push(new_entry);
    }
}

impl MemoryData {
    /// Get the value for the next default symbol that would be generated.
    fn get_default_value(&self, data_type: DataType) -> SymExpr {
        SymExpr::Sym(Symbol(data_type, self.name, self.symbols))
    }

    /// Generate a default symbol for uninitialized memory.
    fn generate_default_symbol(&mut self, addr: SymExpr, data_type: DataType) -> SymExpr {
        let value = self.get_default_value(data_type);
        self.entries.push(MemoryEntry {
            addr,
            value: value.clone(),
            epoch: 0,
        });
        self.symbols += 1;
        value
    }
}

impl Display for SymMemory {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "SymMemory [")?;
        let data = self.data.borrow();
        if !data.entries.is_empty() { writeln!(f)?; }
        for entry in &data.entries {
            writeln!(f, "    [{}] {} => {}", entry.epoch, entry.addr, entry.value)?;
        }
        writeln!(f, "]")
    }
}

/// When and where to find the symbolic values in memory in a real execution.
pub type SymbolMap = HashMap<Symbol, AbstractLocation>;

/// A CPU location within the context in which it is valid.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct AbstractLocation {
    /// The address at which the location is valid.
    pub addr: u64,
    /// The path that should have been taken to make the location valid.
    pub path: Vec<u64>,
    pub location: CpuLocation,
}

/// A location in a real execution.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum CpuLocation {
    /// A register.
    Reg(Register),
    /// An address in main memory.
    DirectMemory(u64),
    /// The address stored in the register combined with the offset.
    IndirectMemory(Register, i64),
}

impl Display for AbstractLocation {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{} at {:x} by ", self.location, self.addr)?;
        let mut first = true;
        for &addr in &self.path {
            if !first { write!(f, " -> ")?; } first = false;
            write!(f, "{:x}", addr)?;
        }
        Ok(())
    }
}

impl Display for CpuLocation {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use CpuLocation::*;
        match self {
            Reg(reg) => write!(f, "{}", reg),
            DirectMemory(addr) => write!(f, "[{:#x}]", addr),
            IndirectMemory(reg, offset) => {
                write!(f, "[{}", reg)?;
                if *offset > 0 {
                    write!(f, "+{:#x}", offset)?
                } else if *offset < 0 {
                    write!(f, "-{:#x}", -offset)?
                }
                write!(f, "]")
            }
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
