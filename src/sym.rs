//! Symbolic microcode execution.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};

use crate::x86_64::{Register, Operand};
use crate::ir::{MicroOperation, Location, Temporary, MemoryMapped};
use crate::ir::{JumpCondition, FlaggedOperation};
use crate::num::{Integer, DataType};
use crate::expr::{SymExpr, SymCondition, Symbol};
use DataType::*;


/// The symbolic execution state.
#[derive(Debug, Clone)]
pub struct SymState {
    pub temporaries: HashMap<usize, SymExpr>,
    pub memory: [SymMemory; 2],
    pub symbol_map: SymbolMap,
    pub path: Vec<u64>,
    pub ip: u64,
}

impl SymState {
    /// Create a blank symbolic state.
    pub fn new() -> SymState {
        SymState {
            temporaries: HashMap::new(),
            memory: [SymMemory::new(0), SymMemory::new(1)],
            symbol_map: SymbolMap::new(),
            path: Vec::new(),
            ip: 0,
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

            Op::And { and, a, b } => self.do_binop(and, a, b, SymExpr::bit_and),
            Op::Or { or, a, b } => self.do_binop(or, a, b, SymExpr::bit_or),
            Op::Not { not, a } => self.set_temp(not, self.get_temp(a).bit_not()),

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
                    let symbol = self.memory[0].new_symbol(N8);
                    let value = SymExpr::Sym(symbol);
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

    /// Evaulate a condition.
    fn evaluate(&self, condition: JumpCondition) -> SymCondition {
        use JumpCondition::*;
        use FlaggedOperation as Op;

        match condition {
            True => SymCondition::Bool(true),

            Equal(Op::Sub { a, b }) => self.get_temp(a).equal(self.get_temp(b)),
            Less(Op::Sub { a, b }) => self.get_temp(a).less(self.get_temp(b)),
            LessEqual(Op::Sub { a, b }) => self.get_temp(a).less_equal(self.get_temp(b)),
            Greater(Op::Sub { a, b }) => self.get_temp(a).greater(self.get_temp(b)),
            GreaterEqual(Op::Sub { a, b }) => self.get_temp(a).greater_equal(self.get_temp(b)),

            Equal(Op::And { a, b }) => self.get_temp(a).bit_and(self.get_temp(b))
                .equal(SymExpr::int(a.0, 0)),

            _ => panic!("evaluate: unhandled condition/comparison pair"),
        }
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
    id: usize,
    data: RefCell<MemoryData>,
}

/// The actual memory data, which is wrapped in an interior mutability type
/// to make reads on immutable borrows possible while performing some extra work
/// requiring mutable access.
#[derive(Debug, Clone)]
struct MemoryData {
    map: HashMap<SymExpr, SymExpr>,
    symbols: usize,
}

impl SymMemory {
    /// Create a new blank symbolic memory where symbolics start at `base`.
    pub fn new(id: usize) -> SymMemory {
        SymMemory {
            id,
            data: RefCell::new(MemoryData {
                map: HashMap::new(),
                symbols: 0,
            })
        }
    }

    /// Read from a direct address.
    pub fn read_direct(&self, addr: u64, data_type: DataType) -> SymExpr {
        self.read_expr(SymExpr::from_ptr(addr), data_type)
    }

    /// Read from a symbolic address.
    pub fn read_expr(&self, addr: SymExpr, data_type: DataType) -> SymExpr {
        let mut data = self.data.borrow_mut();
        match data.map.get(&addr) {
            Some(expr) => if expr.data_type() == data_type {
                expr.clone()
            } else {
                expr.clone().cast(data_type, false)
            },
            None => {
                let value = SymExpr::Sym(Symbol(data_type, self.id, data.symbols));
                data.map.insert(addr, value.clone());
                data.symbols += 1;
                value
            },
        }
    }

    /// Write a value to a direct address.
    pub fn write_direct(&mut self, addr: u64, value: SymExpr) {
        self.write_expr(SymExpr::from_ptr(addr), value)
    }

    /// Write a value to a symbolic address.
    pub fn write_expr(&mut self, addr: SymExpr, value: SymExpr) {
        self.data.borrow_mut().map.insert(addr, value);
    }

    /// Generate a new symbol.
    pub fn new_symbol(&mut self, data_type: DataType) -> Symbol {
        let mut data = self.data.borrow_mut();
        data.symbols += 1;
        Symbol(data_type, self.id, data.symbols - 1)
    }
}

impl Display for SymMemory {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "SymMemory [")?;
        let data = self.data.borrow();
        if !data.map.is_empty() { writeln!(f)?; }
        for (location, value) in &data.map {
            writeln!(f, "    {} => {}", location, value)?;
        }
        writeln!(f, "]")
    }
}

/// When and where to find the symbolic values in memory in a real execution.
pub type SymbolMap = HashMap<Symbol, AbstractLocation>;

/// Identifies a CPU location with an address at which it is valid and a path
/// that defines the context in which it is valid.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct AbstractLocation {
    pub addr: u64,
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
                if *offset >= 0 {
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
