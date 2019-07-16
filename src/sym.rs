//! Symbolic microcode execution.

use std::collections::HashMap;
use std::cell::RefCell;
use crate::amd64::Register;
use crate::ir::{MicroOperation, Location, Temporary, Condition, MemoryMapped};
use crate::num::{Integer, DataType};
use self::SymExpr::*;


/// The symbolic execution state.
#[derive(Debug, Clone)]
pub struct SymState {
    pub temporaries: HashMap<usize, SymExpr>,
    pub memory: [SymMemory; 2],
}

impl SymState {
    /// Create a blank symbolic state.
    pub fn new() -> SymState {
        SymState {
            temporaries: HashMap::new(),
            memory: [SymMemory::new(), SymMemory::new()],
        }
    }

    /// Execute a micro operation.
    pub fn step(&mut self, operation: MicroOperation) -> Option<Event> {
        use MicroOperation as Op;

        match operation {
            Op::Mov { dest, src } => self.do_move(dest, src),
            Op::Const { dest, constant } => self.write_location(dest, SymExpr::Int(constant)),
            Op::Cast { target, new, signed } => {
                let new_value = self.get_temp(target).cast(new, signed);
                self.set_temp(Temporary(new, target.1), new_value);
            },

            Op::Add { sum, a, b } => self.set_temp(sum, self.get_temp(a).add(self.get_temp(b))),
            Op::Sub { diff, a, b } => self.set_temp(diff, self.get_temp(a).sub(self.get_temp(b))),
            Op::Mul { prod, a, b } => self.set_temp(prod, self.get_temp(a).mul(self.get_temp(b))),

            Op::And { and, a, b } => unimplemented!("and"),
            Op::Or { or, a, b } => unimplemented!("or"),
            Op::Not { not, a } => unimplemented!("not"),

            Op::Set { target, condition } => unimplemented!("set"),
            Op::Jump { target, condition, relative } => {
                return Some(Event::Jump { target: self.get_temp(target), condition, relative });
            },

            Op::Syscall => {
                if let SymExpr::Int(int) = self.get_reg(Register::RAX) {
                    match int.1 {
                        // Exit syscall
                        0x3c => return Some(Event::Exit),
                        s => panic!("unimplemented syscall: {:#x}", s),
                    }
                } else {
                    panic!("step: unhandled symbolic syscall");
                }
            },
        }

        None
    }

    /// Move a value from a location to another location.
    fn do_move(&mut self, dest: Location, src: Location) {
        assert_eq!(dest.data_type(), src.data_type(),
            "do_move: incompatible data types for move");
        let value = self.read_location(src);

        self.write_location(dest, value);
    }

    /// Retrieve data from a location.
    fn read_location(&self, src: Location) -> SymExpr {
        match src {
            Location::Temp(temp) => self.get_temp(temp),
            Location::Direct(data_type, space, addr) => {
                self.memory[space].read_direct(addr, data_type)
            },
            Location::Indirect(data_type, space, temp) => {
                let addr = self.get_temp(temp);
                assert_eq!(addr.data_type(), DataType::N64,
                    "read_location: address has to be 64-bit");
                self.memory[space].read_expr(addr, data_type)
            }
        }
    }

    /// Write data to a location.
    fn write_location(&mut self, dest: Location, value: SymExpr) {
        assert_eq!(dest.data_type(), value.data_type(),
            "write_location: incompatible data types for write");

        match dest {
            Location::Temp(temp) => self.set_temp(temp, value),
            Location::Direct(_, space, addr) => {
                self.memory[space].write_direct(addr, value);
            },
            Location::Indirect(_, space, temp) => {
                let addr = self.get_temp(temp);
                assert_eq!(addr.data_type(), DataType::N64,
                    "write_location: address has to be 64-bit");
                self.memory[space].write_expr(addr, value);
            }
        }
    }

    /// Return the integer stored in the temporary.
    fn get_temp(&self, temp: Temporary) -> SymExpr {
        let expr = self.temporaries[&temp.1].clone();
        assert_eq!(temp.0, expr.data_type(), "get_temp: incompatible data types");
        expr
    }

    /// Set the temporary to a new value.
    fn set_temp(&mut self, temp: Temporary, value: SymExpr) {
        assert_eq!(temp.0, value.data_type(), "set_temp: incompatible data types");
        self.temporaries.insert(temp.1, value);
    }

    /// Get a value from a register.
    fn get_reg(&self, reg: Register) -> SymExpr {
        self.memory[1].read_direct(reg.address(), reg.data_type())
    }

    /// Set a register to a value.
    fn set_reg(&mut self, reg: Register, value: SymExpr) {
        self.memory[1].write_direct(reg.address(), value);
    }
}

/// Events occuring during symbolic execution.
#[derive(Debug, Clone)]
pub enum Event {
    Jump { target: SymExpr, condition: Condition, relative: bool },
    Exit,
}

/// Symbolic memory handling writes and reads involving symbolic
/// values and addresses.
#[derive(Debug, Clone)]
pub struct SymMemory {
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
    /// Create a new blank symbolic memory.
    pub fn new() -> SymMemory {
        SymMemory {
            data: RefCell::new(MemoryData {
                map: HashMap::new(),
                symbols: 0,
            })
        }
    }

    /// Read from a direct address.
    pub fn read_direct(&self, addr: u64, data_type: DataType) -> SymExpr {
        self.read_expr(SymExpr::Int(Integer::ptr(addr)), data_type)
    }

    /// Read from a symbolic address.
    pub fn read_expr(&self, addr: SymExpr, data_type: DataType) -> SymExpr {
        let mut data = self.data.borrow_mut();
        match data.map.get(&addr) {
            Some(expr) => expr.clone(),
            None => {
                let value = SymExpr::Sym(Symbol(data_type, data.symbols));
                data.map.insert(addr, value.clone());
                data.symbols += 1;
                value
            },
        }
    }

    /// Write a value to a direct address.
    pub fn write_direct(&mut self, addr: u64, value: SymExpr) {
        self.write_expr(SymExpr::Int(Integer::ptr(addr)), value)
    }

    /// Write a value to a symbolic address.
    pub fn write_expr(&mut self, addr: SymExpr, value: SymExpr) {
        self.data.borrow_mut().map.insert(addr, value);
    }
}

/// A possibly composed symbolic expression.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum SymExpr {
    Int(Integer),
    Sym(Symbol),
    Add(Box<SymExpr>, Box<SymExpr>),
    Sub(Box<SymExpr>, Box<SymExpr>),
    Mul(Box<SymExpr>, Box<SymExpr>),
    Cast(Box<SymExpr>, DataType, bool)
}

impl SymExpr {
    /// The data type of the expression.
    pub fn data_type(&self) -> DataType {
        match self {
            Int(int) => int.0,
            Sym(sym) => sym.0,
            Add(a, _) => a.data_type(),
            Sub(a, _) => a.data_type(),
            Mul(a, _) => a.data_type(),
            Cast(_, new, _) => *new,
        }
    }

    pub fn add(self, other: SymExpr) -> SymExpr {
        match (self, other) {
            (Int(a), Int(b)) => Int(a + b),
            (a, b) => Add(Box::new(a), Box::new(b)),
        }
    }

    pub fn sub(self, other: SymExpr) -> SymExpr {
        match (self, other) {
            (Int(a), Int(b)) => Int(a - b),
            (a, b) => Sub(Box::new(a), Box::new(b)),
        }
    }

    pub fn mul(self, other: SymExpr) -> SymExpr {
        match (self, other) {
            (Int(a), Int(b)) => Int(a * b),
            (a, b) => Mul(Box::new(a), Box::new(b)),
        }
    }

    pub fn cast(self, new: DataType, signed: bool) -> SymExpr {
        match self {
            Int(x) => Int(x.cast(new, signed)),
            s => Cast(Box::new(s), new, signed),
        }
    }
}

/// A symbol value identified by an index.
#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct Symbol(pub DataType, pub usize);
