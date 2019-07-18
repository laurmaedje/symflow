//! Symbolic microcode execution.

use std::collections::HashMap;
use std::cell::RefCell;
use std::fmt::{self, Debug, Display, Formatter};
use crate::amd64::Register;
use crate::ir::{MicroOperation, Location, Temporary, Condition, Comparison, MemoryMapped};
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
    pub fn step(&mut self, addr: u64, operation: MicroOperation) -> Option<Event> {
        use MicroOperation as Op;

        self.set_reg(Register::RIP, Int(Integer::from_ptr(addr)));
        match operation {
            Op::Mov { dest, src } => self.do_move(dest, src),
            Op::Const { dest, constant } => self.write_location(dest, SymExpr::Int(constant)),
            Op::Cast { target, new, signed } => {
                let new_value = self.get_temp(target).cast(new, signed);
                self.set_temp(Temporary(new, target.1), new_value);
            },

            Op::Add { sum, a, b } => self.do_binop(sum, a, b, SymExpr::add),
            Op::Sub { diff, a, b } => self.do_binop(diff, a, b, SymExpr::sub),
            Op::Mul { prod, a, b } => self.do_binop(prod, a, b, SymExpr::mul),

            Op::And { and, a, b } => self.do_binop(and, a, b, SymExpr::and),
            Op::Or { or, a, b } => self.do_binop(or, a, b, SymExpr::or),
            Op::Not { not, a } => self.set_temp(not, self.get_temp(a).not()),

            Op::Set { target, condition } => {
                self.set_temp(target, self.evaluate(condition, target.0));
            },
            Op::Jump { target, condition, relative } => {
                return Some(Event::Jump {
                    target: self.get_temp(target),
                    condition: self.evaluate(condition, DataType::N8),
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

    /// Do a binary operation.
    fn do_binop<F>(&mut self, target: Temporary, a: Temporary, b: Temporary, binop: F)
    where F: FnOnce(SymExpr, SymExpr) -> SymExpr {
        self.set_temp(target, binop(self.get_temp(a), self.get_temp(b)));
    }

    /// Move a value from a location to another location.
    fn do_move(&mut self, dest: Location, src: Location) {
        assert_eq!(dest.data_type(), src.data_type(),
            "do_move: incompatible data types for move");
        let value = self.read_location(src);

        self.write_location(dest, value);
    }

    /// Emulate a linux syscall.
    fn do_syscall(&mut self, num: u64) -> Option<Event> {
        match num {
            // Read from file descriptor.
            0 => {
                let fd = self.get_reg(Register::RDI);
                let buf = self.get_reg(Register::RSI);
                let count = self.get_reg(Register::RDX);
                let byte_count = match count {
                    Int(Integer(DataType::N64, bytes)) => bytes,
                    _ => panic!("do_syscall: read: unknown byte count"),
                };

                for i in 0 .. byte_count {
                    let target = buf.clone().add(Int(Integer(DataType::N64, i)));
                    let value = Sym(self.memory[0].new_symbol(DataType::N8));
                    self.memory[0].write_expr(target, value);
                }
            },

            // Write to file descriptor.
            1 => {},

            // System exit
            60 => return Some(Event::Exit),
            s => panic!("do_syscall: unimplemented syscall number {}", s),
        }
        None
    }

    /// Evaulate a condition.
    fn evaluate(&self, condition: Condition, data_type: DataType) -> SymExpr {
        use Condition::*;
        use Comparison as Cmp;
        match condition {
            True => Int(Integer(data_type, 1)),

            Equal(Cmp::Sub(a, b)) => self.get_temp(a).equal(self.get_temp(b), data_type),
            Greater(Cmp::Sub(a, b)) => self.get_temp(a).greater(self.get_temp(b), data_type),
            Less(Cmp::Sub(a, b)) => self.get_temp(a).less(self.get_temp(b), data_type),
            LessEqual(Cmp::Sub(a, b)) => self.get_temp(a).less_equal(self.get_temp(b), data_type),

            Equal(Cmp::And(a, b)) => self.get_temp(a).and(self.get_temp(b))
                .equal(Int(Integer(a.0, 0)), data_type),

            _ => panic!("evaluate: unhandled condition/comparison pair"),
        }
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
    Jump { target: SymExpr, condition: SymExpr, relative: bool },
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
        self.read_expr(SymExpr::Int(Integer::from_ptr(addr)), data_type)
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
                let value = SymExpr::Sym(Symbol(data_type, data.symbols));
                data.map.insert(addr, value.clone());
                data.symbols += 1;
                value
            },
        }
    }

    /// Write a value to a direct address.
    pub fn write_direct(&mut self, addr: u64, value: SymExpr) {
        self.write_expr(SymExpr::Int(Integer::from_ptr(addr)), value)
    }

    /// Write a value to a symbolic address.
    pub fn write_expr(&mut self, addr: SymExpr, value: SymExpr) {
        self.data.borrow_mut().map.insert(addr, value);
    }

    /// Generate a new symbol.
    pub fn new_symbol(&mut self, data_type: DataType) -> Symbol {
        let mut data = self.data.borrow_mut();
        data.symbols += 1;
        Symbol(data_type, data.symbols - 1)
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

/// A possibly composed symbolic expression.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum SymExpr {
    Int(Integer),
    Sym(Symbol),
    Add(Box<SymExpr>, Box<SymExpr>),
    Sub(Box<SymExpr>, Box<SymExpr>),
    Mul(Box<SymExpr>, Box<SymExpr>),
    And(Box<SymExpr>, Box<SymExpr>),
    Or(Box<SymExpr>, Box<SymExpr>),
    Not(Box<SymExpr>),
    Equal(Box<SymExpr>, Box<SymExpr>, DataType),
    Less(Box<SymExpr>, Box<SymExpr>, DataType),
    LessEqual(Box<SymExpr>, Box<SymExpr>, DataType),
    Greater(Box<SymExpr>, Box<SymExpr>, DataType),
    GreaterEqual(Box<SymExpr>, Box<SymExpr>, DataType),
    Cast(Box<SymExpr>, DataType, bool),
}

macro_rules! bin_expr {
    ($func:ident, $op:tt, $variant:ident) => {
        pub fn $func(self, other: SymExpr) -> SymExpr {
            match (self, other) {
                (Int(a), Int(b)) => Int(a $op b),
                (a, b) => $variant(Box::new(a), Box::new(b)),
            }
        }
    };
}

macro_rules! bin_expr_simplifying {
    ($func:ident, $a:ident, $b:ident, $target:expr) => {
        pub fn $func(self, other: SymExpr) -> SymExpr {
            fn boxed(expr: SymExpr) -> Box<SymExpr> { Box::new(expr) }
            fn add_or_sub(expr: SymExpr, a: Integer, b: Integer) -> SymExpr {
                if a > b {
                    Add(boxed(expr), boxed(Int(a - b)))
                } else {
                    Sub(boxed(expr), boxed(Int(b - a)))
                }
            }
            let $a = self;
            let $b = other;
            $target
        }
    };
}

macro_rules! cmp_expr {
    ($func:ident, $op:ident, $variant:ident) => {
        pub fn $func(self, other: SymExpr, data_type: DataType) -> SymExpr {
            match (self, other) {
                (Int(a), Int(b)) => Int(Integer::from_bool(a.$op(&b), data_type)),
                (a, b) => $variant(Box::new(a), Box::new(b), data_type),
            }
        }
    };
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
            And(a, _) => a.data_type(),
            Or(a, _) => a.data_type(),
            Not(a) => a.data_type(),
            Cast(_, new, _) => *new,
            Equal(_, _, d) => *d,
            Less(_, _, d) => *d,
            LessEqual(_, _, d) => *d,
            Greater(_, _, d) => *d,
            GreaterEqual(_, _, d) => *d,
        }
    }

    // Add and simplify.
    bin_expr_simplifying!(add, a, b, match (a, b) {
        (a, Int(Integer(_, 0))) | (Int(Integer(_, 0)), a) => a,
        (Int(a), Int(b)) => Int(a + b),
        (Int(a), Add(b, c)) | (Add(b, c), Int(a)) => match (*b, *c) {
            (Int(b), c) | (c, Int(b)) => Add(boxed(c), boxed(Int(a + b))),
            (b, c) => Add(boxed(b), boxed(c)),
        },
        (Int(a), Sub(b, c)) | (Sub(b, c), Int(a)) => match (*b, *c) {
            (b, Int(c)) => add_or_sub(b, a, c),
            (Int(b), c) => Sub(boxed(Int(a + b)), boxed(c)),
            (b, c) => Add(boxed(b), boxed(c)),
        }
        (a, b) => Add(boxed(a), boxed(b)),
    });

    // Subtract and simplify.
    bin_expr_simplifying!(sub, a, b, match (a, b) {
        (a, Int(Integer(_, 0))) | (Int(Integer(_, 0)), a) => a,
        (Int(a), Int(b)) => Int(a - b),
        (Int(a), Sub(b, c)) => match (*b, *c) {
            (Int(b), c) => add_or_sub(c, a, b),
            (b, Int(c)) => Sub(boxed(Int(a + c)), boxed(b)),
            (b, c) => Sub(boxed(b), boxed(c)),
        },
        (Sub(a, b), Int(c)) => match (*a, *b) {
            (Int(a), b) => Sub(boxed(Int(a - c)), boxed(b)),
            (a, Int(b)) => Sub(boxed(a), boxed(Int(b + c))),
            (a, b) => Sub(boxed(a), boxed(b)),
        },
        (Int(a), Add(b, c)) => match (*b, *c) {
            (Int(b), c) => Sub(boxed(Int(a - b)), boxed(c)),
            (b, Int(c)) => Sub(boxed(Int(a + c)), boxed(b)),
            (b, c) => Sub(boxed(b), boxed(c)),
        },
        (Add(a, b), Int(c)) => match (*a, *b) {
            (Int(a), b) | (b, Int(a)) => add_or_sub(b, a, c),
            (a, b) => Sub(boxed(a), boxed(b)),
        }
        (a, b) => Sub(boxed(a), boxed(b)),
    });

    bin_expr!(mul, *, Mul);
    bin_expr!(and, &, And);
    bin_expr!(or, *, Or);

    pub fn not(self) -> SymExpr {
        match self {
            Int(x) => Int(!x),
            x => Not(Box::new(x)),
        }
    }

    cmp_expr!(equal, eq, Equal);
    cmp_expr!(less, lt, Less);
    cmp_expr!(less_equal, le, LessEqual);
    cmp_expr!(greater, gt, Greater);
    cmp_expr!(greater_equal, ge, GreaterEqual);

    pub fn cast(self, new: DataType, signed: bool) -> SymExpr {
        match self {
            Int(x) => Int(x.cast(new, signed)),
            s => Cast(Box::new(s), new, signed),
        }
    }
}

impl Display for SymExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Int(int) => write!(f, "{}", int),
            Sym(sym) => write!(f, "{}", sym),
            Add(a, b) => write!(f, "({} + {})", a, b),
            Sub(a, b) => write!(f, "({} - {})", a, b),
            Mul(a, b) => write!(f, "({} * {})", a, b),
            And(a, b) => write!(f, "({} & {})", a, b),
            Or(a, b) => write!(f, "({} | {})", a, b),
            Not(a) => write!(f, "(!{})", a),
            Equal(a, b, d) => write!(f, "({} == {} [{}])", a, b, d),
            Less(a, b, d) => write!(f, "({} < {} [{}])", a, b, d),
            LessEqual(a, b, d) => write!(f, "({} <= {} [{}])", a, b, d),
            Greater(a, b, d) => write!(f, "({} > {} [{}])", a, b, d),
            GreaterEqual(a, b, d) => write!(f, "({} >= {} [{}])", a, b, d),
            Cast(x, new, signed) => write!(f, "({} as {}{})", x, new,
                if *signed { " signed "} else { "" }),
        }
    }
}

/// A symbol value identified by an index.
#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct Symbol(pub DataType, pub usize);

impl Display for Symbol {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "s{}:{}", self.1, self.0)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::num::{Integer, DataType::*};

    #[test]
    fn expr() {
        let zero = || Int(Integer(N64, 0));
        let five = || Int(Integer(N64, 5));
        let eight = || Int(Integer(N64, 8));
        let ten = || Int(Integer(N64, 10));
        let fifteen  = || Int(Integer(N64, 15));
        let x = || Sym(Symbol(N64, 0));

        assert_eq!(x().add(zero()), x());
        assert_eq!(ten().add(zero()), ten());

        assert_eq!(x().add(five()).add(ten()),
            Add(Box::new(x()), Box::new(fifteen())));

        assert_eq!(x().sub(five()).add(ten()),
            Add(Box::new(x()), Box::new(five())));

        assert_eq!(x().sub(ten()).sub(five()),
            Sub(Box::new(x()), Box::new(fifteen())));

        assert_eq!(x().add(ten()).sub(five()),
            Add(Box::new(x()), Box::new(five())));

        assert_eq!(x().sub(eight()).sub(eight()).add(eight()),
            Sub(Box::new(x()), Box::new(Int(Integer(N64, 8)))));
    }
}
