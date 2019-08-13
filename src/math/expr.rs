//! Symbolic integer expressions.

use std::fmt::{self, Display, Formatter};
use z3::Context as Z3Context;
use z3::ast::{BV as Z3BitVec};

use crate::helper::{check_compatible, boxed};
use super::{Integer, DataType, Symbol, SymCondition, Traversed};
use super::smt::{Z3Parser, FromAstError};
use SymExpr::*;
use SymCondition::*;


/// A possibly nested symbolic machine integer expression.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum SymExpr {
    Int(Integer),
    Sym(Symbol),
    Add(Box<SymExpr>, Box<SymExpr>),
    Sub(Box<SymExpr>, Box<SymExpr>),
    Mul(Box<SymExpr>, Box<SymExpr>),
    BitAnd(Box<SymExpr>, Box<SymExpr>),
    BitOr(Box<SymExpr>, Box<SymExpr>),
    BitNot(Box<SymExpr>),
    Cast(Box<SymExpr>, DataType, bool),
    AsExpr(Box<SymCondition>, DataType),
    IfThenElse(Box<SymCondition>, Box<SymExpr>, Box<SymExpr>),
}

macro_rules! bin_expr {
    ($func:ident,$variant:ident) => {
        pub fn $func(self, other: SymExpr) -> SymExpr {
            check_compatible(self.data_type(), other.data_type(), "operation");
            match (self, other) {
                (Int(a), Int(b)) => Int(a.$func(b)),
                (a, b) => $variant(Box::new(a), Box::new(b)),
            }
        }
    };
}

macro_rules! bin_expr_simplifying {
    ($func:ident, $a:ident, $b:ident, $target:expr) => {
        pub fn $func(self, other: SymExpr) -> SymExpr {
            check_compatible(self.data_type(), other.data_type(), "operation");
            fn add_or_sub(expr: SymExpr, a: Integer, b: Integer) -> SymExpr {
                if a.flagged_sub(b).1.sign {
                    Sub(boxed(expr), boxed(Int(b.sub(a))))
                } else {
                    Add(boxed(expr), boxed(Int(a.sub(b))))
                }
            }
            let $a = self;
            let $b = other;
            $target
        }
    };
}

macro_rules! cmp_signed {
    ($func:ident, $variant:ident) => {
        pub fn $func(self, other: SymExpr, signed: bool) -> SymCondition {
            check_compatible(self.data_type(), other.data_type(), "comparison");
            match (self, other) {
                (Int(a), Int(b)) => Bool(a.$func(b, signed)),
                (a, b) => $variant(Box::new(a), Box::new(b), signed),
            }
        }
    };
}

macro_rules! forward {
    ($self:expr, $func:ident, $arg:expr) => {
        match $self {
            Int(_) => {},
            Sym(_) => {},
            Add(a, b) => { a.$func($arg); b.$func($arg); },
            Sub(a, b) => { a.$func($arg); b.$func($arg); },
            Mul(a, b) => { a.$func($arg); b.$func($arg); },
            BitAnd(a, b) => { a.$func($arg); b.$func($arg); },
            BitOr(a, b) => { a.$func($arg); b.$func($arg); },
            BitNot(a) => a.$func($arg),
            Cast(a, _, _) => a.$func($arg),
            AsExpr(a, _) => a.$func($arg),
            IfThenElse(c, a, b) => { a.$func($arg); b.$func($arg); c.$func($arg); },
        }
    };
}

impl SymExpr {
    /// Create a new integer expression.
    pub fn from_int(data_type: DataType, value: u64) -> SymExpr {
        SymExpr::Int(Integer(data_type, value))
    }

    /// Create a new pointer-sized integer expression.
    pub fn from_ptr(value: u64) -> SymExpr {
        SymExpr::Int(Integer::from_ptr(value))
    }

    /// Convert the Z3-solver Ast into an expression if possible.
    pub fn from_z3_ast(ast: &Z3BitVec) -> Result<SymExpr, FromAstError> {
        let repr = ast.to_string();
        let mut parser = Z3Parser::new(&repr);
        parser.parse_expr()
    }

    /// Convert this expression into a Z3-solver Ast.
    pub fn to_z3_ast<'ctx>(&self, ctx: &'ctx Z3Context) -> Z3BitVec<'ctx> {
        match self {
            Int(int) => Z3BitVec::from_u64(ctx, int.1, int.0.bits() as u32),
            Sym(sym) => Z3BitVec::new_const(ctx, sym.to_string(), sym.0.bits() as u32),

            Add(a, b) => z3_binop!(ctx, a, b, bvadd),
            Sub(a, b) => z3_binop!(ctx, a, b, bvsub),
            Mul(a, b) => z3_binop!(ctx, a, b, bvmul),
            BitAnd(a, b) => z3_binop!(ctx, a, b, bvand),
            BitOr(a, b) => z3_binop!(ctx, a, b, bvor),
            BitNot(a) => a.to_z3_ast(ctx).bvnot(),

            Cast(x, new, signed) => {
                let x_ast = x.to_z3_ast(ctx);
                let src_len = x.data_type().bits() as u32;
                let dest_len = new.bits() as u32;

                if src_len < dest_len {
                    let extra_bits = dest_len - src_len;
                    if *signed {
                        x_ast.sign_ext(extra_bits)
                    } else {
                        x_ast.zero_ext(extra_bits)
                    }
                } else if src_len > dest_len {
                    x_ast.extract(dest_len, 0)
                } else {
                    x_ast
                }
            },
            AsExpr(x, new) => {
                x.to_z3_ast(ctx).ite(
                    &Z3BitVec::from_u64(ctx, 1, new.bits() as u32),
                    &Z3BitVec::from_u64(ctx, 0, new.bits() as u32),
                )
            },
            IfThenElse(c, a, b) => c.to_z3_ast(ctx).ite(&a.to_z3_ast(ctx), &b.to_z3_ast(ctx)),
        }
    }

    // Add and simplify.
    bin_expr_simplifying!(add, a, b, match (a, b) {
        (a, Int(Integer(_, 0))) | (Int(Integer(_, 0)), a) => a,
        (Int(a), Int(b)) => Int(a.add(b)),
        (Int(a), Add(b, c)) | (Add(b, c), Int(a)) => match (*b, *c) {
            (Int(b), c) | (c, Int(b)) => Add(boxed(c), boxed(Int(a.add(b)))),
            (b, c) => Add(boxed(Int(a)), boxed(Add(boxed(b), boxed(c)))),
        },
        (Int(a), Sub(b, c)) | (Sub(b, c), Int(a)) => match (*b, *c) {
            (b, Int(c)) => add_or_sub(b, a, c),
            (Int(b), c) => Sub(boxed(Int(a.add(b))), boxed(c)),
            (b, c) => Add(boxed(Int(a)), boxed(Sub(boxed(b), boxed(c))))
        }
        (a, b) => Add(boxed(a), boxed(b)),
    });

    // Subtract and simplify.
    bin_expr_simplifying!(sub, a, b, match (a, b) {
        (a, Int(Integer(_, 0))) | (Int(Integer(_, 0)), a) => a,
        (Int(a), Int(b)) => Int(a.sub(b)),
        (Int(a), Sub(b, c)) => match (*b, *c) {
            (Int(b), c) => add_or_sub(c, a, b),
            (b, Int(c)) => Sub(boxed(Int(a.add(c))), boxed(b)),
            (b, c) => Sub(boxed(Int(a)), boxed(Sub(boxed(b), boxed(c)))),
        },
        (Sub(a, b), Int(c)) => match (*a, *b) {
            (Int(a), b) => Sub(boxed(Int(a.sub(c))), boxed(b)),
            (a, Int(b)) => Sub(boxed(a), boxed(Int(b.add(c)))),
            (a, b) => Sub(boxed(Sub(boxed(a), boxed(b))), boxed(Int(c))),
        },
        (Int(a), Add(b, c)) => match (*b, *c) {
            (Int(b), c) => Sub(boxed(Int(a.sub(b))), boxed(c)),
            (b, Int(c)) => Sub(boxed(Int(a.add(c))), boxed(b)),
            (b, c) => Sub(boxed(Int(a)), boxed(Add(boxed(b), boxed(c)))),
        },
        (Add(a, b), Int(c)) => match (*a, *b) {
            (Int(a), b) | (b, Int(a)) => add_or_sub(b, a, c),
            (a, b) => Sub(boxed(Add(boxed(a), boxed(b))), boxed(Int(c))),
        }
        (a, b) => Sub(boxed(a), boxed(b)),
    });

    bin_expr!(mul, Mul);
    bin_expr!(bitand, BitAnd);
    bin_expr!(bitor, BitOr);

    pub fn bitnot(self) -> SymExpr {
        match self {
            Int(x) => Int(x.bitnot()),
            x => BitNot(Box::new(x)),
        }
    }

    pub fn equal(self, other: SymExpr) -> SymCondition {
        check_compatible(self.data_type(), other.data_type(), "comparison");
        match (self, other) {
            (Int(a), Int(b)) => Bool(a.equal(b)),
            (a, b) => Equal(Box::new(a), Box::new(b)),
        }
    }

    cmp_signed!(less_than, LessThan);
    cmp_signed!(less_equal, LessEqual);
    cmp_signed!(greater_than, GreaterThan);
    cmp_signed!(greater_equal, GreaterEqual);

    pub fn cast(self, new: DataType, signed: bool) -> SymExpr {
        match self {
            Int(x) => Int(x.cast(new, signed)),
            Cast(x, t, false) => {
                if x.data_type() == new {
                    *x
                } else if t.bytes() < new.bytes() {
                    Cast(x, new, false)
                } else {
                    Cast(boxed(Cast(x, t, false)), new, signed)
                }
            },
            s => if s.data_type() == new { s } else { Cast(boxed(s), new, signed) },
        }
    }

    /// The data type of the expression.
    pub fn data_type(&self) -> DataType {
        match self {
            Int(int) => int.0,
            Sym(sym) => sym.0,
            Add(a, _)    => a.data_type(),
            Sub(a, _)    => a.data_type(),
            Mul(a, _)    => a.data_type(),
            BitAnd(a, _) => a.data_type(),
            BitOr(a, _)  => a.data_type(),
            BitNot(a)    => a.data_type(),
            Cast(_, new, _) => *new,
            AsExpr(_, new)  => *new,
            IfThenElse(_, a, _) => a.data_type(),
        }
    }

    /// Evaluate the expression with the given values for the symbols.
    pub fn evaluate<S>(&self, symbols: &S) -> Integer where S: Fn(Symbol) -> Option<Integer> {
        match self {
            Int(int) => *int,
            Sym(sym) => symbols(*sym).unwrap_or_else(|| {
                panic!("evaluate: missing symbol: {}", sym);
            }),
            Add(a, b) => a.evaluate(symbols).add(b.evaluate(symbols)),
            Sub(a, b) => a.evaluate(symbols).sub(b.evaluate(symbols)),
            Mul(a, b) => a.evaluate(symbols).mul(b.evaluate(symbols)),
            BitAnd(a, b) => a.evaluate(symbols).bitand(b.evaluate(symbols)),
            BitOr(a, b) => a.evaluate(symbols).bitor(b.evaluate(symbols)),
            BitNot(a) => a.evaluate(symbols).bitnot(),
            Cast(a, data_type, signed) => a.evaluate(symbols).cast(*data_type, *signed),
            AsExpr(a, data_type) => Integer::from_bool(a.evaluate(symbols), *data_type),
            IfThenElse(c, a, b) => if c.evaluate(symbols) {
                a.evaluate(symbols)
            } else {
                b.evaluate(symbols)
            }
        }
    }

    /// Call a function for every node in the expression/condition tree.
    pub fn traverse<F>(&self, f: &mut F) where F: FnMut(Traversed) {
        f(Traversed::Expr(self));
        forward!(self, traverse, f);
    }

    /// Replace the symbols with new expressions.
    pub fn replace_symbols<S>(&mut self, symbols: &S) where S: Fn(Symbol) -> SymExpr {
        match self {
            Sym(sym) => *self = symbols(*sym),
            s => forward!(s, replace_symbols, symbols),
        }
    }
}

impl Display for SymExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use crate::helper::signed_name;

        match self {
            Int(int) => write!(f, "{}", int),
            Sym(sym) => write!(f, "{}", sym),
            Add(a, b) => write!(f, "({} + {})", a, b),
            Sub(a, b) => write!(f, "({} - {})", a, b),
            Mul(a, b) => write!(f, "({} * {})", a, b),
            BitAnd(a, b) => write!(f, "({} & {})", a, b),
            BitOr(a, b) => write!(f, "({} | {})", a, b),
            BitNot(a) => write!(f, "(!{})", a),
            Cast(x, new, signed) => write!(f, "({} as {} {})", x, new, signed_name(*signed)),
            AsExpr(c, data_type) => write!(f, "({} as {})", c, data_type),
            IfThenElse(c, a, b) => write!(f, "if {} then {} else {}", c, a, b),
        }
    }
}
