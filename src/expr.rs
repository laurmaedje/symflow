//! Symbolic expressions and calculations.

use std::fmt::{self, Display, Formatter};
use z3::Context as Z3Context;
use z3::ast::{Ast, BV as Z3BitVec, Bool as Z3Bool};

use crate::num::{Integer, DataType};
use crate::smt::{Z3Parser, FromAstError};
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

/// A possibly nested symbolic boolean expression.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum SymCondition {
    Bool(bool),
    Equal(Box<SymExpr>, Box<SymExpr>),
    /// If the bool is true, the operation is signed.
    LessThan(Box<SymExpr>, Box<SymExpr>, bool),
    LessEqual(Box<SymExpr>, Box<SymExpr>, bool),
    GreaterThan(Box<SymExpr>, Box<SymExpr>, bool),
    GreaterEqual(Box<SymExpr>, Box<SymExpr>, bool),
    And(Box<SymCondition>, Box<SymCondition>),
    Or(Box<SymCondition>, Box<SymCondition>),
    Not(Box<SymCondition>),
}

/// A dynamically typed symbolic value.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum SymDynamic {
    Expr(SymExpr),
    Condition(SymCondition),
}

/// A symbol value identified by an index.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Symbol(pub DataType, pub &'static str, pub usize);

/// Make sure operations only happen on same expressions.
fn check_compatible(a: DataType, b: DataType, operation: &str) {
    assert_eq!(a, b, "incompatible data types for symbolic {}", operation);
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

macro_rules! cmp_maybe_signed {
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

macro_rules! contains {
    ($symbol:expr, $($exprs:expr),*) => {
        $($exprs.contains_symbol($symbol) ||)* false
    };
}

macro_rules! z3_binop {
    ($ctx:expr, $a:expr, $b:expr, $op:ident) => { $a.to_z3_ast($ctx).$op(&$b.to_z3_ast($ctx)) };
}


impl SymExpr {
    /// Create a new integer expression.
    pub fn int(data_type: DataType, value: u64) -> SymExpr {
        SymExpr::Int(Integer(data_type, value))
    }

    /// Create a new pointer-sized integer expression.
    pub fn from_ptr(value: u64) -> SymExpr {
        SymExpr::Int(Integer::from_ptr(value))
    }

    /// Evaluate the condition with the given values for symbols.
    pub fn evaluate_with<S>(&self, symbols: S) -> Integer
    where S: Fn(Symbol) -> Option<Integer> {
        self.evaluate_inner(&symbols)
    }

    fn evaluate_inner<S>(&self, symbols: &S) -> Integer
    where S: Fn(Symbol) -> Option<Integer> {
        match self {
            Int(int) => *int,
            Sym(sym) => symbols(*sym)
                .unwrap_or_else(|| panic!("evaluate_inner: missing symbol: {}", sym)),
            Add(a, b)    => a.evaluate_inner(symbols).add(b.evaluate_inner(symbols)),
            Sub(a, b)    => a.evaluate_inner(symbols).sub(b.evaluate_inner(symbols)),
            Mul(a, b)    => a.evaluate_inner(symbols).mul(b.evaluate_inner(symbols)),
            BitAnd(a, b) => a.evaluate_inner(symbols).bitand(b.evaluate_inner(symbols)),
            BitOr(a, b)  => a.evaluate_inner(symbols).bitor(b.evaluate_inner(symbols)),
            BitNot(a)    => a.evaluate_inner(symbols).bitnot(),
            Cast(a, data_type, signed) => a.evaluate_inner(symbols).cast(*data_type, *signed),
            AsExpr(a, data_type)  => Integer::from_bool(a.evaluate_inner(symbols), *data_type),
            IfThenElse(c, a, b) => if c.evaluate_inner(symbols) {
                a.evaluate_inner(symbols)
            } else {
                b.evaluate_inner(symbols)
            }
        }
    }

    /// Whether the given symbol appears somewhere in this tree.
    pub fn contains_symbol(&self, symbol: Symbol) -> bool {
        match self {
            Int(_) => false,
            Sym(sym) => *sym == symbol,
            Add(a, b)    => contains!(symbol, a, b),
            Sub(a, b)    => contains!(symbol, a, b),
            Mul(a, b)    => contains!(symbol, a, b),
            BitAnd(a, b) => contains!(symbol, a, b),
            BitOr(a, b)  => contains!(symbol, a, b),
            BitNot(a)    => contains!(symbol, a),
            Cast(a, _, _) => contains!(symbol, a),
            AsExpr(a, _)  => contains!(symbol, a),
            IfThenElse(c, a, b) => contains!(symbol, c, a, b),
        }
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
            BitOr(a, b)  => z3_binop!(ctx, a, b, bvor),
            BitNot(a)    => a.to_z3_ast(ctx).bvnot(),

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
            (Int(a), Int(b)) => Bool(a.equal(&b)),
            (a, b) => Equal(Box::new(a), Box::new(b)),
        }
    }

    cmp_maybe_signed!(less_than, LessThan);
    cmp_maybe_signed!(less_equal, LessEqual);
    cmp_maybe_signed!(greater_than, GreaterThan);
    cmp_maybe_signed!(greater_equal, GreaterEqual);
}

macro_rules! bin_cond  {
    ($func:ident, $op:tt, $variant:ident) => {
        pub fn $func(self, other: SymCondition) -> SymCondition {
            match (self, other) {
                (Bool(a), Bool(b)) => Bool(a $op b),
                (a, b) => $variant(Box::new(a), Box::new(b)),
            }
        }
    };
}

impl SymCondition {
    pub const TRUE: SymCondition = SymCondition::Bool(true);
    pub const FALSE: SymCondition = SymCondition::Bool(false);

    /// Evaluate the condition with the given values for symbols.
    pub fn evaluate_with<S>(&self, symbols: S) -> bool
    where S: Fn(Symbol) -> Option<Integer> {
        self.evaluate_inner(&symbols)
    }

    fn evaluate_inner<S>(&self, symbols: &S) -> bool
    where S: Fn(Symbol) -> Option<Integer> {
        match self {
            Bool(b) => *b,
            Equal(a, b) => a.evaluate_inner(symbols) == b.evaluate_inner(symbols),
            LessThan(a, b, s)     => a.evaluate_inner(symbols).less_than(b.evaluate_inner(symbols), *s),
            LessEqual(a, b, s)    => a.evaluate_inner(symbols).less_equal(b.evaluate_inner(symbols), *s),
            GreaterThan(a, b, s)  => a.evaluate_inner(symbols).greater_than(b.evaluate_inner(symbols), *s),
            GreaterEqual(a, b, s) => a.evaluate_inner(symbols).greater_equal(b.evaluate_inner(symbols), *s),
            And(a, b) => a.evaluate_inner(symbols) && b.evaluate_inner(symbols),
            Or(a, b)  => a.evaluate_inner(symbols) && b.evaluate_inner(symbols),
            Not(a)    => !a.evaluate_inner(symbols),
        }
    }

    /// Whether the given symbol appears somewhere in this tree.
    pub fn contains_symbol(&self, symbol: Symbol) -> bool {
        match self {
            Bool(_) => false,
            Equal(a, b) => contains!(symbol, a, b),
            LessThan(a, b, _)     => contains!(symbol, a, b),
            LessEqual(a, b, _)    => contains!(symbol, a, b),
            GreaterThan(a, b, _)  => contains!(symbol, a, b),
            GreaterEqual(a, b, _) => contains!(symbol, a, b),
            And(a, b) => contains!(symbol, a, b),
            Or(a, b)  => contains!(symbol, a, b),
            Not(a)    => contains!(symbol, a),
        }
    }

    /// Convert the Z3-solver Ast into an expression if possible.
    pub fn from_z3_ast(ast: &Z3Bool) -> Result<SymCondition, FromAstError> {
        let repr = ast.to_string();
        let mut parser = Z3Parser::new(&repr);
        parser.parse_condition()
    }

    /// Convert this condition into a Z3-solver Ast.
    pub fn to_z3_ast<'ctx>(&self, ctx: &'ctx Z3Context) -> Z3Bool<'ctx> {
        match self {
            Bool(b) => Z3Bool::from_bool(ctx, *b),

            Equal(a, b) => z3_binop!(ctx, a, b, _eq),

            LessThan(a, b, false)         => z3_binop!(ctx, a, b, bvult),
            LessEqual(a, b, false)    => z3_binop!(ctx, a, b, bvule),
            GreaterThan(a, b, false)      => z3_binop!(ctx, a, b, bvugt),
            GreaterEqual(a, b, false) => z3_binop!(ctx, a, b, bvuge),

            LessThan(a, b, true)          => z3_binop!(ctx, a, b, bvslt),
            LessEqual(a, b, true)     => z3_binop!(ctx, a, b, bvsle),
            GreaterThan(a, b, true)       => z3_binop!(ctx, a, b, bvsgt),
            GreaterEqual(a, b, true)  => z3_binop!(ctx, a, b, bvsge),

            And(a, b) => a.to_z3_ast(ctx).and(&[&b.to_z3_ast(ctx)]),
            Or(a, b)  => a.to_z3_ast(ctx).or(&[&b.to_z3_ast(ctx)]),
            Not(a)    => a.to_z3_ast(ctx).not(),
        }
    }

    /// Convert this condition into an expression, where `true` is represented by
    /// 1 and `false` by 0.
    pub fn as_expr(self, data_type: DataType) -> SymExpr {
        match self {
            Bool(b) => Int(Integer::from_bool(b, data_type)),
            c => AsExpr(Box::new(c), data_type),
        }
    }

    bin_cond!(and, &&, And);
    bin_cond!(or, ||, Or);

    pub fn not(self) -> SymCondition {
        match self {
            Bool(x) => Bool(!x),
            x => Not(Box::new(x)),
        }
    }

    pub fn if_then_else(self, a: SymExpr, b: SymExpr) -> SymExpr {
        check_compatible(a.data_type(), b.data_type(), "if-then-else");
        IfThenElse(boxed(self), boxed(a), boxed(b))
    }
}

impl From<SymExpr> for SymDynamic {
    fn from(expr: SymExpr) -> SymDynamic { SymDynamic::Expr(expr) }
}

impl From<SymCondition> for SymDynamic {
    fn from(cond: SymCondition) -> SymDynamic { SymDynamic::Condition(cond) }
}

impl Display for SymExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Int(int) => write!(f, "{}", int),
            Sym(sym) => write!(f, "{}", sym),
            Add(a, b) => write!(f, "({} + {})", a, b),
            Sub(a, b) => write!(f, "({} - {})", a, b),
            Mul(a, b) => write!(f, "({} * {})", a, b),
            BitAnd(a, b) => write!(f, "({} & {})", a, b),
            BitOr(a, b) => write!(f, "({} | {})", a, b),
            BitNot(a) => write!(f, "(!{})", a),
            Cast(x, new, signed) => write!(f, "({} as {}{})", x, new,
                if *signed { " signed"} else { "" }),
            AsExpr(c, data_type) => write!(f, "({} as {})", c, data_type),
            IfThenElse(c, a, b) => write!(f, "if {} then {} else {}", c, a, b),
        }
    }
}

impl Display for SymCondition {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        fn signed(s: bool) -> &'static str { if s { "signed" } else { "unsigned" } }
        match self {
            Bool(b) => write!(f, "{}", b),
            Equal(a, b) => write!(f, "({} == {})", a, b),
            LessThan(a, b, s) => write!(f, "({} < {} {})", a, b, signed(*s)),
            LessEqual(a, b, s) => write!(f, "({} <= {} {})", a, b, signed(*s)),
            GreaterThan(a, b, s) => write!(f, "({} > {} {})", a, b, signed(*s)),
            GreaterEqual(a, b, s) => write!(f, "({} >= {} {})", a, b, signed(*s)),
            And(a, b) => write!(f, "({} and {})", a, b),
            Or(a, b) => write!(f, "({} or {})", a, b),
            Not(a) => write!(f, "(not {})", a),
        }
    }
}

impl Display for Symbol {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}-{}:{}", self.1, self.2, self.0)
    }
}

fn boxed<T>(value: T) -> Box<T> { Box::new(value) }


#[cfg(test)]
mod tests {
    use super::*;
    use crate::num::Integer;
    use crate::num::DataType::*;

    fn n(x: u64) -> SymExpr { Int(Integer(N64, x)) }
    fn x() -> SymExpr { Sym(Symbol(N64, "stdin", 0)) }
    fn y() -> SymExpr { Sym(Symbol(N8, "stdin", 1)) }
    fn z() -> SymExpr { Sym(Symbol(N64, "stdin", 1)) }

    #[test]
    fn calculations() {
        assert_eq!(x().add(n(0)), x());
        assert_eq!(n(10).add(n(0)), n(10));
        assert_eq!(x().add(n(5)).add(n(10)), Add(boxed(x()), boxed(n(15))));
        assert_eq!(x().sub(n(5)).add(n(10)), Add(boxed(x()), boxed(n(5))));
        assert_eq!(x().sub(n(10)).sub(n(5)), Sub(boxed(x()), boxed(n(15))));
        assert_eq!(x().add(n(10)).sub(n(5)), Add(boxed(x()), boxed(n(5))));
        assert_eq!(x().sub(n(8)).sub(n(8)).add(n(8)), Sub(boxed(x()), boxed(n(8))));

        assert_ne!(n(10).add(x()).add(x()).add(n(5)), n(10).add(x()).add(x()));

        assert_eq!(y().cast(N32, false).cast(N8, false), y());
        assert_eq!(y().cast(N32, false).cast(N64, false), y().cast(N64, false));
        assert_eq!(y().cast(N8, false), y());
    }

    #[test]
    fn ast() {
        let config = z3::Config::new();
        let ctx = Z3Context::new(&config);

        let expr = n(10).add(x()).add(x()).add(n(5));
        let ast = expr.to_z3_ast(&ctx);
        let simple_ast = ast.simplify();
        let simple_expr = SymExpr::from_z3_ast(&simple_ast).unwrap();

        assert_eq!(simple_expr, n(15).add(n(2).mul(x())));

        let expr = n(10).add(x()).add(n(20)).bitor(z()).sub(n(3)).mul(n(40));
        assert_eq!(expr, SymExpr::from_z3_ast(&expr.to_z3_ast(&ctx)).unwrap());
    }
}
