//! Symbolic boolean expressions.

use std::fmt::{self, Display, Formatter};
use z3::Context as Z3Context;
use z3::ast::{Ast, Bool as Z3Bool};

use crate::helper::{check_compatible, boxed};
use super::{Integer, DataType, SymExpr, Symbol, Traversed};
use super::smt::{Z3Parser, FromAstError};
use SymCondition::*;
use SymExpr::*;


/// A possibly nested symbolic boolean expression.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum SymCondition {
    Bool(bool),
    And(Box<SymCondition>, Box<SymCondition>),
    Or(Box<SymCondition>, Box<SymCondition>),
    Not(Box<SymCondition>),
    Equal(Box<SymExpr>, Box<SymExpr>),
    /// If the bool is true, the operation is signed.
    LessThan(Box<SymExpr>, Box<SymExpr>, bool),
    LessEqual(Box<SymExpr>, Box<SymExpr>, bool),
    GreaterThan(Box<SymExpr>, Box<SymExpr>, bool),
    GreaterEqual(Box<SymExpr>, Box<SymExpr>, bool),
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

macro_rules! forward {
    ($self:expr, $func:ident, $arg:expr) => {
        match $self {
            Bool(_) => {},
            Equal(a, b) => { a.$func($arg); b.$func($arg); },
            LessThan(a, b, _) => { a.$func($arg); b.$func($arg); },
            LessEqual(a, b, _) => { a.$func($arg); b.$func($arg); },
            GreaterThan(a, b, _) => { a.$func($arg); b.$func($arg); },
            GreaterEqual(a, b, _) => { a.$func($arg); b.$func($arg); },
            And(a, b) => { a.$func($arg); b.$func($arg); },
            Or(a, b) => { a.$func($arg); b.$func($arg); },
            Not(a) => a.$func($arg),
        }
    };
}

impl SymCondition {
    pub const TRUE: SymCondition = SymCondition::Bool(true);
    pub const FALSE: SymCondition = SymCondition::Bool(false);

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

            And(a, b) => a.to_z3_ast(ctx).and(&[&b.to_z3_ast(ctx)]),
            Or(a, b) => a.to_z3_ast(ctx).or(&[&b.to_z3_ast(ctx)]),
            Not(a) => a.to_z3_ast(ctx).not(),

            Equal(a, b) => z3_binop!(ctx, a, b, _eq),
            LessThan(a, b, false) => z3_binop!(ctx, a, b, bvult),
            LessEqual(a, b, false) => z3_binop!(ctx, a, b, bvule),
            GreaterThan(a, b, false) => z3_binop!(ctx, a, b, bvugt),
            GreaterEqual(a, b, false) => z3_binop!(ctx, a, b, bvuge),
            LessThan(a, b, true) => z3_binop!(ctx, a, b, bvslt),
            LessEqual(a, b, true) => z3_binop!(ctx, a, b, bvsle),
            GreaterThan(a, b, true) => z3_binop!(ctx, a, b, bvsgt),
            GreaterEqual(a, b, true) => z3_binop!(ctx, a, b, bvsge),
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

    /// Convert this condition into an expression, where `true` is represented by
    /// 1 and `false` by 0.
    pub fn as_expr(self, data_type: DataType) -> SymExpr {
        match self {
            Bool(b) => Int(Integer::from_bool(b, data_type)),
            c => AsExpr(Box::new(c), data_type),
        }
    }

    /// Evaluate the condition with the given values for the symbols.
    pub fn evaluate<S>(&self, symbols: &S) -> bool where S: Fn(Symbol) -> Option<Integer> {
        match self {
            Bool(b) => *b,
            Equal(a, b) => a.evaluate(symbols).equal(b.evaluate(symbols)),
            LessThan(a, b, s) => a.evaluate(symbols).less_than(b.evaluate(symbols), *s),
            LessEqual(a, b, s) => a.evaluate(symbols).less_equal(b.evaluate(symbols), *s),
            GreaterThan(a, b, s) => a.evaluate(symbols).greater_than(b.evaluate(symbols), *s),
            GreaterEqual(a, b, s) => a.evaluate(symbols).greater_equal(b.evaluate(symbols), *s),
            And(a, b) => a.evaluate(symbols) && b.evaluate(symbols),
            Or(a, b) => a.evaluate(symbols) && b.evaluate(symbols),
            Not(a) => !a.evaluate(symbols),
        }
    }

    /// Call a function for every node in the expression/condition tree.
    pub fn traverse<F>(&self, f: &mut F) where F: FnMut(Traversed) {
        f(Traversed::Condition(self));
        forward!(self, traverse, f);
    }

    /// Replace the symbols with new expressions.
    pub fn replace_symbols<S>(&mut self, symbols: &S) where S: Fn(Symbol) -> SymExpr {
        forward!(self, replace_symbols, symbols);
    }
}

impl Display for SymCondition {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use crate::helper::signed_name;
        match self {
            Bool(b) => write!(f, "{}", b),
            Equal(a, b) => write!(f, "({} == {})", a, b),
            LessThan(a, b, s) => write!(f, "({} < {} {})", a, b, signed_name(*s)),
            LessEqual(a, b, s) => write!(f, "({} <= {} {})", a, b, signed_name(*s)),
            GreaterThan(a, b, s) => write!(f, "({} > {} {})", a, b, signed_name(*s)),
            GreaterEqual(a, b, s) => write!(f, "({} >= {} {})", a, b, signed_name(*s)),
            And(a, b) => write!(f, "({} and {})", a, b),
            Or(a, b) => write!(f, "({} or {})", a, b),
            Not(a) => write!(f, "(not {})", a),
        }
    }
}
