//! Machine numbers and symbolic expressions.

#![macro_use]

use std::fmt::{self, Display, Formatter};

macro_rules! z3_binop {
    ($ctx:expr, $a:expr, $b:expr, $op:ident) => {
        $a.to_z3_ast($ctx).$op(&$b.to_z3_ast($ctx))
    };
}

mod num;
mod expr;
mod cond;
mod smt;

pub use num::*;
pub use expr::*;
pub use cond::*;
pub use smt::{Solver, SharedSolver, FromAstError};


/// A dynamically typed symbolic value.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum SymDynamic {
    Expr(SymExpr),
    Condition(SymCondition),
}

impl From<SymExpr> for SymDynamic {
    fn from(expr: SymExpr) -> SymDynamic { SymDynamic::Expr(expr) }
}

impl From<SymCondition> for SymDynamic {
    fn from(cond: SymCondition) -> SymDynamic { SymDynamic::Condition(cond) }
}

/// A symbol value identified by an index.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Symbol(pub DataType, pub &'static str, pub usize);

impl Display for Symbol {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}{}:{}", self.1, self.2, self.0)
    }
}

/// A reference to an expression or condition node in the traversed tree.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Traversed<'a> {
    Expr(&'a SymExpr),
    Condition(&'a SymCondition),
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::helper::boxed;
    use crate::math::Integer;
    use crate::math::DataType::*;
    use SymExpr::*;

    fn n(x: u64) -> SymExpr { Int(Integer(N64, x)) }
    fn x() -> SymExpr { Sym(Symbol(N64, "stdin", 0)) }
    fn y() -> SymExpr { Sym(Symbol(N8, "stdin", 1)) }

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
        let solver = Solver::new();

        let expr = n(10).add(x()).add(x()).add(n(5));
        assert_eq!(solver.simplify_expr(&expr), n(15).add(n(2).mul(x())));
    }
}
