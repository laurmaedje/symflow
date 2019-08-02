//! Symbolic expressions and calculations.

use std::fmt::{self, Debug, Display, Formatter};
use z3::Context as Z3Context;
use z3::ast::{self, Ast, BV as Z3BitVec, Bool as Z3Bool};

use crate::num::{Integer, DataType, DataType::*};
use self::{SymExpr::*, SymCondition::*};


/// A possibly nested symbolic machine integer expression.
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
    Cast(Box<SymExpr>, DataType, bool),
    AsExpr(Box<SymCondition>, DataType),
}

/// A possibly nested symbolic boolean expression.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum SymCondition {
    Bool(bool),
    Equal(Box<SymExpr>, Box<SymExpr>),
    Less(Box<SymExpr>, Box<SymExpr>),
    LessEqual(Box<SymExpr>, Box<SymExpr>),
    Greater(Box<SymExpr>, Box<SymExpr>),
    GreaterEqual(Box<SymExpr>, Box<SymExpr>),
}

/// A symbol value identified by an index.
#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct Symbol(pub DataType, pub usize, pub usize);

/// Make sure operations only happen on same expressions.
fn check_compatible(a: DataType, b: DataType, operation: &str) {
    assert_eq!(a, b, "incompatible data types for symbolic {}", operation);
}

macro_rules! bin_expr {
    ($func:ident, $op:tt, $variant:ident) => {
        pub fn $func(self, other: SymExpr) -> SymExpr {
            check_compatible(self.data_type(), other.data_type(), "operation");
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
            check_compatible(self.data_type(), other.data_type(), "operation");
            fn add_or_sub(expr: SymExpr, a: Integer, b: Integer) -> SymExpr {
                if a.flagged_sub(b).1.sign {
                    Sub(boxed(expr), boxed(Int(b - a)))
                } else {
                    Add(boxed(expr), boxed(Int(a - b)))
                }
            }
            let $a = self;
            let $b = other;
            $target
        }
    };
}

macro_rules! z3_binop {
    ($ctx:expr, $a:expr, $b:expr, $op:ident) => { $a.to_z3_ast($ctx).$op(&$b.to_z3_ast($ctx)) };
}

macro_rules! cmp_expr {
    ($func:ident, $op:ident, $variant:ident) => {
        pub fn $func(self, other: SymExpr) -> SymCondition {
            check_compatible(self.data_type(), other.data_type(), "comparison");
            match (self, other) {
                (Int(a), Int(b)) => Bool(a.$op(&b)),
                (a, b) => $variant(Box::new(a), Box::new(b)),
            }
        }
    };
}

impl SymExpr {
    /// The data type of the expression if it is an integer type.
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
            AsExpr(_, new) => *new,
        }
    }

    // Add and simplify.
    bin_expr_simplifying!(add, a, b, match (a, b) {
        (a, Int(Integer(_, 0))) | (Int(Integer(_, 0)), a) => a,
        (Int(a), Int(b)) => Int(a + b),
        (Int(a), Add(b, c)) | (Add(b, c), Int(a)) => match (*b, *c) {
            (Int(b), c) | (c, Int(b)) => Add(boxed(c), boxed(Int(a + b))),
            (b, c) => Add(boxed(Int(a)), boxed(Add(boxed(b), boxed(c)))),
        },
        (Int(a), Sub(b, c)) | (Sub(b, c), Int(a)) => match (*b, *c) {
            (b, Int(c)) => add_or_sub(b, a, c),
            (Int(b), c) => Sub(boxed(Int(a + b)), boxed(c)),
            (b, c) => Add(boxed(Int(a)), boxed(Sub(boxed(b), boxed(c))))
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
            (b, c) => Sub(boxed(Int(a)), boxed(Sub(boxed(b), boxed(c)))),
        },
        (Sub(a, b), Int(c)) => match (*a, *b) {
            (Int(a), b) => Sub(boxed(Int(a - c)), boxed(b)),
            (a, Int(b)) => Sub(boxed(a), boxed(Int(b + c))),
            (a, b) => Sub(boxed(Sub(boxed(a), boxed(b))), boxed(Int(c))),
        },
        (Int(a), Add(b, c)) => match (*b, *c) {
            (Int(b), c) => Sub(boxed(Int(a - b)), boxed(c)),
            (b, Int(c)) => Sub(boxed(Int(a + c)), boxed(b)),
            (b, c) => Sub(boxed(Int(a)), boxed(Sub(boxed(b), boxed(c)))),
        },
        (Add(a, b), Int(c)) => match (*a, *b) {
            (Int(a), b) | (b, Int(a)) => add_or_sub(b, a, c),
            (a, b) => Sub(boxed(Sub(boxed(a), boxed(b))), boxed(Int(c))),
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

    cmp_expr!(equal, eq, Equal);
    cmp_expr!(less, lt, Less);
    cmp_expr!(less_equal, le, LessEqual);
    cmp_expr!(greater, gt, Greater);
    cmp_expr!(greater_equal, ge, GreaterEqual);

    /// Convert the Z3-solver Ast into an expression if possible.
    pub fn from_z3_ast(ast: &Z3BitVec) -> Result<SymExpr, FromAstError> {
        let repr = ast.to_string();
        Z3Parser::new(&repr).parse_bitvec().map_err(|message| FromAstError::new(ast, message))
    }

    /// Convert this expression into a Z3-solver Ast.
    pub fn to_z3_ast<'ctx>(&self, ctx: &'ctx Z3Context) -> Z3BitVec<'ctx> {
        match self {
            Int(int) => Z3BitVec::from_u64(ctx, int.1, int.0.bits() as u32),
            Sym(sym) => Z3BitVec::new_const(ctx, sym.to_string(), sym.0.bits() as u32),

            Add(a, b) => z3_binop!(ctx, a, b, bvadd),
            Sub(a, b) => z3_binop!(ctx, a, b, bvsub),
            Mul(a, b) => z3_binop!(ctx, a, b, bvmul),
            And(a, b) => z3_binop!(ctx, a, b, bvand),
            Or(a, b)  => z3_binop!(ctx, a, b, bvor),
            Not(a)    => a.to_z3_ast(ctx).bvnot(),

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
            }
        }
    }
}

impl SymCondition {
    /// Convert this condition into an expression, where `true` is represented by
    /// 1 and `false` by 0.
    pub fn as_expr(self, data_type: DataType) -> SymExpr {
        match self {
            Bool(b) => Int(Integer::from_bool(b, data_type)),
            c => AsExpr(Box::new(c), data_type),
        }
    }

    /// Convert the Z3-solver Ast into an expression if possible.
    pub fn from_z3_ast(ast: &Z3Bool) -> Result<SymCondition, FromAstError> {
        let repr = ast.to_string();
        Z3Parser::new(&repr).parse_bool().map_err(|message| FromAstError::new(ast, message))
    }

    /// Convert this condition into a Z3-solver Ast.
    pub fn to_z3_ast<'ctx>(&self, ctx: &'ctx Z3Context) -> Z3Bool<'ctx> {
        match self {
            Bool(b) => Z3Bool::from_bool(ctx, *b),

            Equal(a, b)        => z3_binop!(ctx, a, b, _eq),
            Less(a, b)         => z3_binop!(ctx, a, b, bvult),
            LessEqual(a, b)    => z3_binop!(ctx, a, b, bvule),
            Greater(a, b)      => z3_binop!(ctx, a, b, bvugt),
            GreaterEqual(a, b) => z3_binop!(ctx, a, b, bvuge),
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
            Cast(x, new, signed) => write!(f, "({} as {}{})", x, new,
                if *signed { " signed "} else { "" }),
            AsExpr(c, data_type) => write!(f, "({} as {})", c, data_type),
        }
    }
}

impl Display for SymCondition {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Bool(b) => write!(f, "{}", b),
            Equal(a, b) => write!(f, "({} == {})", a, b),
            Less(a, b) => write!(f, "({} < {})", a, b),
            LessEqual(a, b) => write!(f, "({} <= {})", a, b),
            Greater(a, b) => write!(f, "({} > {})", a, b),
            GreaterEqual(a, b) => write!(f, "({} >= {})", a, b),
        }
    }
}

impl Display for Symbol {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "s{}-{}:{}", self.1, self.2, self.0)
    }
}

/// Parses Z3 string representations.
struct Z3Parser<'a> {
    ast: &'a str,
}

impl<'a> Z3Parser<'a> {
    /// Create a new parser.
    pub fn new(ast: &'a str) -> Z3Parser<'a> {
        Z3Parser { ast }
    }

    /// Parse a Z3 Ast into a symbolic expression.
    pub fn parse_bitvec(&mut self) -> Result<SymExpr, String> {
        self.skip_white();
        match self.peek() {
            Some('(') => self.parse_bv_func(),
            Some('#') => {
                let (bits, value) = self.parse_bv_immediate()?;
                let data_type = match bits {
                    8 => N8,
                    16 => N16,
                    32 => N32,
                    64 => N64,
                    s => return err(format!("invalid bitvec immediate size: {}", s)),
                };
                Ok(Int(Integer(data_type, value)))
            },
            Some('|') => self.parse_bv_symbol(),
            Some(c) => err(format!("unexpected symbol: {:?} while parsing bitvec", c)),
            p => return err(format!("expected expression while parsing bitvec, found {:?}", p)),
        }
    }

    /// Parse a Z3 Ast into a symbolic condition.
    pub fn parse_bool(&mut self) -> Result<SymCondition, String> {
        self.skip_white();
        match self.peek() {
            Some('(') => self.parse_bool_func(),
            Some(c) => err(format!("unexpected symbol: {:?} while parsing bool", c)),
            p => return err(format!("expected expression while parsing bool, found {:?}", p)),
        }
    }

    /// Parse a bitvector function.
    fn parse_bv_func(&mut self) -> Result<SymExpr, String> {
        self.expect('(')?;
        let func = self.parse_word();
        let expr = match func {
            "(_" => {
                self.skip_white();
                let kind = self.parse_word();
                match kind {
                    "zero_extend" => {
                        self.skip_white();
                        let bits = self.parse_word_while(|c| c.is_digit(10)).parse::<usize>()
                            .map_err(|_| "expected bits for zero extension")?;

                        self.expect(')')?;
                        self.skip_white();
                        let right = self.parse_bitvec()?;

                        zero_extend(bits, right)?
                    },
                    _ => return err("unknown _ function kind"),
                }
            },

            "bvadd" => self.parse_bv_varop(Add)?,
            "bvsub" => self.parse_bv_varop(Sub)?,
            "bvmul" => self.parse_bv_varop(Mul)?,
            "bvand" => self.parse_bv_varop(And)?,
            "bvor"  => self.parse_bv_varop(Or)?,
            "bvnot" => Not(boxed(self.parse_bitvec()?)),

            "concat" => {
                self.skip_white();
                if self.peek() != Some('#') {
                    return err("can only use in concat with left-hand zero immediate");
                }

                let (bits, value) = self.parse_bv_immediate()?;
                if value != 0 {
                    return err("unhandled concat: non-zero left-hand value");
                }

                let right = self.parse_bitvec()?;

                zero_extend(bits, right)?
            }
            _ => return err(format!("unknown bitvec function: {}", func)),
        };
        self.expect(')')?;
        Ok(expr)
    }

    /// Parse a bitvector function with variable number of arguments.
    fn parse_bv_varop<F>(&mut self, op: F) -> Result<SymExpr, String>
    where F: Fn(Box<SymExpr>, Box<SymExpr>) -> SymExpr {
        let mut expr = op(boxed(self.parse_bitvec()?), boxed(self.parse_bitvec()?));

        self.skip_white();
        while self.peek() != Some(')') {
            expr = op(boxed(expr), boxed(self.parse_bitvec()?));
            self.skip_white();
        }

        Ok(expr)
    }

    /// Parse a bitvector immediate value.
    fn parse_bv_immediate(&mut self) -> Result<(usize, u64), String> {
        self.expect('#')?;

        enum Kind { Hex, Bin }
        let (kind, radix) = match self.next() {
            Some('x') => (Kind::Hex, 16),
            Some('b') => (Kind::Bin, 2),
            n => return err(format!("expected radix for immediate, found {:?}", n)),
        };

        let word = self.parse_word_while(|c| c.is_digit(radix));

        let bits = word.len() * match kind {
            Kind::Hex => 4,
            Kind::Bin => 1,
        };

        let value = u64::from_str_radix(word, radix)
            .map_err(|_| "invalid immediate")?;

        Ok((bits, value))
    }

    /// Parse a bitvector symbol.
    fn parse_bv_symbol(&mut self) -> Result<SymExpr, String> {
        self.expect('|')?;
        self.expect('s')?;
        let space = self.parse_digit()?;
        self.expect('-')?;
        let index = self.parse_digit()?;

        self.expect(':')?;
        self.expect('n')?;
        let data_type = match (self.next(), self.next()) {
            (Some('8'), Some('|')) => N8,
            (Some('1'), Some('6')) => N16,
            (Some('3'), Some('2')) => N32,
            (Some('6'), Some('4')) => N64,
            _ => return err("expected data type for symbol"),
        };

        if data_type != N8 {
            self.expect('|')?;
        }

        Ok(Sym(Symbol(data_type, space, index)))
    }

    /// Parse a boolean function.
    fn parse_bool_func(&mut self) -> Result<SymCondition, String> {
        self.expect('(')?;
        let func = self.parse_word();
        let cond = match func {
            "=" => Equal(boxed(self.parse_bitvec()?), boxed(self.parse_bitvec()?)),
            "bvult" => Less(boxed(self.parse_bitvec()?), boxed(self.parse_bitvec()?)),
            "bvule" => LessEqual(boxed(self.parse_bitvec()?), boxed(self.parse_bitvec()?)),
            "bvugt" => Greater(boxed(self.parse_bitvec()?), boxed(self.parse_bitvec()?)),
            "bvuge" => GreaterEqual(boxed(self.parse_bitvec()?), boxed(self.parse_bitvec()?)),
            _ => return err(format!("unknown boolean function: {}", func)),
        };
        self.expect(')')?;
        Ok(cond)
    }

    /// Return everything until the next whitespace.
    fn parse_word(&mut self) -> &'a str {
        self.parse_word_while(|c| !c.is_whitespace())
    }

    /// Return everything until the predicate is false.
    fn parse_word_while<F>(&mut self, predicate: F) -> &'a str where F: Fn(char) -> bool {
        let mut end = 0;
        for (index, c) in self.ast.char_indices() {
            if !predicate(c) {
                break;
            }
            end = index + c.len_utf8();
        }

        let name = &self.ast[..end];
        self.ast = &self.ast[end ..];
        name
    }

    /// Try to parse a digit from the first letter.
    fn parse_digit(&mut self) -> Result<usize, String> {
        match self.next() {
            Some(c @ ('0' ..= '9')) => Ok((c as usize) - ('0' as usize)),
            _ => err("expected digit"),
        }
    }

    /// Skip leading whitespace.
    fn skip_white(&mut self) {
        while let Some(c) = self.peek() {
            if !c.is_whitespace() {
                break;
            }
            self.ast = &self.ast[c.len_utf8() ..];
        }
    }

    /// Return an error if the first letter is not the expected one.
    fn expect(&mut self, expected: char) -> Result<(), String> {
        match self.next() {
            Some(first) if first == expected => Ok(()),
            n => err(format!("expected char {:?}, found {:?}", expected, n)),
        }
    }

    /// Return the next letter if there is one.
    fn next(&mut self) -> Option<char> {
        self.ast.chars().next().map(|first| {
            self.ast = &self.ast[first.len_utf8() ..];
            first
        })
    }

    /// Return the next letter without consuming it.
    fn peek(&self) -> Option<char> {
        self.ast.chars().next()
    }
}

/// Extend `right` by `bits` bits.
fn zero_extend(bits: usize, right: SymExpr) -> Result<SymExpr, String> {
    match bits + right.data_type().bits() {
        16 => Ok(right.cast(N16, false)),
        32 => Ok(right.cast(N32, false)),
        64 => Ok(right.cast(N64, false)),
        s => err(format!("unhandled zero extension: invalid target size {}", s)),
    }
}

/// Fast way to make an error.
fn err<T, S: Into<String>>(message: S) -> Result<T, String> {
    Err(message.into())
}

/// Shorthand for `Box::new`.
fn boxed<T>(value: T) -> Box<T> {
    Box::new(value)
}


/// The error type for decoding a Z3 Ast into a [`SymExpr`].
pub struct FromAstError {
    ast: String,
    message: String
}

impl FromAstError {
    fn new<'ctx, T, S: Into<String>>(ast: &'ctx T, message: S) -> FromAstError
    where T: Ast<'ctx> + Display {
        FromAstError {
            ast: ast.to_string(),
            message: message.into(),
        }
    }
}

impl std::error::Error for FromAstError {}
impl Display for FromAstError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        writeln!(f, "failed to parse Z3 value: {} [", self.message)?;
        for line in self.ast.lines() {
            writeln!(f, "    {}", line)?;
        }
        write!(f, "]")
    }
}

impl Debug for FromAstError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Display::fmt(self, f)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::num::Integer;

    fn n(x: u64) -> SymExpr { Int(Integer(N64, x)) }
    fn x() -> SymExpr { Sym(Symbol(N64, 0, 0)) }
    fn y() -> SymExpr { Sym(Symbol(N8, 0, 1)) }
    fn z() -> SymExpr { Sym(Symbol(N64, 0, 1)) }

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
        let ctx = z3::Context::new(&config);

        let expr = n(10).add(x()).add(x()).add(n(5));
        let ast = expr.to_z3_ast(&ctx);
        let simple_ast = ast.simplify();
        let simple_expr = SymExpr::from_z3_ast(&simple_ast).unwrap();

        assert_eq!(simple_expr, n(15).add(n(2).mul(x())));

        let expr = n(10).add(x()).add(n(20)).or(z()).sub(n(3)).mul(n(40));
        assert_eq!(expr, SymExpr::from_z3_ast(&expr.to_z3_ast(&ctx)).unwrap());
    }
}
