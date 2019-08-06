//! Symbolic expressions and calculations.

use std::collections::HashMap;
use std::fmt::{self, Debug, Display, Formatter};
use z3::Context as Z3Context;
use z3::ast::{Ast, BV as Z3BitVec, Bool as Z3Bool};

use crate::num::{Integer, DataType};
use DataType::*;
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
    And(Box<SymCondition>, Box<SymCondition>),
    Or(Box<SymCondition>, Box<SymCondition>),
    Not(Box<SymCondition>),
}

/// A symbol value identified by an index.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
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

macro_rules! contains {
    ($symbol:expr, $($exprs:expr),*) => {
        $($exprs.contains_symbol($symbol) ||)* false
    };
}

impl SymExpr {
    /// The data type of the expression if it is an integer type.
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
    bin_expr!(bit_and, &, BitAnd);
    bin_expr!(bit_or, *, BitOr);

    pub fn bit_not(self) -> SymExpr {
        match self {
            Int(x) => Int(!x),
            x => BitNot(Box::new(x)),
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

    /// Whether the given symbol appears somewhere in this tree.
    pub fn contains_symbol(&self, symbol: Symbol) -> bool {
        match self {
            Int(int) => false,
            Sym(sym) => *sym == symbol,
            Add(a, b)    => contains!(symbol, a, b),
            Sub(a, b)    => contains!(symbol, a, b),
            Mul(a, b)    => contains!(symbol, a, b),
            BitAnd(a, b) => contains!(symbol, a, b),
            BitOr(a, b)  => contains!(symbol, a, b),
            BitNot(a)    => contains!(symbol, a),
            Cast(a, _, _) => contains!(symbol, a),
            AsExpr(a, _)  => contains!(symbol, a),
        }
    }

    /// Convert the Z3-solver Ast into an expression if possible.
    pub fn from_z3_ast(ast: &Z3BitVec) -> Result<SymExpr, FromAstError> {
        let repr = ast.to_string();
        let mut parser = Z3Parser::new(&repr);
        parser.parse_bitvec()
            .map_err(|message| FromAstError::new(ast, parser.index(), message))
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
            }
        }
    }
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

    /// Whether the given symbol appears somewhere in this tree.
    pub fn contains_symbol(&self, symbol: Symbol) -> bool {
        match self {
            Bool(b) => false,
            Equal(a, b)        => contains!(symbol, a, b),
            Less(a, b)         => contains!(symbol, a, b),
            LessEqual(a, b)    => contains!(symbol, a, b),
            Greater(a, b)      => contains!(symbol, a, b),
            GreaterEqual(a, b) => contains!(symbol, a, b),
            And(a, b) => contains!(symbol, a, b),
            Or(a, b)  => contains!(symbol, a, b),
            Not(a)    => contains!(symbol, a),
        }
    }

    /// Convert the Z3-solver Ast into an expression if possible.
    pub fn from_z3_ast(ast: &Z3Bool) -> Result<SymCondition, FromAstError> {
        let repr = ast.to_string();
        let mut parser = Z3Parser::new(&repr);
        parser.parse_bool()
            .map_err(|message| FromAstError::new(ast, parser.index(), message))
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

            And(a, b) => a.to_z3_ast(ctx).and(&[&b.to_z3_ast(ctx)]),
            Or(a, b)  => a.to_z3_ast(ctx).or(&[&b.to_z3_ast(ctx)]),
            Not(a)    => a.to_z3_ast(ctx).not(),
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
            BitAnd(a, b) => write!(f, "({} & {})", a, b),
            BitOr(a, b) => write!(f, "({} | {})", a, b),
            BitNot(a) => write!(f, "(!{})", a),
            Cast(x, new, signed) => write!(f, "({} as {}{})", x, new,
                if *signed { " signed"} else { "" }),
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
            And(a, b) => write!(f, "({} and {})", a, b),
            Or(a, b) => write!(f, "({} or {})", a, b),
            Not(a) => write!(f, "(not {})", a),
        }
    }
}

impl Display for Symbol {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "s{}-{}:{}", self.1, self.2, self.0)
    }
}

/// Parses Z3 string representations.
#[derive(Debug, Clone)]
struct Z3Parser<'a> {
    ast: &'a str,
    active: &'a str,
    bindings: HashMap<String, SymDynamic>,
}

type ParseResult<T> = Result<T, String>;
impl<'a> Z3Parser<'a> {
    /// Create a new parser.
    pub fn new(ast: &'a str) -> Z3Parser<'a> {
        Z3Parser {
            ast,
            active: ast,
            bindings: HashMap::new(),
        }
    }

    /// The index in the string the parse is at currently.
    pub fn index(&self) -> usize {
        self.active.as_ptr() as usize - self.ast.as_ptr() as usize
    }

    /// Parse a Z3 Ast into a symbolic expression.
    pub fn parse_bitvec(&mut self) -> ParseResult<SymExpr> {
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
            Some(c) => match self.parse_variable()? {
                SymDynamic::Expr(expr) => Ok(expr),
                SymDynamic::Condition(_) => err("let-binding has wrong type,
                    expected expression, found condition"),
            },
            p => return err(format!("expected expression while parsing bitvec, found {:?}", p)),
        }
    }

    /// Parse a bitvector function.
    fn parse_bv_func(&mut self) -> ParseResult<SymExpr> {
        self.expect('(')?;
        let func = self.parse_ident();
        let expr = match func {
            "let" => { self.parse_let_bindings(Self::parse_bitvec)?; self.parse_bitvec()? },
            "(_" => {
                self.skip_white();
                let kind = self.parse_ident();
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
            "bvand" => self.parse_bv_varop(BitAnd)?,
            "bvor"  => self.parse_bv_varop(BitOr)?,
            "bvnot" => BitNot(boxed(self.parse_bitvec()?)),

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
        self.skip_white();
        self.expect(')')?;
        Ok(expr)
    }

    /// Parse a bitvector function with variable number of arguments.
    fn parse_bv_varop<F>(&mut self, op: F) -> ParseResult<SymExpr>
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
    fn parse_bv_immediate(&mut self) -> ParseResult<(usize, u64)> {
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
    fn parse_bv_symbol(&mut self) -> ParseResult<SymExpr> {
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

    /// Parse a Z3 Ast into a symbolic condition.
    pub fn parse_bool(&mut self) -> ParseResult<SymCondition> {
        self.skip_white();
        match self.peek() {
            Some('(') => self.parse_bool_func(),
            Some(c) => match self.parse_variable()? {
                SymDynamic::Condition(condition) => Ok(condition),
                SymDynamic::Expr(_) => err("let-binding has wrong type,
                    expected condition, found expression"),
            },
            p => return err(format!("expected expression while parsing bool, found {:?}", p)),
        }
    }

    /// Parse a boolean function.
    fn parse_bool_func(&mut self) -> ParseResult<SymCondition> {
        self.expect('(')?;
        let func = self.parse_ident();
        let cond = match func {
            "let" => { self.parse_let_bindings(Self::parse_bool)?; self.parse_bool()? },
            "=" => Equal(boxed(self.parse_bitvec()?), boxed(self.parse_bitvec()?)),

            "bvult" => Less(boxed(self.parse_bitvec()?), boxed(self.parse_bitvec()?)),
            "bvule" => LessEqual(boxed(self.parse_bitvec()?), boxed(self.parse_bitvec()?)),
            "bvugt" => Greater(boxed(self.parse_bitvec()?), boxed(self.parse_bitvec()?)),
            "bvuge" => GreaterEqual(boxed(self.parse_bitvec()?), boxed(self.parse_bitvec()?)),

            "and" => And(boxed(self.parse_bool()?), boxed(self.parse_bool()?)),
            "or" => And(boxed(self.parse_bool()?), boxed(self.parse_bool()?)),
            "not" => Not(boxed(self.parse_bool()?)),

            _ => return err(format!("unknown boolean function: {}", func)),
        };
        self.skip_white();
        self.expect(')')?;
        Ok(cond)
    }

    /// Parse variable assignments.
    fn parse_let_bindings<P, T: Into<SymDynamic>>(&mut self, subparser: P) -> ParseResult<()>
    where P: Fn(&mut Self) -> ParseResult<T> {
        self.skip_white();
        self.expect('(')?;

        while self.peek() == Some('(') {
            self.expect('(')?;

            let name = self.parse_ident();
            let value = subparser(self)?;
            self.bindings.insert(name.to_string(), value.into());

            self.skip_white();
            self.expect(')')?;
            self.skip_white();
        }

        self.expect(')')?;

        Ok(())
    }

    /// Parse variable usages.
    fn parse_variable(&mut self) -> ParseResult<SymDynamic> {
        let name = self.parse_ident();
        match self.bindings.get(name) {
            Some(value) => Ok(value.clone()),
            None => err(format!("let binding with name {} does not exist", name))
        }
    }

    /// Return everything until the next whitespace or parens.
    fn parse_ident(&mut self) -> &'a str {
        self.parse_word_while(|c| !c.is_whitespace() && c != ')' && c != '(')
    }

    /// Return everything until the predicate is false.
    fn parse_word_while<F>(&mut self, predicate: F) -> &'a str where F: Fn(char) -> bool {
        let mut end = 0;
        for (index, c) in self.active.char_indices() {
            if !predicate(c) {
                break;
            }
            end = index + c.len_utf8();
        }

        let name = &self.active[..end];
        self.active = &self.active[end ..];
        name
    }

    /// Try to parse a digit from the first letter.
    fn parse_digit(&mut self) -> ParseResult<usize> {
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
            self.active = &self.active[c.len_utf8() ..];
        }
    }

    /// Return an error if the first letter is not the expected one.
    fn expect(&mut self, expected: char) -> ParseResult<()> {
        match self.next() {
            Some(first) if first == expected => Ok(()),
            n => err(format!("expected char {:?}, found {:?}", expected, n)),
        }
    }

    /// Return the next letter if there is one.
    fn next(&mut self) -> Option<char> {
        self.active.chars().next().map(|first| {
            self.active = &self.active[first.len_utf8() ..];
            first
        })
    }

    /// Return the next letter without consuming it.
    fn peek(&self) -> Option<char> {
        self.active.chars().next()
    }
}

/// Extend `right` by `bits` bits.
fn zero_extend(bits: usize, right: SymExpr) -> ParseResult<SymExpr> {
    match bits + right.data_type().bits() {
        16 => Ok(right.cast(N16, false)),
        32 => Ok(right.cast(N32, false)),
        64 => Ok(right.cast(N64, false)),
        s => err(format!("unhandled zero extension: invalid target size {}", s)),
    }
}

/// Fast way to make an error.
fn err<T, S: Into<String>>(message: S) -> ParseResult<T> {
    Err(message.into())
}

/// Shorthand for `Box::new`.
fn boxed<T>(value: T) -> Box<T> {
    Box::new(value)
}

/// A dynamically typed symbolic value.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
enum SymDynamic {
    Expr(SymExpr),
    Condition(SymCondition),
}

impl From<SymExpr> for SymDynamic {
    fn from(expr: SymExpr) -> SymDynamic { SymDynamic::Expr(expr) }
}

impl From<SymCondition> for SymDynamic {
    fn from(cond: SymCondition) -> SymDynamic { SymDynamic::Condition(cond) }
}


/// The error type for decoding a Z3 Ast into a [`SymExpr`].
pub struct FromAstError {
    ast: String,
    index: usize,
    message: String
}

impl FromAstError {
    fn new<'ctx, T, S: Into<String>>(ast: &'ctx T, index: usize, message: S) -> FromAstError
    where T: Ast<'ctx> + Display {
        FromAstError {
            ast: ast.to_string(),
            index,
            message: message.into(),
        }
    }
}

impl std::error::Error for FromAstError {}
impl Display for FromAstError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        writeln!(f, "Failed to parse Z3 value at index {}: {} [", self.index, self.message)?;
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

        let expr = n(10).add(x()).add(n(20)).bit_or(z()).sub(n(3)).mul(n(40));
        assert_eq!(expr, SymExpr::from_z3_ast(&expr.to_z3_ast(&ctx)).unwrap());
    }
}
