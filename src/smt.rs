//! Functionality for using the Z3 SMT solver.

use std::collections::HashMap;
use std::fmt::{self, Display, Debug, Formatter};
use z3::Context as Z3Context;
use z3::ast::Ast;

use crate::expr::{SymExpr, SymCondition, SymDynamic, Symbol};
use crate::num::{Integer, DataType::*};
use SymExpr::*;
use SymCondition::*;


/// Solves and simplifies conditions and expressions using Z3.
pub struct Solver {
    ctx: Z3Context,
}

/// A reference-counted condition solver.
pub type SharedSolver = std::rc::Rc<Solver>;

impl Solver {
    /// Create a new condition solver with it's own Z3 context.
    pub fn new() -> Solver {
        let config = z3::Config::new();
        let ctx = Z3Context::new(&config);
        Solver { ctx }
    }

    /// Simplify an expression.
    pub fn simplify_expr(&self, expr: SymExpr) -> SymExpr {
        let z3_expr = expr.to_z3_ast(&self.ctx);
        let params = self.params();
        let z3_simplified = z3_expr.simplify_ex(&params).simplify_ex(&params);
        SymExpr::from_z3_ast(&z3_simplified).unwrap_or_else(|_| {
            println!("warning: condition solver: failed to simplify expression: {}", expr);
            expr
        })
    }

    /// Simplify a condition.
    pub fn simplify_condition(&self, cond: SymCondition) -> SymCondition {
        if self.check_sat(&cond) {
            let z3_cond = cond.to_z3_ast(&self.ctx);
            let params = self.params();
            let z3_simplified = z3_cond.simplify_ex(&params).simplify_ex(&params);
            SymCondition::from_z3_ast(&z3_simplified).unwrap_or_else(|_| {
                println!("warning: condition solver: failed to simplify condition: {}", cond);
                cond
            })
        } else {
            SymCondition::FALSE
        }
    }

    /// Check whether the condition is satisfiable.
    pub fn check_sat(&self, cond: &SymCondition) -> bool {
        let z3_cond = cond.to_z3_ast(&self.ctx);
        let solver = z3::Solver::new(&self.ctx);
        solver.assert(&z3_cond);
        solver.check()
    }

    /// Check whether two expressions are possibly equal.
    pub fn check_equal_sat(&self, a: &SymExpr, b: &SymExpr) -> bool {
        let z3_a = a.to_z3_ast(&self.ctx);
        let z3_b = b.to_z3_ast(&self.ctx);
        let solver = z3::Solver::new(&self.ctx);
        solver.assert(&z3_a._eq(&z3_b));
        solver.check()
    }

    /// Builds the default simplifaction params.
    fn params(&self) -> z3::Params {
        let mut params = z3::Params::new(&self.ctx);
        params.set_bool("elim_sign_ext", false);
        params
    }
}

impl Debug for Solver {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Solver")
    }
}

/// Parses Z3 string representations.
#[derive(Debug, Clone)]
pub struct Z3Parser<'a> {
    ast: &'a str,
    active: &'a str,
    bindings: HashMap<String, SymDynamic>,
}

impl<'a> Z3Parser<'a> {
    /// Create a new parser.
    pub fn new(ast: &'a str) -> Z3Parser<'a> {
        Z3Parser {
            ast,
            active: ast,
            bindings: HashMap::new(),
        }
    }

    /// Parse a Z3 Ast into a symbolic expression.
    pub fn parse_expr(&mut self) -> Result<SymExpr, FromAstError> {
        self.parse_bitvec()
            .map_err(|message| FromAstError::new(self.ast, self.index(), message))
    }

    /// Parse a Z3 Ast into a symbolic condition.
    pub fn parse_condition(&mut self) -> Result<SymCondition, FromAstError> {
        self.parse_bool()
            .map_err(|message| FromAstError::new(self.ast, self.index(), message))
    }

    /// Parse a bitvector expression.
    fn parse_bitvec(&mut self) -> ParseResult<SymExpr> {
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
            Some(_) => match self.parse_variable()? {
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
        let func = self.parse_func_name();
        let expr = match func {
            "let" => { self.parse_let_bindings()?; self.parse_bitvec()? },
            "(_" => {
                self.skip_white();
                let kind = self.parse_ident();
                match kind {
                    "zero_extend" | "sign_extend" => {
                        self.skip_white();
                        let bits = self.parse_word_while(|c| c.is_digit(10)).parse::<usize>()
                            .map_err(|_| "expected value for bit extension")?;

                        self.expect(')')?;
                        self.skip_white();
                        let right = self.parse_bitvec()?;

                        bit_extend(bits, right, kind == "sign_extend")?
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

            "ite" => {
                let condition = self.parse_bool()?;
                condition.if_then_else(self.parse_bitvec()?, self.parse_bitvec()?)
            },

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
                bit_extend(bits, right, false)?
            },

            _ => return err(format!("unknown bitvec function: {:?}", func)),
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
        let space = match self.parse_word_while(char::is_alphabetic) {
            "mem" => "mem",
            "reg" => "reg",
            "stdin" => "stdin",
            s => return err(format!("invalid space name for symbol: {:?}", s)),
        };
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

    /// Parse a Z3 Ast into a boolean expression.
    fn parse_bool(&mut self) -> ParseResult<SymCondition> {
        self.skip_white();
        match self.peek() {
            Some('(') => self.parse_bool_func(),
            Some(_) => match self.parse_variable()? {
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
        let func = self.parse_func_name();
        let cond = match func {
            "let" => { self.parse_let_bindings()?; self.parse_bool()? },

            "=" => Equal(boxed(self.parse_bitvec()?), boxed(self.parse_bitvec()?)),

            "bvult" => LessThan(boxed(self.parse_bitvec()?), boxed(self.parse_bitvec()?), false),
            "bvule" => LessEqual(boxed(self.parse_bitvec()?), boxed(self.parse_bitvec()?), false),
            "bvugt" => GreaterThan(boxed(self.parse_bitvec()?), boxed(self.parse_bitvec()?), false),
            "bvuge" => GreaterEqual(boxed(self.parse_bitvec()?), boxed(self.parse_bitvec()?), false),

            "bvslt" => LessThan(boxed(self.parse_bitvec()?), boxed(self.parse_bitvec()?), true),
            "bvsle" => LessEqual(boxed(self.parse_bitvec()?), boxed(self.parse_bitvec()?), true),
            "bvsgt" => GreaterThan(boxed(self.parse_bitvec()?), boxed(self.parse_bitvec()?), true),
            "bvsge" => GreaterEqual(boxed(self.parse_bitvec()?), boxed(self.parse_bitvec()?), true),

            "and" => And(boxed(self.parse_bool()?), boxed(self.parse_bool()?)),
            "or" => Or(boxed(self.parse_bool()?), boxed(self.parse_bool()?)),
            "not" => Not(boxed(self.parse_bool()?)),

            _ => return err(format!("unknown boolean function: {:?}", func)),
        };
        self.skip_white();
        self.expect(')')?;
        Ok(cond)
    }

    /// Parse variable assignments.
    fn parse_let_bindings(&mut self) -> ParseResult<()> {
        self.skip_white();
        self.expect('(')?;

        while self.peek() == Some('(') {
            self.expect('(')?;

            let name = self.parse_ident();

            let value = match self.parse_bitvec() {
                Ok(bitvec) => bitvec.into(),
                Err(_) => self.parse_bool()?.into(),
            };
            self.bindings.insert(name.to_string(), value);

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
        match name {
            "true" => Ok(SymDynamic::Condition(SymCondition::TRUE)),
            "false" => Ok(SymDynamic::Condition(SymCondition::FALSE)),
            var => match self.bindings.get(var) {
                Some(value) => Ok(value.clone()),
                None => err(format!("let binding with name {:?} does not exist", name))
            }
        }
    }

    /// Return everything until the next whitespace or parens.
    fn parse_ident(&mut self) -> &'a str {
        self.parse_word_while(|c| !c.is_whitespace() && c != ')' && c != '(')
    }

    /// Parse function name.
    fn parse_func_name(&mut self) -> &'a str {
        self.parse_word_while(|c| !c.is_whitespace())
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

    /// The index in the string the parser is at currently.
    fn index(&self) -> usize {
        self.active.as_ptr() as usize - self.ast.as_ptr() as usize
    }
}

type ParseResult<T> = Result<T, String>;

/// Extend `right` by `bits` bits (ones if signed, zeros otherwise).
fn bit_extend(bits: usize, right: SymExpr, signed: bool) -> ParseResult<SymExpr> {
    match bits + right.data_type().bits() {
        16 => Ok(right.cast(N16, signed)),
        32 => Ok(right.cast(N32, signed)),
        64 => Ok(right.cast(N64, signed)),
        s => err(format!("unhandled bit extension: invalid target size {}", s)),
    }
}

/// Fast way to make an error.
fn err<T, S: Into<String>>(message: S) -> ParseResult<T> {
    Err(message.into())
}

fn boxed<T>(value: T) -> Box<T> { Box::new(value) }


/// The error type for decoding a Z3 Ast into a symbolic expression/condition.
pub struct FromAstError {
    ast: String,
    index: usize,
    message: String
}

impl FromAstError {
    fn new(ast: &str, index: usize, message: String) -> FromAstError {
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
