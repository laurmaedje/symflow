//! Microcode encoding of instructions.

use std::fmt::{self, Display, Formatter};

use crate::x86_64::{Instruction, Mnemoic, Operand, Register};
use crate::math::{Integer, DataType, SymExpr, SymCondition, Symbol};
use Register::*;
use SymCondition::*;


/// A sequence of micro operations.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Microcode {
    pub ops: Vec<MicroOperation>,
}

/// A minimal executable action.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum MicroOperation {
    /// Store the value at location `src` in location `dest`.
    Mov { dest: Location, src: Location },

    /// Store a constant in the temporary `dest`.
    Const { dest: Temporary, constant: Integer },
    /// Cast the temporary `target` to another type.
    /// - If the target type is smaller, it will get truncated.
    /// - If the target type is bigger, if signed is true the value will be
    ///   sign-extended and otherwise zero-extended.
    Cast { target: Temporary, new: DataType, signed: bool },

    /// Store the sum of `a` and `b` in `sum`. Set flags if active.
    Add { sum: Temporary, a: Temporary, b: Temporary },
    /// Store the difference of `a` and `b` in `diff`. Set flags if active.
    Sub { diff: Temporary, a: Temporary, b: Temporary },
    /// Store the product of `a` and `b` in `prod`. Set flags if active.
    Mul { prod: Temporary, a: Temporary, b: Temporary },

    /// Store the bitwise AND of `a` and `b` in and. Set flags if active.
    And { and: Temporary, a: Temporary, b: Temporary },
    /// Store the bitwise OR of `a` and `b` in or. Set flags if active.
    Or { or: Temporary, a: Temporary, b: Temporary },
    /// Store the bitwise NOT of `a` in `not`.
    Not { not: Temporary, a: Temporary },

    /// Set the target temporary to one if the condition is true and to zero otherwise.
    Set { target: Temporary, condition: SymCondition },
    /// Jump to the current address plus the `offset` if `relative` is true,
    /// otherwise directly to the target if the condition specified by `condition`
    /// is fulfilled.
    Jump { target: Temporary, condition: SymCondition, relative: bool },

    /// Perform a syscall.
    Syscall,
}

impl Display for Microcode {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Microcode [")?;
        if !self.ops.is_empty() { writeln!(f)?; }
        for operation in &self.ops {
            writeln!(f, "    {}", operation)?;
        }
        write!(f, "]")
    }
}

impl Display for MicroOperation {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use MicroOperation::*;
        use crate::helper::signed_name;

        fn show_condition(condition: &SymCondition) -> String {
            match condition {
                &SymCondition::TRUE => "".to_string(),
                cond => format!(" if {}", cond),
            }
        }

        match self {
            Mov { dest, src } => write!(f, "mov {} = {}", dest, src),
            Const { dest, constant } => write!(f, "const {} = {}", dest, constant),
            Cast { target, new, signed } => write!(f, "cast {} to {} {}",
                target, new, signed_name(*signed)),

            Add { sum, a, b } => write!(f, "add {} = {} + {}", sum, a, b),
            Sub { diff, a, b } => write!(f, "sub {} = {} - {}", diff, a, b),
            Mul { prod, a, b } => write!(f, "mul {} = {} * {}", prod, a, b),

            And { and, a, b } => write!(f, "and {} = {} & {}", and, a, b),
            Or { or, a, b } => write!(f, "or {} = {} | {}", or, a, b),
            Not { not, a } => write!(f, "not {} = !{}", not, a),

            Set { target, condition } => write!(f, "set {}{}",
                target, show_condition(condition)),
            Jump { target, condition, relative } => write!(f, "jump {} {}{}",
                if *relative { "by" } else { "to" }, target, show_condition(condition)),

            Syscall => write!(f, "syscall"),
        }
    }
}

/// Encodes instructions into microcode.
#[derive(Debug, Clone)]
pub struct MicroEncoder {
    ops: Vec<MicroOperation>,
    temps: usize,
    last_flag_op: Option<SymExpr>,
}

type EncoderResult<T> = Result<T, String>;

/// Constructs a `SymCondition` from the last flag-modifying operation.
macro_rules! condition {
    ($self:expr; $($flag_op:pat => $cond:expr),* ) => {
        match $self.last_flag_op.clone() {
            $(Some($flag_op) => Ok($cond),)*
            Some(op) => panic!("condition!: unhandled last op: {:?}", op),
            None => Err("get_comparison: no previous flag-modifying operation".to_string()),
        }
    };
}

impl MicroEncoder {
    /// Create a new encoder.
    pub fn new() -> MicroEncoder {
        MicroEncoder {
            ops: vec![],
            temps: 0,
            last_flag_op: None
        }
    }

    /// Encode an instruction into microcode.
    pub fn encode(&mut self, inst: &Instruction) -> EncodeResult<Microcode> {
        self.encode_internal(inst)
            .map_err(|msg| EncodingError::new(inst.clone(), msg))
    }

    /// The actual encoding but with a different result type that the public interface.
    fn encode_internal(&mut self, inst: &Instruction) -> EncoderResult<Microcode> {
        use MicroOperation as Op;
        use Mnemoic::*;

        match inst.mnemoic {
            // Load both operands, perform an operation and write the result back.
            Add => {
                let (a, b) = self.encode_binop(inst, |sum, a, b| Op::Add { sum, a, b });
                self.last_flag_op = Some(a.to_expr().add(b.to_expr()));
            },
            Sub => {
                let (a, b) = self.encode_binop(inst, |diff, a, b| Op::Sub { diff, a, b });
                self.last_flag_op = Some(a.to_expr().sub(b.to_expr()));
            },
            Imul => {
                let (a, b) = self.encode_binop(inst, |prod, a, b| Op::Mul { prod, a, b });
                self.last_flag_op = Some(a.to_expr().mul(b.to_expr()));
            },

            Cmp => {
                let ((_dest, left), (_src, right)) = self.encode_load_both(inst);
                self.last_flag_op = Some(left.to_expr().sub(right.to_expr()));
            },
            Test => {
                let ((_dest, left), (_src, right)) = self.encode_load_both(inst);
                self.last_flag_op = Some(left.to_expr().bitand(right.to_expr()));
            },

            // Retrieve both locations and move from source to destination.
            Mov => {
                let dest = self.encode_get_location(inst.operands[0]);
                let src = self.encode_get_location(inst.operands[1]);
                if dest.data_type() != src.data_type() {
                    self.ops.clear();
                    self.temps = 0;
                    self.encode_move_casted(inst.operands[0], inst.operands[1], true)?;
                } else {
                    self.encode_move(dest, src)?;
                }
            },

            // Load the source, cast it to the destination type and move it there.
            Movzx => self.encode_move_casted(inst.operands[0], inst.operands[1], false)?,
            Movsx => self.encode_move_casted(inst.operands[0], inst.operands[1], true)?,

            Cwde => self.encode_move_casted(Operand::Direct(EAX), Operand::Direct(AX), true)?,
            Cdqe => self.encode_move_casted(Operand::Direct(RAX), Operand::Direct(EAX), true)?,

            // Retrieve both locations, but instead of loading just move the
            // address into the destination.
            Lea => {
                let dest = self.encode_get_location(inst.operands[0]);
                let src = self.encode_get_location(inst.operands[1]);

                if let Location::Indirect(_, _, temp) = src {
                    self.encode_move(dest, Location::Temp(temp))?;
                } else {
                    return Err("invalid source operand for lea".to_string());
                }
            },

            // Store or load data on the stack and adjust the stack pointer.
            Push => {
                let src = self.encode_get_location(inst.operands[0]);
                self.encode_push(src)?;
            },
            Pop => {
                let dest = self.encode_get_location(inst.operands[0]);
                self.encode_pop(dest)?;
            },

            // Jump to the first operand under specific conditions.
            Jmp => self.encode_jump(inst, SymCondition::TRUE),
            Je  => self.encode_jump(inst, condition!(self;
                SymExpr::Sub(a, b) => Equal(a, b),
                SymExpr::BitAnd(a, b) => SymExpr::from_int(a.data_type(), 0).equal(a.bitand(*b))
            )?),
            Jbe => self.encode_jump(inst, condition!(self; SymExpr::Sub(a, b) => LessEqual(a, b, false))?),
            Jl  => self.encode_jump(inst, condition!(self; SymExpr::Sub(a, b) => LessThan(a, b, true))?),
            Jle => self.encode_jump(inst, condition!(self; SymExpr::Sub(a, b) => LessEqual(a, b, true))?),
            Jg  => self.encode_jump(inst, condition!(self; SymExpr::Sub(a, b) => GreaterThan(a, b, true))?),
            Jge => self.encode_jump(inst, condition!(self; SymExpr::Sub(a, b) => GreaterEqual(a, b, true))?),
            Setl => self.encode_set(inst, condition!(self; SymExpr::Sub(a, b) => LessThan(a, b, true))?)?,

            // Save the procedure linking information on the stack and jump.
            Call => {
                let rip = self.encode_get_location(Operand::Direct(Register::RIP));
                self.encode_push(rip)?;
                self.encode_jump(inst, SymCondition::TRUE);
            },

            // Copies the base pointer into the stack pointer register and pops the
            // old base pointer from the stack.
            Leave => {
                let rbp = self.encode_get_location(Operand::Direct(Register::RBP));
                let rsp = self.encode_get_location(Operand::Direct(Register::RSP));
                self.encode_move(rsp, rbp)?;
                self.encode_pop(rbp)?;
            },

            // Jumps back to the address located on top of the stack.
            Ret => {
                let target = Temporary(DataType::N64, self.temps);
                self.temps += 1;
                self.encode_pop(Location::Temp(target))?;
                self.ops.push(Op::Jump { target, condition: SymCondition::TRUE, relative: false });
            },

            Syscall => { self.ops.push(Op::Syscall); },
            Nop => {},
        }

        let mut ops = Vec::new();
        std::mem::swap(&mut ops, &mut self.ops);
        Ok(Microcode { ops })
    }

    /// Encode a binary operation like an add or a subtract.
    fn encode_binop<F>(&mut self, inst: &Instruction, binop: F) -> (Temporary, Temporary)
    where F: FnOnce(Temporary, Temporary, Temporary) -> MicroOperation {
        let ((dest, left), (_, right)) = self.encode_load_both(inst);

        // Encode the actual binary operation and the move from the target temporary
        // into the destination.
        let target = Temporary(left.0, self.temps);
        self.temps += 1;
        self.ops.push(binop(target, left, right));
        self.ops.push(MicroOperation::Mov { dest, src: Location::Temp(target) });

        (left, right)
    }

    /// Encode a conditional, relative jump.
    fn encode_jump(&mut self, inst: &Instruction, condition: SymCondition) {
        let operand = inst.operands[0];
        let relative = match operand {
            Operand::Offset(_) => true,
            _ => false,
        };
        let (_, target) = self.encode_load_operand(operand);
        self.ops.push(MicroOperation::Jump { target, condition, relative });
    }

    /// Encode a set instruction, which sets a bit based on a condition.
    fn encode_set(&mut self, inst: &Instruction, condition: SymCondition) -> EncoderResult<()> {
        let location = self.encode_get_location(inst.operands[0]);
        let temp = Temporary(location.data_type(), self.temps);
        self.temps += 1;
        self.ops.push(MicroOperation::Set { target: temp, condition });
        self.encode_move(location, Location::Temp(temp))
    }

    /// Encode a push instruction.
    fn encode_push(&mut self, src: Location) -> EncoderResult<()> {
        // Load the stack pointer.
        let (sp, stack) = self.encode_load_operand(Operand::Direct(Register::RSP));
        let data_type = src.data_type();

        // Load the width of the moved thing as a constant and subtract it from the
        // stack pointer.
        let offset = Temporary(DataType::N64, self.temps);
        let constant = Integer(DataType::N64, data_type.bytes() as u64);
        self.ops.push(MicroOperation::Const { dest: offset, constant });
        self.ops.push(MicroOperation::Sub { diff: stack, a: stack, b: offset });
        self.temps += 1;

        // Move the value from the source onto the stack.
        let stack_space = Location::Indirect(data_type, 0, stack);
        self.encode_move(stack_space, src)?;

        // Copy back the stack pointer.
        self.encode_move(sp, Location::Temp(stack))
    }

    /// Encode a pop instruction.
    fn encode_pop(&mut self, dest: Location) -> EncoderResult<()> {
        // Load the stack pointer.
        let (sp, stack) = self.encode_load_operand(Operand::Direct(Register::RSP));
        let data_type = dest.data_type();

        // Move the value from the stack into the destination.
        let stack_space = Location::Indirect(data_type, 0, stack);
        self.encode_move(dest, stack_space)?;

        // Load the width of the moved thing as a constant and add it to the
        // stack pointer. Then copy the stack pointer back into it's register.
        let offset = Temporary(DataType::N64, self.temps);
        let constant = Integer(DataType::N64, data_type.bytes() as u64);
        self.ops.push(MicroOperation::Const { dest: offset, constant });
        self.ops.push(MicroOperation::Add { sum: stack, a: stack, b: offset });
        self.temps += 1;
        self.encode_move(sp, Location::Temp(stack))
    }

    /// Encode the operations to load both operands of a binary instruction and return
    /// the locations and temporaries that the operands where loaded into.
    /// Casts the second operand to the type of the first one if necessary.
    fn encode_load_both(&mut self, inst: &Instruction)
    -> ((Location, Temporary), (Location, Temporary)) {
        // Encode the loading of both operands into a temporary.
        let (dest, left) = self.encode_load_operand(inst.operands[0]);
        let (src, mut right) = self.encode_load_operand(inst.operands[1]);

        // Enforce that both operands have the exact same data type.
        if left.0 != right.0 {
            self.ops.push(MicroOperation::Cast { target: right, new: left.0, signed: true });
            right.0 = left.0;
        }

        ((dest, left), (src, right))
    }

    /// Encode the loading of an operand into a temporary.
    fn encode_load_operand(&mut self, operand: Operand) -> (Location, Temporary) {
        let location = self.encode_get_location(operand);
        if let Location::Temp(temp) = location {
            (location, temp)
        } else {
            let temp = Temporary(location.data_type(), self.temps);
            self.ops.push(MicroOperation::Mov { dest: Location::Temp(temp), src: location });
            self.temps += 1;
            (location, temp)
        }
    }

    /// Encode the micro operations necessary to prepare the location of the operand.
    /// The operand itself will not be loaded and the operations may be empty
    /// if it is a direct operand.
    fn encode_get_location(&mut self, operand: Operand) -> Location {
        use Location::*;

        match operand {
            Operand::Direct(reg) => Direct(reg.data_type(), 1, reg.address()),

            Operand::Indirect { data_type, base, scaled_offset, displacement } => {
                // Load the base into a register.
                let reg = self.encode_load_reg(base);

                // Load index and scale into temporaries, multiply them and add them to the base.
                if let Some((index, scale)) = scaled_offset {
                    let index_reg = self.encode_load_reg(index);
                    let scale = self.encode_load_constant(Integer::from_ptr(scale as u64));

                    self.ops.push(MicroOperation::Mul { prod: index_reg, a: index_reg, b: scale });
                    self.ops.push(MicroOperation::Add { sum: reg, a: reg, b: index_reg });
                }

                // Add the displacement, too.
                if let Some(disp) = displacement {
                    let disp = self.encode_load_constant(Integer::from_ptr(disp as u64));
                    self.ops.push(MicroOperation::Add { sum: reg, a: reg, b: disp });
                }

                Indirect(data_type, 0, reg)
            },

            Operand::Immediate(int) => {
                // Load the immediate into a temporary.
                Temp(self.encode_load_constant(int))
            },

            Operand::Offset(offset) => {
                Temp(self.encode_load_constant(Integer(DataType::N64, offset as u64)))
            },
        }
    }

    /// Encode the loading of a register from memory into a temporary.
    /// The resulting temporary will have the data type matching the registers width.
    fn encode_load_reg(&mut self, reg: Register) -> Temporary {
        let data_type = reg.data_type();
        let temp = Temporary(data_type, self.temps);

        let src = Location::Direct(data_type, 1, reg.address());
        self.ops.push(MicroOperation::Mov { dest: Location::Temp(temp), src });
        self.temps += 1;

        temp
    }

    /// Encode the loading of a constant into a temporary.
    fn encode_load_constant(&mut self, constant: Integer) -> Temporary {
        let dest = Temporary(constant.0, self.temps);
        self.ops.push(MicroOperation::Const { dest, constant });
        self.temps += 1;
        dest
    }

    /// Encode moving with a cast to the destination source type.
    fn encode_move_casted(&mut self, dest: Operand, src: Operand, signed: bool) -> EncoderResult<()> {
        let dest = self.encode_get_location(dest);
        let (_, mut temp) = self.encode_load_operand(src);
        let new = dest.data_type();
        self.ops.push(MicroOperation::Cast { target: temp, new, signed });
        temp.0 = new;
        self.encode_move(dest, Location::Temp(temp))
    }

    /// Encode a move operation.
    fn encode_move(&mut self, dest: Location, src: Location) -> EncoderResult<()> {
        // Enforce that both operands have the exact same data type.
        if src.data_type() != dest.data_type() {
            return Err(format!("incompatible data types for move: {} and {}",
                src.data_type(), dest.data_type()));
        }

        Ok(self.ops.push(MicroOperation::Mov { dest, src }))
    }
}

/// Pinpoints a target in memory or temporaries.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Location {
    Temp(Temporary),
    Direct(DataType, usize, u64),
    Indirect(DataType, usize, Temporary),
}

impl Location {
    /// The underlying data type of the value at the location.
    pub fn data_type(&self) -> DataType {
        match *self {
            Location::Temp(temp) => temp.0,
            Location::Direct(data, _, _) => data,
            Location::Indirect(data, _, _) => data,
        }
    }
}

impl Display for Location {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use Location::*;
        match self {
            Temp(temp) => write!(f, "{}", temp),
            Direct(data, space, offset) => write!(f, "[m{}][{:#x}:{}]", space, offset, data),
            Indirect(data, space, temp) => write!(f, "[m{}][({}):{}]", space, temp, data),
        }
    }
}

/// Temporary variable identified by an index.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Temporary(pub DataType, pub usize);

impl Temporary {
    /// Create a symbol matching this temporary.
    pub fn to_symbol(self) -> Symbol {
        Symbol(self.0, "T", self.1)
    }

    /// Convert this into a symbol expression.
    pub fn to_expr(self) -> SymExpr {
        SymExpr::Sym(self.to_symbol())
    }
}

impl Display for Temporary {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "T{}:{}", self.1, self.0)
    }
}

/// Addresses of things stored in memory (registers).
pub trait MemoryMapped {
    /// Address of the memory mapped thing.
    fn address(&self) -> u64;
}

impl MemoryMapped for Register {
    /// Address of a register in the register memory space.
    fn address(&self) -> u64 {
        match self {
            AL | AX | EAX | RAX => 0x00,
            CL | CX | ECX | RCX => 0x08,
            DL | DX | EDX | RDX => 0x10,
            BL | BX | EBX | RBX => 0x18,
            AH | SP | ESP | RSP => 0x20,
            CH | BP | EBP | RBP => 0x28,
            DH | SI | ESI | RSI => 0x30,
            BH | DI | EDI | RDI => 0x38,
            R8  => 0x40,
            R9  => 0x48,
            R10 => 0x50,
            R11 => 0x58,
            R12 => 0x60,
            R13 => 0x68,
            R14 => 0x70,
            R15 => 0x78,
            IP | EIP | RIP => 0x80,
        }
    }
}


/// The error type for microcode encoding.
#[derive(Eq, PartialEq)]
pub struct EncodingError(Instruction, String);
pub(in super) type EncodeResult<T> = Result<T, EncodingError>;

impl EncodingError {
    /// Create a new encoding error with a message.
    fn new<S: Into<String>>(inst: Instruction, message: S) -> EncodingError {
        EncodingError(inst, message.into())
    }
}

impl Display for EncodingError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Failed to encode instruction: {} [{}]", self.1, self.0)
    }
}

impl std::error::Error for EncodingError {}
debug_display!(EncodingError);


#[cfg(test)]
mod tests {
    use crate::x86_64::*;
    use super::*;

    fn test(bytes: &[u8], display: &str) {
        test_with_encoder(&mut MicroEncoder::new(), bytes, display);
    }

    fn test_with_encoder(encoder: &mut MicroEncoder, bytes: &[u8], display: &str) {
        let instruction = Instruction::decode(bytes).unwrap();
        let code = encoder.encode(&instruction).unwrap();
        let display = codify(display);
        println!("==================================");
        println!("Instruction: {:#02x?}", bytes);
        println!("Encoded: {}", code);
        println!("Display: {}\n", display);
        assert_eq!(code.to_string(), display);
    }

    fn codify(code: &str) -> String {
        let mut output = "Microcode [\n".to_string();
        for line in code.lines() {
            if !line.chars().all(|c| c.is_whitespace()) {
                output.push_str("    ");
                output.push_str(line.trim());
                output.push('\n');
            }
        }
        output.push(']');
        output
    }

    #[test]
    fn bin_ops() {
        // Instruction: add r8, qword ptr [rdi+0xa]
        // The microcode works as follows:
        // - Move r8 into t0
        // - Move rdi into t1, move 0xa into t2, sum them up into t3
        // - Load the value at address t3 into t4
        // - Compute the sum of t0 and t4 and store it in t5
        // - Move t5 into r8
        test(&[0x4c, 0x03, 0x47, 0x0a], "
            mov T0:n64 = [m1][0x40:n64]
            mov T1:n64 = [m1][0x38:n64]
            const T2:n64 = 0xa:n64
            add T1:n64 = T1:n64 + T2:n64
            mov T3:n64 = [m0][(T1:n64):n64]
            add T4:n64 = T0:n64 + T3:n64
            mov [m1][0x40:n64] = T4:n64
        ");

        // Instruction: sub rsp, 0x10
        test(&[0x48, 0x83, 0xec, 0x10], "
            mov T0:n64 = [m1][0x20:n64]
            const T1:n8 = 0x10:n8
            cast T1:n8 to n64 signed
            sub T2:n64 = T0:n64 - T1:n64
            mov [m1][0x20:n64] = T2:n64
        ");

        // Instruction: sub eax, 0x20
        test(&[0x83, 0xe8, 0x20], "
            mov T0:n32 = [m1][0x0:n32]
            const T1:n8 = 0x20:n8
            cast T1:n8 to n32 signed
            sub T2:n32 = T0:n32 - T1:n32
            mov [m1][0x0:n32] = T2:n32
        ");
    }

    #[test]
    fn moves() {
        // Instruction: mov esi, edx
        test(&[0x89, 0xd6], "mov [m1][0x30:n32] = [m1][0x10:n32]");

        // Instruction: mov rax, 0x3c
        test(&[0x48, 0xc7, 0xc0, 0x3c, 0x00, 0x00, 0x00], "
            const T0:n32 = 0x3c:n32
            cast T0:n32 to n64 signed
            mov [m1][0x0:n64] = T0:n64
        ");

        // Instruction: mov dword ptr [rbp-0x4], edi
        test(&[0x89, 0x7d, 0xfc], "
            mov T0:n64 = [m1][0x28:n64]
            const T1:n64 = 0xfffffffffffffffc:n64
            add T0:n64 = T0:n64 + T1:n64
            mov [m0][(T0:n64):n32] = [m1][0x38:n32]
        ");

        // Instruction: mov dword ptr [rbp-0x8], 0xa
        test(&[0xc7, 0x45, 0xf8, 0x0a, 0x00, 0x00, 0x00], "
            mov T0:n64 = [m1][0x28:n64]
            const T1:n64 = 0xfffffffffffffff8:n64
            add T0:n64 = T0:n64 + T1:n64
            const T2:n32 = 0xa:n32
            mov [m0][(T0:n64):n32] = T2:n32
        ");

        // Instruction: lea rax, qword ptr [rbp-0xc]
        test(&[0x48, 0x8d, 0x45, 0xf4], "
            mov T0:n64 = [m1][0x28:n64]
            const T1:n64 = 0xfffffffffffffff4:n64
            add T0:n64 = T0:n64 + T1:n64
            mov [m1][0x0:n64] = T0:n64
        ");

        // Instruction: lea rbx, qword ptr [rdx+rax*1]
        test(&[0x48, 0x8d, 0x1c, 0x02], "
            mov T0:n64 = [m1][0x10:n64]
            mov T1:n64 = [m1][0x0:n64]
            const T2:n64 = 0x1:n64
            mul T1:n64 = T1:n64 * T2:n64
            add T0:n64 = T0:n64 + T1:n64
            mov [m1][0x18:n64] = T0:n64
        ");

        // Instruction: movzx eax, al
        test(&[0x0f, 0xb6, 0xc0], "
            mov T0:n8 = [m1][0x0:n8]
            cast T0:n8 to n32 unsigned
            mov [m1][0x0:n32] = T0:n32
        ");

        // Instruction: push rbp
        test(&[0x55], "
            mov T0:n64 = [m1][0x20:n64]
            const T1:n64 = 0x8:n64
            sub T0:n64 = T0:n64 - T1:n64
            mov [m0][(T0:n64):n64] = [m1][0x28:n64]
            mov [m1][0x20:n64] = T0:n64
        ");

        // Instruction: pop rbp
        test(&[0x5d], "
            mov T0:n64 = [m1][0x20:n64]
            mov [m1][0x28:n64] = [m0][(T0:n64):n64]
            const T1:n64 = 0x8:n64
            add T0:n64 = T0:n64 + T1:n64
            mov [m1][0x20:n64] = T0:n64
        ");
    }

    #[test]
    fn compares() {
        // Instruction: cmp eax, dword ptr [rbp-0x8]
        test(&[0x3b, 0x45, 0xf8], "
            mov T0:n32 = [m1][0x0:n32]
            mov T1:n64 = [m1][0x28:n64]
            const T2:n64 = 0xfffffffffffffff8:n64
            add T1:n64 = T1:n64 + T2:n64
            mov T3:n32 = [m0][(T1:n64):n32]
        ");

        let mut enc = MicroEncoder::new();

        // Instruction: cmp eax, eax
        test_with_encoder(&mut enc, &[0x39, 0xc0], "
            mov T0:n32 = [m1][0x0:n32]
            mov T1:n32 = [m1][0x0:n32]
        ");

        // Instruction: setl al
        test_with_encoder(&mut enc, &[0x0f, 0x9c, 0xc0], "
            set T2:n8 if (T0:n32 < T1:n32 signed)
            mov [m1][0x0:n8] = T2:n8
        ");
    }

    #[test]
    fn jumps() {
        // Instruction: jmp +0x7
        test(&[0xeb, 0x07], "
            const T0:n64 = 0x7:n64
            jump by T0:n64
        ");

        let mut enc = MicroEncoder::new();

        // Instruction: test eax, eax
        test_with_encoder(&mut enc, &[0x85, 0xc0], "
            mov T0:n32 = [m1][0x0:n32]
            mov T1:n32 = [m1][0x0:n32]
        ");

        // Instruction: sub rsp, 0x10
        test_with_encoder(&mut enc, &[0x48, 0x83, 0xec, 0x10], "
            mov T2:n64 = [m1][0x20:n64]
            const T3:n8 = 0x10:n8
            cast T3:n8 to n64 signed
            sub T4:n64 = T2:n64 - T3:n64
            mov [m1][0x20:n64] = T4:n64
        ");

        // Instruction: je +0xe
        test_with_encoder(&mut enc, &[0x74, 0x0e], "
            const T5:n64 = 0xe:n64
            jump by T5:n64 if (T2:n64 == T3:n64)
        ");

        // Instruction: call -0x76
        test(&[0xe8, 0x8a, 0xff, 0xff, 0xff], "
            mov T0:n64 = [m1][0x20:n64]
            const T1:n64 = 0x8:n64
            sub T0:n64 = T0:n64 - T1:n64
            mov [m0][(T0:n64):n64] = [m1][0x80:n64]
            mov [m1][0x20:n64] = T0:n64
            const T2:n64 = 0xffffffffffffff8a:n64
            jump by T2:n64
        ");

        // Instruction: call rdx
        test(&[0xff, 0xd2], "
            mov T0:n64 = [m1][0x20:n64]
            const T1:n64 = 0x8:n64
            sub T0:n64 = T0:n64 - T1:n64
            mov [m0][(T0:n64):n64] = [m1][0x80:n64]
            mov [m1][0x20:n64] = T0:n64
            mov T2:n64 = [m1][0x10:n64]
            jump to T2:n64
        ");

        // Instruction: leave
        test(&[0xc9], "
            mov [m1][0x20:n64] = [m1][0x28:n64]
            mov T0:n64 = [m1][0x20:n64]
            mov [m1][0x28:n64] = [m0][(T0:n64):n64]
            const T1:n64 = 0x8:n64
            add T0:n64 = T0:n64 + T1:n64
            mov [m1][0x20:n64] = T0:n64
        ");

        // Instruction: ret
        test(&[0xc3], "
            mov T1:n64 = [m1][0x20:n64]
            mov T0:n64 = [m0][(T1:n64):n64]
            const T2:n64 = 0x8:n64
            add T1:n64 = T1:n64 + T2:n64
            mov [m1][0x20:n64] = T1:n64
            jump to T0:n64
        ");
    }
}
