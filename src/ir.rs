//! Microcode representation of instructions.

use std::fmt::{self, Debug, Display, Formatter};
use crate::amd64::{Instruction, Mnemoic, Operand, Register};
use crate::num::{DataType, Integer};


/// Microcode composed of any number of micro operations.
#[derive(Debug, Clone, PartialEq)]
pub struct Microcode {
    pub ops: Vec<MicroOperation>,
}

impl Microcode {
    /// Express the given instruction in microcode.
    pub fn from_instruction(instruction: &Instruction) -> EncodeResult<Microcode> {
        Encoder::new(instruction).encode()
    }
}

/// Encodes instructions into microcode.
#[derive(Debug)]
struct Encoder<'a> {
    inst: &'a Instruction,
    ops: Vec<MicroOperation>,
    temps: usize,
}

impl<'a> Encoder<'a> {
    /// Create a new encoder.
    fn new(inst: &'a Instruction) -> Encoder<'a> {
        Encoder { inst, ops: vec![], temps: 0 }
    }

    /// Encode the instruction into microcode.
    fn encode(mut self) -> EncodeResult<Microcode> {
        use MicroOperation as Op;
        use Mnemoic::*;

        match self.inst.mnemoic {
            // Load both operands, perform an operation and write the result back.
            Add => self.encode_binop(|sum, a, b| Op::Add { sum, a, b, flags: true }, true),
            Sub => self.encode_binop(|diff, a, b| Op::Sub { diff, a, b, flags: true }, true),
            Imul => self.encode_binop(|prod, a, b| Op::Mul { prod, a, b, flags: true }, true),

            // Retrieve both locations and move from source to destination.
            Mov => {
                let dest = self.encode_get_location(self.inst.operands[0]);
                let src = self.encode_get_location(self.inst.operands[1]);
                if dest.data_type() != src.data_type() {
                    self.ops.clear();
                    self.temps = 0;
                    self.encode_move_casted(true);
                } else {
                    self.encode_move(dest, src)?;
                }
            },

            // Load the source, cast it to the destination type and move it there.
            Movzx => self.encode_move_casted(false),

            // Retrieve both locations, but instead of loading just move the
            // address into the destination.
            Lea => {
                let dest = self.encode_get_location(self.inst.operands[0]);
                let src = self.encode_get_location(self.inst.operands[1]);

                if let Location::Indirect(_, _, temp) = src {
                    self.encode_move(dest, Location::Temp(temp))?;
                } else {
                    panic!("encode: invalid source for lea");
                }
            },

            // Store or load data on the stack and adjust the stack pointer.
            Push => {
                let src = self.encode_get_location(self.inst.operands[0]);
                self.encode_push(src);
            },
            Pop => {
                let dest = self.encode_get_location(self.inst.operands[0]);
                self.encode_pop(dest);
            },

            // Jump to the first operand under specific conditions.
            Jmp => self.encode_jump(Condition::Always),
            Je => self.encode_jump(Condition::Equal),
            Jg => self.encode_jump(Condition::Greater),

            // Save the procedure linking information on the stack and jump.
            Call => {
                let rip = self.encode_get_location(Operand::Direct(Register::RIP));
                self.encode_push(rip);
                self.encode_jump(Condition::Always);
            },

            // Copies the base pointer into the stack pointer register and pops the
            // old base pointer from the stack.
            Leave => {
                let rbp = self.encode_get_location(Operand::Direct(Register::RBP));
                let rsp = self.encode_get_location(Operand::Direct(Register::RSP));
                self.encode_move(rsp, rbp).unwrap();
                self.encode_pop(rbp);
            },

            // Jumps back to the address located on top of the stack.
            Ret => {
                let target = Temporary(DataType::N64, self.temps);
                self.temps += 1;
                self.encode_pop(Location::Temp(target));
                self.ops.push(Op::Jump { target, condition: Condition::Always, relative: false });
            },

            Cmp => self.encode_binop(|diff, a, b| Op::Sub { diff, a, b, flags: true }, false),
            Test => self.encode_binop(|and, a, b| Op::And { and, a, b, flags: true }, false),
            Setl => self.encode_set(Condition::Less),

            Syscall => { self.ops.push(Op::Syscall); },
            Nop => {},
        };

        Ok(Microcode { ops: self.ops })
    }

    /// Encode a binary operation like add or subtract.
    fn encode_binop<F>(&mut self, binop: F, store: bool)
    where F: FnOnce(Temporary, Temporary, Temporary) -> MicroOperation {
        // Encode the loading of both operands into a temporary.
        let (dest, left) = self.encode_load_operand(self.inst.operands[0]);
        let (_, mut right) = self.encode_load_operand(self.inst.operands[1]);

        // Enforce that both operands have the exact same data type.
        if left.0 != right.0 {
            self.ops.push(MicroOperation::Cast { target: right, new: left.0, signed: true });
            right.0 = left.0;
        }

        // Encode the actual binary operation and the move from the target temporary
        // into the destination.
        let target = Temporary(left.0, self.temps);
        self.temps += 1;
        self.ops.push(binop(target, left, right));
        if store {
            self.ops.push(MicroOperation::Mov { dest, src: Location::Temp(target) });
        }
    }

    /// Encode a relative jump.
    fn encode_jump(&mut self, condition: Condition) {
        if let Operand::Offset(offset) = self.inst.operands[0] {
            let target = Temporary(DataType::N64, self.temps);
            let constant = Integer(DataType::N64, offset as u64);
            self.ops.push(MicroOperation::Const { dest: Location::Temp(target), constant });
            self.ops.push(MicroOperation::Jump { target, condition, relative: true });
            self.temps += 1;
        } else {
            panic!("encode_jump: invalid operand for jump");
        }
    }

    /// Encode a set instruction.
    fn encode_set(&mut self, condition: Condition) {
        let location = self.encode_get_location(self.inst.operands[0]);
        let temp = Temporary(location.data_type(), self.temps);
        self.temps += 1;
        self.ops.push(MicroOperation::Set { target: temp, condition });
        self.encode_move(location, Location::Temp(temp)).unwrap();
    }

    /// Load the stack pointer, decrement it by the operand size, store the
    /// operand on the stack and store the new stack pointer in the register.
    fn encode_push(&mut self, src: Location) {
        // Load the stack pointer.
        let (sp, stack) = self.encode_load_operand(Operand::Direct(Register::RSP));
        let data_type = src.data_type();

        // Load the width of the moved thing as a constant and subtract it from the
        // stack pointer.
        let offset = Temporary(DataType::N64, self.temps);
        let constant = Integer(DataType::N64, data_type.bytes());
        self.ops.push(MicroOperation::Const { dest: Location::Temp(offset), constant });
        self.ops.push(MicroOperation::Sub { diff: stack, a: stack, b: offset, flags: false });
        self.temps += 1;

        // Move the value from the source onto the stack.
        let stack_space = Location::Indirect(data_type, 0, stack);
        self.encode_move(stack_space, src).unwrap();

        // Copy back the stack pointer.
        self.encode_move(sp, Location::Temp(stack)).unwrap();
    }

    /// Load the operand from the stack, load the stack pointer, increment it
    /// by the operand size and store the new stack pointer in the register.
    fn encode_pop(&mut self, dest: Location) {
        // Load the stack pointer.
        let (sp, stack) = self.encode_load_operand(Operand::Direct(Register::RSP));
        let data_type = dest.data_type();

        // Move the value from the stack into the destination.
        let stack_space = Location::Indirect(data_type, 0, stack);
        self.encode_move(dest, stack_space).unwrap();

        // Load the width of the moved thing as a constant and add it to the
        // stack pointer. Then copy the stack pointer back into it's register.
        let offset = Temporary(DataType::N64, self.temps);
        let constant = Integer(DataType::N64, data_type.bytes());
        self.ops.push(MicroOperation::Const { dest: Location::Temp(offset), constant });
        self.ops.push(MicroOperation::Add { sum: stack, a: stack, b: offset, flags: false });
        self.temps += 1;
        self.encode_move(sp, Location::Temp(stack)).unwrap();
    }

    /// Encode moving with a cast to the destination source type.
    fn encode_move_casted(&mut self, signed: bool) {
        let dest = self.encode_get_location(self.inst.operands[0]);
        let (_, mut temp) = self.encode_load_operand(self.inst.operands[1]);
        let new = dest.data_type();
        self.ops.push(MicroOperation::Cast { target: temp, new, signed });
        temp.0 = new;
        self.encode_move(dest, Location::Temp(temp)).unwrap();
    }

    /// Encode the micro operations to load the operand into a temporary.
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

    /// Encode the micro operations to prepare the location of the operand.
    /// The operand itself will not be loaded and the operations may be empty
    /// if it is a direct operand.
    fn encode_get_location(&mut self, operand: Operand) -> Location {
        use Location::*;

        match operand {
            Operand::Direct(reg) => Direct(reg.data_type(), 1, reg.address()),

            Operand::Indirect(data_type, reg) => {
                // Load the address into a temporary.
                let reg = self.encode_load_reg(reg);
                Indirect(data_type, 0, reg)
            },

            Operand::IndirectDisplaced(data_type, reg, displace) => {
                // Load the base address into a temporary.
                let reg = self.encode_load_reg(reg);

                // Load the displacement constant into a temporary.
                let constant = Temporary(DataType::N64, self.temps);
                self.ops.push(MicroOperation::Const {
                    dest: Location::Temp(constant),
                    constant: Integer(DataType::N64, displace as u64)
                });

                // Compute the final address.
                self.ops.push(MicroOperation::Add {
                    sum: Temporary(DataType::N64, self.temps + 1),
                    a: reg, b: constant, flags: false
                });
                self.temps += 2;

                Indirect(data_type, 0, Temporary(DataType::N64, self.temps - 1))
            },

            Operand::Immediate(data_type, immediate) => {
                // Load the immediate into a temporary.
                Temp(self.encode_load_constant(data_type, immediate))
            },

            Operand::Offset(offset) => {
                Temp(self.encode_load_constant(DataType::N64, offset as u64))
            },
        }
    }

    /// Encode the micro operations to load a register from memory into a temporary.
    /// The resulting temporary will have the data type matching the registers width.
    fn encode_load_reg(&mut self, reg: Register) -> Temporary {
        let data_type = reg.data_type();
        let temp = Temporary(data_type, self.temps);

        let src = Location::Direct(data_type, 1, reg.address());
        self.ops.push(MicroOperation::Mov { dest: Location::Temp(temp), src });
        self.temps += 1;

        temp
    }

    /// Encode the micro operations to load a constant into a temporary.
    fn encode_load_constant(&mut self, data_type: DataType, constant: u64) -> Temporary {
        let temp = Temporary(data_type, self.temps);

        self.ops.push(MicroOperation::Const {
            dest: Location::Temp(temp),
            constant: Integer(data_type, constant)
        });
        self.temps += 1;

        temp
    }

    /// Encode a move operation with check for matching data types.
    fn encode_move(&mut self, dest: Location, src: Location) -> EncodeResult<()> {
        // Enforce that both operands have the exact same data type.
        if src.data_type() != dest.data_type() {
            return Err(EncodeError::new(self.inst.clone(),
                format!("incompatible data types for move: {} and {}",
                    src.data_type(), dest.data_type())));
        }
        Ok(self.ops.push(MicroOperation::Mov { dest, src }))
    }
}

impl Display for Microcode {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Microcode [")?;
        if !self.ops.is_empty() {
            writeln!(f)?;
        }
        for operation in &self.ops {
            writeln!(f, "    {}", operation)?;
        }
        write!(f, "]")
    }
}

/// Describes one atomic operation.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum MicroOperation {
    /// Store the value at location `src` in location `dest`.
    Mov { dest: Location, src: Location },
    /// Store a constant in location `dest`.
    Const { dest: Location, constant: Integer },
    /// Cast the temporary `target` to another type.
    /// - If the target type is smaller, it will get truncated.
    /// - If the target type is bigger, if signed is true the value will be
    ///   sign-extended and otherwise zero-extended.
    Cast { target: Temporary, new: DataType, signed: bool },

    /// Store the sum of `a` and `b` in `sum`. Set flags if active.
    Add { sum: Temporary, a: Temporary, b: Temporary, flags: bool },
    /// Store the difference of `a` and `b` in `diff`. Set flags if active.
    Sub { diff: Temporary, a: Temporary, b: Temporary, flags: bool },
    /// Store the product of `a` and `b` in `prod`. Set flags if active.
    Mul { prod: Temporary, a: Temporary, b: Temporary, flags: bool },

    /// Store the bitwise AND of `a` and `b` in and. Set flags if active.
    And { and: Temporary, a: Temporary, b: Temporary, flags: bool },
    /// Store the bitwise OR of `a` and `b` in or. Set flags if active.
    Or { or: Temporary, a: Temporary, b: Temporary, flags: bool },
    /// Store the bitwise NOT of `a` in `not`.
    Not { not: Temporary, a: Temporary },

    /// Set the target temporary to one if the condition is true and to zero otherwise.
    Set { target: Temporary, condition: Condition },
    /// Jump to the current address plus the `offset` if `relative` is true,
    /// otherwise directly to the target if the condition specified by `condition`
    /// is fulfilled.
    Jump { target: Temporary, condition: Condition, relative: bool },

    /// Perform a syscall.
    Syscall,
}

impl Display for MicroOperation {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use MicroOperation::*;
        fn flag(flags: bool) -> &'static str { if flags { "[flags]" } else { "" } }
        match *self {
            Mov { dest, src } => write!(f, "mov {} = {}", dest, src),
            Const { dest, constant } => write!(f, "const {} = {}", dest, constant),
            Cast { target, new, signed } => write!(f, "cast {} to {} {}", target, new,
                if signed { "signed" } else { "unsigned" }),

            Add { sum, a, b, flags } => write!(f, "add{} {} = {} + {}", flag(flags), sum, a, b),
            Sub { diff, a, b, flags } => write!(f, "sub{} {} = {} - {}", flag(flags), diff, a, b),
            Mul { prod, a, b, flags } => write!(f, "mul{} {} = {} * {}", flag(flags), prod, a, b),

            And { and, a, b, flags } => write!(f, "and{} {} = {} & {}", flag(flags), and, a, b),
            Or { or, a, b, flags } => write!(f, "or{} {} = {} | {}", flag(flags), or, a, b),
            Not { not, a } => write!(f, "not {} = !{}", not, a),

            Set { target, condition } => write!(f, "set {} {}", target, condition),
            Jump { target, condition, relative } => write!(f, "jump {} {} {}",
                if relative { "by" } else { "to" }, target, condition),

            Syscall => write!(f, "syscall"),
        }
    }
}

/// Strongly typed target for moves.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Location {
    Temp(Temporary),
    Direct(DataType, usize, u64),
    Indirect(DataType, usize, Temporary),
}

impl Location {
    /// Underlying data type of values at the location.
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

/// Strongly typed temporary identified by an index.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Temporary(pub DataType, pub usize);

impl Display for Temporary {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "T{}:{}", self.1, self.0)
    }
}

/// Condition for jumps and sets.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Condition {
    Always,
    Equal,
    Less,
    Greater,
}

impl Display for Condition {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Condition::Always => write!(f, "always"),
            Condition::Equal => write!(f, "if equal"),
            Condition::Less => write!(f, "if less"),
            Condition::Greater => write!(f, "if greater"),
        }
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
        use Register::*;
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

/// Error type for microcode encoding.
#[derive(Eq, PartialEq)]
pub struct EncodeError {
    pub instruction: Instruction,
    pub message: String,
}

impl EncodeError {
    /// Create a new encoding error with a mesage.
    fn new<S: Into<String>>(instruction: Instruction, message: S) -> EncodeError {
        EncodeError { instruction, message: message.into() }
    }
}

/// Result type for instruction decoding.
pub(in super) type EncodeResult<T> = Result<T, EncodeError>;
impl std::error::Error for EncodeError {}

impl Display for EncodeError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Failed to encode instruction {}: {}.", self.instruction, self.message)
    }
}

impl Debug for EncodeError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Display::fmt(self, f)
    }
}


#[cfg(test)]
mod test {
    use crate::amd64::*;
    use super::*;

    fn test(bytes: &[u8], display: &str) {
        let instruction = Instruction::decode(bytes).unwrap();
        let code = Microcode::from_instruction(&instruction).unwrap();
        let display = codify(display);
        println!("==================================");
        println!("bytes: {:#02x?}", bytes);
        println!("encoded: {}", code);
        println!("display: {}", display);
        println!();
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
    fn binops() {
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
            add T3:n64 = T1:n64 + T2:n64
            mov T4:n64 = [m0][(T3:n64):n64]
            add[flags] T5:n64 = T0:n64 + T4:n64
            mov [m1][0x40:n64] = T5:n64
        ");

        // Instruction: sub rsp, 0x10
        test(&[0x48, 0x83, 0xec, 0x10], "
            mov T0:n64 = [m1][0x20:n64]
            const T1:n8 = 0x10:n8
            cast T1:n8 to n64 signed
            sub[flags] T2:n64 = T0:n64 - T1:n64
            mov [m1][0x20:n64] = T2:n64
        ");

        // Instruction: sub eax, 0x20
        test(&[0x83, 0xe8, 0x20], "
            mov T0:n32 = [m1][0x0:n32]
            const T1:n8 = 0x20:n8
            cast T1:n8 to n32 signed
            sub[flags] T2:n32 = T0:n32 - T1:n32
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
            add T2:n64 = T0:n64 + T1:n64
            mov [m0][(T2:n64):n32] = [m1][0x38:n32]
        ");

        // Instruction: mov dword ptr [rbp-0x8], 0xa
        test(&[0xc7, 0x45, 0xf8, 0x0a, 0x00, 0x00, 0x00], "
            mov T0:n64 = [m1][0x28:n64]
            const T1:n64 = 0xfffffffffffffff8:n64
            add T2:n64 = T0:n64 + T1:n64
            const T3:n32 = 0xa:n32
            mov [m0][(T2:n64):n32] = T3:n32
        ");

        // Instruction: lea rax, qword ptr [rbp-0xc]
        test(&[0x48, 0x8d, 0x45, 0xf4], "
            mov T0:n64 = [m1][0x28:n64]
            const T1:n64 = 0xfffffffffffffff4:n64
            add T2:n64 = T0:n64 + T1:n64
            mov [m1][0x0:n64] = T2:n64
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
            add T3:n64 = T1:n64 + T2:n64
            mov T4:n32 = [m0][(T3:n64):n32]
            sub[flags] T5:n32 = T0:n32 - T4:n32
        ");

        // Instruction: test eax, eax
        test(&[0x85, 0xc0], "
            mov T0:n32 = [m1][0x0:n32]
            mov T1:n32 = [m1][0x0:n32]
            and[flags] T2:n32 = T0:n32 & T1:n32
        ");

        // Instruction: setl al
        test(&[0x0f, 0x9c, 0xc0], "
            set T0:n8 if less
            mov [m1][0x0:n8] = T0:n8
        ");
    }

    #[test]
    fn jumps() {
        // Instruction: jmp +0x7
        test(&[0xeb, 0x07], "
            const T0:n64 = 0x7:n64
            jump by T0:n64 always
        ");

        // Instruction: jg +0x9
        test(&[0x7f, 0x09], "
            const T0:n64 = 0x9:n64
            jump by T0:n64 if greater
        ");

        // Instruction: je +0xe
        test(&[0x74, 0x0e], "
            const T0:n64 = 0xe:n64
            jump by T0:n64 if equal
        ");

        // Instruction: call -0x76
        test(&[0xe8, 0x8a, 0xff, 0xff, 0xff], "
            mov T0:n64 = [m1][0x20:n64]
            const T1:n64 = 0x8:n64
            sub T0:n64 = T0:n64 - T1:n64
            mov [m0][(T0:n64):n64] = [m1][0x80:n64]
            mov [m1][0x20:n64] = T0:n64
            const T2:n64 = 0xffffffffffffff8a:n64
            jump by T2:n64 always
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
            jump to T0:n64 always
        ");
    }
}
