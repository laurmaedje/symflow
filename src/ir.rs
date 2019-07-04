//! Microcode representation of instructions.

use std::fmt::{self, Debug, Display, Formatter};
use crate::amd64::{Instruction, Mnemoic, Operand, Register, Flag};
use crate::num::{DataType, DataWidth, Integer};


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
        use {Mnemoic::*, DataType::*, MicroOperation as Op};

        match self.inst.mnemoic {
            // Load both operands, perform an operation and write the result back.
            Add => self.encode_binop(|sum, a, b| Op::Add { sum, a, b, flags: true }, true),
            Sub => self.encode_binop(|diff, a, b| Op::Sub { diff, a, b, flags: true }, true),
            Imul => self.encode_binop(|prod, a, b| Op::Mul { prod, a, b, flags: true }, true),

            // Retrieve both locations and move from source to destination.
            Mov => {
                let dest = self.encode_get_location(self.inst.operands[0], true);
                let src = self.encode_get_location(self.inst.operands[1], true);
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
                let dest = self.encode_get_location(self.inst.operands[0], false);
                let src = self.encode_get_location(self.inst.operands[1], true);

                if let Location::Indirect(_, _, temp) = src {
                    self.encode_move(dest, Location::Temp(temp))?;
                } else {
                    panic!("encode: invalid source for lea");
                }
            },

            // Store or load data on the stack and adjust the stack pointer.
            Push => {
                let src = self.encode_get_location(self.inst.operands[0], true);
                self.encode_push(src);
            },
            Pop => {
                let dest = self.encode_get_location(self.inst.operands[0], true);
                self.encode_pop(dest);
            },

            // Jump to the first operand under specific conditions.
            Jmp => self.encode_jump(Condition::Always),
            Je => self.encode_jump(Condition::Equal),
            Jg => self.encode_jump(Condition::Greater),

            // Save the procedure linking information on the stack and jump.
            Call => {
                let rip = self.encode_get_location(Operand::Direct(Register::RIP), false);
                self.encode_push(rip);
                self.encode_jump(Condition::Always);
            },

            // Copies the base pointer into the stack pointer register and pops the
            // old base pointer from the stack.
            Leave => {
                let rbp = self.encode_get_location(Operand::Direct(Register::RBP), false);
                let rsp = self.encode_get_location(Operand::Direct(Register::RSP), false);
                self.encode_move(rsp, rbp).unwrap();
                self.encode_pop(rbp);
            },

            // Jumps back to the address located on top of the stack.
            Ret => {
                let target = Temporary(U64, self.temps);
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
        let (dest, left) = self.encode_load_operand(self.inst.operands[0], true);
        let (_, mut right) = self.encode_load_operand(self.inst.operands[1], true);

        // Enforce that both operands have the exact same data type.
        if left.0 != right.0 {
            self.ops.push(MicroOperation::Cast { target: right, new: left.0 });
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
            let target = Temporary(DataType::I64, self.temps);
            let constant = Integer(DataType::I64, offset as u64);
            self.ops.push(MicroOperation::Const { dest: Location::Temp(target), constant });
            self.ops.push(MicroOperation::Jump { target, condition, relative: true });
            self.temps += 1;
        } else {
            panic!("encode_jump: invalid operand for jump");
        }
    }

    /// Encode a set instruction.
    fn encode_set(&mut self, condition: Condition) {
        let location = self.encode_get_location(self.inst.operands[0], false);
        let temp = Temporary(location.data_type(), self.temps);
        self.temps += 1;
        self.ops.push(MicroOperation::Set { target: temp, condition });
        self.encode_move(location, Location::Temp(temp)).unwrap();
    }

    /// Load the stack pointer, decrement it by the operand size, store the
    /// operand on the stack and store the new stack pointer in the register.
    fn encode_push(&mut self, src: Location) {
        // Load the stack pointer.
        let (sp, stack) = self.encode_load_operand(Operand::Direct(Register::RSP), false);
        let data_type = src.data_type();

        // Load the width of the moved thing as a constant and subtract it from the
        // stack pointer.
        let offset = Temporary(DataType::U64, self.temps);
        let constant = Integer(DataType::U64, data_type.width().bytes());
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
        let (sp, stack) = self.encode_load_operand(Operand::Direct(Register::RSP), false);
        let data_type = dest.data_type();

        // Move the value from the stack into the destination.
        let stack_space = Location::Indirect(data_type, 0, stack);
        self.encode_move(dest, stack_space).unwrap();

        // Load the width of the moved thing as a constant and add it to the
        // stack pointer. Then copy the stack pointer back into it's register.
        let offset = Temporary(DataType::U64, self.temps);
        let constant = Integer(DataType::U64, data_type.width().bytes());
        self.ops.push(MicroOperation::Const { dest: Location::Temp(offset), constant });
        self.ops.push(MicroOperation::Add { sum: stack, a: stack, b: offset, flags: false });
        self.temps += 1;
        self.encode_move(sp, Location::Temp(stack)).unwrap();
    }

    /// Encode moving with a cast to the destination source type.
    fn encode_move_casted(&mut self, signed: bool) {
        let dest = self.encode_get_location(self.inst.operands[0], signed);
        let (_, mut temp) = self.encode_load_operand(self.inst.operands[1], signed);
        let new = dest.data_type();
        self.ops.push(MicroOperation::Cast { target: temp, new });
        temp.0 = new;
        self.encode_move(dest, Location::Temp(temp)).unwrap();
    }

    /// Encode the micro operations to load the operand into a temporary.
    fn encode_load_operand(&mut self, operand: Operand, signed: bool) -> (Location, Temporary) {
        let location = self.encode_get_location(operand, signed);
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
    fn encode_get_location(&mut self, operand: Operand, signed: bool) -> Location {
        use {Location::*, DataType::*};

        match operand {
            Operand::Direct(reg) => {
                // Locate the registers in memory.
                let data_type = DataType::from_width(reg.width(), signed);
                Direct(data_type, 1, reg.address())
            },

            Operand::Indirect(width, reg) => {
                // Load the address into a temporary.
                let reg = self.encode_load_reg(reg, false);
                let data_type = DataType::from_width(width, signed);
                Indirect(data_type, 0, reg)
            },

            Operand::IndirectDisplaced(width, reg, displace) => {
                // Load the base address into a temporary.
                let reg = self.encode_load_reg(reg, false);

                // Load the displacement constant into a temporary.
                let constant = Temporary(U64, self.temps);
                self.ops.push(MicroOperation::Const {
                    dest: Location::Temp(constant),
                    constant: Integer(U64, displace as u64)
                });

                // Compute the final address.
                self.ops.push(MicroOperation::Add {
                    sum: Temporary(U64, self.temps + 1),
                    a: reg, b: constant, flags: false
                });
                self.temps += 2;

                let data_type = DataType::from_width(width, signed);
                Indirect(data_type, 0, Temporary(U64, self.temps - 1))
            },

            Operand::Immediate(width, immediate) => {
                // Load the immediate into a temporary.
                let data_type = DataType::from_width(width, signed);
                Temp(self.encode_load_constant(data_type, immediate))
            },

            Operand::Offset(offset) => {
                let data_type = if signed { I64 } else { U64 };
                Temp(self.encode_load_constant(data_type, offset as u64))
            },
        }
    }

    /// Encode the micro operations to load a register from memory into a temporary.
    /// The resulting temporary will have the data type matching the registers width.
    fn encode_load_reg(&mut self, reg: Register, signed: bool) -> Temporary {
        let data_type = DataType::from_width(reg.width(), signed);
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
    /// Cast the temporary `target` to another type
    /// (possibly zero-extending, sign-extending or truncating).
    Cast { target: Temporary, new: DataType },

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
        fn flag(flags: bool) -> &'static str { if flags { " (flags)" } else { "" } }
        match *self {
            Mov { dest, src } => write!(f, "mov {} = {}", dest, src),
            Const { dest, constant } => write!(f, "const {} = {}", dest, constant),
            Cast { target, new } => write!(f, "cast {} to {}", target, new),

            Add { sum, a, b, flags } => write!(f, "add {} = {} + {}{}", sum, a, b, flag(flags)),
            Sub { diff, a, b, flags } => write!(f, "sub {} = {} - {}{}", diff, a, b, flag(flags)),
            Mul { prod, a, b, flags } => write!(f, "mul {} = {} * {}{}", prod, a, b, flag(flags)),

            And { and, a, b, flags } => write!(f, "and {} = {} & {}{}", and, a, b, flag(flags)),
            Or { or, a, b, flags } => write!(f, "or {} = {} | {}{}", or, a, b, flag(flags)),
            Not { not, a } => write!(f, "not {} = !{}", not, a),

            Set { target, condition } => write!(f, "set {} {}", target, condition),
            Jump { target, condition, relative } => write!(f, "jump {} {} {}",
                if relative { "by" } else { "to" }, target, condition),

            Syscall => write!(f, "syscall"),
        }
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
        use Location::*;
        match *self {
            Temp(temp) => temp.0,
            Direct(data, _, _) => data,
            Indirect(data, _, _) => data,
        }
    }
}

impl Display for Location {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use Location::*;
        match self {
            Temp(temp) => write!(f, "{}", temp),
            Direct(data, space, offset) => write!(f, "[mem{}][{:#x}:{}]", space, offset, data),
            Indirect(data, space, temp) => write!(f, "[mem{}][({}):{}]", space, temp, data),
        }
    }
}

/// Strongly typed temporary identified by an index.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Temporary(pub DataType, pub usize);

impl Display for Temporary {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "T{}:{}", self.1, self.0)
    }
}

/// Addresses of things stored in memory (registers & flags).
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

impl MemoryMapped for Flag {
    /// Address of a flag register in the register memory space.
    fn address(&self) -> u64 {
        use Flag::*;
        match self {
            Carry => 0x100,
            Parity => 0x101,
            Adjust => 0x102,
            Zero => 0x103,
            Sign => 0x104,
            Overflow => 0x105,
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
            mov T0:i64 = [mem1][0x40:i64]
            mov T1:u64 = [mem1][0x38:u64]
            const T2:u64 = 0xa:u64
            add T3:u64 = T1:u64 + T2:u64
            mov T4:i64 = [mem0][(T3:u64):i64]
            add T5:i64 = T0:i64 + T4:i64 (flags)
            mov [mem1][0x40:i64] = T5:i64
        ");

        // Instruction: sub rsp, 0x10
        test(&[0x48, 0x83, 0xec, 0x10], "
            mov T0:i64 = [mem1][0x20:i64]
            const T1:i8 = 0x10:i8
            cast T1:i8 to i64
            sub T2:i64 = T0:i64 - T1:i64 (flags)
            mov [mem1][0x20:i64] = T2:i64
        ");

        // Instruction: sub eax, 0x20
        test(&[0x83, 0xe8, 0x20], "
            mov T0:i32 = [mem1][0x0:i32]
            const T1:i8 = 0x20:i8
            cast T1:i8 to i32
            sub T2:i32 = T0:i32 - T1:i32 (flags)
            mov [mem1][0x0:i32] = T2:i32
        ");
    }

    #[test]
    fn moves() {
        // Instruction: mov esi, edx
        test(&[0x89, 0xd6], "mov [mem1][0x30:i32] = [mem1][0x10:i32]");

        // Instruction: mov rax, 0x3c
        test(&[0x48, 0xc7, 0xc0, 0x3c, 0x00, 0x00, 0x00], "
            const T0:i32 = 0x3c:i32
            cast T0:i32 to i64
            mov [mem1][0x0:i64] = T0:i64
        ");

        // Instruction: mov dword ptr [rbp-0x4], edi
        test(&[0x89, 0x7d, 0xfc], "
            mov T0:u64 = [mem1][0x28:u64]
            const T1:u64 = 0xfffffffffffffffc:u64
            add T2:u64 = T0:u64 + T1:u64
            mov [mem0][(T2:u64):i32] = [mem1][0x38:i32]
        ");

        // Instruction: mov dword ptr [rbp-0x8], 0xa
        test(&[0xc7, 0x45, 0xf8, 0x0a, 0x00, 0x00, 0x00], "
            mov T0:u64 = [mem1][0x28:u64]
            const T1:u64 = 0xfffffffffffffff8:u64
            add T2:u64 = T0:u64 + T1:u64
            const T3:i32 = 0xa:i32
            mov [mem0][(T2:u64):i32] = T3:i32
        ");

        // Instruction: lea rax, qword ptr [rbp-0xc]
        test(&[0x48, 0x8d, 0x45, 0xf4], "
            mov T0:u64 = [mem1][0x28:u64]
            const T1:u64 = 0xfffffffffffffff4:u64
            add T2:u64 = T0:u64 + T1:u64
            mov [mem1][0x0:u64] = T2:u64
        ");

        // Instruction: movzx eax, al
        test(&[0x0f, 0xb6, 0xc0], "
            mov T0:u8 = [mem1][0x0:u8]
            cast T0:u8 to u32
            mov [mem1][0x0:u32] = T0:u32
        ");

        // Instruction: push rbp
        test(&[0x55], "
            mov T0:u64 = [mem1][0x20:u64]
            const T1:u64 = 0x8:u64
            sub T0:u64 = T0:u64 - T1:u64
            mov [mem0][(T0:u64):i64] = [mem1][0x28:i64]
            mov [mem1][0x20:u64] = T0:u64
        ");

        // Instruction: pop rbp
        test(&[0x5d], "
            mov T0:u64 = [mem1][0x20:u64]
            mov [mem1][0x28:i64] = [mem0][(T0:u64):i64]
            const T1:u64 = 0x8:u64
            add T0:u64 = T0:u64 + T1:u64
            mov [mem1][0x20:u64] = T0:u64
        ");
    }

    #[test]
    fn compares() {
        // Instruction: cmp eax, dword ptr [rbp-0x8]
        test(&[0x3b, 0x45, 0xf8], "
            mov T0:i32 = [mem1][0x0:i32]
            mov T1:u64 = [mem1][0x28:u64]
            const T2:u64 = 0xfffffffffffffff8:u64
            add T3:u64 = T1:u64 + T2:u64
            mov T4:i32 = [mem0][(T3:u64):i32]
            sub T5:i32 = T0:i32 - T4:i32 (flags)
        ");

        // Instruction: test eax, eax
        test(&[0x85, 0xc0], "
            mov T0:i32 = [mem1][0x0:i32]
            mov T1:i32 = [mem1][0x0:i32]
            and T2:i32 = T0:i32 & T1:i32 (flags)
        ");

        // Instruction: setl al
        test(&[0x0f, 0x9c, 0xc0], "
            set T0:u8 if less
            mov [mem1][0x0:u8] = T0:u8
        ");
    }

    #[test]
    fn jumps() {
        // Instruction: jmp +0x7
        test(&[0xeb, 0x07], "
            const T0:i64 = 0x7:i64
            jump by T0:i64 always
        ");

        // Instruction: jg +0x9
        test(&[0x7f, 0x09], "
            const T0:i64 = 0x9:i64
            jump by T0:i64 if greater
        ");

        // Instruction: je +0xe
        test(&[0x74, 0x0e], "
            const T0:i64 = 0xe:i64
            jump by T0:i64 if equal
        ");

        // Instruction: call -0x76
        test(&[0xe8, 0x8a, 0xff, 0xff, 0xff], "
            mov T0:u64 = [mem1][0x20:u64]
            const T1:u64 = 0x8:u64
            sub T0:u64 = T0:u64 - T1:u64
            mov [mem0][(T0:u64):u64] = [mem1][0x80:u64]
            mov [mem1][0x20:u64] = T0:u64
            const T2:i64 = -0x76:i64
            jump by T2:i64 always
        ");

        // Instruction: leave
        test(&[0xc9], "
            mov [mem1][0x20:u64] = [mem1][0x28:u64]
            mov T0:u64 = [mem1][0x20:u64]
            mov [mem1][0x28:u64] = [mem0][(T0:u64):u64]
            const T1:u64 = 0x8:u64
            add T0:u64 = T0:u64 + T1:u64
            mov [mem1][0x20:u64] = T0:u64
        ");

        // Instruction: ret
        test(&[0xc3], "
            mov T1:u64 = [mem1][0x20:u64]
            mov T0:u64 = [mem0][(T1:u64):u64]
            const T2:u64 = 0x8:u64
            add T1:u64 = T1:u64 + T2:u64
            mov [mem1][0x20:u64] = T1:u64
            jump to T0:u64 always
        ");
    }
}
