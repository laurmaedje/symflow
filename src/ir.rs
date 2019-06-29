//! Intermediate representation of instructions.

use std::fmt::{self, Debug, Display, Formatter};
use crate::amd64::{Instruction, Mnemoic, Operand, Register, DataWidth};


/// Microcode composed of any number of micro operations.
#[derive(Debug, Clone, PartialEq)]
pub struct Microcode {
    pub ops: Vec<MicroOperation>,
}

impl Microcode {
    /// Express the given instruction in microcode.
    fn from_instruction(instruction: &Instruction) -> Microcode {
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
    fn encode(mut self) -> Microcode {
        use Mnemoic::*;
        use MicroOperation as Op;

        match self.inst.mnemoic {
            Add => self.encode_binop(|sum, a, b| Op::Add { sum, a, b, flags: true }),
            Sub => self.encode_binop(|diff, a, b| Op::Sub { diff, a, b, flags: true }),
            Imul => self.encode_binop(|prod, a, b| Op::Mul { prod, a, b, flags: true }),

            Mov => {
                // Prepare the source and destination locations.
                let dest = self.encode_get_location(self.inst.operands[0]);
                let src = self.encode_get_location(self.inst.operands[1]);

                // Enforce that both operands have the exact same data type.
                assert_eq!(src.data_type(), dest.data_type(), "incompatible data types for move");
                self.ops.push(Op::Mov { dest, src });
            },

            Movzx => unimplemented!(),
            Lea => unimplemented!(),
            Push => unimplemented!(),
            Pop => unimplemented!(),
            Jmp => unimplemented!(),
            Je => unimplemented!(),
            Jg => unimplemented!(),
            Call => unimplemented!(),
            Leave => unimplemented!(),
            Ret => unimplemented!(),
            Cmp => unimplemented!(),
            Test => unimplemented!(),
            Set => unimplemented!(),

            Syscall => { self.ops.push(Op::Syscall); },
            Nop => {},
        };

        Microcode { ops: self.ops }
    }

    /// Encode a binary operation like add or subtract.
    fn encode_binop<F>(&mut self, bin_op: F)
    where F: FnOnce(Temporary, Temporary, Temporary) -> MicroOperation {
        // Encode the loading of both operands into a temporary.
        let (dest_loc, dest) = self.encode_load_operand(self.inst.operands[0]);
        let (src_loc, src) = self.encode_load_operand(self.inst.operands[1]);

        // Enforce that both operands have the exact same data type.
        assert_eq!(dest.0, src.0, "incompatible data types for binary operation");

        // Encode the actual binary operation and the move from the target temporary
        // into the destination.
        let target = Temporary(src.0, self.temps);
        self.ops.push(bin_op(target, dest, src));
        self.ops.push(MicroOperation::Mov { dest: dest_loc, src: Location::Temp(target) });
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
        use {Location::*, DataType::*, MicroOperation as Op};

        match operand {
            Operand::Direct(reg) => {
                // Locate the registers in memory.
                let data_type = DataType::from_width(reg.width(), false);
                Direct(data_type, MemorySpace(1), reg_mem(reg))
            },

            Operand::Indirect(reg) => {
                // Load the address into a temporary.
                let reg = self.encode_load_reg(reg);
                Indirect(U64, MemorySpace(0), reg) // FIXME: Don't assume U64
            },

            Operand::IndirectDisplaced(reg, displace) => {
                // Load the base register into a temporary.
                let reg = self.encode_load_reg(reg);

                // Load the displacement constant into a temporary.
                let constant = Temporary(U64, self.temps);
                self.ops.push(Op::Const {
                    dest: Location::Temp(constant),
                    value: Immediate(U64, displace as u64)
                });

                // Compute the final address.
                self.ops.push(Op::Add {
                    sum: Temporary(U64, self.temps + 1),
                    a: reg, b: constant, flags: false
                });
                self.temps += 2;

                Indirect(U64, MemorySpace(0), Temporary(U64, self.temps - 1)) // FIXME: U64
            },

            Operand::Immediate(immediate) => Temp(self.encode_load_constant(immediate)),
            Operand::Offset(offset) => Temp(self.encode_load_constant(offset as u64)),
        }
    }

    /// Encode the micro operations to load a register from memory into a temporary.
    /// The resulting temporary will have the data type matching the registers width.
    fn encode_load_reg(&mut self, reg: Register) -> Temporary {
        let data_type = DataType::from_width(reg.width(), false);
        let temp = Temporary(data_type, self.temps);

        let src = Location::Direct(data_type, MemorySpace(1), reg_mem(reg));
        self.ops.push(MicroOperation::Mov { dest: Location::Temp(temp), src });
        self.temps += 1;

        temp
    }

    /// Encode the micro operations to load a constant into a temporary.
    fn encode_load_constant(&mut self, constant: u64) -> Temporary {
        let temp = Temporary(DataType::U64, self.temps);

        self.ops.push(MicroOperation::Const {
            dest: Location::Temp(temp),
            value: Immediate(DataType::U64, constant)
        });
        self.temps += 1;

        temp
    }
}

impl Display for Microcode {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        writeln!(f, "Microcode [")?;
        for operation in &self.ops {
            writeln!(f, "    {}", operation)?;
        }
        write!(f, "]")
    }
}

/// Describes one atomic operation.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum MicroOperation {
    /// Store the sum of `a` and `b` in `sum`. Set flags if active.
    Add { sum: Temporary, a: Temporary, b: Temporary, flags: bool },
    /// Store the difference of `a` and `b` in `diff`. Set flags if active.
    Sub { diff: Temporary, a: Temporary, b: Temporary, flags: bool },
    /// Store the product of `a` and `b` in `prod`. Set flags if active.
    Mul { prod: Temporary, a: Temporary, b: Temporary, flags: bool },
    /// Store the bitwise AND of `a` and `b` in and. Set flags if active.
    BitAnd { and: Temporary, a: Temporary, b: Temporary, flags: bool },
    /// Store the bitwise OR of `a` and `b` in or. Set flags if active.
    BitOr { or: Temporary, a: Temporary, b: Temporary, flags: bool },
    /// Store the value at location `src` in location `dest`.
    Mov { dest: Location, src: Location },
    /// Store a constant in location `dest`.
    Const { dest: Location, value: Immediate },
    /// Cast the temporary `target` to another type
    /// (possibly zero-extending, sign-extending or truncating).
    Cast { target: Temporary, new: DataType },
    /// Jump to the address given in `target` if the value of the
    /// condition temporary is not zero.
    Jump { target: Temporary, condition: Temporary },
    /// Perform a syscall.
    Syscall,
}

impl Display for MicroOperation {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use MicroOperation::*;
        fn flag(flags: bool) -> &'static str { if flags { " (flags)" } else { "" } }
        match *self {
            Add { sum, a, b, flags } => write!(f, "add {} = {} + {}{}", sum, a, b, flag(flags)),
            Sub { diff, a, b, flags } => write!(f, "sub {} = {} + {}{}", diff, a, b, flag(flags)),
            Mul { prod, a, b, flags } => write!(f, "mul {} = {} + {}{}", prod, a, b, flag(flags)),
            BitAnd { and, a, b, flags } => write!(f, "and {} = {} & {}{}", and, a, b, flag(flags)),
            BitOr { or, a, b, flags } => write!(f, "or {} = {} | {}{}", or, a, b, flag(flags)),
            Mov { dest, src } => write!(f, "mov {} = {}", dest, src),
            Const { dest, value } => write!(f, "const {} = {}", dest, value),
            Cast { target, new } => write!(f, "cast {} to {}", target, new),
            Jump { target, condition } => write!(f, "jump {} if {} != 0", target, condition),
            Syscall => write!(f, "syscall"),
        }
    }
}

/// Strongly typed target for moves.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Location {
    Temp(Temporary),
    Direct(DataType, MemorySpace, Address),
    Indirect(DataType, MemorySpace, Temporary),
}

impl Location {
    fn data_type(&self) -> DataType {
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
            Direct(data, space, offset) => write!(f, "[{}][{}:{}]", space, offset, data),
            Indirect(data, space, temp) => write!(f, "[{}][({}):{}]", space, temp, data),
        }
    }
}

/// Memory space identified by an index.
///
/// Typically,
/// - memory space `0` is used for the flat primary memory space
/// - memory space `1` is used for registers (their addresses can be
///   retrieve through the [`reg_mem`] function).
/// - memory space `2` is used for flags.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct MemorySpace(pub usize);

impl Display for MemorySpace {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "mem{}", self.0)
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

/// Strongly typed immediate value.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Immediate(pub DataType, pub u64);

impl Display for Immediate {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "0x{:x}:{}", self.1, self.0)
    }
}

/// Memory address.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Address(pub u64);

impl Display for Address {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "0x{:x}", self.0)
    }
}

/// Basic numeric types.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum DataType {
    I8, U8, I16, U16, I32, U32, I64, U64,
}

impl DataType {
    /// The data type for the specified bit width. (signed or unsigned).
    fn from_width(width: DataWidth, signed: bool) -> DataType {
        use DataType::*;
        match width {
            DataWidth::Bits8  => [U8, I8][signed as usize],
            DataWidth::Bits16 => [U16, I16][signed as usize],
            DataWidth::Bits32 => [U32, I32][signed as usize],
            DataWidth::Bits64 => [U64, I64][signed as usize],
        }
    }
}

impl Display for DataType {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", format!("{:?}", self).to_lowercase())
    }
}

/// Returns the address of a register in the register memory space.
pub fn reg_mem(reg: Register) -> Address {
    use Register::*;
    Address(match reg {
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
    })
}


#[cfg(test)]
mod test {
    use crate::amd64::*;
    use super::*;

    fn test(instruction: Instruction, display: &str) {
        let code = Microcode::from_instruction(&instruction);
        let display = codify(display);
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
    fn microcode() {
        // Instruction: add r8, [rdi+0xa]
        // The microcode works as follows:
        // - Move r8 into t0
        // - Move rdi into t1, move 0xa into t2, sum them up into t3
        // - Load the value at address t3 into t4
        // - Compute the sum of t0 and t4 and store it in t5
        // - Move t5 into r8
        test(Instruction::decode(&[0x4c, 0x03, 0x47, 0x0a]).unwrap(), "
            mov T0:u64 = [mem1][0x40:u64]
            mov T1:u64 = [mem1][0x38:u64]
            const T2:u64 = 0xa:u64
            add T3:u64 = T1:u64 + T2:u64
            mov T4:u64 = [mem0][(T3:u64):u64]
            add T5:u64 = T0:u64 + T4:u64 (flags)
            mov [mem1][0x40:u64] = T5:u64
        ");

        // Instruction: mov esi, edx
        test(Instruction::decode(&[0x89, 0xd6]).unwrap(),
             "mov [mem1][0x30:u32] = [mem1][0x10:u32]");

        // Instruction: mov [rbp-0x8], 0xa
        test(Instruction::decode(&[0xc7, 0x45, 0xf8, 0x0a, 0x00, 0x00, 0x00]).unwrap(), "
            mov T0:u64 = [mem1][0x28:u64]
            const T1:u64 = 0xfffffffffffffff8:u64
            add T2:u64 = T0:u64 + T1:u64
            const T3:u64 = 0xa:u64
            mov [mem0][(T2:u64):u64] = T3:u64
        ");
    }
}
