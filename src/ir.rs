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
        use Mnemoic::*;
        use Location::*;
        use MicroOperation as Op;

        let mut ops = Vec::new();
        let inst = instruction;
        let mut temps = 0;

        match inst.mnemoic {
            Add => {
                let (a_ops, dest, a) = encode_operand(inst.operands[0], &mut temps);
                ops.extend(a_ops);
                let (b_ops, _, b) = encode_operand(inst.operands[1], &mut temps);
                ops.extend(b_ops);

                let sum = Temporary(a.0, temps);
                assert_eq!(a.0, b.0, "incompatible data types for add");
                ops.push(Op::Add { sum, a, b });
                ops.push(Op::Mov { dest, src: Temp(sum) });
            },
            Sub => {},
            Imul => {},
            Mov => {},
            Movzx => {},
            Lea => {},
            Push => {},
            Pop => {},
            Jmp => {},
            Je => {},
            Jg => {},
            Call => {},
            Leave => {},
            Ret => {},
            Cmp => {},
            Test => {},
            Set => {},
            Syscall => { ops.push(Op::Syscall); },
            Nop => {},
        }

        Microcode { ops }
    }
}

fn encode_operand(operand: Operand, temps: &mut usize)
    -> (Vec<MicroOperation>, Location, Temporary) {
    use Location::*;
    use MicroOperation as Op;
    use DataType::*;

    match operand {
        // Load the register from register memory space into a temporary.
        Operand::Direct(reg) => encode_load_reg(reg, temps, true),
        Operand::Indirect(reg) => unimplemented!(),

        // Load the value in main memory at address reg + offset into a temporary.
        Operand::IndirectDisplaced(reg, offset) => {
            let (mut ops, _, reg) = encode_load_reg(reg, temps, false);

            let constant = Temporary(U64, *temps);
            ops.push(Op::Const { dest: Temp(constant), value: Immediate(U64, offset as u64) });
            ops.push(Op::Add { sum: Temporary(U64, *temps + 1), a: reg, b: constant });

            // FIXME: Don't assume I64, need more information in amd64 instructions.
            let src = Indirect(I64, MemorySpace(0), Temporary(U64, *temps + 1));
            let dest = Temporary(I64, *temps + 2);
            ops.push(Op::Mov { dest: Temp(dest), src });

            *temps += 3;
            (ops, src, dest)
        },

        Operand::Immediate(value) => unimplemented!(),
        Operand::Offset(offset) => unimplemented!(),
    }
}

fn encode_load_reg(reg: Register, temps: &mut usize, signed: bool)
    -> (Vec<MicroOperation>, Location, Temporary) {

    let data_type = DataType::from_width(reg.width(), signed);
    let src = Location::Direct(data_type, MemorySpace(1), reg_mem(reg));
    let dest = Temporary(data_type, *temps);
    let ops = vec![MicroOperation::Mov { dest: Location::Temp(dest), src }];

    *temps += 1;
    (ops, src, dest)
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
    /// Store the sum of `a` and `b` in `sum`.
    Add { sum: Temporary, a: Temporary, b: Temporary },
    /// Store the difference of `a` and `b` in `sum`.
    Sub { difference: Temporary, a: Temporary, b: Temporary },
    /// Store the product of `a` and `b` in `sum`.
    Mul { product: Temporary, a: Temporary, b: Temporary },
    /// Store the bitwise AND of `a` and `b` in and.
    BitAnd { and: Temporary, a: Temporary, b: Temporary },
    /// Store the bitwise OR of `a` and `b` in and.
    BitOr { or: Temporary, a: Temporary, b: Temporary },
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
        match self {
            Add { sum, a, b } => write!(f, "add {} = {} + {}", sum, a, b),
            Sub { difference, a, b } => write!(f, "sub {} = {} + {}", difference, a, b),
            Mul { product, a, b } => write!(f, "mul {} = {} + {}", product, a, b),
            BitAnd { and, a, b } => write!(f, "and {} = {} & {}", and, a, b),
            BitOr { or, a, b } => write!(f, "or {} = {} | {}", or, a, b),
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
/// Typically, index 0 is used for the flat primary memory space and
/// index 1 for registers.
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
        assert_eq!(code.to_string(), codify(display));
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
            mov T0:i64 = [mem1][0x40:i64]
            mov T1:u64 = [mem1][0x38:u64]
            const T2:u64 = 0xa:u64
            add T3:u64 = T1:u64 + T2:u64
            mov T4:i64 = [mem0][(T3:u64):i64]
            add T5:i64 = T0:i64 + T4:i64
            mov [mem1][0x40:i64] = T5:i64
        ");
    }
}
