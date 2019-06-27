//! Decodes `amd64` instructions.

use std::fmt::{self, Debug, Display, Formatter};
use std::io::Cursor;
use byteorder::{ByteOrder, LittleEndian};


/// Decoded machine code instruction.
#[derive(Debug, Clone)]
pub struct Instruction {
    pub bytes: Vec<u8>,
    pub mnemoic: Mnemoic,
    pub operands: Vec<Operand>,
}

impl Instruction {
    /// Tries to decode an instruction from bytes.
    pub fn decode(bytes: &[u8]) -> DecodeResult<Instruction> {
        let mut index = 0;

        // Parse the optional REX prefix.
        let rex = (bytes[index] ^ 0b01000000) < 16;
        let mut rex_w = false;
        let mut rex_r = false;
        let mut rex_x = false;
        let mut rex_b = false;
        if rex {
            rex_w = bytes[index] & 0b00001000 > 0;
            rex_r = bytes[index] & 0b00000100 > 0;
            rex_x = bytes[index] & 0b00000010 > 0;
            rex_b = bytes[index] & 0b00000001 > 0;
            index += 1;
        }

        // Parse the opcode.
        let mut opcode_len = 1;
        if bytes[index] == 0x0f {
            opcode_len += 1;
            if bytes[index + 1] == 0x38 || bytes[index + 1] == 0x3a {
                opcode_len += 1;
            }
        }
        let opcode = &bytes[index .. index + opcode_len];
        index += opcode_len;

        #[derive(Debug, Copy, Clone, Eq, PartialEq)]
        enum Op { Free, Plus(u8), Rm, RegRm, RmReg }
        use Op::*;

        // Parse the mnemoic.
        let (mnemoic, op) = match opcode {
            &[0x03] => (Mnemoic::Add, RegRm),
            &[0x3b] => (Mnemoic::Cmp, RegRm),
            &[x] if 0x50 <= x && x < 0x58 => (Mnemoic::Push, Plus(0x50)),
            &[x] if 0x58 <= x && x < 0x60 => (Mnemoic::Pop, Plus(0x58)),
            &[0x89] => (Mnemoic::Mov, RmReg),
            &[0x8b] => (Mnemoic::Mov, RegRm),
            &[0xc3] => (Mnemoic::Ret, Free),
            &[0x0f, 0x9c] => (Mnemoic::Set, Rm),
            _ => panic!("unknown mnemoic"),
        };

        let mut operands = Vec::new();
        if let Plus(base) = op {
            let reg = Register::from_bits(false, opcode[0] - base, DataWidth::Bits64);
            operands.push(Operand::Direct(reg));

        } else if op == Rm {
            // TODO
        } else if op == RegRm || op == RmReg {
            // Parse the ModR/M byte.
            let modrm_modus = bytes[index] >> 6;
            let modrm_reg = (bytes[index] & 0b00111000) >> 3;
            let modrm_rm = bytes[index] & 0b00000111;
            index += 1;

            let displacement_bytes = match modrm_modus {
                0b00 => Some(0),
                0b01 => Some(1),
                0b10 => Some(4),
                0b11 => None,
                _ => unreachable!(),
            };

            // Parse the SIB byte.
            let sib = false;
            if sib {}

            // Parse the displacement.
            let displacement = displacement_bytes.map(|dis| {
                let offset = if dis == 1 {
                    (bytes[index] as i8) as i64
                } else {
                    LittleEndian::read_i32(&bytes[index .. index + 4]) as i64
                };
                index += dis;
                offset
            });

            // Parse the immediate.
            let immediate = false;
            if immediate {}

            // Construct the operands.
            let a = {
                let width = if rex_w { DataWidth::Bits64 } else { DataWidth::Bits32 };
                let reg = Register::from_bits(rex_r, modrm_reg, width);
                Operand::Direct(reg)
            };
            let b = {
                let reg = Register::from_bits(rex_b, modrm_rm, DataWidth::Bits64);
                match displacement {
                    Some(0) => Operand::Indirect(reg),
                    Some(offset) => Operand::IndirectDisplaced(reg, offset),
                    None => Operand::Direct(reg),
                }
            };

            if op == RegRm {
                operands.push(a);
                operands.push(b);
            } else {
                operands.push(b);
                operands.push(a);
            }
        }

        Ok(Instruction {
            bytes: bytes.to_vec(),
            mnemoic,
            operands,
        })
    }
}

impl Display for Instruction {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.mnemoic)?;
        let mut first = true;
        for operand in &self.operands {
            if !first { write!(f, ",")?; }
            first = false;
            write!(f, " {}", operand)?;
        }
        Ok(())
    }
}

/// Describes an instruction.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Mnemoic {
    Add, Mov, Push, Pop, Ret, Cmp, Set,
}

impl Display for Mnemoic {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", format!("{:?}", self).to_lowercase())
    }
}

/// Operand in an instruction.
#[derive(Debug, Copy, Clone)]
pub enum Operand {
    Direct(Register),
    Indirect(Register),
    IndirectDisplaced(Register, i64),
}

impl Display for Operand {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use Operand::*;
        match self {
            Direct(reg) => write!(f, "{}", reg),
            Indirect(reg) => write!(f, "[{}]", reg),
            IndirectDisplaced(reg, offset) => if *offset >= 0 {
                write!(f, "[{} + 0x{:x}]", reg, offset)
            } else {
                write!(f, "[{} - 0x{:x}]", reg, -offset)
            }
        }
    }
}

/// Identifies a register.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Register {
    RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI,
    EAX, ECX, EDX, EBX, ESP, EBP, ESI, EDI,
    AX, CX, DX, BX, SP, BP, SI, DI,
    AL, CL, DL, BL, AH, CH, DH, BH,
    R8, R9, R10, R11, R12, R13, R14, R15,
}

impl Display for Register {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", format!("{:?}", self).to_lowercase())
    }
}

impl Register {
    fn from_bits(w: bool, reg: u8, width: DataWidth) -> Register {
        use Register::*;
        match (w, reg) {
            (false, 0b000) => [AL, AX, EAX, RAX][width as usize],
            (false, 0b001) => [CL, CX, ECX, RCX][width as usize],
            (false, 0b010) => [DL, DX, EDX, RDX][width as usize],
            (false, 0b011) => [BL, BX, EBX, RBX][width as usize],
            (false, 0b100) => [AH, SP, ESP, RSP][width as usize],
            (false, 0b101) => [CH, BP, EBP, RBP][width as usize],
            (false, 0b110) => [DH, SI, ESI, RSI][width as usize],
            (false, 0b111) => [BH, DI, EDI, RDI][width as usize],
            (true, 0b000)  => R8,
            (true, 0b001)  => R9,
            (true, 0b010)  => R10,
            (true, 0b011)  => R11,
            (true, 0b100)  => R12,
            (true, 0b101)  => R13,
            (true, 0b110)  => R14,
            (true, 0b111)  => R15,
            _ => panic!("from_bits: invalid bits for register"),
        }
    }
}

/// Width of data in bits.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum DataWidth {
    Bits8 = 0,
    Bits16 = 1,
    Bits32 = 2,
    Bits64 = 3,
}

/// Error type for instruction decoding.
pub struct DecodeError {
    pub addr: u64,
    pub bytes: Vec<u8>,
}

/// Result type for instruction decoding.
pub(in super) type DecodeResult<T> = Result<T, DecodeError>;
impl std::error::Error for DecodeError {}

impl Display for DecodeError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Decoding error at {:x}:", self.addr)?;
        for &byte in &self.bytes {
            write!(f, " {:02x}", byte)?;
        }
        Ok(())
    }
}

impl Debug for DecodeError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Display::fmt(self, f)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn test(binary: &[u8], display: &str) {
        let inst = Instruction::decode(binary).unwrap();
        assert_eq!(inst.to_string(), display);
    }

    #[test]
    fn decode() {
        test(&[0x4c, 0x03, 0x47, 0x0a], "add r8, [rdi + 0xa]");

        // // <compare>
        test(&[0x55], "push rbp");
        test(&[0x48, 0x89, 0xe5], "mov rbp, rsp");
        test(&[0x89, 0x7d, 0xfc], "mov [rbp - 0x4], edi");
        test(&[0x89, 0x75, 0xf8], "mov [rbp - 0x8], esi");
        test(&[0x8b, 0x45, 0xfc], "mov eax, [rbp - 0x4]");
        test(&[0x3b, 0x45, 0xf8], "cmp eax, [rbp - 0x8]");
        // test(&[0x0f, 0x9c, 0xc0], "set al");
        // test(&[0x0f, 0xb6, 0xc0], "movzx eax, al");
        test(&[0x5d], "pop rbp");
        test(&[0xc3], "ret");
    }
}
