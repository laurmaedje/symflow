//! Decodes `amd64` instructions.

use std::fmt::{self, Debug, Display, Formatter};
use byteorder::{ByteOrder, LittleEndian};
use crate::num::DataType;


/// Decoded machine code instruction.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Instruction {
    pub bytes: Vec<u8>,
    pub mnemoic: Mnemoic,
    pub operands: Vec<Operand>,
}

impl Instruction {
    /// Tries to decode an instruction from bytes.
    pub fn decode(bytes: &[u8]) -> DecodeResult<Instruction> {
        Decoder::new(bytes).decode()
    }

    /// The length of the first instruction in the slice.
    pub fn length(bytes: &[u8]) -> u64 {
        lde::X64.ld(bytes) as u64
    }
}

/// Decodes instructions.
#[derive(Debug)]
struct Decoder<'a> {
    bytes: &'a [u8],
    index: usize,
}

impl<'a> Decoder<'a> {
    /// Create a new decoder.
    fn new(bytes: &'a [u8]) -> Decoder<'a> {
        Decoder { bytes, index: 0 }
    }

    /// Decode the bytes into an instruction.
    fn decode(mut self) -> DecodeResult<Instruction> {
        use OperandLayout::*;

        // Parse the optional REX prefix.
        let rex = self.decode_rex();

        // Parse the opcode.
        let (opcode, operation) = self.decode_opcode(rex);
        let (mnemoic, op) = operation.ok_or_else(|| DecodeError::new(self.bytes.to_vec()))?;

        // Construct the operands.
        let mut operands = Vec::new();
        if let Plus(base, width) = op {
            // Decode the register from the opcode.
            let reg = Register::from_bits(false, opcode[0] - base, width);
            operands.push(Operand::Direct(reg));

        } else if let PlusIm(base, width, im_w) = op {
            // Decode the register from the opcode.
            let reg = Register::from_bits(false, opcode[0] - base, width);
            operands.push(Operand::Direct(reg));

            // Parse the immediate.
            let immediate = self.decode_immediate(im_w);
            operands.push(Operand::Immediate(im_w, immediate));

        } else if let Rm(rm_w) = op {
            // Parse just the R/M part and add a direct operand with it.
            let modrm_rm = self.decode_modrm().2;
            operands.push(construct_modrm_rm(rex.b, modrm_rm, rm_w, None));

        } else if let RegRm(reg_w, rm_w, ordered) = op {
            // Parse the ModR/M byte with displacement.
            let (reg, rm, displace) = self.decode_modrm_displaced();

            // Construct the operands.
            let p = construct_modrm_reg(rex.r, reg, reg_w);
            let s = construct_modrm_rm(rex.b, rm, rm_w, displace);

            // Insert them in the ordered denoted by the opcode.
            if ordered {
                operands.push(p); operands.push(s);
            } else {
                operands.push(s); operands.push(p);
            }

        } else if let RmIm(rm_w, im_w) = op {
            // Parse the ModR/M byte with displacement.
            let (_, rm, displace) = self.decode_modrm_displaced();

            // Parse the immediate.
            let immediate = self.decode_immediate(im_w);

            // Construct and insert the operands.
            operands.push(construct_modrm_rm(rex.b, rm, rm_w, displace));
            operands.push(Operand::Immediate(im_w, immediate));

        } else if let Rel(width) = op {
            // Parse the relative offset and adjust it by the length of this instruction.
            let offset = self.decode_offset(width);
            operands.push(Operand::Offset(offset));
        }

        Ok(Instruction {
            bytes: self.bytes.to_vec(),
            mnemoic,
            operands,
        })
    }

    /// Decode the REX prefix.
    fn decode_rex(&mut self) -> RexPrefix {
        let byte = self.bytes[self.index];
        let rex = (byte ^ 0b01000000) < 16;
        if rex {
            self.index += 1;
        }

        RexPrefix {
            w: rex && (byte & 0b00001000 > 0),
            r: rex && (byte & 0b00000100 > 0),
            x: rex && (byte & 0b00000010 > 0),
            b: rex && (byte & 0b00000001 > 0),
        }
    }

    /// Decode the opcode.
    fn decode_opcode(&mut self, rex: RexPrefix) -> (&'a [u8], Option<(Mnemoic, OperandLayout)>) {
        use DataType::*;
        use OperandLayout::*;

        // Find out the length of the opcode and adjust the index.
        let mut len = 1;
        if self.bytes[self.index] == 0x0f {
            len += 1;
            if self.bytes[self.index + 1] == 0x38 || self.bytes[self.index + 1] == 0x3a {
                len += 1;
            }
        }
        let opcode = &self.bytes[self.index .. self.index + len];
        self.index += opcode.len();

        // The default widths for reg und r/m depends on the rex prefix.
        let scaled = if rex.w { N64 } else { N32 };

        // The instruction extension (0 - 7), uses the reg field of ModR/M.
        let ext = self.bytes[self.index ..].get(0).map(|byte| {
            (byte & 0b00111000) >> 3
        });

        (opcode, Some(match opcode {
            &[0x01] => (Mnemoic::Add, RegRm(scaled, scaled, false)),
            &[0x03] => (Mnemoic::Add, RegRm(scaled, scaled, true)),
            &[0x83] if ext == Some(5) => (Mnemoic::Sub, RmIm(scaled, N8)),
            &[0x0f, 0xaf] => (Mnemoic::Imul, RegRm(scaled, scaled, true)),

            &[x] if 0x50 <= x && x < 0x58 => (Mnemoic::Push, Plus(0x50, N64)),
            &[x] if 0x58 <= x && x < 0x60 => (Mnemoic::Pop, Plus(0x58, N64)),

            &[0x89] => (Mnemoic::Mov, RegRm(scaled, scaled, false)),
            &[0x8b] => (Mnemoic::Mov, RegRm(scaled, scaled, true)),
            &[0xc7] => (Mnemoic::Mov, RmIm(scaled, N32)),
            &[x] if 0xb8 <= x && x < 0xc0 => (Mnemoic::Mov, PlusIm(0xb8, scaled, scaled)),
            &[0x0f, 0xb6] => (Mnemoic::Movzx, RegRm(scaled, N8, true)),

            &[0x8d] => (Mnemoic::Lea, RegRm(scaled, scaled, true)),

            &[0x83] if ext == Some(7) => (Mnemoic::Cmp, RmIm(scaled, N8)),
            &[0x3b] => (Mnemoic::Cmp, RegRm(scaled, scaled, true)),
            &[0x85] => (Mnemoic::Test, RegRm(scaled, scaled, true)),
            &[0x0f, 0x9c] => (Mnemoic::Setl, Rm(N8)),

            &[0x7f] =>(Mnemoic::Jg, Rel(N8)),
            &[0x74] =>(Mnemoic::Je, Rel(N8)),
            &[0xeb] =>(Mnemoic::Jmp, Rel(N8)),
            &[0xe8] =>(Mnemoic::Call, Rel(N16)),

            &[0x90] => (Mnemoic::Nop, Free),
            &[0xc9] => (Mnemoic::Leave, Free),
            &[0xc3] => (Mnemoic::Ret, Free),
            &[0x0f, 0x05] => (Mnemoic::Syscall, Free),

            _ => return (opcode, None),
        }))
    }

    /// Decodes the ModR/M byte and displacement and returns a
    /// (reg, rm, offset, bytes_read) quadruple, where bytes denotes how
    /// many bytes where used from the slice.
    fn decode_modrm_displaced(&mut self) -> (u8, u8, Option<i64>) {
        let (modus, reg, rm) = self.decode_modrm();
        let displacement = self.decode_displacement(modus);
        (reg, rm, displacement)
    }

    /// Decodes the ModR/M byte and returns a (modus, reg, rm) triple.
    fn decode_modrm(&mut self) -> (u8, u8, u8) {
        let byte = self.bytes[self.index];
        let modus = byte >> 6;
        let reg = (byte & 0b00111000) >> 3;
        let rm = byte & 0b00000111;
        self.index += 1;
        (modus, reg, rm)
    }

    /// Decodes the displacement and returns an (offset, bytes_read) pair.
    fn decode_displacement(&mut self, modrm_modus: u8) -> Option<i64> {
        let (displace, off) = match modrm_modus {
            0b00 => (Some(0), 0),
            0b01 => (Some((self.bytes[self.index] as i8) as i64), 1),
            0b10 => (Some(LittleEndian::read_i32(&self.bytes[self.index ..]) as i64), 4),
            0b11 => (None, 0),
            _ => panic!("decode_displacement: invalid modrm_modus"),
        };
        self.index += off;
        displace
    }

    /// Decodes an immediate value with given bit width and returns an
    /// (immediate, bytes_read) pair.
    fn decode_immediate(&mut self, width: DataType) -> u64 {
        use DataType::*;
        let bytes = &self.bytes[self.index ..];
        let (imm, off) = match width {
            N8 => (bytes[0] as u64, 1),
            N16 => (LittleEndian::read_u16(bytes) as u64, 2),
            N32 => (LittleEndian::read_u32(bytes) as u64, 4),
            N64 => (LittleEndian::read_u64(bytes), 8),
        };
        self.index += off;
        imm
    }

    /// Decodes an offset value similar to [`decode_immediate`].
    fn decode_offset(&mut self, width: DataType) -> i64 {
        use DataType::*;
        let bytes = &self.bytes[self.index ..];
        let (offset, off) = match width {
            N8 => ((bytes[0] as i8) as i64, 1),
            N16 => (LittleEndian::read_i16(bytes) as i64, 2),
            N32 => (LittleEndian::read_i32(bytes) as i64, 4),
            N64 => (LittleEndian::read_i64(bytes), 8),
        };
        self.index += off;
        offset
    }
}

/// REX prefix.
#[derive(Debug, Copy, Clone)]
struct RexPrefix {
    w: bool,
    r: bool,
    x: bool,
    b: bool,
}

/// Describes the operand layout of the instruction.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum OperandLayout {
    Free,
    Plus(u8, DataType),
    PlusIm(u8, DataType, DataType),
    Rm(DataType),
    RegRm(DataType, DataType, bool),
    RmIm(DataType, DataType),
    Rel(DataType),
}

/// Construct an operand from the reg part of ModR/M.
fn construct_modrm_reg(rex_r: bool, reg: u8, reg_w: DataType) -> Operand {
    Operand::Direct(Register::from_bits(rex_r, reg, reg_w))
}

/// Construct an operand from the rm part of ModR/M with a displacement.
fn construct_modrm_rm(rex_b: bool, rm: u8, reg_w: DataType, displace: Option<i64>) -> Operand {
    let direct = Register::from_bits(rex_b, rm, reg_w);
    let indirect = Register::from_bits(rex_b, rm, DataType::N64);
    match displace {
        Some(0) => Operand::Indirect(reg_w, indirect),
        Some(offset) => Operand::IndirectDisplaced(reg_w, indirect, offset),
        None => Operand::Direct(direct),
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
    Add, Sub, Imul,
    Mov, Movzx, Lea,
    Push, Pop,
    Jmp, Je, Jg, Call, Leave, Ret,
    Cmp, Test, Setl,
    Syscall,
    Nop,
}

impl Display for Mnemoic {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", format!("{:?}", self).to_lowercase())
    }
}

/// Operand in an instruction.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Operand {
    Direct(Register),
    Indirect(DataType, Register),
    IndirectDisplaced(DataType, Register, i64),
    Immediate(DataType, u64),
    Offset(i64),
}

impl Display for Operand {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use Operand::*;
        match self {
            Direct(reg) => write!(f, "{}", reg),
            Indirect(width, reg) => write!(f, "{} ptr [{}]", width.name(), reg),
            IndirectDisplaced(width, reg, offset) => if *offset >= 0 {
                write!(f, "{} ptr [{}+{:#x}]", width.name(), reg, offset)
            } else {
                write!(f, "{} ptr [{}-{:#x}]", width.name(), reg, -offset)
            },
            Immediate(_, value) => write!(f, "{:#x}", value),
            Offset(offset) => if *offset >= 0 {
                write!(f, "+{:#x}", offset)
            } else {
                write!(f, "-{:#x}", -offset)
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
    IP, EIP, RIP,
}

impl Display for Register {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", format!("{:?}", self).to_lowercase())
    }
}

impl Register {
    /// Data type of the register.
    pub fn data_type(&self) -> DataType {
        use Register::*;
        match self {
            RAX | RCX | RDX | RBX | RSP | RBP | RSI | RDI |
            R8 | R9 | R10 | R11 | R12 | R13 | R14 | R15 | RIP => DataType::N64,
            EAX | ECX | EDX | EBX | ESP | EBP | ESI | EDI | EIP => DataType::N32,
            AX | CX | DX | BX | SP | BP | SI | DI | IP => DataType::N16,
            AL | CL | DL | BL | AH | CH | DH | BH => DataType::N8,
        }
    }

    /// Decodes the register from the bit pattern in the instruction.
    fn from_bits(alt: bool, reg: u8, width: DataType) -> Register {
        use Register::*;
        match (alt, reg) {
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

/// Error type for instruction decoding.
#[derive(Eq, PartialEq)]
pub struct DecodeError {
    pub bytes: Vec<u8>,
}

impl DecodeError {
    /// Create a new decoding error from bytes.
    fn new(bytes: Vec<u8>) -> DecodeError {
        DecodeError { bytes }
    }
}

/// Result type for instruction decoding.
pub(in super) type DecodeResult<T> = Result<T, DecodeError>;
impl std::error::Error for DecodeError {}

impl Display for DecodeError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Failed to decode instruction bytes {:02x?}.", self.bytes)
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
        test(&[0x4c, 0x03, 0x47, 0x0a], "add r8, qword ptr [rdi+0xa]");

        test(&[0x55], "push rbp");
        test(&[0x48, 0x89, 0xe5], "mov rbp, rsp");
        test(&[0x89, 0x7d, 0xfc], "mov dword ptr [rbp-0x4], edi");
        test(&[0x89, 0x75, 0xf8], "mov dword ptr [rbp-0x8], esi");
        test(&[0x8b, 0x45, 0xfc], "mov eax, dword ptr [rbp-0x4]");
        test(&[0x3b, 0x45, 0xf8], "cmp eax, dword ptr [rbp-0x8]");
        test(&[0x0f, 0x9c, 0xc0], "setl al");
        test(&[0x0f, 0xb6, 0xc0], "movzx eax, al");
        test(&[0x5d], "pop rbp");
        test(&[0xc3], "ret");

        test(&[0x48, 0x89, 0x7d, 0xf8], "mov qword ptr [rbp-0x8], rdi");
        test(&[0x48, 0x8b, 0x45, 0xf8], "mov rax, qword ptr [rbp-0x8]");
        test(&[0xc7, 0x00, 0xef, 0xbe, 0xad, 0xde], "mov dword ptr [rax], 0xdeadbeef");
        test(&[0x90], "nop");

        test(&[0xc7, 0x00, 0xad, 0xde, 0xef, 0xbe], "mov dword ptr [rax], 0xbeefdead");
        test(&[0x48, 0x83, 0xec, 0x10], "sub rsp, 0x10");
        test(&[0xc7, 0x45, 0xf8, 0x0a, 0x00, 0x00, 0x00], "mov dword ptr [rbp-0x8], 0xa");
        test(&[0x83, 0x7d, 0xf8, 0x04], "cmp dword ptr [rbp-0x8], 0x4");
        test(&[0x7f, 0x09], "jg +0x9");
        test(&[0xc7, 0x45, 0xfc, 0x0f, 0x00, 0x00, 0x00], "mov dword ptr [rbp-0x4], 0xf");
        test(&[0xeb, 0x07], "jmp +0x7");
        test(&[0xc7, 0x45, 0xfc, 0x05, 0x00, 0x00, 0x00], "mov dword ptr [rbp-0x4], 0x5");
        test(&[0x8b, 0x55, 0xfc], "mov edx, dword ptr [rbp-0x4]");
        test(&[0x89, 0xd6], "mov esi, edx");
        test(&[0x89, 0xc7], "mov edi, eax");
        test(&[0xe8, 0x8a, 0xff, 0xff, 0xff], "call -0x76");
        test(&[0x85, 0xc0], "test eax, eax");
        test(&[0x74, 0x0e], "je +0xe");
        test(&[0x48, 0x8d, 0x45, 0xf4], "lea rax, qword ptr [rbp-0xc]");
        test(&[0x48, 0x89, 0xc7], "mov rdi, rax");

        test(&[0xe8, 0x92, 0xff, 0xff, 0xff], "call -0x6e");
        test(&[0xeb, 0x0c], "jmp +0xc");
        test(&[0xe8, 0x99, 0xff, 0xff, 0xff], "call -0x67");
        test(&[0xc9], "leave");

        test(&[0xb8, 0x00, 0x00, 0x00, 0x00], "mov eax, 0x0");
        test(&[0xe8, 0x9d, 0xff, 0xff, 0xff], "call -0x63");
        test(&[0x48, 0xc7, 0xc0, 0x3c, 0x00, 0x00, 0x00], "mov rax, 0x3c");
        test(&[0x48, 0xc7, 0xc7, 0x00, 0x00, 0x00, 0x00], "mov rdi, 0x0");
        test(&[0x0f, 0x05], "syscall");

        test(&[0x01, 0xd0], "add eax, edx");
        test(&[0x0f, 0xaf, 0x45, 0xfc], "imul eax, dword ptr [rbp-0x4]");

        assert_eq!(Instruction::decode(&[0x12, 0x34]).unwrap_err(),
                   DecodeError { bytes: vec![0x12, 0x34] });
    }
}
