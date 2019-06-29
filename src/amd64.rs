//! Decodes `amd64` instructions.

use std::fmt::{self, Debug, Display, Formatter};
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
        Decoder::new(bytes).decode()
    }

    /// The length of the first instruction in the slice.
    pub fn length(bytes: &[u8]) -> usize {
        lde::X64.ld(bytes) as usize
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
        let (mnemoic, op) = operation.ok_or_else(|| DecodeError { bytes: self.bytes.to_vec() })?;

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
            operands.push(Operand::Immediate(immediate));

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
            operands.push(Operand::Immediate(immediate));

        } else if let Rel(width) = op {
            // Parse the relative offset and adjust it by the length of this instruction.
            let mut offset = self.decode_offset(width);
            offset += self.bytes.len() as i64;

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
        use DataWidth::*;
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
        let scaled = if rex.w { Bits64 } else { Bits32 };

        // The instruction extension (0 - 7), uses the reg field of ModR/M.
        let ext = self.bytes[self.index ..].get(0).map(|byte| {
            (byte & 0b00111000) >> 3
        });

        (opcode, Some(match opcode {
            &[0x01] => (Mnemoic::Add, RegRm(scaled, scaled, false)),
            &[0x03] => (Mnemoic::Add, RegRm(scaled, scaled, true)),
            &[0x83] if ext == Some(5) => (Mnemoic::Sub, RmIm(scaled, Bits8)),
            &[0x0f, 0xaf] => (Mnemoic::Imul, RegRm(scaled, scaled, true)),

            &[x] if 0x50 <= x && x < 0x58 => (Mnemoic::Push, Plus(0x50, Bits64)),
            &[x] if 0x58 <= x && x < 0x60 => (Mnemoic::Pop, Plus(0x58, Bits64)),

            &[0x89] => (Mnemoic::Mov, RegRm(scaled, scaled, false)),
            &[0x8b] => (Mnemoic::Mov, RegRm(scaled, scaled, true)),
            &[0xc7] => (Mnemoic::Mov, RmIm(scaled, Bits32)),
            &[x] if 0xb8 <= x && x < 0xc0 => (Mnemoic::Mov, PlusIm(0xb8, scaled, scaled)),
            &[0x0f, 0xb6] => (Mnemoic::Movzx, RegRm(scaled, Bits8, true)),

            &[0x8d] => (Mnemoic::Lea, RegRm(scaled, scaled, true)),

            &[0x83] if ext == Some(7) => (Mnemoic::Cmp, RmIm(scaled, Bits8)),
            &[0x3b] => (Mnemoic::Cmp, RegRm(scaled, scaled, true)),
            &[0x85] => (Mnemoic::Test, RegRm(scaled, scaled, true)),
            &[0x0f, 0x9c] => (Mnemoic::Set, Rm(Bits8)),

            &[0x7f] =>(Mnemoic::Jg, Rel(Bits8)),
            &[0x74] =>(Mnemoic::Je, Rel(Bits8)),
            &[0xeb] =>(Mnemoic::Jmp, Rel(Bits8)),
            &[0xe8] =>(Mnemoic::Call, Rel(Bits16)),

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
    fn decode_immediate(&mut self, width: DataWidth) -> u64 {
        use DataWidth::*;
        let bytes = &self.bytes[self.index ..];
        let (imm, off) = match width {
            Bits8 => (bytes[0] as u64, 1),
            Bits16 => (LittleEndian::read_u16(bytes) as u64, 2),
            Bits32 => (LittleEndian::read_u32(bytes) as u64, 4),
            Bits64 => (LittleEndian::read_u64(bytes), 8),
        };
        self.index += off;
        imm
    }

    /// Decodes an offset value similar to [`decode_immediate`].
    fn decode_offset(&mut self, width: DataWidth) -> i64 {
        use DataWidth::*;
        let bytes = &self.bytes[self.index ..];
        let (offset, off) = match width {
            Bits8 => ((bytes[0] as i8) as i64, 1),
            Bits16 => (LittleEndian::read_i16(bytes) as i64, 2),
            Bits32 => (LittleEndian::read_i32(bytes) as i64, 4),
            Bits64 => (LittleEndian::read_i64(bytes), 8),
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
    Plus(u8, DataWidth),
    PlusIm(u8, DataWidth, DataWidth),
    Rm(DataWidth),
    RegRm(DataWidth, DataWidth, bool),
    RmIm(DataWidth, DataWidth),
    Rel(DataWidth),
}

/// Construct an operand from the reg part of ModR/M.
fn construct_modrm_reg(rex_r: bool, reg: u8, reg_w: DataWidth) -> Operand {
    Operand::Direct(Register::from_bits(rex_r, reg, reg_w))
}

/// Construct an operand from the rm part of ModR/M with a displacement.
fn construct_modrm_rm(rex_b: bool, rm: u8, reg_w: DataWidth, displace: Option<i64>) -> Operand {
    let direct = Register::from_bits(rex_b, rm, reg_w);
    let indirect = Register::from_bits(rex_b, rm, DataWidth::Bits64);
    match displace {
        Some(0) => Operand::Indirect(indirect),
        Some(offset) => Operand::IndirectDisplaced(indirect, offset),
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
    Cmp, Test, Set,
    Syscall,
    Nop,
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
    Immediate(u64),
    Offset(i64),
}

impl Display for Operand {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use Operand::*;
        match self {
            Direct(reg) => write!(f, "{}", reg),
            Indirect(reg) => write!(f, "[{}]", reg),
            IndirectDisplaced(reg, offset) => if *offset >= 0 {
                write!(f, "[{}+0x{:x}]", reg, offset)
            } else {
                write!(f, "[{}-0x{:x}]", reg, -offset)
            },
            Immediate(value) => write!(f, "0x{:x}", value),
            Offset(offset) => if *offset >= 0 {
                write!(f, "+0x{:x}", offset)
            } else {
                write!(f, "-0x{:x}", -offset)
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
    /// Bitwidth of the register.
    pub fn width(&self) -> DataWidth {
        use Register::*;
        match self {
            RAX | RCX | RDX | RBX | RSP | RBP | RSI | RDI |
            R8 | R9 | R10 | R11 | R12 | R13 | R14 | R15 => DataWidth::Bits64,
            EAX | ECX | EDX | EBX | ESP | EBP | ESI | EDI => DataWidth::Bits32,
            AX | CX | DX | BX | SP | BP | SI | DI => DataWidth::Bits16,
            AL | CL | DL | BL | AH | CH | DH | BH => DataWidth::Bits8,
        }
    }

    /// Decodes the register from the bit pattern in the instruction.
    fn from_bits(alt: bool, reg: u8, width: DataWidth) -> Register {
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

/// Width of data in bits.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum DataWidth {
    Bits8 = 0,
    Bits16 = 1,
    Bits32 = 2,
    Bits64 = 3,
}

impl DataWidth {
    /// The number of bytes this width equals.
    fn bytes(&self) -> usize {
        match self {
            DataWidth::Bits8 => 1,
            DataWidth::Bits16 => 2,
            DataWidth::Bits32 => 4,
            DataWidth::Bits64 => 8,
        }
    }
}

/// Error type for instruction decoding.
#[derive(Eq, PartialEq)]
pub struct DecodeError {
    pub bytes: Vec<u8>,
}

/// Result type for instruction decoding.
pub(in super) type DecodeResult<T> = Result<T, DecodeError>;
impl std::error::Error for DecodeError {}

impl Display for DecodeError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Decode error: failed to decode instruction {:02x?}", self.bytes)
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
        test(&[0x4c, 0x03, 0x47, 0x0a], "add r8, [rdi+0xa]");

        test(&[0x55], "push rbp");
        test(&[0x48, 0x89, 0xe5], "mov rbp, rsp");
        test(&[0x89, 0x7d, 0xfc], "mov [rbp-0x4], edi");
        test(&[0x89, 0x75, 0xf8], "mov [rbp-0x8], esi");
        test(&[0x8b, 0x45, 0xfc], "mov eax, [rbp-0x4]");
        test(&[0x3b, 0x45, 0xf8], "cmp eax, [rbp-0x8]");
        test(&[0x0f, 0x9c, 0xc0], "set al");
        test(&[0x0f, 0xb6, 0xc0], "movzx eax, al");
        test(&[0x5d], "pop rbp");
        test(&[0xc3], "ret");

        test(&[0x48, 0x89, 0x7d, 0xf8], "mov [rbp-0x8], rdi");
        test(&[0x48, 0x8b, 0x45, 0xf8], "mov rax, [rbp-0x8]");
        test(&[0xc7, 0x00, 0xef, 0xbe, 0xad, 0xde], "mov [rax], 0xdeadbeef");
        test(&[0x90], "nop");

        test(&[0xc7, 0x00, 0xad, 0xde, 0xef, 0xbe], "mov [rax], 0xbeefdead");
        test(&[0x48, 0x83, 0xec, 0x10], "sub rsp, 0x10");
        test(&[0xc7, 0x45, 0xf8, 0x0a, 0x00, 0x00, 0x00], "mov [rbp-0x8], 0xa");
        test(&[0x83, 0x7d, 0xf8, 0x04], "cmp [rbp-0x8], 0x4");
        test(&[0x7f, 0x09], "jg +0xb");
        test(&[0xc7, 0x45, 0xfc, 0x0f, 0x00, 0x00, 0x00], "mov [rbp-0x4], 0xf");
        test(&[0xeb, 0x07], "jmp +0x9");
        test(&[0xc7, 0x45, 0xfc, 0x05, 0x00, 0x00, 0x00], "mov [rbp-0x4], 0x5");
        test(&[0x8b, 0x55, 0xfc], "mov edx, [rbp-0x4]");
        test(&[0x89, 0xd6], "mov esi, edx");
        test(&[0x89, 0xc7], "mov edi, eax");
        test(&[0xe8, 0x8a, 0xff, 0xff, 0xff], "call -0x71");
        test(&[0x85, 0xc0], "test eax, eax");
        test(&[0x74, 0x0e], "je +0x10");
        test(&[0x48, 0x8d, 0x45, 0xf4], "lea rax, [rbp-0xc]");
        test(&[0x48, 0x89, 0xc7], "mov rdi, rax");

        test(&[0xe8, 0x92, 0xff, 0xff, 0xff], "call -0x69");
        test(&[0xeb, 0x0c], "jmp +0xe");
        test(&[0xe8, 0x99, 0xff, 0xff, 0xff], "call -0x62");
        test(&[0xc9], "leave");

        test(&[0xb8, 0x00, 0x00, 0x00, 0x00], "mov eax, 0x0");
        test(&[0xe8, 0x9d, 0xff, 0xff, 0xff], "call -0x5e");
        test(&[0x48, 0xc7, 0xc0, 0x3c, 0x00, 0x00, 0x00], "mov rax, 0x3c");
        test(&[0x48, 0xc7, 0xc7, 0x00, 0x00, 0x00, 0x00], "mov rdi, 0x0");
        test(&[0x0f, 0x05], "syscall");

        test(&[0x01, 0xd0], "add eax, edx");
        test(&[0x0f, 0xaf, 0x45, 0xfc], "imul eax, [rbp-0x4]");

        assert_eq!(Instruction::decode(&[0x12, 0x34]).unwrap_err(),
                   DecodeError { bytes: vec![0x12, 0x34] });
    }
}
