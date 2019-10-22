//! Decoding of `x86_64` instructions.

use std::fmt::{self, Display, Formatter};
use byteorder::{ByteOrder, LittleEndian};

use crate::flow::{ValueSource, StorageLocation};
use crate::math::{Integer, DataType};
use DataType::*;


/// A decoded machine code instruction.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Instruction {
    pub bytes: Vec<u8>,
    pub mnemoic: Mnemoic,
    pub operands: Vec<Operand>,
}

/// Identifies an instruction.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Mnemoic {
    Add, Sub, Imul,
    Mov, Movzx, Movsx, Lea,
    Cwde, Cdqe,
    Push, Pop,
    Jmp, Je, Jl, Jle, Jg, Jge, Jbe,
    Call, Leave, Ret,
    Cmp, Test,
    Setl,
    Syscall,
    Nop,
}

/// An operand in an instruction.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Operand {
    /// Directly the register.
    Direct(Register),
    /// The value at the address stored in base plus:
    /// - optionally a scaled offset of register * scale.
    /// - optionally an immediate displacement.
    Indirect {
        data_type: DataType,
        base: Register,
        scaled_offset: Option<(Register, u8)>,
        displacement: Option<i64>,
    },
    /// A direct immediate value.
    Immediate(Integer),
    /// A direct offset.
    Offset(i64),
}

impl Instruction {
    /// Tries to decode an instruction from raw bytes.
    pub fn decode(bytes: &[u8]) -> DecodeResult<Instruction> {
        Decoder::new(bytes).decode()
    }

    /// The byte length of the first instruction in the given slice.
    pub fn length(bytes: &[u8]) -> u64 {
        lde::X64.ld(bytes) as u64
    }

    /// Pairs of (source, sink) describing data dependencies in the instruction.
    pub fn flows(&self) -> Vec<(ValueSource, StorageLocation)> {
        use Mnemoic::*;
        use Register::*;

        let loc = StorageLocation::from_operand;
        let reg = StorageLocation::Direct;
        let stg = ValueSource::Storage;
        let src = |op| match op {
            Operand::Immediate(int) => Some(ValueSource::Const(int)),
            op => loc(op).map(|s| ValueSource::Storage(s)),
        };

        macro_rules! get {
            ($op:expr) => { if let Some(s) = loc($op) { s } else { return vec![] } };
        }

        fn stack(data_type: DataType, push: bool) -> StorageLocation {
            StorageLocation::Indirect {
                data_type,
                base: RSP,
                scaled_offset: None,
                displacement: if push { Some(-(data_type.bytes() as i64)) } else { None },
            }
        }

        match self.mnemoic {
            Add | Sub | Imul => {
                let target = get!(self.operands[0]);
                let mut source_iter = self.operands.iter();
                if self.operands.len() > 2 {
                    source_iter.next().unwrap();
                }
                source_iter.take(2)
                    .filter_map(|&op| src(op).map(|s| (s, target)))
                    .collect()
            },

            Mov | Movzx | Movsx => match (src(self.operands[1]), loc(self.operands[0])) {
                (Some(a), Some(b)) => vec![(a, b)],
                _ => vec![],
            },

            Lea => {
                let target = get!(self.operands[0]);
                let source = get!(self.operands[1]);
                let mut pairs = vec![];
                if let StorageLocation::Indirect { base, scaled_offset, .. } = source {
                    pairs.push((stg(reg(base)), target));
                    if let Some((offset, _)) = scaled_offset {
                        pairs.push((stg(reg(offset)), target));
                    }
                }
                pairs
            },

            Cwde => vec![(stg(reg(AX)), reg(EAX))],
            Cdqe => vec![(stg(reg(EAX)), reg(RAX))],

            Push => {
                let target = get!(self.operands[0]);
                vec![(stg(reg(RSP)), reg(RSP)), (stg(target), stack(target.data_type(), true))]
            },
            Pop => {
                let target = get!(self.operands[0]);
                vec![(stg(reg(RSP)), reg(RSP)), (stg(stack(target.data_type(), false)), target)]
            },

            Call => vec![(stg(reg(RIP)), stack(N64, true))],
            Leave => vec![
                (stg(reg(RBP)), reg(RSP)),
                (stg(StorageLocation::indirect_reg(N64, RBP)), reg(RBP))
            ],
            Ret => vec![(stg(stack(N64, false)), reg(RIP))],

            _ => vec![],
        }
    }
}

/// Decodes an instruction.
#[derive(Debug, Clone)]
struct Decoder<'a> {
    bytes: &'a [u8],
    index: usize,
}

impl<'a> Decoder<'a> {
    /// Create a new decoder.
    fn new(bytes: &'a [u8]) -> Decoder<'a> {
        Decoder { bytes, index: 0 }
    }

    /// Decodes the bytes into an instruction.
    fn decode(mut self) -> DecodeResult<Instruction> {
        use OperandLayout::*;

        // Parse the optional REX prefix.
        let rex = self.decode_rex();

        // Parse the opcode.
        let (opcode, operation) = self.decode_opcode(rex);
        let (mnemoic, op) = operation.ok_or_else(|| DecodingError::new(self.bytes.to_vec()))?;

        // Construct the operands.
        let mut operands = Vec::new();
        if let Plus(base, width) = op {
            // Decode the register from the opcode.
            let reg = Register::from_bits(false, opcode[0] - base, width);
            operands.push(Operand::Direct(reg));

        } else if let PlusIm(base, width, im_w) = op {
            // Decode the register from the opcode.
            let reg = Register::from_bits(false, opcode[0] - base, width);
            let immediate = self.decode_immediate(im_w);

            operands.push(Operand::Direct(reg));
            operands.push(immediate);

        } else if let Rm(rm_w) = op {
            // Parse just the R/M part and add a direct operand with it.
            let (_, rm) = self.decode_modrm_operands(rex, N64, rm_w);
            operands.push(rm);

        } else if let RegRm(reg_w, rm_w, ordered) = op {
            let (reg, rm) = self.decode_modrm_operands(rex, reg_w, rm_w);

            // Insert them in the ordered denoted by the opcode.
            if ordered {
                operands.push(reg); operands.push(rm);
            } else {
                operands.push(rm); operands.push(reg);
            }

        } else if let RmIm(rm_w, im_w) = op {
            let (_, rm) = self.decode_modrm_operands(rex, N64, rm_w);
            let immediate = self.decode_immediate(im_w);

            // Construct and insert the operands.
            operands.push(rm);
            operands.push(immediate);

        } else if let FixIm(left, im_w) = op {
            let immediate = self.decode_immediate(im_w);

            // The left operand is already given.
            operands.push(left);
            operands.push(immediate)

        } else if let Rel(width) = op {
            let offset = self.decode_offset(width);
            operands.push(offset);
        }

        Ok(Instruction {
            bytes: self.bytes.to_vec(),
            mnemoic,
            operands,
        })
    }

    /// Decodes the REX prefix.
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

    /// Decodes the opcode.
    fn decode_opcode(&mut self, rex: RexPrefix) -> (&'a [u8], Option<(Mnemoic, OperandLayout)>) {
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

        // Handle all the opcodes.
        (opcode, Some(match opcode {
            &[0x00] => (Mnemoic::Add, RegRm(N8, N8, false)),
            &[0x01] => (Mnemoic::Add, RegRm(scaled, scaled, false)),
            &[0x03] => (Mnemoic::Add, RegRm(scaled, scaled, true)),
            &[0x05] => (Mnemoic::Add, FixIm(Operand::Direct(Register::EAX), N32)),
            &[0x81] if ext == Some(0) => (Mnemoic::Add, RmIm(scaled, N32)),
            &[0x83] if ext == Some(0) => (Mnemoic::Add, RmIm(scaled, N8)),
            &[0x81] if ext == Some(5) => (Mnemoic::Sub, RmIm(scaled, N32)),
            &[0x83] if ext == Some(5) => (Mnemoic::Sub, RmIm(scaled, N8)),
            &[0x0f, 0xaf] => (Mnemoic::Imul, RegRm(scaled, scaled, true)),

            &[x] if 0x50 <= x && x < 0x58 => (Mnemoic::Push, Plus(0x50, N64)),
            &[x] if 0x58 <= x && x < 0x60 => (Mnemoic::Pop, Plus(0x58, N64)),

            &[0x88] => (Mnemoic::Mov, RegRm(N8, N8, false)),
            &[0x89] => (Mnemoic::Mov, RegRm(scaled, scaled, false)),
            &[0x8b] => (Mnemoic::Mov, RegRm(scaled, scaled, true)),
            &[0xc6] if ext == Some(0) => (Mnemoic::Mov, RmIm(N8, N8)),
            &[0xc7] => (Mnemoic::Mov, RmIm(scaled, N32)),
            &[x] if 0xb8 <= x && x < 0xc0 => (Mnemoic::Mov, PlusIm(0xb8, scaled, scaled)),
            &[0x0f, 0xb6] => (Mnemoic::Movzx, RegRm(scaled, N8, true)),
            &[0x0f, 0xbe] => (Mnemoic::Movsx, RegRm(scaled, N8, true)),

            &[0x8d] => (Mnemoic::Lea, RegRm(scaled, scaled, true)),

            &[0x80] if ext == Some(7) => (Mnemoic::Cmp, RmIm(N8, N8)),
            &[0x83] if ext == Some(7) => (Mnemoic::Cmp, RmIm(scaled, N8)),
            &[0x3c] => (Mnemoic::Cmp, FixIm(Operand::Direct(Register::AL), N8)),
            &[0x39] => (Mnemoic::Cmp, RegRm(scaled, scaled, false)),
            &[0x3b] => (Mnemoic::Cmp, RegRm(scaled, scaled, true)),
            &[0x85] => (Mnemoic::Test, RegRm(scaled, scaled, true)),
            &[0x0f, 0x9c] => (Mnemoic::Setl, Rm(N8)),

            &[0x74] =>(Mnemoic::Je, Rel(N8)),
            &[0x7c] =>(Mnemoic::Jl, Rel(N8)),
            &[0x7e] =>(Mnemoic::Jle, Rel(N8)),
            &[0x76] =>(Mnemoic::Jbe, Rel(N8)),
            &[0x7f] =>(Mnemoic::Jg, Rel(N8)),
            &[0x7d] =>(Mnemoic::Jge, Rel(N8)),
            &[0xeb] =>(Mnemoic::Jmp, Rel(N8)),
            &[0xe8] =>(Mnemoic::Call, Rel(N16)),
            &[0xff] if ext == Some(2) =>(Mnemoic::Call, Rm(N64)),

            &[0x90] => (Mnemoic::Nop, Free),
            &[0x98] => (if rex.w { Mnemoic::Cdqe } else { Mnemoic::Cwde }, Free),
            &[0xc9] => (Mnemoic::Leave, Free),
            &[0xc3] => (Mnemoic::Ret, Free),
            &[0x0f, 0x05] => (Mnemoic::Syscall, Free),

            _ => return (opcode, None),
        }))
    }

    /// Decodes the ModR/M byte and displacement.
    fn decode_modrm_operands(&mut self, rex: RexPrefix, reg_w: DataType, rm_w: DataType)
    -> (Operand, Operand) {
        let (modus, reg, rm) = self.decode_modrm();

        let reg_op = Operand::Direct(Register::from_bits(rex.r, reg, reg_w));

        let rm_op = match modus {
            0b00 => {
                // Check if we use SIB, RIP-relative or R/M.
                if rm == 0b100 {
                    let (scale, index, base) = self.decode_sib(rex);
                    Operand::Indirect {
                        data_type: rm_w,
                        base,
                        scaled_offset: Some((index, scale)),
                        displacement: None,
                    }

                } else if rm == 0b101 {
                    let disp = self.decode_signed_value(N32);
                    Operand::Indirect {
                        data_type: rm_w,
                        base: Register::RIP,
                        scaled_offset: None,
                        displacement: Some(disp),
                    }

                } else {
                    let base = Register::from_bits(rex.b, rm, N64);
                    Operand::Indirect {
                        data_type: rm_w,
                        base,
                        scaled_offset: None,
                        displacement: None,
                    }
                }
            },
            0b01 | 0b10 => {
                let displace_width = if modus == 0b01 { N8 } else { N32 };

                // Check if we use SIB or just R/M.
                if rm == 0b100 {
                    let (scale, index, base) = self.decode_sib(rex);
                    let disp = self.decode_signed_value(displace_width);
                    Operand::Indirect {
                        data_type: rm_w,
                        base,
                        scaled_offset: Some((index, scale)),
                        displacement: Some(disp),
                    }

                } else {
                    let base = Register::from_bits(rex.b, rm, N64);
                    let disp = self.decode_signed_value(displace_width);
                    Operand::Indirect {
                        data_type: rm_w,
                        base,
                        scaled_offset: None,
                        displacement: Some(disp),
                    }
                }
            },
            0b11 => Operand::Direct(Register::from_bits(rex.b, rm, rm_w)),
            _ => panic!("decode_modrm_operands: invalid modus"),
        };

        (reg_op, rm_op)
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

    /// Decodes the SIB byte and returns a (scale, index, base) triple.
    fn decode_sib(&mut self, rex: RexPrefix) -> (u8, Register, Register) {
        let byte = self.bytes[self.index];
        let scale = 2u8.pow((byte >> 6) as u32);
        let index = Register::from_bits(rex.x, (byte & 0b00111000) >> 3, N64);
        let base = Register::from_bits(rex.b, byte & 0b00000111, N64);
        self.index += 1;
        (scale, index, base)
    }

    /// Decodes an immediate value with given bit width.
    fn decode_immediate(&mut self, width: DataType) -> Operand {
        Operand::Immediate(Integer(width, self.decode_unsigned_value(width)))
    }

    /// Decodes an offset value similar to [`decode_immediate`].
    fn decode_offset(&mut self, width: DataType) -> Operand {
        Operand::Offset(self.decode_signed_value(width))
    }

    /// Decode a variable width unsigned value.
    fn decode_unsigned_value(&mut self, width: DataType) -> u64 {
        let bytes = &self.bytes[self.index ..];
        let (value, off) = match width {
            N8 => (bytes[0] as u64, 1),
            N16 => (LittleEndian::read_u16(bytes) as u64, 2),
            N32 => (LittleEndian::read_u32(bytes) as u64, 4),
            N64 => (LittleEndian::read_u64(bytes), 8),
        };
        self.index += off;
        value
    }

    /// Decode a variable width signed value.
    fn decode_signed_value(&mut self, width: DataType) -> i64 {
        let bytes = &self.bytes[self.index ..];
        let (value, off) = match width {
            N8 => (bytes[0] as i8 as i64, 1),
            N16 => (LittleEndian::read_i16(bytes) as i64, 2),
            N32 => (LittleEndian::read_i32(bytes) as i64, 4),
            N64 => (LittleEndian::read_i64(bytes), 8),
        };
        self.index += off;
        value
    }
}

/// A REX prefix.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
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
    FixIm(Operand, DataType),
    Rel(DataType),
}

impl Display for Instruction {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.mnemoic)?;
        let mut first = true;
        for operand in &self.operands {
            if !first { write!(f, ",")?; } first = false;
            write!(f, " {}", operand)?;
        }
        Ok(())
    }
}

impl Display for Mnemoic {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", format!("{:?}", self).to_lowercase())
    }
}

impl Display for Operand {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use Operand::*;
        use crate::helper::write_signed_hex;

        match *self {
            Direct(reg) => write!(f, "{}", reg),
            Indirect { data_type, base, scaled_offset, displacement } => {
                write!(f, "{} ptr [{}", data_type.name(), base)?;
                if let Some((index, scale)) = scaled_offset {
                    write!(f, "+{}*{}", index, scale)?;
                }
                if let Some(disp) = displacement {
                    write_signed_hex(f, disp)?;
                }
                write!(f, "]")
            },
            Immediate(int) => write!(f, "{:#x}", int.1),
            Offset(offset) => write_signed_hex(f, offset),
        }
    }
}

/// Identifies a register.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
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
    /// The data type (bit width) of the register.
    pub fn data_type(&self) -> DataType {
        use Register::*;
        match self {
            RAX | RCX | RDX | RBX | RSP | RBP | RSI | RDI |
            R8 | R9 | R10 | R11 | R12 | R13 | R14 | R15 | RIP => N64,
            EAX | ECX | EDX | EBX | ESP | EBP | ESI | EDI | EIP => N32,
            AX | CX | DX | BX | SP | BP | SI | DI | IP => N16,
            AL | CL | DL | BL | AH | CH | DH | BH => N8,
        }
    }

    /// The base (64-bit) version of the register.
    pub fn base(self) -> Register {
        use Register::*;
        match self {
            AL | AX | EAX => RAX,
            CL | CX | ECX => RCX,
            DL | DX | EDX => RDX,
            BL | BX | EBX => RBX,
            AH | SP | ESP => RSP,
            CH | BP | EBP => RBP,
            DH | SI | ESI => RSI,
            BH | DI | EDI => RDI,
            r => r
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


/// The error type for instruction decoding.
pub struct DecodingError(Vec<u8>);
pub(in super) type DecodeResult<T> = Result<T, DecodingError>;

impl DecodingError {
    /// Create a new decoding error from bytes.
    fn new(bytes: Vec<u8>) -> DecodingError {
        DecodingError(bytes)
    }
}

impl Display for DecodingError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Failed to decode instruction: {:02x?}", self.0)
    }
}

impl std::error::Error for DecodingError {}
debug_display!(DecodingError);


#[cfg(test)]
mod tests {
    use super::*;

    fn test(binary: &[u8], display: &str) {
        let inst = Instruction::decode(binary).unwrap();
        assert_eq!(inst.to_string(), display);
    }

    #[test]
    fn decode() {
        // Calculations
        test(&[0x01, 0xd0], "add eax, edx");
        test(&[0x4c, 0x03, 0x47, 0x0a], "add r8, qword ptr [rdi+0xa]");
        test(&[0x83, 0xc0, 0x01], "add eax, 0x1");
        test(&[0x00, 0x45, 0xff], "add byte ptr [rbp-0x1], al");
        test(&[0x48, 0x81, 0xc4, 0x20, 0x04, 0x00, 0x00], "add rsp, 0x420");
        test(&[0x48, 0x81, 0xec, 0x20, 0x04, 0x00, 0x00], "sub rsp, 0x420");
        test(&[0x0f, 0xaf, 0x45, 0xfc], "imul eax, dword ptr [rbp-0x4]");

        // Comparisons
        test(&[0x80, 0x7d, 0xff, 0x60], "cmp byte ptr [rbp-0x1], 0x60");
        test(&[0x39, 0x45, 0xfc], "cmp dword ptr [rbp-0x4], eax");
        test(&[0x3c, 0x40], "cmp al, 0x40");
        test(&[0x80, 0x7d, 0xfe, 0x80], "cmp byte ptr [rbp-0x2], 0x80");

        // Moves
        test(&[0x88, 0x45, 0xec], "mov byte ptr [rbp-0x14], al");
        test(&[0xc6, 0x00, 0x21], "mov byte ptr [rax], 0x21");
        test(&[0x0f, 0xbe, 0xc0], "movsx eax, al");
        test(&[0x48, 0x8d, 0x05, 0xcb, 0xff, 0xff, 0xff], "lea rax, qword ptr [rip-0x35]");
        test(&[0x48, 0x8d, 0x1c, 0x02], "lea rbx, qword ptr [rdx+rax*1]");

        // Jumps
        test(&[0x7e, 0x19], "jle +0x19");
        test(&[0xff, 0xd2], "call rdx");
    }

    #[test]
    fn decode_block() {
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
    }

    #[test]
    fn decode_err() {
        assert_eq!(Instruction::decode(&[0x12, 0x34]).unwrap_err().0, vec![0x12, 0x34]);
    }
}
