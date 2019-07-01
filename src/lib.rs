//! Machine code slicer for the AMD-64 architecture based on symbolic execution. 💻

#![allow(unused)]

use std::fmt::{self, Display, Debug, Formatter};
use crate::amd64::{Instruction, Mnemoic, DecodeError, DecodeResult};
use crate::ir::{Microcode, EncodeError};

pub mod elf;
pub mod amd64;
pub mod ir;


/// View into a slice of machine code.
#[derive(Debug)]
pub struct Code<'a> {
    base: u64,
    code: &'a [u8],
}

impl<'a> Code<'a> {
    /// Create a code view that inspects a given binary loaded at a base address.
    pub fn new(base: u64, code: &'a [u8]) -> Code<'a> {
        Code { base, code }
    }

    /// Disassemble the whole code.
    pub fn disassemble_all(&self) -> DisassembleResult<Block> {
        let mut instructions = Vec::new();

        let mut addr = self.base;
        while addr - self.base < self.code.len() as u64 {
            let inst = self.disassemble_instruction(addr)?;
            let len = inst.bytes.len() as u64;
            let microcode = Microcode::from_instruction(&inst)?;
            instructions.push((addr, inst, microcode));
            addr += len;
        }

        Ok(Block { instructions })
    }

    /// Disassemble a basic block at an address.
    pub fn disassemble_block(&self, mut addr: u64) -> DisassembleResult<Block> {
        let mut instructions = Vec::new();

        let mut finished = false;
        while !finished && addr - self.base < self.code.len() as u64 {
            let inst = self.disassemble_instruction(addr)?;

            // If this is a `ret` instruction, the block is ended.
            if inst.mnemoic == Mnemoic::Ret {
                finished = true;
            }

            let len = inst.bytes.len() as u64;
            let microcode = Microcode::from_instruction(&inst)?;
            instructions.push((addr, inst, microcode));
            addr += len;
        }

        Ok(Block { instructions })
    }

    /// Tries to decode a single instruction at an address.
    pub fn disassemble_instruction(&self, addr: u64) -> DecodeResult<Instruction> {
        let len = Instruction::length(&self.code[(addr - self.base) as usize ..]) as u64;
        let bytes = &self.code[(addr - self.base) as usize .. (addr + len - self.base) as usize];
        Instruction::decode(bytes)
    }
}

/// Block of machine code instructions.
#[derive(Debug, Clone)]
pub struct Block {
    pub instructions: Vec<(u64, Instruction, Microcode)>,
}

impl Display for Block {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Block [")?;
        if !self.instructions.is_empty() {
            writeln!(f)?;
        }
        let mut start = true;
        for (addr, instruction, microcode) in &self.instructions {
            if !start {
                writeln!(f)?;
            }
            start = false;
            writeln!(f, "    {:x}: {}", addr, instruction)?;
            for line in microcode.to_string().lines() {
                if !line.starts_with("Microcode") && line != "]" {
                    writeln!(f, "         | {}", &line[4..])?;
                }
            }
        }
        write!(f, "]")
    }
}

/// Error type for disassembling.
#[derive(Eq, PartialEq)]
pub enum DisassembleError {
    Decode(DecodeError),
    Encode(EncodeError),
}

/// Result type for disassembling.
type DisassembleResult<T> = Result<T, DisassembleError>;
impl std::error::Error for DisassembleError {}

impl Display for DisassembleError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use DisassembleError::*;
        match self {
            Decode(err) => write!(f, "Decoding error: {}", err),
            Encode(err) => write!(f, "Encoding error: {}", err),
        }
    }
}

impl Debug for DisassembleError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl From<DecodeError> for DisassembleError {
    fn from(err: DecodeError) -> DisassembleError {
        DisassembleError::Decode(err)
    }
}

impl From<EncodeError> for DisassembleError {
    fn from(err: EncodeError) -> DisassembleError {
        DisassembleError::Encode(err)
    }
}


#[cfg(test)]
mod tests {
    use super::elf::*;
    use super::*;

    #[test]
    fn disassemble() {
        // Read the text section from the binary file.
        let bin = std::fs::read("test/block").unwrap();
        let mut file = ElfFile::from_slice(&bin).unwrap();
        let text = file.get_section(".text").unwrap();

        // Disassemble the whole code and print it.
        let code = Code::new(text.header.addr, &text.data);
        let all = code.disassemble_all().unwrap();
        println!("{}", all);
    }
}
