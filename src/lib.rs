//! Decodes `amd64` binaries.

#![allow(unused)]

use std::fmt::{self, Debug, Display, Formatter};
pub mod elf;


/// A view into machine code.
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
    pub fn disassemble_all(&self) -> Block {
        let mut instructions = Vec::new();

        let mut addr = self.base;
        while addr - self.base < self.code.len() as u64 {
            let inst = self.disassemble_instruction(addr);
            addr += inst.bytes.len() as u64;
            instructions.push(inst);
        }

        Block { instructions }
    }

    /// Disassemble a basic block at an address.
    pub fn disassemble_block(&self, mut addr: u64) -> Block {
        let mut instructions = Vec::new();

        let mut finished = false;
        while !finished && addr - self.base < self.code.len() as u64 {
            let inst = self.disassemble_instruction(addr);
            addr += inst.bytes.len() as u64;

            // If this is a `retq` instruction, the block is ended.
            if &inst.bytes == &[0xc3] {
                finished = true;
            }

            instructions.push(inst);
        }

        Block { instructions }
    }

    /// Decode a single instruction at an address.
    pub fn disassemble_instruction(&self, addr: u64) -> Instruction {
        let len = lde::X64.ld(&self.code[(addr - self.base) as usize ..]) as u64;
        Instruction {
            addr,
            bytes: self.code[(addr - self.base) as usize
                             .. (addr + len - self.base) as usize].to_vec(),
        }
    }
}

/// A block of machine code instructions.
#[derive(Debug)]
pub struct Block {
    pub instructions: Vec<Instruction>,
}

/// A machine code instruction.
pub struct Instruction {
    pub addr: u64,
    pub bytes: Vec<u8>,
}

impl Display for Instruction {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "0x{:x}:", self.addr)?;
        for &byte in &self.bytes {
            write!(f, " {:02x}", byte)?;
        }
        Ok(())
    }
}

impl Debug for Instruction {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Display::fmt(self, f)

    }
}


#[cfg(test)]
mod tests {
    use std::fs;
    use super::elf::ElfReader;
    use super::*;

    #[test]
    fn disassemble() {
        // Read the text section from the file.
        let bin = fs::read("test/block").unwrap();
        let mut reader = ElfReader::from_slice(&bin);
        let text = reader.get_section(".text").unwrap();

        let code = Code::new(text.header.addr, &text.data);
        let block = code.disassemble_block(0x00000000000002b1);
        println!("compare: {:#?}", block);
    }
}
