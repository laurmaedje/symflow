//! Machine code slicer for the AMD-64 architecture based on symbolic execution. 💻

#![allow(unused)]

use std::fmt::{self, Debug, Display, Formatter};
use amd64::{Instruction, DecodeResult};

pub mod elf;
pub mod amd64;


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
    pub fn disassemble_all(&self) -> DecodeResult<Block> {
        let mut instructions = Vec::new();

        let mut addr = self.base;
        while addr - self.base < self.code.len() as u64 {
            let inst = self.disassemble_instruction(addr)?;
            addr += inst.bytes.len() as u64;
            instructions.push(inst);
        }

        Ok(Block { instructions })
    }

    /// Disassemble a basic block at an address.
    pub fn disassemble_block(&self, mut addr: u64) -> DecodeResult<Block> {
        let mut instructions = Vec::new();

        let mut finished = false;
        while !finished && addr - self.base < self.code.len() as u64 {
            let inst = self.disassemble_instruction(addr)?;
            addr += inst.bytes.len() as u64;

            // If this is a `retq` instruction, the block is ended.
            if &inst.bytes == &[0xc3] {
                finished = true;
            }

            instructions.push(inst);
        }

        Ok(Block { instructions })
    }

    /// Tries to decode a single instruction at an address.
    pub fn disassemble_instruction(&self, addr: u64) -> DecodeResult<Instruction> {
        let len = lde::X64.ld(&self.code[(addr - self.base) as usize ..]) as u64;
        let bytes = &self.code[(addr - self.base) as usize .. (addr + len - self.base) as usize];
        Instruction::decode(bytes)
    }
}

/// Block of machine code instructions.
#[derive(Debug, Clone)]
pub struct Block {
    pub instructions: Vec<Instruction>,
}


#[cfg(test)]
mod tests {
    use std::fs;
    use super::{elf::*, *};

    #[test]
    fn disassemble() {
        // Read the text section from the binary file.
        let bin = fs::read("test/block").unwrap();
        let mut file = ElfFile::from_slice(&bin).unwrap();
        let text = file.get_section(".text").unwrap();

        // Disassemble one basic block, the <compare> function at address 0x2b1.
        let code = Code::new(text.header.addr, &text.data);
        let all = code.disassemble_all();
        println!("{:#?}", all);
        // let block = code.disassemble_block(0x2b1);
        // println!("compare: {:#?}", block);
    }
}
