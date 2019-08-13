//! A machine code flow analyzer 🚀 for the `x86_64` architecture based on symbolic execution.

#![allow(unused)]

use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};
use std::path::Path;

use crate::elf::ElfFile;
use crate::x86_64::Instruction;
use crate::ir::{Microcode, MicroEncoder};

pub mod elf;
pub mod x86_64;
pub mod ir;
pub mod num;
pub mod expr;
pub mod smt;
pub mod sym;
pub mod control_flow;
pub mod data_flow;


/// A decoded binary file.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Program {
    pub base: u64,
    pub entry: u64,
    pub binary: Vec<u8>,
    pub code: Vec<(u64, u64, Instruction, Microcode)>,
    pub symbols: HashMap<u64, String>,
}

impl Program {
    /// Create a new program from a 64-bit ELF file.
    pub fn new<P: AsRef<Path>>(filename: P) -> Program {
        let mut file = ElfFile::new(filename).unwrap();
        let text = file.get_section(".text").unwrap();

        let entry = file.header.entry;
        let base = text.header.addr;
        let binary = text.data;

        let mut code = Vec::new();
        let mut index = 0;
        let mut encoder = MicroEncoder::new();

        while index < binary.len() as u64 {
            let len = Instruction::length(&binary[index as usize ..]);
            let bytes = &binary[index as usize .. (index + len) as usize];
            let instruction = Instruction::decode(bytes).unwrap();
            let microcode = encoder.encode(&instruction).unwrap();
            code.push((base + index, len, instruction, microcode));
            index += len;
        }

        let mut symbols = HashMap::new();
        if let Ok(symbol_entries) = file.get_symbols() {
            for entry in symbol_entries {
                if !entry.name.is_empty() {
                    symbols.insert(entry.value, entry.name);
                }
            }
        }

        Program {
            base,
            entry,
            binary,
            code,
            symbols
        }
    }

    /// Get the instruction at the given address.
    pub fn get_instruction(&self, addr: u64) -> Option<&Instruction> {
        self.code.iter()
            .find(|entry| entry.0 == addr)
            .map(|entry| &entry.2)
    }
}

impl Display for Program {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Program [")?;
        if !self.code.is_empty() { writeln!(f)?; }
        let mut first = true;
        for (addr, _, instruction, microcode) in &self.code {
            if f.alternate() && !first { writeln!(f)?; } first = false;
            writeln!(f, "    {:x}: {}", addr, instruction)?;
            if f.alternate() {
                for &op in &microcode.ops {
                    writeln!(f, "         | {}", op)?;
                }
            }
        }
        write!(f, "]")
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn program() {
        load_program("block-1");
        load_program("block-2");
        load_program("case");
        load_program("twice");
        load_program("loop");
        load_program("recursive-1");
        load_program("recursive-2");
        load_program("func");
        load_program("bufs-1");
        load_program("bufs-2");
        load_program("bufs-3");
    }

    fn load_program(filename: &str) {
        let path = format!("target/bin/{}", filename);
        println!("{}: {}\n", filename, Program::new(path));
    }
}
