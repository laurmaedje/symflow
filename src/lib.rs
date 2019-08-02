//! A machine code slicer 🚀 for the AMD-64 architecture based on symbolic execution.

#![allow(unused)]

use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};
use std::path::Path;
use crate::elf::ElfFile;
use crate::amd64::Instruction;
use crate::ir::{MicroEncoder, Microcode};

pub mod amd64;
pub mod elf;
pub mod ir;
pub mod flow;
pub mod slice;
pub mod sym;
pub mod expr;
pub mod num;


/// A decoded binary file.
#[derive(Debug, Clone)]
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
                for line in microcode.to_string().lines() {
                    if !line.starts_with("Microcode") && line != "]" {
                        writeln!(f, "         | {}", &line[4..])?;
                    }
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
        // load_program("target/block-1");
        // load_program("target/block-2");
        // load_program("target/case");
        // load_program("target/twice");
        // load_program("target/loop");
        // load_program("target/recursive-1");
        // load_program("target/recursive-2");
        // load_program("target/func");
        load_program("target/bufs");
    }

    fn load_program(filename: &str) {
        println!("{}: {:#}\n", filename, Program::new(filename));
    }
}
