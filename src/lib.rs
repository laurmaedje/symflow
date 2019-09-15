//! Value flow analysis for x86-64 ELF binaries based on symbolic execution 🚀

use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};
use std::path::Path;

use crate::elf::ElfFile;
use crate::ir::{Microcode, MicroEncoder};
use crate::x86_64::Instruction;


/// Helper functions and macros that are used across the crate.
#[macro_use]
mod helper {
    use std::fmt::{self, Formatter};
    use crate::math::DataType;

    pub fn write_signed_hex(f: &mut Formatter, value: i64) -> fmt::Result {
        if value > 0 {
            write!(f, "+{:#x}", value)
        } else if value < 0 {
            write!(f, "-{:#x}", -value)
        } else {
            Ok(())
        }
    }

    pub fn signed_name(s: bool) -> &'static str {
        if s { " signed" } else { "" }
    }

    pub fn boxed<T>(value: T) -> Box<T> { Box::new(value) }

    /// Make sure operations only happen on same expressions.
    pub fn check_compatible(a: DataType, b: DataType, operation: &str) {
        assert_eq!(a, b, "incompatible data types for {}", operation);
    }

    macro_rules! debug_display {
        ($type:ty) => {
            impl std::fmt::Debug for $type {
                fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    std::fmt::Display::fmt(self, f)
                }
            }
        };
    }
}

pub mod flow;
pub mod math;
pub mod sym;
pub mod elf;
pub mod ir;
pub mod x86_64;

#[cfg(feature = "timings")]
pub mod timings;
#[cfg(not(feature = "timings"))]
mod timings {
    pub(crate) fn with<S: Into<String>, F, T>(_: S, f: F) -> T where F: FnOnce() -> T { f() }
    pub(crate) fn start<S>(_: S) {}
    pub(crate) fn stop() {}
}

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
        crate::timings::start("program");

        let mut file = ElfFile::new(filename).unwrap();
        let text = file.get_section(".text").unwrap();

        let base = text.header.addr;
        let binary = text.data;

        let mut index = 0;
        let mut code = Vec::new();
        let mut encoder = MicroEncoder::new();

        // Decode the whole text section.
        while index < binary.len() as u64 {
            let len = Instruction::length(&binary[index as usize ..]);
            let bytes = &binary[index as usize .. (index + len) as usize];
            let instruction = Instruction::decode(bytes).unwrap();
            let microcode = encoder.encode(&instruction).unwrap();
            code.push((base + index, len, instruction, microcode));
            index += len;
        }

        // Extract the symbol names for functions and other things.
        let mut symbols = HashMap::new();
        if let Ok(symbol_entries) = file.get_symbols() {
            for entry in symbol_entries {
                if !entry.name.is_empty() {
                    symbols.insert(entry.value, entry.name);
                }
            }
        }

        crate::timings::stop();

        Program {
            base,
            entry: file.header.entry,
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
                for op in &microcode.ops {
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

    fn test(filename: &str) {
        let path = format!("target/bin/{}", filename);
        Program::new(path);
    }

    #[test]
    fn program() {
        test("block-1");
        test("block-2");
        test("case");
        test("twice");
        test("loop");
        test("recursive-1");
        test("recursive-2");
        test("func");
        test("bufs");
        test("paths");
        test("deep");
        test("overwrite");
    }
}
