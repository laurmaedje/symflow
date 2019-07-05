//! Machine code slicer for the AMD-64 architecture based on symbolic execution. 💻

use std::collections::HashMap;
use std::fmt::{self, Display, Debug, Formatter};

use crate::elf::Section;
use crate::amd64::{Instruction, DecodeError};
use crate::ir::{Microcode, EncodeError};

pub mod amd64;
pub mod elf;
pub mod ir;
pub mod num;
pub mod sim;


/// A microcode program composed of microcode at addresses.
#[derive(Debug, Clone)]
pub struct Program {
    pub mapping: HashMap<u64, (u64, Instruction, Microcode)>,
}

impl Program {
    /// Create a new program from an ELF file.
    pub fn new(text: &Section) -> SynthResult<Program> {
        let base = text.header.addr;
        let code = &text.data;

        let mut mapping = HashMap::new();
        let mut index = 0;

        while index < code.len() as u64 {
            let len = Instruction::length(&code[index as usize ..]) as u64;
            let bytes = &code[index as usize .. (index + len) as usize];
            let instruction = Instruction::decode(bytes)?;
            let microcode = Microcode::from_instruction(&instruction)?;
            mapping.insert(base + index, (len, instruction, microcode));
            index += len;
        }

        Ok(Program { mapping })
    }

    /// Create an empty program.
    pub fn empty() -> Program {
        Program { mapping: HashMap::new() }
    }
}

impl Display for Program {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut values: Vec<_> = self.mapping.iter().map(|(a, (_, i, m))| (a, i, m)).collect();
        values.sort_by_key(|triple| triple.0);
        write!(f, "Program [")?;
        if !values.is_empty() {
            writeln!(f)?;
        }
        let mut start = true;
        for (addr, instruction, microcode) in values {
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

/// Error type for synthesis.
#[derive(Eq, PartialEq)]
pub enum SynthError {
    Decode(DecodeError),
    Encode(EncodeError),
}

/// Result type for synthesis.
type SynthResult<T> = Result<T, SynthError>;
impl std::error::Error for SynthError {}

impl Display for SynthError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            SynthError::Decode(err) => write!(f, "Decoding error: {}", err),
            SynthError::Encode(err) => write!(f, "Encoding error: {}", err),
        }
    }
}

impl Debug for SynthError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result { Display::fmt(self, f) }
}

impl From<DecodeError> for SynthError {
    fn from(err: DecodeError) -> SynthError { SynthError::Decode(err) }
}

impl From<EncodeError> for SynthError {
    fn from(err: EncodeError) -> SynthError { SynthError::Encode(err) }
}
