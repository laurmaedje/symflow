//! Machine code slicer for the AMD-64 architecture based on symbolic execution. 💻

#![allow(unused)]

use std::collections::HashMap;
use std::fmt::{self, Display, Debug, Formatter};
use crate::amd64::{Instruction, Mnemoic, DecodeError, DecodeResult};
use crate::ir::{Microcode, EncodeError};

pub mod amd64;
pub mod elf;
pub mod ir;
pub mod num;
pub mod sim;


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

    /// Disassemble the whole code into a microcode program.
    pub fn disassemble_program(&self) -> DisassembleResult<Program> {
        let mut mapping = HashMap::new();

        let mut addr = self.base;
        while addr - self.base < self.code.len() as u64 {
            let inst = self.disassemble_instruction(addr)?;
            let len = inst.bytes.len() as u64;
            let microcode = Microcode::from_instruction(&inst)?;
            mapping.insert(addr, (len, inst, microcode));
            addr += len;
        }

        Ok(Program { mapping })
    }

    /// Tries to decode a single instruction at an address.
    pub fn disassemble_instruction(&self, addr: u64) -> DecodeResult<Instruction> {
        let len = Instruction::length(&self.code[(addr - self.base) as usize ..]) as u64;
        let bytes = &self.code[(addr - self.base) as usize .. (addr + len - self.base) as usize];
        Instruction::decode(bytes)
    }
}

/// A microcode program composed of microcode at addresses.
#[derive(Debug, Clone)]
pub struct Program {
    pub mapping: HashMap<u64, (u64, Instruction , Microcode)>,
}

impl Program {
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
