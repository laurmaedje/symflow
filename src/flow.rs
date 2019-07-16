//! Flow graph calculation.

use std::fmt::{self, Display, Formatter};
use crate::elf::Section;
use crate::amd64::Instruction;
use crate::ir::{Microcode, MicroEncoder, Condition};
use crate::sym::{SymState, SymExpr, Event};
use crate::num::{Integer, DataType};


/// Control flow graph representation of a program.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FlowGraph {
    pub blocks: Vec<(BasicBlock, FlowEdges)>,
}

/// A single flat jump-free sequence of micro operations.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BasicBlock {
    pub addr: u64,
    pub code: Microcode,
    pub exit: (SymExpr, Condition),
}

/// The incoming and outgoing edges of a basic block in the flow graph.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FlowEdges {
    pub incoming: Vec<(usize, Condition)>,
    pub outgoing: Vec<(usize, Condition)>,
}

impl FlowGraph {
    /// Generate a flow graph from the `.text` section of a program.
    pub fn new(text: Section, entry: u64) -> FlowGraph {
        let base = text.header.addr;
        let binary = &text.data;

        let mut stack = vec![entry];
        let mut blocks = Vec::new();

        while let Some(addr) = stack.pop() {
            // Parse the first block.
            let start = addr - base;
            let block = parse_block(&binary[start as usize ..], addr);
            println!("discovered block: {}", block);

            // Add the possible exits to the stack.
            match block.exit {
                (SymExpr::Int(Integer(DataType::N64, target)), Condition::True)
                    => stack.push(target),
                e => panic!("flow graph: unhandled block exit: {:?}", e),
            }

            blocks.push((block, FlowEdges {
                incoming: vec![],
                outgoing: vec![],
            }));
        }


        FlowGraph {
            blocks,
        }
    }
}

/// Parse the basic block at the beginning of the given binary code.
fn parse_block(binary: &[u8], entry: u64) -> BasicBlock {
    // Symbolically execute the block and keep the microcode.
    let mut code = Vec::new();
    let mut encoder = MicroEncoder::new();
    let mut state = SymState::new();
    let mut addr = 0;

    // Execute instructions until an exit is found.
    loop {
        // Parse the instruction.
        let bytes = &binary[addr as usize ..];
        let len = Instruction::length(bytes);
        let instruction = Instruction::decode(bytes).unwrap();
        addr += len;

        // Encode the instruction in microcode.
        let ops = encoder.encode(&instruction).unwrap().ops;
        code.extend(&ops);

        // Execute the microcode.
        for &op in &ops {
            if let Some(event) = state.step(op) {
                match &event {
                    // If it is a jump, add the exit to the list.
                    Event::Jump { target, condition, relative } => {
                        let exit = (if *relative {
                            target.clone().add(SymExpr::Int(Integer::ptr(entry + addr)))
                        } else {
                            target.clone()
                        }, *condition);

                        return BasicBlock {
                            addr: entry,
                            code: Microcode { ops: code },
                            exit
                        };
                    },
                    Event::Exit => unimplemented!("parse_block: sys-exit"),
                }
            }
        }

    }
}

impl Display for FlowGraph {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "FlowGraph [")?;
        if !self.blocks.is_empty() {
            writeln!(f)?;
        }
        for (block, _) in &self.blocks {
            for line in block.to_string().lines() {
                writeln!(f, "    {}", line)?;
            }
        }
        write!(f, "]")
    }
}

impl Display for BasicBlock {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        writeln!(f, "Block: {:#x}", self.addr)?;
        for op in &self.code.ops {
            writeln!(f, "| {}", op)?;
        }
        writeln!(f, "> Exit: {:?}", self.exit)
    }
}


#[cfg(test)]
mod tests {
    use std::fs::File;
    use super::*;
    use crate::elf::ElfFile;

    #[test]
    fn flow() {
        let mut file = ElfFile::new(File::open("test/block-1").unwrap()).unwrap();
        let text = file.get_section(".text").unwrap();

        let graph = FlowGraph::new(text, file.header.entry);
        println!("flow graph: {}", graph);
    }
}
