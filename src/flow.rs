//! Flow graph calculation.

use std::collections::{HashMap, HashSet};
use std::fmt::{self, Display, Formatter};
use crate::elf::Section;
use crate::amd64::Instruction;
use crate::ir::{Microcode, MicroEncoder, Condition};
use crate::sym::{SymState, SymExpr, Event};
use crate::num::{Integer, DataType};


/// Control flow graph representation of a program.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FlowGraph {
    pub blocks: HashMap<u64, (BasicBlock, FlowEdges)>,
}

/// A single flat jump-free sequence of micro operations.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BasicBlock {
    pub addr: u64,
    pub len: u64,
    pub code: Microcode,
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

        let mut stack = vec![(entry, SymState::new())];
        let mut visited = HashSet::new();
        let mut blocks = HashMap::new();

        while let Some((addr, state)) = stack.pop() {
            // Parse the first block.
            let start = addr - base;
            let (block, exit) = parse_block(&binary[start as usize ..], addr, state.clone());
            visited.insert(addr);

            print!("[Block {:#x}]", block.addr);

            // Add the possible exits to the stack.
            if let Some(exit) = exit {
                match exit.target {
                    SymExpr::Int(Integer(DataType::N64, target)) => {
                        let alt_target = addr + block.len;

                        print!(" Condition: {}", exit.condition);

                        // Determine which paths are reachable.
                        let mut jmp = false;
                        let mut alt = false;
                        match exit.condition {
                            SymExpr::Int(Integer(DataType::N8, 0)) => alt = true,
                            SymExpr::Int(Integer(DataType::N8, 1)) => jmp = true,
                            _ => {
                                alt = true;
                                jmp = true;
                            },
                        }

                        if alt /* && !visited.contains(&alt_target) */ {
                            stack.push((alt_target, exit.state.clone()));
                        }

                        if jmp /* && !visited.contains(&target) */ {
                            stack.push((target, exit.state.clone()));
                        }
                    },
                    _ => panic!("flow graph: unhandled jump target: {}", exit.target),
                }
            }

            println!();

            blocks.insert(block.addr, (block, FlowEdges {
                incoming: vec![],
                outgoing: vec![],
            }));
        }

        FlowGraph { blocks }
    }
}

/// Parse the basic block at the beginning of the given binary code.
fn parse_block(binary: &[u8], entry: u64, mut state: SymState) -> (BasicBlock, Option<BlockExit>) {
    // Symbolically execute the block and keep the microcode.
    let mut code = Vec::new();
    let mut encoder = MicroEncoder::new();
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
            if let Some(event) = state.step(entry + addr, op) {
                let exit = match event {
                    // If it is a jump, add the exit to the list.
                    Event::Jump { target, condition, relative } => Some(BlockExit {
                        target: if relative {
                            target.clone().add(SymExpr::Int(Integer::from_ptr(entry + addr)))
                        } else {
                            target.clone()
                        },
                        condition,
                        state,
                    }),
                    Event::Exit => None,
                };

                return (BasicBlock {
                    addr: entry,
                    len: addr,
                    code: Microcode { ops: code },
                }, exit);
            }
        }

    }
}

/// An exit of a block.
#[derive(Debug, Clone)]
struct BlockExit {
    target: SymExpr,
    condition: SymExpr,
    state: SymState,
}

impl Display for FlowGraph {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "FlowGraph [")?;
        if !self.blocks.is_empty() { writeln!(f)?; }
        let mut blocks = self.blocks.values().map(|(b, _)| b).collect::<Vec<_>>();
        blocks.sort_by_key(|block| block.addr);
        for block in blocks {
            for line in block.to_string().lines() {
                writeln!(f, "    {}", line)?;
            }
            writeln!(f)?;
        }
        write!(f, "]")
    }
}

impl Display for BasicBlock {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        writeln!(f, "[Block: {:#x}]", self.addr)?;
        for op in &self.code.ops {
            writeln!(f, "| {}", op)?;
        }
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use std::fs::File;
    use super::*;
    use crate::elf::ElfFile;

    fn gen(file: &str) {
        println!("Generating flow graph for <{}>", file);
        let mut file = ElfFile::new(File::open(file).unwrap()).unwrap();
        let text = file.get_section(".text").unwrap();
        let graph = FlowGraph::new(text, file.header.entry);
        // println!();
        // println!("flow graph: {}", graph);
        // println!();
        println!();
    }

    #[test]
    fn flow() {
        gen("test/block-1");
        gen("test/read");
    }
}
