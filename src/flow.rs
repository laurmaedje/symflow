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
    pub blocks: HashMap<u64, Block>,
}

/// A single flat jump-free sequence of micro operations.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Block {
    pub addr: u64,
    pub len: u64,
    pub code: Microcode,
}

impl FlowGraph {
    /// Generate a flow graph from the `.text` section of a program.
    pub fn new(text: &Section, entry: u64) -> FlowGraph {
        FlowConstructor::new(text).construct(entry)
    }
}

#[derive(Debug, Clone)]
struct FlowConstructor<'a> {
    bin: &'a [u8],
    base: u64,
    blocks: HashMap<u64, Block>,
    stack: Vec<(u64, SymState, Vec<(u64, u64)>)>,
}

/// An exit of a block.
#[derive(Debug, Clone)]
struct Exit {
    target: SymExpr,
    condition: Condition,
    state: SymState,
}

impl<'a> FlowConstructor<'a> {
    /// Construct a new flow graph builder.
    fn new(text: &'a Section) -> FlowConstructor<'a> {
        FlowConstructor {
            bin: &text.data,
            base: text.header.addr,
            blocks: HashMap::new(),
            stack: Vec::new(),
        }
    }

    /// Build the flow graph.
    fn construct(mut self, entry: u64) -> FlowGraph {
        self.stack.push((entry, SymState::new(), vec![]));

        while let Some((addr, state, path)) = self.stack.pop() {
            // Parse the first block.
            println!("Parsing block {:#x}", addr);
            let (block, maybe_exit) = self.parse_block(addr, state.clone());

            // Add blocks reachable from this one.
            if let Some(exit) = maybe_exit {
                self.handle_exit(&block, exit, path);
            }

            // Add the block to the map if we have not already found it
            // through another path.
            if !self.blocks.contains_key(&block.addr) {
                self.blocks.insert(block.addr, block);
            }
        }

        FlowGraph { blocks: self.blocks }
    }

    /// Parse the basic block at the beginning of the given binary code.
    fn parse_block(&self, entry: u64, mut state: SymState) -> (Block, Option<Exit>) {
        let start = (entry - self.base) as usize;
        let binary = &self.bin[start ..];

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
                let maybe_event = state.step(entry + addr, op);

                // Check for exiting.
                if let Some(event) = maybe_event {
                    let exit = match event {
                        // If it is a jump, add the exit to the list.
                        Event::Jump { target, condition, relative } => Some(Exit {
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

                    return (Block {
                        addr: entry,
                        len: addr,
                        code: Microcode { ops: code },
                    }, exit);
                }
            }

        }
    }

    /// Add reachable blocks to the stack depending on the exit conditions of the
    /// just parsed block.
    fn handle_exit(&mut self, block: &Block, exit: Exit, path: Vec<(u64, u64)>) {
        match exit.target {
            SymExpr::Int(Integer(DataType::N64, target)) => {
                if exit.condition != Condition::True {
                    let alt_target = block.addr + block.len;
                    self.add_if_acyclic(alt_target, block.addr, &path, &exit.state);
                }

                self.add_if_acyclic(target, block.addr, &path, &exit.state);
            },

            _ => panic!("handle_exit: unresolved jump target: {}", exit.target),
        }
    }

    /// Add a target to the stack if it was not visitied already by this path
    /// (e.g. if is not cyclic).
    fn add_if_acyclic(&mut self, target: u64, current: u64, path: &[(u64, u64)], state: &SymState) {
        if !path.contains(&(current, target)) {
            println!("  Adding search target {:#x}", target);
            let mut new_path = path.to_vec();
            new_path.push((current, target));
            self.stack.push((target, state.clone(), new_path));
        }
    }
}

impl Display for FlowGraph {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "FlowGraph [")?;
        if !self.blocks.is_empty() { writeln!(f)?; }
        let mut blocks = self.blocks.values().collect::<Vec<_>>();
        blocks.sort_by_key(|block| block.addr);
        let mut first = true;
        for block in blocks {
            if !first { writeln!(f)?; }
            first = false;
            for line in block.to_string().lines() {
                writeln!(f, "    {}", line)?;
            }
        }
        write!(f, "]")
    }
}

impl Display for Block {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Block: {:#x} [", self.addr)?;
        if !self.code.ops.is_empty() { writeln!(f)?; }
        for op in &self.code.ops {
            writeln!(f, "   {}", op)?;
        }
        writeln!(f, "]")
    }
}


#[cfg(test)]
mod tests {
    use std::fs::File;
    use super::*;
    use crate::elf::ElfFile;

    fn test(file: &str) {
        println!("Generating flow graph for <{}>", file);
        let mut file = ElfFile::new(File::open(file).unwrap()).unwrap();
        let text = file.get_section(".text").unwrap();
        let graph = FlowGraph::new(&text, file.header.entry);
        println!();
        println!("flow graph: {}", graph);
        println!();
    }

    #[test]
    fn flow() {
        test("test/block-1");
        test("test/block-2");
        test("test/read");
        test("test/paths");
    }
}
