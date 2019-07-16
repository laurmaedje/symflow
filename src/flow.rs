//! Flow graph calculation.

use std::fmt::{self, Debug, Display, Formatter};
use crate::elf::Section;
use crate::amd64::Instruction;
use crate::ir::{Microcode, MicroEncoder, MicroOperation};
use crate::sym::SymExpr;


/// Control flow graph representation of a program.
#[derive(Debug, Clone)]
pub struct FlowGraph {
    blocks: Vec<(u64, Microcode, FlowEdges)>,
}

#[derive(Debug, Clone)]
pub struct FlowEdges {
    incoming: Vec<(usize, FlowCondition)>,
    outgoing: Vec<(usize, FlowCondition)>,
}

#[derive(Debug, Copy, Clone)]
pub struct FlowCondition;

impl FlowGraph {
    /// Generate a flow graph from the `.text` section of a program.
    fn new(text: Section, entry_point: u64) -> FlowGraph {
        let base = text.header.addr;
        let code = &text.data;

        let mut blocks = Vec::new();

        let start = entry_point - base;
        let (microcode, exits) = parse_block(&code[start as usize ..]);

        blocks.push((base + start, microcode, FlowEdges {
            incoming: vec![],
            outgoing: vec![],
        }));

        FlowGraph {
            blocks,
        }
    }
}

fn parse_block(code: &[u8]) -> (Microcode, Vec<SymExpr>) {
    let mut exits = Vec::new();

    let mut index = 0;
    let mut encoder = MicroEncoder::new();
    loop {
        let bytes = &code[index as usize ..];
        let len = Instruction::length(bytes);
        let instruction = Instruction::decode(bytes).unwrap();

        let op_count = encoder.ops.len();
        encoder.encode(&instruction).unwrap();

        println!("{}", instruction);
        if encoder.ops[op_count..].iter().any(MicroOperation::diverges) {
            println!("   > diverges");
            break;
        }

        index += len;
    }

    (encoder.finish(), exits)
}

impl Display for FlowGraph {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "FlowGraph [")?;
        if !self.blocks.is_empty() {
            writeln!(f)?;
        }

        for (addr, code, edges) in &self.blocks {
            writeln!(f, "    Block: {:#x}", addr)?;
            for op in &code.ops {
                writeln!(f, "    | {}", op)?;
            }
        }

        write!(f, "]")
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::elf::ElfFile;

    #[test]
    fn flow() {
        let mut file = ElfFile::new(std::fs::File::open("test/block-1").unwrap()).unwrap();
        let text = file.get_section(".text").unwrap();

        let graph = FlowGraph::new(text, file.header.entry);
        println!("flow graph: {}", graph);
    }
}
