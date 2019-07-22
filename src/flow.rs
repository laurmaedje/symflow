//! Flow graph calculation.

use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use std::fmt::{self, Display, Formatter};

use crate::elf::Section;
use crate::amd64::{Instruction, Mnemoic};
use crate::ir::{Microcode, MicroEncoder, Condition};
use crate::sym::{SymState, SymExpr, Event};
use crate::num::{Integer, DataType};


/// Control flow graph representation of a program.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FlowGraph {
    pub blocks: HashMap<BlockId, Block>,
    pub flow: HashMap<(BlockId, BlockId), (Condition, bool)>,
}

/// A single flat jump-free sequence of micro operations.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Block {
    pub id: BlockId,
    pub len: u64,
    pub code: Microcode,
    pub instructions: Vec<Instruction>,
}

/// A block identifier denoted by address and call site.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct BlockId {
    pub addr: u64,
    pub jumpsite: u64,
    pub trace: Vec<u64>,
}

impl FlowGraph {
    /// Generate a flow graph from the `.text` section of a program.
    pub fn new(text: &Section, entry: u64) -> FlowGraph {
        FlowConstructor::new(text).construct(entry)
    }

    /// Visualize this flow graph in a graphviz dot file.
    ///
    /// If `micro` is true, this will print out the microcode, otherwise the
    /// normal instructions.
    pub fn visualize<W: Write>(&self, title: &str, target: W, micro: bool) -> io::Result<()> {
        const BR: &str = "<br align=\"left\"/>";
        let mut f = target;

        writeln!(f, "digraph Flow {{")?;
        writeln!(f, "label=<Flow graph for {}<br/><br/>>", title)?;
        writeln!(f, "labelloc=\"t\"")?;

        writeln!(f, "graph [fontname=\"Source Code Pro\"]")?;
        writeln!(f, "node [fontname=\"Source Code Pro\"]")?;
        writeln!(f, "edge [fontname=\"Source Code Pro\"]")?;

        let mut blocks = self.blocks.values().collect::<Vec<_>>();
        blocks.sort_by_key(|block| &block.id);
        for block in blocks {
            write!(f, "b{} [label=<<b>{:#x}:</b>{}", block.id, block.id.addr, BR)?;
            if micro {
                for op in &block.code.ops {
                    write!(f, "{}{}", op.to_string().replace("&", "&amp;"), BR)?;
                }
            } else {
                for instruction in &block.instructions {
                    write!(f, "{}{}", instruction, BR)?;
                }
            }
            write!(f, ">, shape=box")?;

            if self.flow.keys().find(|edge| edge.0 == block.id).is_none() ||
               self.flow.keys().find(|edge| edge.1 == block.id).is_none() {
                write!(f, ", style=filled, fillcolor=\"#dddddd\"")?;
            }
            writeln!(f, "]")?;
        }

        let mut edges = self.flow.iter().collect::<Vec<_>>();
        edges.sort_by_key(|edge| edge.0);
        for ((start, end), &(condition, value)) in edges {
            write!(f, "b{} -> b{} [", start, end)?;
            if condition != Condition::True {
                write!(f, "label=\"[{}]\", ", condition.pretty_format(value))?;
            }
            writeln!(f, "style=dashed, color=grey]")?;
        }

        writeln!(f, "}}")?;


        Ok(())
    }
}

#[derive(Debug, Clone)]
struct FlowConstructor<'a> {
    bin: &'a [u8],
    base: u64,
    blocks: HashMap<BlockId, Block>,
    flow: HashMap<(BlockId, BlockId), (Condition, bool)>,
    stack: Vec<(BlockId, SymState, Vec<(u64, Vec<u64>)>)>,
}

/// An exit of a block.
#[derive(Debug, Clone)]
struct Exit {
    target: SymExpr,
    kind: ExitKind,
    jumpsite: u64,
    condition: Condition,
    state: SymState,
}

/// How the block is exited.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum ExitKind {
    Call,
    Return,
    Jump,
}

impl<'a> FlowConstructor<'a> {
    /// Construct a new flow graph builder.
    fn new(text: &'a Section) -> FlowConstructor<'a> {
        FlowConstructor {
            bin: &text.data,
            base: text.header.addr,
            blocks: HashMap::new(),
            flow: HashMap::new(),
            stack: Vec::new(),
        }
    }

    /// Build the flow graph.
    fn construct(mut self, entry: u64) -> FlowGraph {
        self.stack.push((BlockId {
            addr: entry,
            jumpsite: 0,
            trace: vec![],
        }, SymState::new(), vec![]));

        while let Some((id, state, path)) = self.stack.pop() {
            // Parse the first block.
            let (mut block, maybe_exit) = self.parse_block(&id, state.clone());
            let len = block.len;

            // Add the block to the map if we have not already found it
            // through another path with the same call site and call trace.
            self.blocks.entry(block.id.clone()).or_insert(block);

            // Add blocks reachable from this one.
            if let Some(exit) = maybe_exit {
                self.handle_exit(id, len, exit, &path);
            }
        }

        FlowGraph {
            blocks: self.blocks,
            flow: self.flow,
        }
    }

    /// Parse the basic block at the beginning of the given binary code.
    fn parse_block(&self, id: &BlockId, mut state: SymState)
    -> (Block, Option<Exit>) {
        let start = (id.addr - self.base) as usize;
        let binary = &self.bin[start ..];

        // Symbolically execute the block and keep the microcode.
        let mut instructions = Vec::new();
        let mut code = Vec::new();
        let mut encoder = MicroEncoder::new();
        let mut index = 0;

        // Execute instructions until an exit is found.
        loop {
            // Parse the instruction.
            let bytes = &binary[index as usize ..];
            let len = Instruction::length(bytes);
            let instruction = Instruction::decode(bytes).unwrap();
            let mnemoic = instruction.mnemoic;
            let current_addr = id.addr + index;
            index += len;

            // Encode the instruction in microcode.
            let ops = encoder.encode(&instruction).unwrap().ops;
            instructions.push(instruction);
            code.extend(&ops);

            // Execute the microcode.
            for &op in &ops {
                let next_addr = id.addr + index;
                let maybe_event = state.step(next_addr, op);

                // Check for exiting.
                if let Some(event) = maybe_event {
                    let exit = match event {
                        // If it is a jump, add the exit to the list.
                        Event::Jump { target, condition, relative } => Some(Exit {
                            target: if relative {
                                target.clone().add(SymExpr::Int(Integer::from_ptr(next_addr)))
                            } else {
                                target.clone()
                            },
                            kind: match mnemoic {
                                Mnemoic::Call => ExitKind::Call,
                                Mnemoic::Ret => ExitKind::Return,
                                _ => ExitKind::Jump,
                            },
                            jumpsite: current_addr,
                            condition,
                            state,
                        }),
                        Event::Exit => None,
                    };

                    return (Block {
                        id: id.clone(),
                        len: index,
                        code: Microcode { ops: code },
                        instructions,
                    }, exit);
                }
            }
        }
    }

    /// Add reachable blocks to the stack depending on the exit conditions of the
    /// just parsed block.
    fn handle_exit(&mut self, mut id: BlockId, len: u64, exit: Exit, path: &[(u64, Vec<u64>)]) {
        match exit.target {
            SymExpr::Int(Integer(DataType::N64, target)) => {
                // Try the not-jumping path.
                if exit.condition != Condition::True {
                    self.add_if_acyclic(
                        id.addr + len, exit.jumpsite, id.clone(), exit.kind,
                        (exit.condition, false), &exit.state, path
                    );
                }

                // Try the jumping path.
                self.add_if_acyclic(
                    target, exit.jumpsite, id, exit.kind,
                    (exit.condition, true), &exit.state, path
                );
            },

            _ => panic!("handle_exit: unresolved jump target: {}", exit.target),
        }
    }

    /// Add a target to the stack if it was not visitied already by this path
    /// (e.g. if is not cyclic).
    fn add_if_acyclic(&mut self, addr: u64, jumpsite: u64, id: BlockId, kind: ExitKind,
        condition: (Condition, bool), state: &SymState, path: &[(u64, Vec<u64>)]) {

        // Adjust the trace.
        let mut trace = id.trace.clone();
        let recursive = id.trace.contains(&jumpsite);
        if !recursive {
            match kind {
                ExitKind::Call => trace.push(jumpsite),
                ExitKind::Return => { trace.pop(); },
                ExitKind::Jump => {},
            };
        }

        // Assemble the new id.
        let target_id = BlockId {
            addr,
            jumpsite,
            trace,
        };

        // Adjust the edges.
        self.flow.insert((id.clone(), target_id.clone()), condition);

        // Only consider the target if it is acyclic.
        if !path.contains(&(jumpsite, id.trace.clone())) && !recursive {
            // Add the current block to the path.
            let mut path = path.to_vec();
            path.push((id.jumpsite, id.trace));

            // Put the new target on top of the search stack.
            self.stack.push((target_id, state.clone(), path));
        }
    }
}

impl Display for FlowGraph {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "FlowGraph [")?;
        if !self.blocks.is_empty() { writeln!(f)?; }
        let mut blocks = self.blocks.values().collect::<Vec<_>>();
        blocks.sort_by_key(|block| &block.id);
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
        write!(f, "Block {}", self.id)?;
        if !self.code.ops.is_empty() { writeln!(f)?; }
        for inst in &self.instructions {
            writeln!(f, "   {}", inst)?;
        }
        writeln!(f, "]")
    }
}

impl Display for BlockId {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{:#x}_{:x}_", self.addr, self.jumpsite)?;
        for id in &self.trace {
            write!(f, "_{:x}", id)?;
        }
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use std::fs::{self, File};
    use std::process::Command;
    use super::*;
    use crate::elf::ElfFile;

    fn test(filename: &str) {
        // Generate the flow graph.
        let mut file = ElfFile::new(File::open(filename).unwrap()).unwrap();
        let text = file.get_section(".text").unwrap();
        let graph = FlowGraph::new(&text, file.header.entry);

        // Visualize the graph into a PDF file.
        let flow_temp = "target/temp-flow.gv";
        let mut flow_file = File::create(flow_temp).unwrap();
        graph.visualize(filename, flow_file, false);
        let output = Command::new("bash")
            .arg("-c")
            .arg(format!("dot -Tpdf {} -o {}-flow.pdf", flow_temp, filename))
            .output()
            .expect("failed to run graphviz");
        io::stdout().write_all(&output.stdout).unwrap();
        io::stderr().write_all(&output.stderr).unwrap();
        fs::remove_file(flow_temp);
    }

    #[test]
    fn flow() {
        test("target/block-1");
        test("target/read");
        test("target/merge");
        test("target/loop");
        test("target/recurse");
    }
}
