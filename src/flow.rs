//! Flow graph calculation.

use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use std::fmt::{self, Display, Formatter};

use crate::elf::Section;
use crate::amd64::Instruction;
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
    pub outgoing: HashSet<BlockId>,
    pub incoming: HashSet<BlockId>,
}

/// A block identifier denoted by address and call site.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct BlockId {
    pub addr: u64,
    pub callsite: u64,
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

        for block in self.blocks.values() {
            write!(f, "b{id} [label=<<b>{id}:</b>{}", BR, id=block.id)?;
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
            if block.incoming.is_empty() || block.outgoing.is_empty() {
                write!(f, ", style=filled, fillcolor=\"#dddddd\"")?;
            }
            writeln!(f, "]")?;
        }

        for (&(start, end), &(condition, value)) in self.flow.iter() {
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
    stack: Vec<(u64, u64, SymState, Vec<(BlockId, BlockId)>)>,
}

/// An exit of a block.
#[derive(Debug, Clone)]
struct Exit {
    target: SymExpr,
    callsite: u64,
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
            flow: HashMap::new(),
            stack: Vec::new(),
        }
    }

    /// Build the flow graph.
    fn construct(mut self, entry: u64) -> FlowGraph {
        self.stack.push((entry, 0, SymState::new(), vec![]));

        while let Some((addr, callsite, state, path)) = self.stack.pop() {
            // Parse the first block.
            let (mut block, maybe_exit) = self.parse_block(addr, callsite, state.clone());
            let id = block.id;
            let len = block.len;

            // Add the block to the map if we have not already found it
            // through another path with the same call site.
            {
                let block_ref = self.blocks.entry(block.id).or_insert(block);
                if let Some((from, _)) = path.last() {
                    block_ref.incoming.insert(*from);
                }
            }

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
    fn parse_block(&self, addr: u64, callsite: u64, mut state: SymState) -> (Block, Option<Exit>) {
        let start = (addr - self.base) as usize;
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
            let current_addr = addr + index;
            index += len;

            // Encode the instruction in microcode.
            let ops = encoder.encode(&instruction).unwrap().ops;
            instructions.push(instruction);
            code.extend(&ops);

            // Execute the microcode.
            for &op in &ops {
                let next_addr = addr + index;
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
                            callsite: current_addr,
                            condition,
                            state,
                        }),
                        Event::Exit => None,
                    };

                    return (Block {
                        id: BlockId {
                            addr,
                            callsite,
                        },
                        len: index,
                        code: Microcode { ops: code },
                        instructions,
                        // These are filled in later.
                        outgoing: HashSet::new(),
                        incoming: HashSet::new(),
                    }, exit);
                }
            }

        }
    }

    /// Add reachable blocks to the stack depending on the exit conditions of the
    /// just parsed block.
    fn handle_exit(&mut self, id: BlockId, len: u64, exit: Exit, path: &[(BlockId, BlockId)]) {
        match exit.target {
            SymExpr::Int(Integer(DataType::N64, target)) => {
                if exit.condition != Condition::True {
                    let alt_target = id.addr + len;
                    self.add_if_acyclic(alt_target, exit.callsite, id, path,
                        (exit.condition, false), &exit.state);
                }

                self.add_if_acyclic(target, exit.callsite, id, path,
                    (exit.condition, true), &exit.state);
            },

            _ => panic!("handle_exit: unresolved jump target: {}", exit.target),
        }
    }

    /// Add a target to the stack if it was not visitied already by this path
    /// (e.g. if is not cyclic).
    fn add_if_acyclic(&mut self, addr: u64, callsite: u64, id: BlockId, path: &[(BlockId, BlockId)],
        condition: (Condition, bool), state: &SymState) {
        // Prepare the block id's.
        let target_id = BlockId { addr, callsite };
        let pair = (id, target_id);

        if !path.contains(&pair) {
            // Adjust the edges.
            self.blocks.entry(id).and_modify(|block| {
                block.outgoing.insert(target_id);
            });
            self.flow.insert(pair, condition);

            // Clone the path and add the new entry.
            let mut new_path = path.to_vec();
            new_path.push(pair);

            // Put the new target on top of the search stack.
            self.stack.push((addr, callsite, state.clone(), new_path));
        }
    }
}

impl Display for FlowGraph {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "FlowGraph [")?;
        if !self.blocks.is_empty() { writeln!(f)?; }
        let mut blocks = self.blocks.values().collect::<Vec<_>>();
        blocks.sort_by_key(|block| block.id);
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
        write!(f, "Block {} > {{", self.id)?;
        let mut first = true;
        for id in &self.outgoing {
            if !first { write!(f, ", ")?; }
            first = false;
            write!(f, "{}", id)?;
        }
        write!(f, "}}: [")?;
        if !self.code.ops.is_empty() { writeln!(f)?; }
        for inst in &self.instructions {
            writeln!(f, "   {}", inst)?;
        }
        writeln!(f, "]")
    }
}

impl Display for BlockId {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        if self.callsite == 0 {
            write!(f, "{:#x}", self.addr)
        } else {
            write!(f, "{:#x}_{:#x}", self.addr, self.callsite)
        }
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
        test("target/paths");
    }
}
