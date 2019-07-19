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
    pub blocks: HashMap<u64, Block>,
    pub flow: HashMap<(u64, u64), (Condition, bool)>,
}

/// A single flat jump-free sequence of micro operations.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Block {
    pub addr: u64,
    pub len: u64,
    pub code: Microcode,
    pub instructions: Vec<Instruction>,
    pub outgoing: HashSet<u64>,
    pub incoming: HashSet<u64>,
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
        blocks.sort_by_key(|block| block.addr);
        for block in blocks {
            write!(f, "b{addr:x} [label=<<b>{addr:#x}:</b>{}", BR, addr=block.addr)?;
            if micro {
                for op in &block.code.ops {
                    write!(f, "{}{}", op.to_string().replace("&", "&amp;"), BR)?;
                }
            } else {
                for instruction in &block.instructions {
                    write!(f, "{}{}", instruction, BR)?;
                }
            }
            writeln!(f, ">, shape=box]")?;
        }

        for (&(start, end), &(condition, value)) in self.flow.iter() {
            write!(f, "b{:x} -> b{:x} [", start, end)?;
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
    blocks: HashMap<u64, Block>,
    flow: HashMap<(u64, u64), (Condition, bool)>,
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
            flow: HashMap::new(),
            stack: Vec::new(),
        }
    }

    /// Build the flow graph.
    fn construct(mut self, entry: u64) -> FlowGraph {
        self.stack.push((entry, SymState::new(), vec![]));

        while let Some((addr, state, path)) = self.stack.pop() {
            // Parse the first block.
            let (mut block, maybe_exit) = self.parse_block(addr, state.clone());
            let len = block.len;

            // Add the incoming edge for the block we come from.
            if let Some((from, to)) = path.last() {
                block.incoming.insert(*from);
            }

            // Add the block to the map if we have not already found it
            // through another path.
            self.blocks.entry(block.addr).or_insert(block);

            // Add blocks reachable from this one.
            if let Some(exit) = maybe_exit {
                self.handle_exit(addr, len, exit, path);
            }
        }

        FlowGraph {
            blocks: self.blocks,
            flow: self.flow,
        }
    }

    /// Parse the basic block at the beginning of the given binary code.
    fn parse_block(&self, addr: u64, mut state: SymState) -> (Block, Option<Exit>) {
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
            index += len;

            // Encode the instruction in microcode.
            let ops = encoder.encode(&instruction).unwrap().ops;
            instructions.push(instruction);
            code.extend(&ops);

            // Execute the microcode.
            for &op in &ops {
                let maybe_event = state.step(addr + index, op);

                // Check for exiting.
                if let Some(event) = maybe_event {
                    let exit = match event {
                        // If it is a jump, add the exit to the list.
                        Event::Jump { target, condition, relative } => Some(Exit {
                            target: if relative {
                                target.clone().add(SymExpr::Int(Integer::from_ptr(addr + index)))
                            } else {
                                target.clone()
                            },
                            condition,
                            state,
                        }),
                        Event::Exit => None,
                    };

                    return (Block {
                        addr,
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
    fn handle_exit(&mut self, addr: u64, len: u64, exit: Exit, path: Vec<(u64, u64)>) {
        match exit.target {
            SymExpr::Int(Integer(DataType::N64, target)) => {
                if exit.condition != Condition::True {
                    let alt_target = addr + len;
                    self.add_if_acyclic(alt_target, addr, &path,
                        (exit.condition, false), &exit.state);
                }

                self.add_if_acyclic(target, addr, &path, (exit.condition, true), &exit.state);
            },

            _ => panic!("handle_exit: unresolved jump target: {}", exit.target),
        }
    }

    /// Add a target to the stack if it was not visitied already by this path
    /// (e.g. if is not cyclic).
    fn add_if_acyclic(&mut self, target: u64, addr: u64, path: &[(u64, u64)],
        condition: (Condition, bool), state: &SymState) {
        if !path.contains(&(addr, target)) {

            // Adjust the edges.
            self.blocks.entry(addr).and_modify(|block| {
                block.outgoing.insert(target);
            });
            self.flow.insert((addr, target), condition);

            // Clone the path and add the new entry.
            let mut new_path = path.to_vec();
            new_path.push((addr, target));

            // Put the new target on top of the search stack.
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
        write!(f, "Block {:#x} > {{", self.addr)?;
        let mut first = true;
        for addr in &self.outgoing {
            if !first { write!(f, ", ")?; }
            first = false;
            write!(f, "{:#x}", addr)?;
        }
        write!(f, "}}: [")?;
        if !self.code.ops.is_empty() { writeln!(f)?; }
        for inst in &self.instructions {
            writeln!(f, "   {}", inst)?;
        }
        writeln!(f, "]")
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
        let flow_temp = "test/temp-flow.gv";
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
        test("test/block-1");
        test("test/read");
        test("test/paths");
    }
}
