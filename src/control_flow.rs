//! Control flow graph calculation.

use std::collections::HashMap;
use std::io::{self, Write};

use crate::Program;
use crate::x86_64::{Instruction, Mnemoic};
use crate::ir::{Microcode, MicroEncoder, JumpCondition};
use crate::num::{Integer, DataType};
use crate::expr::SymExpr;
use crate::sym::{SymState, Event};


/// The control flow graph representation of a program.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FlowGraph {
    /// The nodes of the graph (i.e. basic blocks within a context).
    pub nodes: Vec<FlowNode>,
    /// The basic blocks without context.
    pub blocks: HashMap<u64, BasicBlock>,
    /// The control flow between the nodes. The key pairs are indices
    /// into the `nodes` vector.
    pub edges: HashMap<(usize, usize), (JumpCondition, bool)>,
    /// The nodes which the node with the index has edges to.
    pub incoming: Vec<Vec<usize>>,
    /// The nodes which have edges to the node with the index.
    pub outgoing: Vec<Vec<usize>>,
}

/// A node in the control flow graph, that is a basic block in some context.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FlowNode {
    /// The start address of the block.
    pub addr: u64,
    /// The call trace in (callsite, target) pairs.
    pub trace: Vec<(u64, u64)>,
}

impl FlowNode {
    /// Return the same node but without cycles in the trace.
    fn decycled(&self) -> FlowNode {
        FlowNode {
            addr: self.addr,
            trace: decycle(&self.trace, |a, b| a.1 == b.1)
        }
    }
}

/// A single flat jump-free sequence of micro operations.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BasicBlock {
    /// The adress of the block.
    pub addr: u64,
    /// The byte length of the block.
    pub len: u64,
    /// The addresses and lengths of the instructions of the block alongside
    /// their microcode representations.
    pub code: Vec<(u64, u64, Instruction, Microcode)>,
}

impl FlowGraph {
    /// Generate a flow graph of a program.
    pub fn new(program: &Program) -> FlowGraph {
        FlowConstructor::new(program).construct(program.entry)
    }

    /// Visualize this flow graph in a graphviz DOT file.
    pub fn visualize<W: Write>(
        &self,
        target: W,
        program: &Program,
        title: &str,
        style: VisualizationStyle
    ) -> io::Result<()> {

        // Rebind the target file for more shortness later on.
        let mut f = target;
        const BR: &str = "<br align=\"left\"/>";

        writeln!(f, "digraph Flow {{")?;
        writeln!(f, "label=<Flow graph for {}<br/><br/>>", title)?;
        writeln!(f, "labelloc=\"t\"")?;
        writeln!(f, "graph [fontname=\"Source Code Pro\"]")?;
        writeln!(f, "node [fontname=\"Source Code Pro\"]")?;
        writeln!(f, "edge [fontname=\"Source Code Pro\"]")?;

        // Export the blocks.
        for (index, node) in self.nodes.iter().enumerate() {
            // Format the header of the block box.
            write!(f, "b{} [label=<<b>{:x}", index, node.addr)?;
            if let Some(name) = program.symbols.get(&node.addr) {
                write!(f, " &lt;{}&gt;", name)?;
            }
            if !node.trace.is_empty() {
                write!(f, " by ")?;
                let mut first = true;
                for (callsite, _) in &node.trace {
                    if !first { write!(f, " -&gt; ")?; } first = false;
                    write!(f, "{:x}", callsite)?;
                }
            }
            write!(f, "</b>{}", BR)?;

            if style == VisualizationStyle::Instructions || style == VisualizationStyle::Microcode {
                // Write out the body in either micro operations or instructions.
                let block = &self.blocks[&node.addr];
                for (addr, _, instruction, microcode) in &block.code {
                    if style == VisualizationStyle::Microcode {
                        for op in &microcode.ops {
                            write!(f, "{:x}: {}{}", addr,
                                op.to_string().replace("&", "&amp;"), BR)?;
                        }
                    } else {
                        write!(f, "{:x}: {}{}", addr, instruction, BR)?;
                    }
                }
            }
            write!(f, ">, shape=box")?;

            // Change the background if this nodes is either a source or sink.
            if self.outgoing[index].is_empty() || self.incoming[index].is_empty() {
                write!(f, ", style=filled, fillcolor=\"#dddddd\"")?;
            }
            writeln!(f, "]")?;
        }

        // Export the edges, but sort them first to make the graphviz output
        // deterministic eventhough the hash map cannot be traversed in order.
        let mut edges = self.edges.iter().collect::<Vec<_>>();
        edges.sort_by_key(|edge| edge.0);
        for ((start, end), &(condition, value)) in edges {
            write!(f, "b{} -> b{} [", start, end)?;
            if condition != JumpCondition::True {
                write!(f, "label=\"[{}]\", ", condition.pretty_format(value))?;
            }
            writeln!(f, "style=dashed, color=grey]")?;
        }

        writeln!(f, "}}")
    }
}

/// How to visualize the flow graph.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum VisualizationStyle {
    /// Only display the addresses and call traces of blocks.
    Addresses,
    /// Show the assembly instructions.
    Instructions,
    /// Show the whole microcode representation of the instructions.
    Microcode,
}

/// Constructs a flow graph representation of a program.
#[derive(Debug, Clone)]
struct FlowConstructor<'a> {
    binary: &'a [u8],
    base: u64,
    stack: Vec<(FlowNode, Vec<FlowNode>, SymState)>,
    nodes: HashMap<FlowNode, usize>,
    blocks: HashMap<u64, BasicBlock>,
    edges: HashMap<(usize, usize), (JumpCondition, bool)>,
}

/// An exit of a block.
#[derive(Debug, Clone)]
struct Exit {
    target: SymExpr,
    jumpsite: u64,
    condition: JumpCondition,
    kind: ExitKind,
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
    fn new(program: &'a Program) -> FlowConstructor<'a> {
        FlowConstructor {
            binary: &program.binary,
            base: program.base,
            blocks: HashMap::new(),
            nodes: HashMap::new(),
            edges: HashMap::new(),
            stack: Vec::new(),
        }
    }

    /// Build the flow graph.
    fn construct(mut self, entry: u64) -> FlowGraph {
        self.stack.push((FlowNode {
            addr: entry,
            trace: vec![],
        }, vec![], SymState::new()));

        while let Some((node, path, mut state)) = self.stack.pop() {
            let decycled = node.decycled();

            // Parse the first block.
            let maybe_exit = self.parse_block(decycled.clone(), &mut state);

            // Add the block to the map if we have not already found it
            // through another path with the same call site and call trace.
            let new_id = self.nodes.len();
            self.nodes.entry(decycled).or_insert(new_id);

            // Add blocks reachable from this one.
            if let Some(exit) = maybe_exit {
                self.explore_exit(node, &path, exit, state);
            }
        }

        // Arrange the nodes into a vector.
        let count = self.nodes.len();
        let mut nodes = vec![FlowNode { addr: 0, trace: Vec::new() }; count];
        let mut incoming = vec![Vec::new(); count];
        let mut outgoing = vec![Vec::new(); count];

        for (node, index) in self.nodes.into_iter() {
            nodes[index] = node;
        }

        // Add the outgoing and incoming edges to the nodes.
        for &(start, end) in self.edges.keys() {
            outgoing[start].push(end);
            incoming[end].push(start);
        }

        // Sort the outgoing and incoming vectors to make things deterministic.
        for inc in &mut incoming { inc.sort(); }
        for out in &mut outgoing { out.sort(); }

        FlowGraph {
            nodes,
            blocks: self.blocks,
            edges: self.edges,
            incoming,
            outgoing,
        }
    }

    /// Parse the basic block at the beginning of the given binary code.
    fn parse_block(&mut self, node: FlowNode, state: &mut SymState) -> Option<Exit> {
        let mut parser = if let Some(block) = self.blocks.get(&node.addr) {
            BlockParser::from_block(block)
        } else {
            BlockParser::from_binary(&self.binary, self.base, node.addr)
        };

        // Symbolically execute the block until an exit is found.
        loop {
            let (addr, len, instruction, microcode) = parser.next();

            // Execute the microcode.
            for &op in &microcode.ops {
                // Do one single microcode step.
                let next_addr = addr + len;
                let maybe_event = state.step(next_addr, op);

                // Check for exiting.
                let maybe_exit = self.parse_event(maybe_event, instruction, *addr, next_addr);
                if let Some(exit) = maybe_exit {
                    if let Some(block) = parser.export() {
                        self.blocks.insert(node.addr, block);
                    }
                    return exit;
                }
            }
        }
    }

    /// Determine the kind of exit resulting from a symbolic execution event.
    fn parse_event(
        &self,
        maybe_event: Option<Event>,
        inst: &Instruction,
        current_addr: u64,
        next_addr: u64
    ) -> Option<Option<Exit>> {

        match maybe_event {
            // If it is a jump, add the exit to the list.
            Some(Event::Jump { target, condition, relative }) => Some(
                Some(Exit {
                    target: if relative {
                        target.clone().add(SymExpr::Int(Integer::from_ptr(next_addr)))
                    } else {
                        target.clone()
                    },
                    kind: match inst.mnemoic {
                        Mnemoic::Call => ExitKind::Call,
                        Mnemoic::Ret => ExitKind::Return,
                        _ => ExitKind::Jump,
                    },
                    jumpsite: current_addr,
                    condition,
                })
            ),
            Some(Event::Exit) => Some(None),
            _ => None,
        }
    }

    /// Add reachable blocks to the stack depending on the exit conditions
    /// of the just parsed block.
    fn explore_exit(
        &mut self,
        node: FlowNode,
        path: &[FlowNode],
        exit: Exit,
        state: SymState
    ) {
        match exit.target {
            SymExpr::Int(Integer(DataType::N64, target)) => {
                // Try the not-jumping path if it is viable.
                if exit.condition != JumpCondition::True {
                    let len = self.blocks[&node.addr].len;
                    self.explore_acyclic(
                        node.addr + len,
                        exit.jumpsite,
                        exit.kind,
                        (exit.condition, false),
                        node.clone(),
                        path,
                        &state
                    );
                }

                // Try the jumping path.
                self.explore_acyclic(
                    target,
                    exit.jumpsite,
                    exit.kind,
                    (exit.condition, true),
                    node,
                    path,
                    &state
                );
            },

            _ => panic!("handle_exit: unresolved jump target: {}", exit.target),
        }
    }

    /// Add a target to the stack if it was not visited already through some kind of cycle.
    fn explore_acyclic(
        &mut self,
        addr: u64,
        jumpsite: u64,
        kind: ExitKind,
        condition: (JumpCondition, bool),
        node: FlowNode,
        path: &[FlowNode],
        state: &SymState
    ) {
        // Check if we are already recursing.
        // We allow to recursive twice because we want to capture the returns
        // of the recursing function to itself and the outside.
        let fully_recursive = node.trace.iter()
            .filter(|&&jump| jump == (jumpsite, addr)).count() >= 2;

        // Assemble the new ID.
        let mut target_node = FlowNode {
            addr,
            trace: node.trace.clone(),
        };

        // Adjust the trace.
        match kind {
            ExitKind::Call => target_node.trace.push((jumpsite, addr)),
            ExitKind::Return => { target_node.trace.pop(); },
            _ => {},
        }

        // Only consider the target if it is acyclic or recursing in the allowed limits.
        let looping = path.contains(&target_node);

        // Insert a new edge for the jump.
        let new = self.nodes.len();
        let start = *self.nodes.entry(node.decycled()).or_insert(new);
        let new = self.nodes.len();
        let end = *self.nodes.entry(target_node.decycled()).or_insert(new);
        self.edges.insert((start, end), condition);

        if !looping && !fully_recursive {
            // Add the current block to the path.
            let mut path = path.to_vec();
            path.push(node);

            // Put the new target on top of the search stack.
            self.stack.push((target_node, path, state.clone()));
        }
    }
}

/// Either reuses an existing block or parses a block from binary.
#[derive(Debug, Clone)]
enum BlockParser<'a> {
    BasicBlock {
        block: &'a BasicBlock,
        index: usize,
    },
    Binary {
        entry: u64,
        index: u64,
        binary: &'a [u8],
        encoder: MicroEncoder,
        code: Vec<(u64, u64, Instruction, Microcode)>,
    },
}

impl<'a> BlockParser<'a> {
    /// Create a new block parser from an existing block.
    fn from_block(block: &'a BasicBlock) -> BlockParser {
        BlockParser::BasicBlock {
            block,
            index: 0,
        }
    }

    /// Create a new block parser from unparsed binary.
    fn from_binary(binary: &'a [u8], base: u64, entry: u64) -> BlockParser {
        BlockParser::Binary {
            entry,
            index: 0,
            binary: &binary[(entry - base) as usize ..],
            encoder: MicroEncoder::new(),
            code: Vec::new(),
        }
    }

    /// Retrieve the next parsed element.
    fn next(&mut self) -> &(u64, u64, Instruction, Microcode) {
        match self {
            BlockParser::BasicBlock { block, index } => {
                *index += 1;
                &block.code[*index - 1]
            },
            BlockParser::Binary { entry, index, binary, encoder, code } => {
                let bytes = &binary[*index as usize ..];

                let len = Instruction::length(bytes);
                let instruction = Instruction::decode(bytes).unwrap();
                let microcode = encoder.encode(&instruction).unwrap();
                code.push((*index + *entry, len, instruction, microcode));
                *index += len;

                code.last().unwrap()
            }
        }
    }

    /// Return the parsed block if this parser actually parsing something.
    fn export(self) -> Option<BasicBlock> {
        match self {
            BlockParser::BasicBlock { .. } => None,
            BlockParser::Binary { entry, index, code, .. } => {
                Some(BasicBlock {
                    addr: entry,
                    len: index,
                    code,
                })
            }
        }
    }
}

/// Remove all cycles from a list of comparable items, where `cmp` determines
/// if two items are equal.
fn decycle<T: Clone, F>(sequence: &[T], cmp: F) -> Vec<T> where F: Fn(&T, &T) -> bool {
    let mut out = Vec::new();

    for item in sequence {
        if let Some(pos) = out.iter().position(|x| cmp(item, x)) {
            for _ in 0 .. out.len() - pos - 1 {
                out.pop();
            }
        } else {
            out.push(item.clone());
        }
    }

    out
}


#[cfg(test)]
mod tests {
    use std::fs::{self, File};
    use std::process::Command;
    use super::*;

    fn test(filename: &str) {
        let path = format!("target/bin/{}", filename);

        // Generate the flow graph.
        let program = Program::new(path);
        let graph = FlowGraph::new(&program);

        // Visualize the graph into a PDF file.
        fs::create_dir("target/graphs").ok();
        let flow_temp = "target/temp-flow.gv";
        let flow_file = File::create(flow_temp).unwrap();
        graph.visualize(flow_file, &program, filename, VisualizationStyle::Instructions).unwrap();
        let output = Command::new("dot")
            .arg("-Tpdf")
            .arg(flow_temp)
            .arg("-o")
            .arg(format!("target/graphs/{}.pdf", filename))
            .output()
            .expect("failed to run graphviz");
        io::stdout().write_all(&output.stdout).unwrap();
        io::stderr().write_all(&output.stderr).unwrap();
        fs::remove_file(flow_temp).unwrap();
    }

    #[test]
    fn flow_graph() {
        test("block-1");
        test("case");
        test("twice");
        test("loop");
        test("recursive-1");
        test("recursive-2");
        test("func");
        test("bufs");
    }

    fn test_decycle(left: Vec<&str>, right: Vec<&str>) {
        assert_eq!(decycle(&left, |a, b| a == b), right);
    }

    #[test]
    fn decycling() {
        test_decycle(vec!["main", "fib", "fib"], vec!["main", "fib"]);
        test_decycle(vec!["main", "fib", "fib", "a"], vec!["main", "fib", "a"]);
        test_decycle(vec!["main", "foo", "a"], vec!["main", "foo", "a"]);
        test_decycle(vec!["main", "foo", "bar", "a"], vec!["main", "foo", "bar", "a"]);
        test_decycle(vec!["main", "foo", "bar", "foo"], vec!["main", "foo"]);
        test_decycle(vec!["main", "foo", "bar", "foo", "a"], vec!["main", "foo", "a"]);
        test_decycle(vec!["main", "foo", "bar", "foo", "bar", "a"],
                     vec!["main", "foo", "bar", "a"]);
        test_decycle(vec!["main", "foo", "bar", "bar", "foo", "a"], vec!["main", "foo", "a"]);
    }
}
