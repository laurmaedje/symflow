//! Control flow graph calculation.

use std::collections::HashMap;
use std::io::{self, Write};
use std::rc::Rc;

use crate::Program;
use crate::x86_64::{Instruction, Mnemoic};
use crate::ir::{Microcode, MicroEncoder};
use crate::math::{Integer, DataType, SymExpr, SymCondition, Solver};
use crate::sym::{SymState, MemoryStrategy, Event};


/// The control flow graph representation of a program.
#[derive(Debug, Clone)]
pub struct ControlFlowGraph {
    /// The basic blocks in their respective contexts.
    pub nodes: Vec<ControlFlowNode>,
    /// The basic blocks without context.
    pub blocks: HashMap<u64, BasicBlock>,
    /// The control flow between the nodes. The key pairs are indices
    /// into the `nodes` vector.
    pub edges: HashMap<(usize, usize), SymCondition>,
    /// The nodes which the node with the index has edges to.
    pub incoming: Vec<Vec<usize>>,
    /// The nodes which have edges to the node with the index.
    pub outgoing: Vec<Vec<usize>>,
}

/// A node in the control flow graph, that is a basic block in some context.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ControlFlowNode {
    /// The start address of the block. Can be used as an index
    /// into the `blocks` hash map of the graph.
    pub addr: u64,
    /// The call trace in (callsite, target) pairs.
    pub trace: Vec<(u64, u64)>,
}

impl ControlFlowNode {
    /// Return the same node but without cycles in the trace.
    fn decycled(&self) -> ControlFlowNode {
        ControlFlowNode {
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

impl ControlFlowGraph {
    /// Generate a control flow graph of a program.
    pub fn new(program: &Program) -> ControlFlowGraph {
        ControlFlowExplorer::new(program).run()
    }

    /// Visualize this flow graph in a graphviz DOT file.
    pub fn visualize<W: Write>(
        &self,
        target: W,
        program: &Program,
        title: &str,
        style: VisualizationStyle
    ) -> io::Result<()> {
        use super::visualize::*;
        let mut f = target;

        write_header(&mut f, &format!("Control flow graph for {}", title))?;

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

        write_edges(&mut f, &self.edges, |f, (_, condition)| {
            if condition != &SymCondition::TRUE {
                write!(f, "label=\"{}\", ", condition)?;
            }
            Ok(())
        })?;

        write_footer(&mut f)
    }
}

/// How to visualize the control flow graph.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum VisualizationStyle {
    /// Only display the addresses and call traces of blocks.
    Addresses,
    /// Show the assembly instructions.
    Instructions,
    /// Show the whole microcode representation of the instructions.
    Microcode,
}

/// Constructs a control flow graph representation of a program.
#[derive(Clone)]
struct ControlFlowExplorer<'a> {
    program: &'a Program,
    stack: Vec<ExplorationTarget>,
    nodes: HashMap<ControlFlowNode, usize>,
    blocks: HashMap<u64, BasicBlock>,
    edges: HashMap<(usize, usize), SymCondition>,
}

#[derive(Clone)]
struct ExplorationTarget {
    node: ControlFlowNode,
    state: SymState,
    path: Vec<usize>,
}

#[derive(Clone)]
struct Exit {
    target: SymExpr,
    jumpsite: u64,
    condition: SymCondition,
    kind: ExitKind,
}

#[derive(Copy, Clone)]
enum ExitKind {
    Call,
    Return,
    Jump,
}

impl<'a> ControlFlowExplorer<'a> {
    fn new(program: &'a Program) -> ControlFlowExplorer<'a> {
        ControlFlowExplorer {
            program,
            blocks: HashMap::new(),
            nodes: HashMap::new(),
            edges: HashMap::new(),
            stack: Vec::new(),
        }
    }

    /// Build the control flow graph.
    fn run(mut self) -> ControlFlowGraph {
        let node = ControlFlowNode { addr: self.program.entry, trace: vec![], };
        let base_state = SymState::new(MemoryStrategy::PerfectMatches, Rc::new(Solver::new()));

        self.stack.push(ExplorationTarget {
            node,
            state: base_state,
            path: Vec::new(),
        });

        while let Some(mut exp) = self.stack.pop() {
            // Explore this block and find all the ones reachable from this one.
            if let Some(exit) = self.execute_block(&mut exp) {
                self.explore_exit(&exp, exit);
            }
        }

        self.finish()
    }

    /// Arrange all data in the way expected for the flow graph.
    fn finish(self) -> ControlFlowGraph {
        // Arrange the nodes into a vector.
        let count = self.nodes.len();
        let mut nodes = vec![ControlFlowNode { addr: 0, trace: Vec::new() }; count];
        for (node, index) in self.nodes.into_iter() {
            nodes[index] = node;
        }

        // Add the outgoing and incoming edges to the nodes.
        let mut incoming = vec![Vec::new(); count];
        let mut outgoing = vec![Vec::new(); count];
        for &(start, end) in self.edges.keys() {
            outgoing[start].push(end);
            incoming[end].push(start);
        }

        // Sort the outgoing and incoming vectors to make things deterministic.
        for inc in &mut incoming { inc.sort(); }
        for out in &mut outgoing { out.sort(); }

        ControlFlowGraph {
            nodes,
            blocks: self.blocks,
            edges: self.edges,
            incoming,
            outgoing,
        }
    }

    /// Parse and execute the basic block determined by the exploration
    /// target and find all exits. Returns `None` if there is no exit to
    /// a new block, that is, if there was a sys_exit.
    fn execute_block(&mut self, exp: &mut ExplorationTarget) -> Option<Exit> {
        // Create a new binary parser or reuse an existing block.
        let mut parser = match self.blocks.get(&exp.node.addr) {
            Some(block) => BlockParser::from_block(block),
            None => {
                BlockParser::from_binary(&self.program.binary, self.program.base, exp.node.addr)
            },
        };

        // Symbolically execute the block until an exit is found.
        loop {
            let (addr, len, instruction, microcode) = parser.next();

            // Execute the microcode.
            for op in &microcode.ops {
                let next_addr = addr + len;

                if let Some(event) = exp.state.step(next_addr, op) {
                    if let Some(exit) = self.find_exits(event, instruction, *addr, next_addr) {
                        if let Some(block) = parser.export() {
                            self.blocks.insert(exp.node.addr, block);
                        }
                        return exit;
                    }
                }
            }
        }
    }

    /// Determine the kind of exit resulting from a symbolic execution event:
    /// - Returns Some(Some(exit)) if there are new blocks resulting.
    /// - Returns Some(None) if it exited without new blocks (sys_exit).
    /// - Returns None if it was no exit at all.
    fn find_exits(
        &self,
        event: Event,
        inst: &Instruction,
        current_addr: u64,
        next_addr: u64
    ) -> Option<Option<Exit>> {
        match event {
            // If it is a jump, add the exit to the list.
            Event::Jump { target, condition, relative } => Some(Some(Exit {
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
            })),
            Event::Exit => Some(None),
            _ => None,
        }
    }

    /// Add reachable blocks to the stack depending on the exit conditions
    /// of the just parsed block.
    fn explore_exit(&mut self, exp: &ExplorationTarget, exit: Exit) {
        if let SymExpr::Int(Integer(DataType::N64, target)) = exit.target {
            // Try the not-jumping path if it is viable.
            if exit.condition != SymCondition::TRUE {
                let len = self.blocks[&exp.node.addr].len;

                let condition = exit.condition.clone().not();
                let cond = exp.state.solver.simplify_condition(&condition);
                self.explore_acyclic(&exp, exp.node.addr + len, exit.jumpsite, exit.kind, cond);
            }

            // Try the jumping path anyways.
            self.explore_acyclic(&exp, target, exit.jumpsite, exit.kind, exit.condition);
        } else {
            panic!("handle_exit: unresolved jump target: {}", exit.target);
        }
    }

    /// Add a target to the search stack if it was not visited already
    /// through some kind of cycle.
    fn explore_acyclic(
        &mut self,
        exp: &ExplorationTarget,
        addr: u64,
        jumpsite: u64,
        exit_kind: ExitKind,
        condition: SymCondition
    ) {
        // Assemble the new node.
        let mut target_node = ControlFlowNode { addr, trace: exp.node.trace.clone() };
        match exit_kind {
            ExitKind::Call => target_node.trace.push((jumpsite, addr)),
            ExitKind::Return => { target_node.trace.pop(); },
            _ => {},
        }

        // Insert a new edge for the jump.
        let start = self.insert_node(exp.node.decycled());
        let end = self.insert_node(target_node.decycled());
        self.edges.insert((start, end), condition);

        // Only consider the target if it is acyclic or recursing in the allowed limits.
        let looping = exp.path.contains(&end);
        if !looping {
            // Check if we are already recursing.
            // We allow to recursive twice because we want to capture the returns
            // of the recursing function to itself and the outside.
            let fully_recursive = exp.node.trace.iter()
                .filter(|&&jump| jump == (jumpsite, addr)).count() >= 2;

            if !fully_recursive {
                // Add the current block to the path.
                let mut path = exp.path.to_vec();
                path.push(start);

                self.stack.push(ExplorationTarget {
                    node: target_node,
                    path,
                    state: exp.state.clone(),
                });
            }
        }
    }

    /// Add a node to the list.
    fn insert_node(&mut self, node: ControlFlowNode) -> usize {
        let new_index = self.nodes.len();
        *self.nodes.entry(node).or_insert(new_index)
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
        BlockParser::BasicBlock { block, index: 0 }
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
/// if two items are equal. For example this turns 1 -> 2 -> 3 -> 2 -> 4 into
/// 1 -> 2 -> 4.
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
    use crate::flow::visualize::test::compile;
    use super::*;

    fn test(filename: &str) {
        let path = format!("target/bin/{}", filename);

        // Generate the flow graph.
        let program = Program::new(path);
        let graph = ControlFlowGraph::new(&program);

        compile("target/control-flow", filename, |file| {
            graph.visualize(file, &program, filename, VisualizationStyle::Instructions)
        });
    }

    #[test]
    fn control_flow_graph() {
        test("block-1");
        test("block-2");
        test("case");
        test("twice");
        test("loop");
        test("recursive-1");
        test("recursive-2");
        test("func");
        test("bufs");
        test("paths");
        test("deep");
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
