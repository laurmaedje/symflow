//! Value flow analysis.

use std::collections::HashMap;
use std::io::{self, Write};
use std::rc::Rc;

use crate::x86_64::Register;
use crate::math::{SymCondition, Symbol, SharedSolver, Solver};
use crate::sym::{SymState, Event, MemoryStrategy, TypedMemoryAccess, StdioKind};
use super::{ControlFlowGraph, AbstractLocation, StorageLocation};
use super::alias::determine_aliasing_condition;


/// Contains the conditions under which values flow between abstract locations.
#[derive(Debug, Clone)]
pub struct ValueFlowGraph {
    /// The nodes in the graph, i.e. all abstract locations in the executable.
    pub nodes: Vec<AbstractLocation>,
    /// The conditions for value flow between the abstract locations.
    /// The key pairs are indices into the `nodes` vector.
    pub edges: HashMap<(usize, usize), SymCondition>,
    /// For reads or writes through standard interfaces we keep the node they go to,
    /// the kind of I/O and the name of the symbol we read from or wrote to.
    pub io: Vec<(usize, StdioKind, Symbol)>,
}

impl ValueFlowGraph {
    /// Create a new value flow graph for all abstract locations.
    pub fn new(graph: &ControlFlowGraph) -> ValueFlowGraph {
        ValueFlowExplorer::new(graph).run()
    }

    /// Visualize this flow graph in a graphviz DOT file.
    pub fn visualize<W: Write>(
        &self,
        target: W,
        title: &str,
    ) -> io::Result<()> {
        use super::visualize::*;
        let mut f = target;

        write_header(&mut f, &format!("Value flow graph for {}", title))?;

        for (index, (loc, kind, symbol)) in self.io.iter().enumerate() {
            let color = match kind {
                StdioKind::Stdin => "#4caf50",
                StdioKind::Stdout => "#03a9f4",
            };
            writeln!(f, "k{} [label=<<b>{}</b>>, shape=box, style=filled, fillcolor=\"{}\"]",
                index, symbol, color)?;

            match kind {
                StdioKind::Stdin => writeln!(f, "k{} -> b{}", index, loc)?,
                StdioKind::Stdout => writeln!(f, "b{} -> k{}", loc, index)?,
            }
        }

        for (index, node) in self.nodes.iter().enumerate() {
            let fmt = node.to_string().replace(">", "&gt;");
            let mut splitter = fmt.splitn(2, ' ');
            writeln!(f, "b{} [label=<<b>{}</b> {}>, shape=box]", index,
                    splitter.next().unwrap(), splitter.next().unwrap())?;
        }

        write_edges(&mut f, &self.edges, |f, ((start, end), condition)| {
            if condition != &SymCondition::TRUE {
                write!(f, "label=\"{}\", ", condition)?;
            }
            if self.nodes[start].storage.normalized() == self.nodes[end].storage.normalized() {
                write!(f, "style=dashed, ")?;
            }
            write!(f, "color=grey")
        })?;

        write_footer(&mut f)
    }
}

/// Analyses the value flow in the whole executable, building a value flow graph.
struct ValueFlowExplorer<'g> {
    graph: &'g ControlFlowGraph,
    solver: SharedSolver,
    nodes: HashMap<AbstractLocation, usize>,
    edges: HashMap<(usize, usize), SymCondition>,
    io: HashMap<AbstractLocation, (StdioKind, Symbol)>,
}

#[derive(Clone)]
struct ExplorationTarget {
    target: usize,
    state: SymState,
    location_links: HashMap<StorageLocation, usize>,

    /// All past writing memory accesses with their abstract location index (node id).
    /// Whenever a reading memory access happens we can compare it with all these.
    write_accesses: Vec<(usize, TypedMemoryAccess)>
}

impl<'g> ValueFlowExplorer<'g> {
    fn new(graph: &'g ControlFlowGraph) -> ValueFlowExplorer<'g> {
        ValueFlowExplorer {
            graph,
            solver: Rc::new(Solver::new()),
            nodes: HashMap::new(),
            edges: HashMap::new(),
            io: HashMap::new(),
        }
    }

    /// Build the value flow graph.
    /// -----------------------------------------------------------------------
    /// This function traverses the complete control flow graph,
    /// symbolically executing each path.
    ///
    /// All direct flows are translated into edges with condition _True_ in the
    /// graph. Indirect flows through memory can have more complex conditions
    /// associated with them.
    pub fn run(mut self) -> ValueFlowGraph {
        let base_state = SymState::new(MemoryStrategy::ConditionalTrees, self.solver.clone());

        let mut targets = vec![ExplorationTarget {
            target: 0,
            state: base_state,
            location_links: HashMap::new(),
            write_accesses: Vec::new(),
        }];

        while let Some(mut exp) = targets.pop() {
            let node = &self.graph.nodes[exp.target];
            let block = &self.graph.blocks[&node.addr];

            // Simulate a basic block.
            for (addr, len, instruction, microcode) in &block.code {
                let addr = *addr;
                let next_addr = addr + len;

                exp.state.track(&instruction, addr);

                for (source, sink) in instruction.flows() {
                    let source_index = self.insert_node(add_context(source, addr, &exp.state.trace));
                    let sink_index = self.insert_node(add_context(sink, addr, &exp.state.trace));

                    // Connect the abstract locations to their predecessors.
                    self.connect_previous(&mut exp.location_links, source, source_index, false);
                    self.connect_previous(&mut exp.location_links, sink, sink_index, true);

                    // For flows inherent to an instruction the condition is
                    // obviously always true.
                    self.edges.insert((source_index, sink_index), SymCondition::TRUE);

                    // For writing memory accesses we need to store them so we
                    // can check reading accesses later on for aliasing.
                    if let Some(access) = exp.state.get_access_for_location(sink) {
                        exp.write_accesses.push((sink_index, access));
                    }

                    // For reading memory accesses we need to check if the alias
                    // with any of the previous writing ones.
                    if let Some(access) = exp.state.get_access_for_location(source) {
                        self.handle_access(source_index, access, &exp);
                    }
                }

                // Execute the microcode for the instruction.
                for op in &microcode.ops {
                    if let Some(event) = exp.state.step(next_addr, op) {
                        match event {
                            // Generate I/O nodes.
                            Event::Stdio(kind, symbols) => {
                                for (symbol, access) in symbols {
                                    // Add to the previous links list.
                                    let location = exp.state.symbol_map[&symbol].clone();
                                    let index = self.insert_node(location.clone());
                                    exp.location_links.insert(location.storage.normalized(), index);

                                    // If it is a stdin read, that is, a memory write, add it
                                    // to the write access list.
                                    match kind {
                                        StdioKind::Stdin => exp.write_accesses.push((index, access)),
                                        StdioKind::Stdout => self.handle_access(index, access, &exp),
                                    }

                                    self.io.insert(location, (kind, symbol));
                                }
                            },
                            _ => {}
                        }
                    }
                }
            }

            // Add all nodes reachable from that one as targets.
            for &id in &self.graph.outgoing[exp.target] {
                targets.push(ExplorationTarget {
                    target: id,
                    .. exp.clone()
                });
            }
        }

        // Arrange the I/O into a vector.
        let io = self.io.iter()
            .map(|(loc, &(kind, symbol))| (self.nodes[loc], kind, symbol)).collect();

        let count = self.nodes.len();
        let default = AbstractLocation {
            addr: 0, trace: Vec::new(),
            storage: StorageLocation::Direct(Register::RAX)
        };

        // Arrange the nodes into a vector.
        let mut nodes = vec![default; count];
        for (node, index) in self.nodes.into_iter() {
            nodes[index] = node;
        }

        ValueFlowGraph {
            nodes,
            edges: self.edges,
            io,
        }
    }

    /// Add a node to the list. This
    /// - inserts it and returns the new index if it didn't exist before
    /// - finds the existing node and returns it's index
    fn insert_node(&mut self, node: AbstractLocation) -> usize {
        let new_index = self.nodes.len();
        *self.nodes.entry(node).or_insert(new_index)
    }

    /// Adds edges between abstract locations that share the same storage
    /// location and don't change in between their contexts.
    fn connect_previous(
        &mut self,
        links: &mut HashMap<StorageLocation, usize>,
        location: StorageLocation,
        index: usize,
        overwritten: bool
    ) {
        // We only consider direct registers here and not memory storage slots
        // because eventhough they may have the same "location", i.e. the
        // same address formula based on registers the actual address may
        // differ depending on the register values.
        if location.accesses_memory() {
            return;
        }

        // We make sure that different versions of the same register
        // (like EAX and RAX) shared the same slot in the link map.
        let location = location.normalized();

        // Add a link to the previous abstract location of the same
        // storage location if it was used before.
        if !overwritten {
            if let Some(prev) = links.get(&location) {
                self.edges.insert((*prev, index), SymCondition::TRUE);
            }
        }

        links.insert(location, index);
    }

    fn handle_access(&mut self, index: usize, access: TypedMemoryAccess, exp: &ExplorationTarget) {
        for (prev_index, prev) in &exp.write_accesses {
            let condition = determine_aliasing_condition(
                &self.solver,
                prev,
                &access
            );

            if condition != SymCondition::FALSE {
                self.edges.insert((*prev_index, index), condition);
            }
        }
    }
}

fn add_context(storage: StorageLocation, addr: u64, trace: &[u64]) -> AbstractLocation {
    AbstractLocation {
        addr,
        trace: trace.to_vec(),
        storage
    }
}


#[cfg(test)]
mod tests {
    use crate::Program;
    use crate::flow::visualize::test::compile;
    use super::*;

    fn test(filename: &str) {
        let path = format!("target/bin/{}", filename);

        let program = Program::new(path);
        let control_graph = ControlFlowGraph::new(&program);
        let value_graph = ValueFlowGraph::new(&control_graph);

        compile("target/value-flow", filename, |file| {
            value_graph.visualize(file, filename)
        });
    }

    #[test]
    fn value_flow_graph() {
        test("bufs");
        test("paths");
        test("deep");
    }
}
