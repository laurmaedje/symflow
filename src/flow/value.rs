//! Value flow analysis.

use std::collections::HashMap;
use std::io::{self, Write};
use std::rc::Rc;

use crate::x86_64::Register;
use crate::math::{SymCondition, SharedSolver, Solver};
use crate::sym::{SymState, MemoryStrategy};
use super::{ControlFlowGraph, AbstractLocation, StorageLocation};


/// Contains the conditions under which values flow between abstract locations.
#[derive(Debug, Clone)]
pub struct ValueFlowGraph {
    /// The nodes in the graph, i.e. all abstract locations in the executable.
    pub nodes: Vec<AbstractLocation>,
    /// The conditions for value flow between the abstract locations.
    /// The key pairs are indices into the `nodes` vector.
    pub edges: HashMap<(usize, usize), SymCondition>,
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

        for (index, node) in self.nodes.iter().enumerate() {
            let fmt = node.to_string().replace(">", "&lt;");
            let mut splitter = fmt.splitn(2, ' ');
            writeln!(f, "b{} [label=<<b>{}</b> {}>, shape=box]", index,
                    splitter.next().unwrap(), splitter.next().unwrap())?;
        }

        write_edges(&mut f, &self.edges, |f, ((start, end), condition)| {
            if condition != &SymCondition::TRUE {
                write!(f, "label=\"{}\", ", condition)?;
            }
            if self.nodes[start].location.normalized() == self.nodes[end].location.normalized() {
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
}

impl<'g> ValueFlowExplorer<'g> {
    fn new(graph: &'g ControlFlowGraph) -> ValueFlowExplorer<'g> {
        ValueFlowExplorer {
            graph,
            solver: Rc::new(Solver::new()),
            nodes: HashMap::new(),
            edges: HashMap::new(),
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
        let mut targets = vec![(0, base_state, HashMap::new())];

        while let Some((target, mut state, mut location_links)) = targets.pop() {
            let node = &self.graph.nodes[target];
            let block = &self.graph.blocks[&node.addr];

            // Simulate a basic block.
            for (addr, len, instruction, microcode) in &block.code {
                let addr = *addr;
                let next_addr = addr + len;

                state.track(&instruction, addr);

                for (source, sink) in instruction.flows() {
                    let source_index = self.insert_node(add_context(source, addr, &state.trace));
                    let sink_index = self.insert_node(add_context(sink, addr, &state.trace));

                    // Connect the abstract locations to their predecessors.
                    self.connect_previous(&mut location_links, source, source_index);
                    self.connect_previous(&mut location_links, sink, sink_index);

                    // For flows inherent to an instruction the condition is
                    // obviously always true.
                    self.edges.insert((source_index, sink_index), SymCondition::TRUE);
                }

                // Execute the microcode for the instruction.
                for op in &microcode.ops {
                    state.step(next_addr, op);
                }
            }

            // Add all nodes reachable from that one as targets.
            for &id in &self.graph.outgoing[target] {
                targets.push((id, state.clone(), location_links.clone()));
            }
        }

        // Arrange the nodes into a vector.
        let count = self.nodes.len();
        let default = AbstractLocation {
            addr: 0, trace: Vec::new(),
            location: StorageLocation::Direct(Register::RAX)
        };
        let mut nodes = vec![default; count];

        for (node, index) in self.nodes.into_iter() {
            nodes[index] = node;
        }

        ValueFlowGraph {
            nodes,
            edges: self.edges,
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
    ) {
        // We only consider direct registers here and not memory storage slots
        // because eventhough they may have the same "location", i.e. the
        // same address formula based on registers the actual address may
        // differ depending on the register values.
        if let StorageLocation::Indirect { .. } = location {
            return;
        }

        // We make sure that different versions of the same register
        // (like EAX and RAX) shared the same slot in the link map.
        let location = location.normalized();

        // Add a link to the previous abstract location of the same
        // storage location if it was used before.
        if let Some(prev) = links.get(&location) {
            self.edges.insert((*prev, index), SymCondition::TRUE);
        }

        links.insert(location, index);
    }
}

fn add_context(location: StorageLocation, addr: u64, trace: &[u64]) -> AbstractLocation {
    AbstractLocation {
        addr,
        trace: trace.to_vec(),
        location
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
