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
    ///
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
                    let source_node = AbstractLocation::new(addr, exp.state.trace.clone(), source);
                    let sink_node = AbstractLocation::new(addr, exp.state.trace.clone(), sink);
                    let source_index = self.insert_node(source_node);
                    let sink_index = self.insert_node(sink_node);

                    // Connect the abstract locations to their previous abstract
                    // location with the same storage location.
                    self.link_location(&mut exp, source, source_index, false);
                    self.link_location(&mut exp, sink, sink_index, true);

                    // For flows inherent to an instruction the condition is
                    // obviously always true.
                    self.edges.insert((source_index, sink_index), SymCondition::TRUE);

                    // Writing memory accesses are stored in the `write_accesses` list
                    // so we can check aliasing with reading accesses later on.
                    if let Some(access) = exp.state.get_access_for_location(sink) {
                        exp.write_accesses.push((sink_index, access));
                    }

                    // For reading memory accesses we need to check if they alias
                    // with any of the previous writing accesses.
                    if let Some(access) = exp.state.get_access_for_location(source) {
                        self.handle_read_access(&exp, access, source_index);
                    }
                }

                for op in &microcode.ops {
                    if let Some(event) = exp.state.step(next_addr, op) {
                        match event {
                            Event::Stdio(kind, ios) => self.handle_io(&mut exp, kind, ios),
                            _ => {},
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
        let mut io: Vec<_> = self.io.iter()
            .map(|(loc, &(kind, symbol))| (self.nodes[loc], kind, symbol))
            .collect();
        io.sort_by_key(|x| x.0);

        // Arrange the nodes into a vector.
        let default = AbstractLocation::new(0, vec![], StorageLocation::Direct(Register::RAX));
        let mut nodes = vec![default; self.nodes.len()];
        for (node, index) in self.nodes.into_iter() {
            nodes[index] = node;
        }

        ValueFlowGraph {
            nodes,
            edges: self.edges,
            io,
        }
    }

    /// Add I/O nodes and abstract locations for reads and writes.
    fn handle_io(
        &mut self,
        exp: &mut ExplorationTarget,
        kind: StdioKind,
        ios: Vec<(Symbol, TypedMemoryAccess)>
    ) {
        for (symbol, access) in ios {
            // Add to the previous links list.
            let location = exp.state.symbol_map[&symbol].clone();
            let index = self.insert_node(location.clone());
            exp.location_links.insert(location.storage.normalized(), index);

            // If it is a stdin read, that is, a memory write, add it
            // to the write access list.
            match kind {
                StdioKind::Stdin => exp.write_accesses.push((index, access)),
                StdioKind::Stdout => self.handle_read_access(exp, access, index),
            }

            self.io.insert(location, (kind, symbol));
        }
    }

    /// Check all previous write accesses for aliasing with the current read access and add
    /// conditional edges in between if necessary.
    fn handle_read_access(
        &mut self,
        exp: &ExplorationTarget,
        access: TypedMemoryAccess,
        location_index: usize,
    ) {
        for (prev_index, prev) in &exp.write_accesses {
            let condition = determine_aliasing_condition(&self.solver, prev, &access);

            if condition != SymCondition::FALSE {
                self.edges.insert((*prev_index, location_index), condition);
            }
        }
    }

    /// Add an edge to the previous abstract location with the same storage.
    /// If the location has been just overwritten, we remember that for the next
    /// time it is used by inserting it in the `location_links` map.
    fn link_location(
        &mut self,
        exp: &mut ExplorationTarget,
        location: StorageLocation,
        location_index: usize,
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
            if let Some(prev) = exp.location_links.get(&location) {
                self.edges.insert((*prev, location_index), SymCondition::TRUE);
            }
        }

        exp.location_links.insert(location, location_index);
    }

    /// Add a node to the list. This
    /// - inserts it and returns the new index if it didn't exist before
    /// - finds the existing node and returns it's index
    fn insert_node(&mut self, node: AbstractLocation) -> usize {
        let new_index = self.nodes.len();
        *self.nodes.entry(node).or_insert(new_index)
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
        // test("paths");
        // test("deep");
    }
}
