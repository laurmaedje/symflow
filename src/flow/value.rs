//! Value flow analysis.

use std::collections::HashMap;
use std::io::{self, Write};
use std::rc::Rc;

use crate::x86_64::Register;
use crate::math::{SymCondition, Symbol, SharedSolver, Solver, DataType};
use crate::sym::{SymState, Event, MemoryStrategy, TypedMemoryAccess, StdioKind};
use super::{ControlFlowGraph, AbstractLocation, StorageLocation};
use super::alias::determine_aliasing_condition;
use DataType::*;


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
                write!(f, "style=dashed, color=\"#ababab\"")?;
            }
            Ok(())
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

    /// The set of all conditions (through ifs) met on this path.
    preconditions: Vec<SymCondition>,

    /// For each storage location that was already used we stored the abstract
    /// location node were it was written, so we can add an edge back to it
    /// when we use this storage.
    location_links: HashMap<StorageLocation, usize>,

    /// All past writing memory accesses with their abstract location index (node id).
    /// The last usize holds the number of preconditions that were already active
    /// when this access happened. This allows us to discern the new preconditions
    /// for a read access from those that already were before.
    write_accesses: Vec<(usize, TypedMemoryAccess, usize)>
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
    fn run(mut self) -> ValueFlowGraph {
        let base_state = SymState::new(MemoryStrategy::ConditionalTrees, self.solver.clone());

        let mut targets = vec![ExplorationTarget {
            target: 0,
            state: base_state,
            preconditions: Vec::new(),
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

                for (source, sink) in instruction.flows() {
                    let sink_node = AbstractLocation::new(addr, exp.state.trace.clone(), sink);
                    let sink_index = self.insert_node(sink_node);

                    // If source and sink are basically the same, there is only a
                    // stay-flow to be linked.
                    if source.normalized() == sink.normalized() {
                        self.link_location(&mut exp, sink, sink_index, false);
                        continue;
                    }

                    let source_node = AbstractLocation::new(addr, exp.state.trace.clone(), source);
                    let source_index = self.insert_node(source_node);

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
                        exp.write_accesses.push((sink_index, access, exp.preconditions.len()));
                    }

                    // For reading memory accesses we need to check if they alias
                    // with any of the previous writing accesses.
                    if let Some(access) = exp.state.get_access_for_location(source) {
                        self.handle_read_access(&exp, access, source_index);
                    }
                }

                exp.state.track(&instruction, addr);

                // Execute the instruction.
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
                let condition = &self.graph.edges[&(exp.target, id)];

                // If the arrow to the next basic block has a condition, we
                // have an additional precondition which we AND with the existing one.
                let evaluated = exp.state.evaluate_condition(condition);

                let mut preconditions = exp.preconditions.clone();
                preconditions.push(evaluated);

                targets.push(ExplorationTarget {
                    target: id,
                    state: exp.state.clone(),
                    preconditions,
                    location_links: exp.location_links.clone(),
                    write_accesses: exp.write_accesses.clone(),
                });
            }
        }

        self.finish()
    }

    /// Arrange all data in the way expected for the flow graph.
    fn finish(self) -> ValueFlowGraph {
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
                StdioKind::Stdin => exp.write_accesses.push((index, access, exp.preconditions.len())),
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
        location_index: usize
    ) {
        for (prev_index, prev, num_preconditions) in exp.write_accesses.iter().rev() {
            let edge = (*prev_index, location_index);

            let (mut condition, perfect_match) =
                determine_aliasing_condition(&self.solver, prev, &access);

            // Any condition that has to be met on this path *additionally* to those
            // already active for the current write have to be included in the condition.
            for pre in &exp.preconditions[*num_preconditions..] {
                condition = condition.and(pre.clone());
            }

            // There may already be another condition for this edge because we reached
            // it through a different path. If so, both are valid and we have
            // to take the disjunction of them.
            if let Some(prev) = self.edges.remove(&edge) {
                condition = prev.or(condition);
            }

            condition = self.solver.simplify_condition(&condition);

            if condition != SymCondition::FALSE {
                self.edges.insert(edge, condition);
            }

            // If the pointers alias in *any* case and both accesses are the same
            // width this write would have definitely overwritten any previous one
            // and we can quit this loop safely.
            if perfect_match {
                break;
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
        // We only consider direct registers here (except RIP) and not memory storage slots
        // because eventhough they may have the same "location", i.e. the
        // same address formula based on registers the actual address may
        // differ depending on the register values.
        if location.accesses_memory() || location == StorageLocation::Direct(Register::RIP) {
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

    #[test] fn value_bufs() { test("bufs") }
    #[test] fn value_paths() { test("paths") }
    #[test] fn value_deep() { test("deep") }
}
