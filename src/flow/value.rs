//! Value flow analysis.

use std::collections::HashMap;
use std::io::{self, Write};
use std::rc::Rc;

use crate::x86_64::Register;
use crate::math::{SymCondition, Integer, Symbol, SharedSolver, Solver};
use crate::sym::{SymState, Event, MemoryStrategy, TypedMemoryAccess, SymbolMap, StdioKind};
use super::*;


/// Contains the conditions under which values flow between abstract locations.
#[derive(Debug, Clone)]
pub struct ValueFlowGraph {
    /// The nodes in the graph, i.e. all abstract locations in the executable.
    pub nodes: Vec<ValueFlowNode>,
    /// The conditions for value flow between the abstract locations.
    /// The key pairs are indices into the `nodes` vector.
    pub edges: HashMap<(usize, usize), (SymCondition, SymbolMap)>,
}

/// A node in the value flow graph, describing some kind of value.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum ValueFlowNode {
    Location(AbstractLocation),
    Io(StdioKind, Symbol),
    Constant(usize, Integer),
}

impl ValueFlowGraph {
    /// Create a new value flow graph for all abstract locations.
    pub fn new(graph: &ControlFlowGraph) -> ValueFlowGraph {
        crate::timings::with("value-flow-graph", || ValueFlowExplorer::new(graph).run())
    }

    /// Visualize this flow graph in a graphviz DOT file.
    pub fn visualize<W: Write>(
        &self,
        target: W,
        title: &str,
    ) -> io::Result<()> {
        use super::visualize::*;
        let mut f = target;

        write_header(&mut f, &format!("Value flow graph for {}", title), 40)?;

        for (index, node) in self.nodes.iter().enumerate() {
            match node {
                ValueFlowNode::Location(location) => {
                    let fmt = location.to_string().replace(">", "&gt;");
                    let mut splitter = fmt.splitn(2, ' ');
                    writeln!(f, "b{} [label=<<b>{}</b> {}>,shape=box]", index,
                                splitter.next().unwrap(), splitter.next().unwrap())?;
                },

                ValueFlowNode::Io(kind, symbol) => {
                    let color = match kind {
                        StdioKind::Stdin => "#4caf50",
                        StdioKind::Stdout => "#03a9f4",
                    };

                    writeln!(f, "b{} [label=<<b>{}</b>>,shape=box,style=filled,fillcolor=\"{}\"]",
                                index, symbol, color)?;
                },

                ValueFlowNode::Constant(_, int) => {
                    writeln!(f, "b{} [label=<{}>,shape=box,style=filled,fillcolor=\"#f0ce24\"]",
                         index, int)?;
                },
            }
        }

        write_edges(&mut f, &self.edges, |f, ((start, end), (condition, _))| {
            if condition != &SymCondition::TRUE {
                write!(f, "label=< ")?;
                let fmt = condition.to_string().replace("<", "&lt;").replace(">", "&gt;");
                let mut len = 0;
                for part in fmt.split(" ") {
                    write!(f, "{} ", part)?;
                    len += part.len();
                    if len > 40 {
                        write!(f, "<br/>")?;
                        len = 0;
                    }
                }
                write!(f, ">, ")?;
            }

            if let ValueFlowNode::Location(first) = &self.nodes[start] {
                if let ValueFlowNode::Location(second) = &self.nodes[end] {
                    if !first.storage.accesses_memory() &&
                        first.storage.normalized() == second.storage.normalized() {
                        write!(f, "style=dashed, color=\"#ababab\"")?;
                    }
                }
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
    nodes: HashMap<ValueFlowNode, usize>,
    edges: HashMap<(usize, usize), (SymCondition, SymbolMap)>,
}

#[derive(Clone)]
struct ExplorationTarget {
    target: usize,
    state: SymState,

    /// The set of all conditions (through ifs) met on this path.
    preconditions: Vec<SymCondition>,

    /// For each storage location that was already used we store the abstract
    /// location node were it was written, so we can add an edge back to it
    /// when we use this storage. The latter usize holds the number of preconditions
    /// we with the write accesses.
    location_links: HashMap<StorageLocation, (usize, usize)>,

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
            let node = &self.graph.nodes.get(exp.target)
                .expect("value flow explorer: expected node in control flow graph");

            let block = &self.graph.blocks[&node.addr];

            // Simulate a basic block.
            for (addr, len, instruction, microcode) in &block.code {
                let addr = *addr;
                let next_addr = addr + len;

                for (source, sink) in instruction.flows() {
                    let sink_index = self.insert_loc(addr, &exp.state.trace, sink);

                    // The source may be a constant or a storage location.
                    let source_data = match source {
                        ValueSource::Storage(source) => {
                            // If source and sink are the same, we have to add a backlink
                            // because the actual source is the previous storage location,
                            // but we do not add an edge between source and sink (that would
                            // be a reflexive edge).
                            if source.normalized() == sink.normalized() {
                                self.link_location(&mut exp, sink, sink_index, false);

                                Some((sink, sink_index))

                            } else {
                                let source_index = self.insert_loc(addr, &exp.state.trace, source);
                                self.link_location(&mut exp, source, source_index, false);

                                // For flows inherent to an instruction the condition is
                                // obviously always true.
                                self.insert_true_edge(source_index, sink_index);

                                Some((source, source_index))
                            }
                        },

                        ValueSource::Const(int) => {
                            let index = self.insert_node(ValueFlowNode::Constant(sink_index, int));
                            self.insert_pre_edge(&exp, 0, index, sink_index);

                            None
                        }
                    };

                    // Connect the abstract location to their previous abstract
                    // location with the same storage location.
                    self.link_location(&mut exp, sink, sink_index, true);

                    if let Some((source, source_index)) = source_data {
                        // For reading memory accesses we need to check if they alias
                        // with any of the previous writing accesses.
                        if let Some(access) = exp.state.get_access_for_storage(source) {
                            self.handle_read_access(&exp, access, source_index);
                        }
                    }

                    // Writing memory accesses are stored in the `write_accesses` list
                    // so we can check aliasing with reading accesses later on.
                    if let Some(access) = exp.state.get_access_for_storage(sink) {
                        exp.write_accesses.push((sink_index, access, exp.preconditions.len()));
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
        // Arrange the nodes into a vector.
        let default = ValueFlowNode::Constant(0, Integer::from_ptr(0));
        let mut nodes = vec![default; self.nodes.len()];
        for (node, index) in self.nodes.into_iter() {
            nodes[index] = node;
        }

        ValueFlowGraph {
            nodes,
            edges: self.edges,
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
            let location_index = self.insert_node(ValueFlowNode::Location(location.clone()));
            let index = self.insert_node(ValueFlowNode::Io(kind, symbol));

            // Store the location node so it can be backlinked.
            exp.location_links.insert(
                location.storage.normalized(),
                (location_index, exp.preconditions.len())
            );

            // If it is a stdin read, that is, a memory write, add it
            // to the write access list.
            match kind {
                StdioKind::Stdin => {
                    exp.write_accesses.push((location_index, access, exp.preconditions.len()));
                    self.insert_pre_edge(&exp, 0, index, location_index);
                },
                StdioKind::Stdout => {
                    self.handle_read_access(exp, access, location_index);
                    self.insert_pre_edge(&exp, 0, location_index, index);
                },
            }
        }
    }

    /// Check all previous write accesses for aliasing with the current read access and add
    /// conditional edges in between if necessary.
    fn handle_read_access(
        &mut self,
        exp: &ExplorationTarget,
        read: TypedMemoryAccess,
        location_index: usize
    ) {
        let mut overwritten = SymCondition::FALSE;

        for (prev_index, prev, num_preconditions) in exp.write_accesses.iter().rev() {
            let mut alias = determine_alias(prev, &read);

            // Any condition that has to be met on this path *additionally* to those
            // already active for the current write have to be included in the conditions.
            for pre in &exp.preconditions[*num_preconditions..] {
                alias = alias.and(pre.clone());
            }

            // We have to make sure a later write has not overwritten this one.
            alias = alias.and(overwritten.clone().not());

            crate::timings::with("simplify-alias", || {
                alias = self.solver.simplify_condition(&alias);
            });

            // If this cannot match with the neccessary preconditions, we can skip this, too.
            if alias == SymCondition::FALSE {
                continue;
            }

            self.insert_edge(exp, *prev_index, location_index, alias);

            // Determine the condition that the read access was overwritten by the
            // write one.
            let full = determine_full_alias(prev, &read);
            overwritten = overwritten.or(full);

            crate::timings::with("simplify-overwrite", || {
                overwritten = self.solver.simplify_condition(&overwritten);
            });

            // If this was surely overwritten, we don't have to bother with
            // earlier things.
            if overwritten == SymCondition::TRUE {
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
            if let Some((prev, num_preconditions)) = exp.location_links.get(&location) {
                self.insert_pre_edge(&exp, *num_preconditions, *prev, location_index);
            }
        }

        exp.location_links.insert(location, (location_index, exp.preconditions.len()));
    }

    /// Insert a new abstract location node for a storage location in a context.
    fn insert_loc(&mut self, addr: u64, trace: &[u64], storage: StorageLocation) -> usize {
        let location = AbstractLocation::new(addr, trace.to_vec(), storage);
        let node = ValueFlowNode::Location(location);
        self.insert_node(node)
    }

    /// Add a node to the list. This
    /// - inserts it and returns the new index if it didn't exist before
    /// - finds the existing node and returns it's index
    fn insert_node(&mut self, node: ValueFlowNode) -> usize {
        let new_index = self.nodes.len();
        *self.nodes.entry(node).or_insert(new_index)
    }

    /// Insert an edge with condition true overwriting any previous edge.
    fn insert_true_edge(&mut self, from: usize, to: usize) {
        self.edges.insert((from, to), (SymCondition::TRUE, SymbolMap::new()));
    }

    /// Insert an edge with the precondition of the exploration target.
    fn insert_pre_edge(
        &mut self,
        exp: &ExplorationTarget,
        num_preconditions: usize,
        from: usize,
        to: usize
    ) {
        let mut condition = SymCondition::TRUE;
        for pre in &exp.preconditions[num_preconditions..] {
            condition = condition.and(pre.clone());
        }

        self.insert_edge(exp, from, to, condition);
    }

    /// Insert an edge, combining it with an edge that may already be there.
    fn insert_edge(
        &mut self,
        exp: &ExplorationTarget,
        from: usize,
        to: usize,
        mut condition: SymCondition
    ) {
        let edge = (from, to);

        // There may already be another condition for this edge because we reached
        // it through a different path. If so, both are valid and we have
        // to take the disjunction of them.
        if let Some((prev, _)) = self.edges.remove(&edge) {
            crate::timings::start("simplify-combine");
            condition = self.solver.simplify_condition(&prev.or(condition));
            crate::timings::stop();
        }

        if condition != SymCondition::FALSE {
            let symbols = exp.state.get_symbol_map_for(&condition);
            self.edges.insert(edge, (condition, symbols));
        }
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

        compile("value", filename, |file| {
            value_graph.visualize(file, filename)
        });
    }

    #[test] fn value_bufs() { test("bufs") }
    #[test] fn value_paths() { test("paths") }
    #[test] fn value_deep() { test("deep") }
    #[test] fn value_overwrite() { test("overwrite") }
    #[test] fn value_min() { test("min") }
}
