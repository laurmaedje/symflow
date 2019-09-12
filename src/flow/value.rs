//! Value flow analysis.

use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use std::fmt::{self, Display, Formatter};
use std::rc::Rc;

use crate::Program;
use crate::x86_64::Mnemoic;
use crate::math::{Integer, DataType, SymExpr, SymCondition, SharedSolver, Solver, Traversed};
use crate::sym::{SymState, MemoryStrategy, SymbolMap, TypedMemoryAccess};
use super::{ControlFlowGraph, AbstractLocation, StorageLocation};
use DataType::*;


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
        program: &Program,
        title: &str,
    ) -> io::Result<()> {
        use super::visualize::*;
        let mut f = target;

        write_header(&mut f, &format!("Value flow graph for {}", title))?;

        // Export the blocks.
        for (index, node) in self.nodes.iter().enumerate() {
            write!(f, "b{} [label=<{}>]", index, node)?;
        }

        write_edges(&mut f, &self.edges)?;
        write_footer(&mut f)
    }
}

/// Analyzes the value flow in the whole executable, building a value flow graph.
struct ValueFlowExplorer<'g> {
    graph: &'g ControlFlowGraph,
    solver: SharedSolver,
}

impl<'g> ValueFlowExplorer<'g> {
    /// Create a new value flow explorer.
    fn new(graph: &'g ControlFlowGraph) -> ValueFlowExplorer<'g> {
        ValueFlowExplorer {
            graph,
            solver: Rc::new(Solver::new()),
        }
    }

    /// Build the value flow graph.
    pub fn run(mut self) -> ValueFlowGraph {
        unimplemented!()
    }
}


#[cfg(test)]
mod tests {
    use std::fs::{self, File};
    use std::process::Command;
    use crate::Program;
    use crate::x86_64::Register;
    use crate::math::Symbol;
    use crate::flow::visualize::test::compile;
    use crate::flow::{StorageLocation};
    use super::*;

    fn test(filename: &str) {
        let path = format!("target/bin/{}", filename);

        // Generate the flow graph.
        let program = Program::new(path);
        let control_graph = ControlFlowGraph::new(&program);
        let value_graph = ValueFlowGraph::new(&control_graph);

        compile(&program, "target/value-flow", filename, |file| {
            value_graph.visualize(file, &program, filename)
        });
    }

    #[test]
    fn value_flow_graph() {
        test("bufs");
        test("paths");
        test("deep");
    }
}
