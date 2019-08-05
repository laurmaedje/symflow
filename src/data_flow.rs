//! Data flow analysis.

use std::collections::{HashMap, HashSet};
use std::fmt::{self, Display, Formatter};
use z3::ast::Ast;

use crate::ir::Location;
use crate::expr::{SymExpr, SymCondition, Symbol};
use crate::sym::SymState;
use crate::control_flow::FlowGraph;


/// Analyzes the data flow between targets and builds a map of conditions.
#[derive(Debug)]
pub struct DataflowExplorer<'g> {
    graph: &'g FlowGraph,
}

impl<'g> DataflowExplorer<'g> {
    /// Create a new data flow explorer.
    pub fn new(graph: &'g FlowGraph) -> DataflowExplorer<'g> {
        DataflowExplorer { graph }
    }

    /// Build a data flow map for the `target`. The function `get_access_addr`
    /// takes an address and a symbolic state and returns the address of a memory
    /// access of interest happening there if there is any.
    pub fn find_direct_data_flow<F>(&mut self, target_addr: u64, get_access_addr: F) -> DataFlowMap
    where F: Fn(u64, &SymState) -> Option<SymExpr> {

        // Find all nodes that can be backwards reached from first or second to
        // narrow down the search.
        let relevant = self.find_reachable_nodes(target_addr);
        let mut map = HashMap::new();

        // Start the simulation at the entry node (index 0).
        let mut targets = vec![(0, SymState::new(), None)];

        // Explore the relevant part of the flow graph in search of the memory accesses.
        while let Some((target, mut state, mut target_mem)) = targets.pop() {
            let node = &self.graph.nodes[target];
            let block = &self.graph.blocks[&node.addr];

            // Simulate a basic block.
            for (addr, len, _instruction, microcode) in &block.code {
                let addr = *addr;
                let next_addr = addr + len;

                let access = get_access_addr(addr, &state);

                if addr == target_addr {
                    let access_mem = access
                        .expect("find_direct_data_flow: expected access at target address");
                    target_mem = Some(access_mem);
                } else {
                    if let Some(access_mem) = access {
                        let condition = if let Some(target_mem) = target_mem.as_ref() {
                            self.alias_condition(target_mem, &access_mem)
                        } else {
                            SymCondition::Bool(false)
                        };
                        map.insert(addr, condition);
                    }
                }

                // Execute the microcode for the instruction.
                for &op in &microcode.ops {
                    state.step(next_addr, op);
                }
            }

            // Add all nodes reachable from that one as targets.
            for &id in &self.graph.outgoing[target] {
                if relevant.contains(&id) {
                    targets.push((id, state.clone(), target_mem.clone()));
                }
            }
        }

        DataFlowMap { map }
    }

    /// Returns the condition under which `a` and `b` point to overlapping memory.
    fn alias_condition(&self, a: &SymExpr, b: &SymExpr) -> SymCondition {
        let config = z3::Config::new();
        let ctx = z3::Context::new(&config);

        // The byte data accesses are overlapping if the first address
        // equals the second one.
        let condition = a.clone().equal(b.clone());
        let z3_condition = condition.to_z3_ast(&ctx);

        // Use Z3 to simplify the condition.
        let z3_simplified = z3_condition.simplify();
        let simplified_condition = SymCondition::from_z3_ast(&z3_simplified)
            .unwrap_or(condition);

        // Look if the condition is even satisfiable.
        let solver = z3::Solver::new(&ctx);
        solver.assert(&z3_simplified);
        let sat = solver.check();

        if sat {
            simplified_condition
        } else {
            SymCondition::Bool(false)
        }
    }

    /// Find all reachable nodes for this flow analysis by walking through the
    /// flow graph both forwards and backwards from the target node.
    fn find_reachable_nodes(&self, target_addr: u64) -> HashSet<usize> {
        let targets = self.get_containing_nodes(target_addr);

        // Explore all blocks reaschable from here and then all blocks going here.
        let forwards_reachable = self.find_directional_reachable(targets.clone(), true);
        let backwards_reachable = self.find_directional_reachable(targets, false);

        // Join the sets.
        let mut reachable = forwards_reachable;
        reachable.extend(&backwards_reachable);
        reachable
    }

    fn find_directional_reachable(&self, mut targets: Vec<usize>, forward: bool) -> HashSet<usize> {
        let mut reachable = HashSet::new();
        while let Some(target) = targets.pop() {
            if !reachable.contains(&target) {
                targets.extend(if forward {
                    &self.graph.outgoing[target]
                } else {
                    &self.graph.incoming[target]
                });
                reachable.insert(target);
            }
        }
        reachable
    }

    /// Get all nodes whose blocks contain the given address.
    fn get_containing_nodes(&self, addr: u64) -> Vec<usize> {
        let mut nodes = Vec::new();
        for (index, node) in self.graph.nodes.iter().enumerate() {
            let block = &self.graph.blocks[&node.addr];
            if addr >= block.addr && addr <= block.addr + block.len {
                nodes.push(index);
            }
        }
        nodes
    }
}

/// Maps all data flow targets to the conditions under which data flows from the main
/// target into them.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DataFlowMap {
    pub map: HashMap<u64, SymCondition>,
}

impl Display for DataFlowMap {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "DataFlowMap [")?;
        if !self.map.is_empty() { writeln!(f)?; }
        let mut map: Vec<_> = self.map.iter().collect();
        map.sort_by_key(|pair| pair.0);
        for (addr, condition) in map {
            writeln!(f, "    {:x}: {}", addr, condition)?;
        }
        writeln!(f, "]")
    }
}


#[cfg(test)]
mod tests {
    use crate::Program;
    use crate::x86_64::{Mnemoic, Register};
    use super::*;

    #[test]
    fn data_flow() {
        let program = Program::new("target/bin/bufs");
        let graph = FlowGraph::new(&program);

        let mut explorer = DataflowExplorer::new(&graph);

        const SECRET_WRITE_ADDR: u64 = 0x3b8;
        let flow_map = explorer.find_direct_data_flow(SECRET_WRITE_ADDR, |addr, state| {
            // This function returns the address of the byte access of interest at the given
            // address if there is one.
            let inst = program.get_instruction(addr).unwrap();

            if addr == SECRET_WRITE_ADDR {
                // If this is the criterion secret write, we know which instruction it
                // is and thus how to find out the address (as it's in rdx).
                assert_eq!(inst.to_string(), "mov byte ptr [rdx], al");

                let target_addr = state.get_reg(Register::RDX);
                Some(target_addr)

            } else {
                // We consider any move from somewhere as possibly moving our sensible data.
                use Mnemoic::*;
                if [Mov, Movzx, Movsx].contains(&inst.mnemoic) {
                    let source = inst.operands[1];
                    state.get_addr_for_operand(source)
                } else {
                    None
                }
            }
        });

        println!("Data flow: {}", flow_map);
    }
}
