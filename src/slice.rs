//! Data slice analyses.

use std::collections::HashSet;
use z3::ast::Ast;

use crate::Program;
use crate::flow::{FlowGraph, FlowNode};
use crate::sym::{SymState, Event};
use crate::expr::{SymExpr, SymCondition};


/// Check whether the data accesses at `first_addr` and `second_addr`
/// operate on overlapping data.
pub fn has_data_flow(graph: &FlowGraph, first_addr: u64, second_addr: u64) -> SymCondition {
    // Find all nodes that can be backwards reached from first or second to
    // narrow down the search.
    let relevant = find_relevant_nodes(graph, first_addr, second_addr);

    // Start the simulation at the entry node (index 0) if it is even reachable and exists.
    let mut targets = vec![];
    if relevant.contains(&0) {
        targets.push((0, SymState::new(), None, None));
    }

    // Explore the relevant part of the flow graph in search of the memory accesses.
    while let Some((target, mut state, mut first_mem, mut second_mem)) = targets.pop() {
        let node = &graph.nodes[target];
        let block = &graph.blocks[&node.ctx.addr];

        // Simulate a basic block.
        for (addr, len, instruction, microcode) in &block.code {
            let addr = *addr;
            let next_addr = addr + len;
            let mut last_event = None;

            // Execute an instruction while keeping the last event around
            // in case it is a memory access that interests us.
            for &op in &microcode.ops {
                if let Some(event) = state.step(next_addr, op) {
                    last_event = Some(event);
                }
            }

            if addr == first_addr { first_mem = extract_access_from_event(last_event); }
            else if addr == second_addr { second_mem = extract_access_from_event(last_event); }

            // If both memory accesses are found, we are done here.
            if first_mem.is_some() && second_mem.is_some() {
                let config = z3::Config::new();
                let ctx = z3::Context::new(&config);

                // The byte data accesses are overlapping if the first address
                // equals the second one.
                let condition = first_mem.unwrap().equal(second_mem.unwrap());
                let z3_condition = condition.to_z3_ast(&ctx);

                // Use z3 to simplify the condition.
                let z3_simplified = z3_condition.simplify();
                let simplified_condition = SymCondition::from_z3_ast(&z3_simplified)
                    .unwrap_or(condition);

                // Look if the condition is even satisfiable.
                let solver = z3::Solver::new(&ctx);
                solver.assert(&z3_simplified);
                let sat = solver.check();

                return if sat {
                    simplified_condition
                } else {
                    SymCondition::Bool(false)
                }
            }
        }

        // Add all nodes reachable from that one as targets.
        for &id in &node.out {
            if relevant.contains(&id) {
                targets.push((id, state.clone(), first_mem.clone(), second_mem.clone()));
            }
        }
    }

    SymCondition::Bool(false)
}

/// Extract the address of the memory access if there was any.
fn extract_access_from_event(event: Option<Event>) -> Option<SymExpr> {
    match event {
        Some(Event::MemoryAccess { address, .. }) => Some(address),
        _ => None,
    }
}

/// Find all relevant nodes for this flow analysis by walking through the
/// flow graph backwards from the two target nodes.
fn find_relevant_nodes(graph: &FlowGraph, first: u64, second: u64) -> HashSet<usize> {
    let mut relevant = HashSet::new();

    let mut targets = get_containing_nodes(graph, first);
    targets.extend(get_containing_nodes(graph, second));

    while let Some(target) = targets.pop() {
        if !relevant.contains(&target) {
            // Explore all blocks going here.
            targets.extend(&graph.nodes[target].inc);
            relevant.insert(target);
        }
    }

    relevant
}

/// Get all nodes whose blocks contain the given address.
fn get_containing_nodes(graph: &FlowGraph, addr: u64) -> Vec<usize> {
    let mut nodes = Vec::new();
    for (index, node) in graph.nodes.iter().enumerate() {
        let block = &graph.blocks[&node.ctx.addr];
        if addr >= block.addr && addr <= block.addr + block.len {
            nodes.push(index);
        }
    }
    nodes
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_flow() {
        let program = Program::new("target/bufs");
        let graph = FlowGraph::new(&program);
        let condition = has_data_flow(&graph, 0x38c, 0x39a);
        println!("There is data flow if {}", condition);
    }
}
