use std::collections::HashSet;

use crate::Program;
use crate::flow::{FlowGraph, FlowNode};
use crate::sym::{SymState, SymExpr, Event};


/// Check whether the data accesses at `write` and `read` operate on
/// overlapping data.
pub fn data_check(graph: &FlowGraph, first: u64, second: u64) -> Formula {
    let relevant = find_relevant_nodes(graph, first, second);

    let mut targets = vec![(0, SymState::new(), None, None)];

    'explore: while let Some((target, mut state, mut first_addr, mut second_addr)) = targets.pop() {
        let node = &graph.nodes[target];
        let block = &graph.blocks[&node.ctx.addr];

        for (addr, len, instruction, microcode) in &block.code {
            let addr = *addr;
            let next_addr = addr + len;
            let mut last_event = None;

            for &op in &microcode.ops {
                if let Some(event) = state.step(next_addr, op) {
                    last_event = Some(event);
                }
            }

            if addr == first { first_addr = extract_access_from_event(last_event); }
            else if addr == second { second_addr = extract_access_from_event(last_event); }

            if first_addr.is_some() && second_addr.is_some() {
                println!("Halting at address {:#x}", addr);
                println!("first:  {}", first_addr.unwrap());
                println!("second: {}", second_addr.unwrap());
                println!("Finished this path!");
                println!();
                continue 'explore;
            }
        }

        for &id in &node.out {
            if relevant.contains(&id) {
                targets.push((id, state.clone(), first_addr.clone(), second_addr.clone()));
            }
        }
    }

    Formula {}
}

/// Extract the address of the memory access if there was any.
fn extract_access_from_event(event: Option<Event>) -> Option<SymExpr> {
    match event {
        Some(Event::MemoryAccess { address, .. }) => Some(address),
        _ => None,
    }
}

/// Find all relevant nodes for this flow analysis.
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

/// Get all nodes which blocks contain the given address.
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

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Formula {}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_slice() {
        let program = Program::new("target/bufs");
        let graph = FlowGraph::new(&program);
        let condition = data_check(&graph, 0x38c, 0x39a);
        println!("Condition for overlapping data: {:?}", condition);
    }
}
