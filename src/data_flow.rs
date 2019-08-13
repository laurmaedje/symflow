//! Data flow analysis.

use std::collections::{HashMap, HashSet};
use std::fmt::{self, Display, Formatter};
use std::rc::Rc;

use crate::num::{Integer, DataType};
use crate::expr::{SymExpr, SymCondition};
use crate::smt::{SharedSolver, Solver};
use crate::sym::{SymState, MemoryStrategy, SymbolMap, TypedMemoryAccess};
use crate::control_flow::FlowGraph;
use DataType::*;


/// Analyzes the data flow between targets and builds a map of conditions.
pub struct DataflowExplorer<'g> {
    graph: &'g FlowGraph,
    solver: SharedSolver,
}

impl<'g> DataflowExplorer<'g> {
    /// Create a new data flow explorer.
    pub fn new(graph: &'g FlowGraph) -> DataflowExplorer<'g> {
        DataflowExplorer {
            graph,
            solver: Rc::new(Solver::new()),
        }
    }

    /// Build a data flow map for the `target`. The function `get_access_addr`
    /// takes an address and a symbolic state and returns the address of a memory
    /// access of interest happening there if there is any.
    pub fn find_direct_data_flow<F>(&mut self, target_addr: u64, get_access_addr: F) -> DataFlowMap
    where F: Fn(u64, &SymState) -> Option<TypedMemoryAccess> {

        // Find all nodes that can be backwards reached from first or second to
        // narrow down the search.
        let relevant = self.find_reachable_nodes(target_addr);
        let mut map = HashMap::<u64, (SymCondition, SymbolMap)>::new();

        // Start the simulation
        // - at the entry node (index 0)
        // - with no pre conditions (these are generated at conditional jumps)
        // - a blank state
        // - no address yet for the target access
        let base_state = SymState::new(MemoryStrategy::ConditionalTrees, self.solver.clone());
        let mut targets = vec![(0, SymCondition::TRUE, base_state, None)];

        // Explore the relevant part of the flow graph in search of the memory accesses.
        while let Some((target, pre, mut state, mut target_mem)) = targets.pop() {
            let node = &self.graph.nodes[target];
            let block = &self.graph.blocks[&node.addr];

            state.trace(node.addr);

            // Simulate a basic block.
            for (addr, len, _instruction, microcode) in &block.code {
                let addr = *addr;
                let next_addr = addr + len;

                let access = get_access_addr(addr, &state);

                if addr == target_addr {
                    let access_mem = access.expect(
                        "find_direct_data_flow: expected access at target address");
                    target_mem = Some(access_mem);
                } else {
                    if let Some(access_mem) = access {
                        let aliasing = if let Some(target_mem) = target_mem.as_ref() {
                            self.determine_aliasing_condition(target_mem, &access_mem)
                        } else {
                            SymCondition::FALSE
                        };

                        let local_condition = pre.clone().and(aliasing);

                        let condition = self.solver.simplify_condition(match map.remove(&addr) {
                            Some((previous, _)) => previous.or(local_condition),
                            None => local_condition,
                        });

                        let symbols = state.symbol_map.iter()
                            .filter(|&(symbol, _)| condition.contains_symbol(*symbol))
                            .map(|(&sym, loc)| (sym, loc.clone()))
                            .collect();

                        map.insert(addr, (condition, symbols));
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
                    let (condition, value) = self.graph.edges[&(target, id)];
                    let mut evaluated = state.evaluate(condition);
                    if !value {
                        evaluated = evaluated.not();
                    }
                    let new_pre = self.solver.simplify_condition(pre.clone().and(evaluated));
                    targets.push((id, new_pre, state.clone(), target_mem.clone()));
                }
            }
        }

        DataFlowMap { map }
    }

    /// Returns the condition under which `a` and `b` point to overlapping memory.
    fn determine_aliasing_condition(&self, a: &TypedMemoryAccess, b: &TypedMemoryAccess)
    -> SymCondition {
        /// Return the condition under which `ptr` is in the area spanned by
        /// pointer of `area` and the following bytes based on the data type.
        fn contains_ptr(area: &TypedMemoryAccess, ptr: &SymExpr) -> SymCondition {
            let area_len = SymExpr::Int(Integer::from_ptr(area.1.bytes() as u64));
            let left = area.0.clone();
            ptr.clone().sub(left).less_than(area_len, false)
        }

        // The data accesses are overlapping if the area of the first one contains
        // the second one or the other way around, where the area is the memory
        // from the start of the pointer until the start + the byte width.
        let condition = match (a.1, b.1) {
            (N8, N8) => a.0.clone().equal(b.0.clone()),
            (N8, _) => contains_ptr(b, &a.0),
            (_, N8) => contains_ptr(a, &b.0),
            (_, _) => contains_ptr(a, &b.0).or(contains_ptr(b, &a.0)),
        };

        self.solver.simplify_condition(condition)
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

    /// Find nodes that are reachable either forwards or backwards.
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
#[derive(Debug, Clone)]
pub struct DataFlowMap {
    pub map: HashMap<u64, (SymCondition, SymbolMap)>,
}

impl Display for DataFlowMap {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "DataFlowMap [")?;
        if !self.map.is_empty() { writeln!(f)?; }
        let mut map: Vec<_> = self.map.iter().collect();
        map.sort_by_key(|pair| pair.0);
        for (addr, (condition, symbols)) in map {
            writeln!(f, "    {:x}: {}", addr, condition)?;
            if !symbols.is_empty() {
                let mut symbols: Vec<_> = symbols.iter().collect();
                symbols.sort_by_key(|pair| pair.0);
                writeln!(f, "         where ")?;
                for (symbol, location) in symbols {
                    writeln!(f, "             {} is {}", symbol, location)?;
                }
            }
        }
        writeln!(f, "]")
    }
}


#[cfg(test)]
mod tests {
    use crate::Program;
    use crate::x86_64::{Mnemoic, Register};
    use crate::expr::Symbol;
    use super::*;

    #[test]
    fn flow_map_bufs_1() {
        let program = Program::new("target/bin/bufs-1");
        let graph = FlowGraph::new(&program);

        const SECRET_WRITE_ADDR: u64 = 0x39d;

        let mut explorer = DataflowExplorer::new(&graph);
        let flow_map = explorer.find_direct_data_flow(SECRET_WRITE_ADDR, |addr, state| {
            get_access_addr(addr, state, &program,
                SECRET_WRITE_ADDR, "mov byte ptr [rdx], al",
                |state| TypedMemoryAccess(state.get_reg(Register::RDX), N8)
            )
        });

        println!("Data flow for bufs-1: {}", flow_map);

        let secret_flow_condition = &flow_map.map[&0x3aa].0;

        // Ascii 'z' is 122 and ':' is 58, so the difference is exactly 64.
        // This is the only difference that should satisfy the flow condition.
        assert!(secret_flow_condition.evaluate_with(|symbol| match symbol {
            Symbol(N8, "stdin", 0) => Some(Integer(N8, 'z' as u64)),
            Symbol(N8, "stdin", 1) => Some(Integer(N8, ':' as u64)),
            _ => None,
        }));

        assert!(!secret_flow_condition.evaluate_with(|symbol| match symbol {
            Symbol(N8, "stdin", 0) => Some(Integer(N8, 'a' as u64)),
            Symbol(N8, "stdin", 1) => Some(Integer(N8, 'p' as u64)),
            _ => None,
        }));
    }

    #[test]
    fn flow_map_bufs_2() {
        let program = Program::new("target/bin/bufs-2");
        let graph = FlowGraph::new(&program);

        const SECRET_WRITE_ADDR: u64 = 0x399;

        let mut explorer = DataflowExplorer::new(&graph);
        let flow_map = explorer.find_direct_data_flow(SECRET_WRITE_ADDR, |addr, state| {
            get_access_addr(addr, state, &program,
                SECRET_WRITE_ADDR, "mov byte ptr [rdx], al",
                |state| TypedMemoryAccess(state.get_reg(Register::RDX), N8)
            )
        });

        println!("Data flow for bufs-2: {}", flow_map);
    }

    #[test]
    fn flow_map_bufs_3() {
        let program = Program::new("target/bin/bufs-3");
        let graph = FlowGraph::new(&program);

        const SECRET_WRITE_ADDR: u64 = 0x352;

        let mut explorer = DataflowExplorer::new(&graph);
        let flow_map = explorer.find_direct_data_flow(SECRET_WRITE_ADDR, |addr, state| {
            get_access_addr(addr, state, &program,
                SECRET_WRITE_ADDR, "mov byte ptr [rbp+rax*1-0x410], 0x58",
                |state| TypedMemoryAccess(
                    state.get_reg(Register::RBP)
                        .add(state.get_reg(Register::RAX))
                        .sub(SymExpr::from_ptr(0x410)),
                    N8
                )
            )
        });

        println!("Data flow for bufs-3: {}", flow_map);
    }

    /// Generic `get_access_addr` implementation that is customizable for the
    /// specific example.
    fn get_access_addr<F>(
        addr: u64, state: &SymState, program: &Program,
        secret_write_addr: u64,
        expected_instruction: &str,
        target_addr: F
    ) -> Option<TypedMemoryAccess> where F: Fn(&SymState) -> TypedMemoryAccess {
        let inst = program.get_instruction(addr).unwrap();

        if addr == secret_write_addr {
            assert_eq!(inst.to_string(), expected_instruction);
            Some(target_addr(state))
        } else {
            use Mnemoic::*;
            if [Mov, Movzx, Movsx].contains(&inst.mnemoic) {
                let source = inst.operands[1];
                state.get_access_for_operand(source)
            } else {
                None
            }
        }
    }
}
