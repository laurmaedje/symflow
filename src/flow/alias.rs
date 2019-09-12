//! Alias analysis.

use std::collections::{HashMap, HashSet};
use std::fmt::{self, Display, Formatter};
use std::rc::Rc;

use crate::x86_64::Mnemoic;
use crate::math::{Integer, DataType, SymExpr, SymCondition, SharedSolver, Solver, Traversed};
use crate::sym::{SymState, MemoryStrategy, SymbolMap, TypedMemoryAccess};
use super::{ControlFlowGraph, AbstractLocation, StorageLocation};
use DataType::*;


/// Maps all abstract locations to the conditions under which there is a
/// memory access happening at them aliasing with the main access.
#[derive(Debug, Clone)]
pub struct AliasMap {
    pub map: HashMap<AbstractLocation, (SymCondition, SymbolMap)>,
}

impl AliasMap {
    /// Create a new alias map for a target abstract location.
    pub fn new(graph: &ControlFlowGraph, target: &AbstractLocation) -> AliasMap {
        AliasExplorer::new(graph, target).run()
    }
}

/// Analyzes the value flow from a target and builds a map of conditions.
struct AliasExplorer<'g> {
    graph: &'g ControlFlowGraph,
    target: &'g AbstractLocation,
    map: HashMap<AbstractLocation, (SymCondition, SymbolMap)>,
    solver: SharedSolver,
}

impl<'g> AliasExplorer<'g> {
    /// Create a new alias explorer.
    fn new(graph: &'g ControlFlowGraph, target: &'g AbstractLocation) -> AliasExplorer<'g> {
        AliasExplorer {
            graph,
            target,
            map: HashMap::new(),
            solver: Rc::new(Solver::new()),
        }
    }

    /// Build an alias flow map for the target.
    pub fn run(mut self) -> AliasMap {
        // Find all nodes that can be backwards reached from first or second to
        // narrow down the search.
        let relevant = self.find_reachable_nodes();

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

            // Simulate a basic block.
            for (addr, len, instruction, microcode) in &block.code {
                let addr = *addr;
                let next_addr = addr + len;

                state.track(&instruction, addr);

                // Check for the target access.
                if addr == self.target.addr && state.trace == self.target.trace {
                    let access = state.get_access_for_location(self.target.storage)
                        .expect("expected memory access at target");
                    target_mem = Some(access);

                // Check for another access.
                } else {
                    use Mnemoic::*;
                    if [Mov, Movzx, Movsx].contains(&instruction.mnemoic) {
                        let source = instruction.operands[1];

                        if let Some(storage) = StorageLocation::from_operand(source) {
                            if let Some(access) = state.get_access_for_location(storage) {
                                // There is a memory access.
                                let location = AbstractLocation {
                                    addr,
                                    trace: state.trace.clone(),
                                    storage,
                                };

                                self.handle_access(
                                    target_mem.as_ref(),
                                    access,
                                    location,
                                    &pre,
                                    &state
                                );
                            }
                        }
                    }
                }

                // Execute the microcode for the instruction.
                for op in &microcode.ops {
                    state.step(next_addr, op);
                }
            }

            // Add all nodes reachable from that one as targets.
            for &id in &self.graph.outgoing[target] {
                if relevant.contains(&id) {
                    let condition = &self.graph.edges[&(target, id)];
                    let evaluated = state.evaluate_condition(condition);
                    let new_pre = self.solver.simplify_condition(&pre.clone().and(evaluated));
                    targets.push((id, new_pre, state.clone(), target_mem.clone()));
                }
            }
        }

        AliasMap { map: self.map }
    }

    /// Compute the condition for a memory access and add it to the map.
    fn handle_access(
        &mut self,
        target: Option<&TypedMemoryAccess>,
        access: TypedMemoryAccess,
        location: AbstractLocation,
        pre: &SymCondition,
        state: &SymState,
    ) {
        let aliasing = if let Some(target_access) = target {
            determine_aliasing_condition(&self.solver, target_access, &access)
        } else {
            SymCondition::FALSE
        };

        let local_condition = pre.clone().and(aliasing);

        let condition = self.solver.simplify_condition(&match self.map.remove(&location) {
            Some((previous, _)) => previous.or(local_condition),
            None => local_condition,
        });

        let mut symbols = HashMap::new();
        condition.traverse(&mut |node| {
            if let Traversed::Expr(&SymExpr::Sym(symbol)) = node {
                if let Some(loc) = state.symbol_map.get(&symbol) {
                    symbols.insert(symbol, loc.clone());
                }
            }
        });

        self.map.insert(location, (condition, symbols));
    }

    /// Find all reachable nodes for this flow analysis by walking through the
    /// flow graph both forwards and backwards from the target node.
    fn find_reachable_nodes(&self) -> HashSet<usize> {
        let targets = self.get_containing_nodes(self.target);

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
    fn get_containing_nodes(&self, location: &AbstractLocation) -> Vec<usize> {
        let mut nodes = Vec::new();
        for (index, node) in self.graph.nodes.iter().enumerate() {
            let block = &self.graph.blocks[&node.addr];
            if location.addr >= block.addr && location.addr <= block.addr + block.len
               && location.trace.iter().eq(node.trace.iter().map(|(callsite, _)| callsite)) {
                nodes.push(index);
            }
        }
        nodes
    }
}

/// Returns the condition under which `a` and `b` point to overlapping memory.
pub fn determine_aliasing_condition(
    solver: &Solver,
    a: &TypedMemoryAccess,
    b: &TypedMemoryAccess
) -> SymCondition {
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

    solver.simplify_condition(&condition)
}

impl Display for AliasMap {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "AliasMap [")?;
        if !self.map.is_empty() { writeln!(f)?; }
        let mut map: Vec<_> = self.map.iter().collect();
        map.sort_by_key(|pair| pair.0);
        for (location, (condition, symbols)) in map {
            writeln!(f, "    {}: {}", location, condition)?;
            if !symbols.is_empty() {
                let mut symbols: Vec<_> = symbols.iter().collect();
                symbols.sort_by_key(|pair| pair.0);
                writeln!(f, "    where ")?;
                for (symbol, location) in symbols {
                    writeln!(f, "        {} is {}", symbol, location)?;
                }
            }
        }
        writeln!(f, "]")
    }
}


#[cfg(test)]
mod tests {
    use std::fs::{self, File};
    use std::io::Write;
    use crate::Program;
    use crate::x86_64::Register;
    use crate::math::Symbol;
    use crate::flow::StorageLocation;
    use super::*;

    fn test(filename: &str, location: AbstractLocation) -> AliasMap {
        let path = format!("target/bin/{}", filename);

        let program = Program::new(path);
        let graph = ControlFlowGraph::new(&program);
        let map = AliasMap::new(&graph, &location);

        fs::create_dir("target/value-flow").ok();
        let alias_path = format!("target/value-flow/alias-{}.txt", filename);
        let mut alias_file = File::create(alias_path).unwrap();
        write!(alias_file, "{}", map).unwrap();

        map
    }

    #[test]
    fn bufs() {
        let alias_map = test("bufs", AbstractLocation {
            addr: 0x39d,
            trace: vec![0x2ba],
            storage: StorageLocation::indirect_reg(N8, Register::RDX),
        });

        let secret_flow_condition = &alias_map.map[&AbstractLocation {
            addr: 0x3aa,
            trace: vec![0x2ba],
            storage: StorageLocation::indirect_reg(N8, Register::RAX),
        }].0;

        // Ascii 'z' is 122 and ':' is 58, so the difference is exactly 64.
        // This is the only difference that should satisfy the flow condition.
        assert!(secret_flow_condition.evaluate(&|symbol| match symbol {
            Symbol(N8, "stdin", 0) => Some(Integer(N8, 'z' as u64)),
            Symbol(N8, "stdin", 1) => Some(Integer(N8, ':' as u64)),
            _ => None,
        }));

        assert!(!secret_flow_condition.evaluate(&|symbol| match symbol {
            Symbol(N8, "stdin", 0) => Some(Integer(N8, 'a' as u64)),
            Symbol(N8, "stdin", 1) => Some(Integer(N8, 'p' as u64)),
            _ => None,
        }));
    }

    #[test]
    fn paths() {
        test("paths", AbstractLocation {
            addr: 0x363,
            trace: vec![0x2ba],
            storage: StorageLocation::Indirect {
                data_type: N8,
                base: Register::RBP,
                scaled_offset: Some((Register::RAX, 1)),
                displacement: Some(-0x410),
            },
        });
    }

    #[test]
    fn deep() {
        test("deep", AbstractLocation {
            addr: 0x399,
            trace: vec![0x2ba],
            storage: StorageLocation::indirect_reg(N8, Register::RDX),
        });
    }
}
