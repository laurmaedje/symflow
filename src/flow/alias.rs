//! Alias analysis.

use std::collections::{HashMap, HashSet};
use std::fmt::{self, Display, Formatter};
use std::rc::Rc;

use crate::math::{SymExpr, SymCondition, Integer, DataType, SharedSolver, Solver};
use crate::sym::{SymState, MemoryStrategy, SymbolMap, TypedMemoryAccess};
use super::{ControlFlowGraph, ValueSource, AbstractLocation};
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

#[derive(Clone)]
struct ExplorationTarget {
    target: usize,
    state: SymState,
    preconditions: Vec<SymCondition>,
    target_access: Option<(TypedMemoryAccess, usize)>,
}

impl<'g> AliasExplorer<'g> {
    fn new(graph: &'g ControlFlowGraph, target: &'g AbstractLocation) -> AliasExplorer<'g> {
        AliasExplorer {
            graph,
            target,
            map: HashMap::new(),
            solver: Rc::new(Solver::new()),
        }
    }

    /// Build an alias map for the target access.
    fn run(mut self) -> AliasMap {
        // Find all nodes that can be backwards reached from first or second to
        // narrow down the search.
        let relevant = self.find_reachable_nodes();

        // Start the simulation
        // - at the entry node (index 0)
        // - with no pre conditions (these are generated at conditional jumps)
        // - a blank state
        // - no address yet for the target access
        let base_state = SymState::new(MemoryStrategy::ConditionalTrees, self.solver.clone());

        let mut targets = vec![ExplorationTarget {
            target: 0,
            state: base_state,
            preconditions: Vec::new(),
            target_access: None,
        }];

        // Explore the relevant part of the flow graph in search of the memory accesses.
        while let Some(mut exp) = targets.pop() {
            let node = &self.graph.nodes[exp.target];
            let block = &self.graph.blocks[&node.addr];

            // Simulate a basic block.
            for (addr, len, instruction, microcode) in &block.code {
                let addr = *addr;
                let next_addr = addr + len;

                // Check for the target access.
                let is_target = addr == self.target.addr && exp.state.trace == self.target.trace;

                if is_target {
                    let access = exp.state
                        .get_access_for_storage(self.target.storage)
                        .expect("expected memory access at target");

                    exp.target_access = Some((access, exp.preconditions.len()));
                } else {
                    for (source, _) in instruction.flows() {
                        if let ValueSource::Storage(storage) = source {
                            if let Some(access) = exp.state.get_access_for_storage(storage) {
                                let trace = exp.state.trace.clone();
                                let location = AbstractLocation::new(addr, trace, storage);
                                self.handle_read_access(&exp, access, location);
                            }
                        }
                    }
                }

                exp.state.track(&instruction, addr);

                // Execute the microcode for the instruction.
                for op in &microcode.ops {
                    exp.state.step(next_addr, op);
                }
            }

            // Add all nodes reachable from that one as targets.
            for &id in &self.graph.outgoing[exp.target] {
                if relevant.contains(&id) {
                    let condition = &self.graph.edges[&(exp.target, id)];
                    let evaluated = exp.state.evaluate_condition(condition);

                    let mut preconditions = exp.preconditions.clone();
                    preconditions.push(evaluated);

                    targets.push(ExplorationTarget {
                        target: id,
                        state: exp.state.clone(),
                        preconditions,
                        target_access: exp.target_access.clone(),
                    });
                }
            }
        }

        AliasMap { map: self.map }
    }



    /// Compute the condition in which a memory access aliases with the target access
    /// and add this condition to the map.
    fn handle_read_access(
        &mut self,
        exp: &ExplorationTarget,
        access: TypedMemoryAccess,
        location: AbstractLocation
    ) {
        let condition = if let Some((target_access, num_preconditions)) = &exp.target_access {
            let mut condition = determine_any_byte_condition(&self.solver, &target_access, &access);

            for pre in &exp.preconditions[*num_preconditions..] {
                condition = condition.and(pre.clone());
            }

            if let Some((prev, _)) = self.map.remove(&location) {
                condition = prev.or(condition);
            }

            self.solver.simplify_condition(&condition)
        } else {
            SymCondition::FALSE
        };

        let symbols = exp.state.get_symbol_map_for(&condition);
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

/// Returns the condition under which `a` contains any byte from `b`.
pub fn determine_any_byte_condition(
    solver: &Solver,
    a: &TypedMemoryAccess,
    b: &TypedMemoryAccess
) -> SymCondition {
    solver.simplify_condition(&match (a.1, b.1) {
        (N8, N8) => a.0.clone().equal(b.0.clone()),
        (N8, _) => contains_ptr(b, &a.0),
        (_, N8) => contains_ptr(a, &b.0),
        (_, _) => contains_ptr(a, &b.0).or(contains_ptr(b, &a.0)),
    })
}

/// Returns the condition under which `a` contains all bytes from `b`.
pub fn determine_all_bytes_condition(
    solver: &Solver,
    a: &TypedMemoryAccess,
    b: &TypedMemoryAccess
) -> SymCondition {
    solver.simplify_condition(&if a.1 == b.1 {
        a.0.clone().equal(b.0.clone())
    } else if a.1 > b.1 {
        contains_ptr(a, &b.0)
    } else {
        SymCondition::FALSE
    })
}

/// Return the condition under which `ptr` is in the area spanned by the
/// pointer of `area` and the following bytes based on the data type.
fn contains_ptr(area: &TypedMemoryAccess, ptr: &SymExpr) -> SymCondition {
    let area_len = SymExpr::Int(Integer::from_ptr(area.1.bytes() as u64));
    let left = area.0.clone();
    ptr.clone().sub(left).less_than(area_len, false)
}

impl Display for AliasMap {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "AliasMap [")?;
        if !self.map.is_empty() { writeln!(f)?; }
        let mut map: Vec<_> = self.map.iter().collect();
        map.sort_by_key(|pair| pair.0.addr);
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

        fs::create_dir("target/out").ok();
        fs::create_dir("target/out/alias").ok();
        let alias_path = format!("target/out/alias/{}.txt", filename);
        let mut alias_file = File::create(alias_path).unwrap();
        write!(alias_file, "{}", map).unwrap();

        map
    }

    #[test]
    fn alias_bufs() {
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
    fn alias_paths() {
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
    fn alias_deep() {
        test("deep", AbstractLocation {
            addr: 0x399,
            trace: vec![0x2ba],
            storage: StorageLocation::indirect_reg(N8, Register::RDX),
        });
    }
}
