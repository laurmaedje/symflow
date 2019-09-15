//! Symbolic memory models.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt::{self, Display, Formatter};

use crate::math::{SymExpr, SymCondition, DataType, Symbol, SharedSolver};


/// Symbolic memory handling writes and reads involving symbolic
/// values and addresses.
#[derive(Debug, Clone)]
pub struct SymMemory {
    data: RefCell<MemoryData>,
    solver: SharedSolver,
    strategy: MemoryStrategy,
}

/// How the memory handled complex symbolic queries.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum MemoryStrategy {
    /// Build an if-then-else tree of values that could possibly match.
    ConditionalTrees,
    /// Only return perfect matches (faster).
    PerfectMatches,
}

/// The actual memory data, which is wrapped in an interior mutability type
/// to make reads on immutable borrows possible while performing some extra work
/// requiring mutable access.
#[derive(Debug, Clone)]
struct MemoryData {
    name: &'static str,
    writes: BTreeMap<Epoch, MemoryWrite>,
    symbols: usize,
    epoch: Epoch,
}

type Epoch = u32;

/// A piece of data written to memory.
#[derive(Debug, Clone)]
struct MemoryWrite {
    addr: SymExpr,
    value: SymExpr,
}

impl SymMemory {
    /// Create a new blank symbolic memory.
    pub fn new(name: &'static str, strategy: MemoryStrategy, solver: SharedSolver) -> SymMemory {
        SymMemory {
            data: RefCell::new(MemoryData {
                name,
                writes: BTreeMap::new(),
                symbols: 0,
                epoch: 1,
            }),
            solver,
            strategy,
        }
    }

    /// Read from a direct address.
    pub fn read_direct(&self, addr: u64, data_type: DataType) -> SymExpr {
        self.read_expr(SymExpr::from_ptr(addr), data_type)
    }

    /// Write a value to a direct address.
    pub fn write_direct(&mut self, addr: u64, value: SymExpr) {
        self.write_expr(SymExpr::from_ptr(addr), value)
    }

    /// Read from a symbolic address.
    pub fn read_expr(&self, addr: SymExpr, data_type: DataType) -> SymExpr {
        crate::timings::with("sym-mem", || {
            let expr = match self.strategy {
                MemoryStrategy::PerfectMatches => self.read_perfect(addr, data_type),
                MemoryStrategy::ConditionalTrees => self.read_conditional(addr, data_type),
            };

            if expr.data_type() == data_type { expr } else { expr.cast(data_type, false) }
        })
    }

    /// Read from memory using the perfect matches strategy.
    ///
    /// This will only return the value if the address expression match
    /// perfectly. This is not sound but way faster than `read_conditional`.
    fn read_perfect(&self, addr: SymExpr, data_type: DataType) -> SymExpr {
        let mut data = self.data.borrow_mut();

        for (_, write) in data.writes.iter().rev() {
            if write.addr == addr {
                return write.value.clone();
            }
        }

        data.generate_default_symbol(addr, data_type)
    }

    /// Read from memory using the conditional trees strategy.
    ///
    /// This will go through the write map from newest to oldest and build
    /// an if-then-else chain with all values that possibly match and the
    /// conditions under which they match.
    fn read_conditional(&self, addr: SymExpr, data_type: DataType) -> SymExpr {
        let mut data = self.data.borrow_mut();

        let default = data.get_default_value(data_type);
        let mut tree = default.clone();
        let mut active = &mut tree;
        let mut used_default_symbol = true;

        // We traverse the memory writes from latest to oldest. If we find
        // a write that perfectly matches our read, we can quit early
        // because anything before would have been overwritten for sure.
        for (_, write) in data.writes.iter().rev() {
            // If it matches perfectly, we can stop here.
            if write.addr == addr {
                *active = write.value.clone();
                used_default_symbol = false;
                break;
            }

            if self.solver.check_equal_sat(&write.addr, &addr) {
                let condition = write.addr.clone().equal(addr.clone());
                let simplified = self.solver.simplify_condition(&condition);

                // If it didn't match perfectly but still always is the same thing
                // we can also stop here.
                if simplified == SymCondition::TRUE {
                    *active = write.value.clone();
                    used_default_symbol = false;
                    break;

                } else {
                    *active = simplified.if_then_else(write.value.clone(), default.clone());
                    active = match active {
                        SymExpr::IfThenElse(_, _, ref mut b) => b,
                        _ => panic!("read_conditional: expected if-then-else"),
                    };
                }
            }
        }

        if used_default_symbol {
            data.generate_default_symbol(addr, data_type);
        }

        tree
    }

    /// Write a value to a symbolic address.
    pub fn write_expr(&mut self, addr: SymExpr, value: SymExpr) {
        crate::timings::with("sym-mem", || {
            let mut data = self.data.borrow_mut();

            let new_write = MemoryWrite { addr, value };

            for (_, write) in data.writes.iter_mut().rev() {
                if write.addr == new_write.addr {
                    *write = new_write;
                    return;
                }
            }

            let epoch = data.epoch;
            data.writes.insert(epoch, new_write);
            data.epoch += 1;
        })
    }
}

impl MemoryData {
    /// Get the value for the next default symbol that would be generated.
    fn get_default_value(&self, data_type: DataType) -> SymExpr {
        SymExpr::Sym(Symbol(data_type, self.name, self.symbols))
    }

    /// Generate a default symbol for uninitialized memory.
    fn generate_default_symbol(&mut self, addr: SymExpr, data_type: DataType) -> SymExpr {
        let value = self.get_default_value(data_type);
        self.writes.insert(0, MemoryWrite {
            addr,
            value: value.clone(),
        });
        self.symbols += 1;
        value
    }
}

impl Display for SymMemory {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "SymMemory [")?;
        let data = self.data.borrow();
        if !data.writes.is_empty() { writeln!(f)?; }
        for (epoch, write) in &data.writes {
            writeln!(f, "    [{}] {} => {}", epoch, write.addr, write.value)?;
        }
        writeln!(f, "]")
    }
}
