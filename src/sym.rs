//! Symbolic execution of microcode.

use std::collections::{HashMap, HashSet};
use crate::ir::MicroOperation;
use crate::num::Integer;

pub struct SymState {
    temporaries: HashMap<usize, SymExpr>,
    memory: [SymMemory; 2],
}

impl SymState {
    pub fn execute(operation: MicroOperation) -> Option<Event> {
        None
    }
}

pub struct SymMemory {
    map: HashMap<SymExpr, SymExpr>,
}

/// Events occuring during symbolic execution.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Event {
    Jump(SymExpr),
    Exit,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum SymExpr {
    Int(Integer),
    Sym(Symbol)
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Symbol {
    id: u64,
}
