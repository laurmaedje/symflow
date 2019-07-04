//! Simulation of micro code.

use std::collections::{HashMap, HashSet};
use std::ops::{Add, Sub, Mul, Div, BitAnd, BitOr, Not};
use std::fmt::{self, Debug, Display, Formatter};
use std::ops::{self, Index, IndexMut};

use crate::Program;
use crate::amd64::{Register, Flag};
use crate::ir::{MicroOperation, Location, Condition, Temporary, MemoryMapped};
use crate::num::{DataType, Integer};


/// Simulates an execution.
#[derive(Debug)]
pub struct Simulator {
    base: u64,
    ip: u64,
    program: Program,
    spaces: [MemorySpace; 2],
    temporaries: HashMap<usize, Integer>,
    break_points: HashSet<u64>,
}

impl Simulator {
    /// Create a new simulator.
    pub fn new() -> Simulator {
        let main_memory = MemorySpace::new((0x0, 0x8000_0000_0000_0000 - 1));
        let registers = MemorySpace::new((0x0, 0x200));
        Simulator {
            base: 0,
            ip: 0,
            program: Program::empty(),
            spaces: [main_memory, registers],
            temporaries: HashMap::new(),
            break_points: HashSet::new(),
        }
    }

    /// Load a microcode program at an address.
    pub fn load(&mut self, base: u64, entry_point: u64, program: Program) {
        // Reset temporaries, spaces and break points.
        self.temporaries.clear();
        self.break_points.clear();
        self.spaces[0].clear();
        self.spaces[1].clear();

        // Set up the program and program counter.
        self.base = base;
        self.program = program;
        self.jump(entry_point);

        // Set up the stack.
        let stack_start = ptr(0x7fff_ffff_0000_0000);
        self.set_reg(Register::RSP, stack_start);
        self.set_reg(Register::RBP, stack_start);
    }

    /// Add a one-time breakpoint at an address.
    pub fn add_break_point(&mut self, addr: u64) {
        self.break_points.insert(addr);
    }

    /// Execute one instruction.
    pub fn step(&mut self) -> Option<Event> {
        // Check whether we ran into a break point.
        if self.break_points.contains(&self.ip) {
            self.break_points.remove(&self.ip);
            return Some(Event::Break);
        }

        // Reset the temporaries and load the next instruction.
        self.temporaries.clear();
        let (len, inst, microcode) = self.program.mapping[&(self.ip - self.base)].clone();

        // Adjust the
        self.jump(self.ip + len);

        for op in microcode.ops {
            if let Some(event) = self.microstep(op) {
                match event {
                    Event::Break | Event::Exit => return Some(event),
                    Event::Jump(addr) => {
                        self.jump(addr);
                        break;
                    },
                }
            }
        }

        None
    }

    /// Execute instructions until there is an error or exit.
    fn run(&mut self) {
        loop {
            if let Some(event) = self.step() {
                match event {
                    Event::Break | Event::Exit => break,
                    _ => panic!("run: unexpected event"),
                }
            }
        }
    }

    /// Do a micro operation.
    fn microstep(&mut self, op: MicroOperation) -> Option<Event> {
        use MicroOperation as Op;

        match op {
            Op::Mov { dest, src } => self.do_move(dest, src),
            Op::Const { dest, constant } => self.write_location(dest, constant),
            Op::Cast { target, new } => {
                let new_value = self.get_temp(target).cast(new);
                self.set_temp(Temporary(new, target.1), new_value);
            },

            Op::Add { sum, a, b, flags } => self.do_binop(sum, a, b, Add::add, flags),
            Op::Sub { diff, a, b, flags } => self.do_binop(diff, a, b, Sub::sub, flags),
            Op::Mul { prod, a, b, flags } => self.do_binop(prod, a, b, Mul::mul, flags),

            Op::And { and, a, b, flags } => self.do_binop(and, a, b, BitAnd::bitand, flags),
            Op::Or { or, a, b, flags } => self.do_binop(or, a, b, BitOr::bitor, flags),
            Op::Not { not, a } => self.do_unop(not, a, Not::not),

            Op::Set { target, condition } => {
                let bit = self.evaluate_condition(condition);
                self.set_temp(target, Integer(DataType::U8, bit as u64));
            },
            Op::Jump { target, condition, relative } => {
                if self.evaluate_condition(condition) {
                    let value = self.get_temp(target);

                    let mut addr = value.1;
                    return Some(Event::Jump(if relative {
                        assert_eq!(value.0, DataType::I64, "relative address has to be i64");
                        self.ip.wrapping_add(value.1)
                    } else {
                        assert_eq!(value.0, DataType::U64, "absolute address has to be u64");
                        value.1
                    }))
                }
            },

            Op::Syscall => {
                let value = self.get_reg(Register::RAX, false).1;
                match value {
                    // Exit syscall
                    60 => return Some(Event::Exit),
                    _ => panic!("unimplemented syscall: {:#x}", value),
                }
            },
        }

        None
    }

    /// Do a binary operation on temporaries.
    fn do_binop<F>(&mut self, target: Temporary, a: Temporary, b: Temporary, binop: F, flags: bool)
    where F: FnOnce(Integer, Integer) -> Integer {
        let data_type = target.0;
        assert!(data_type == a.0 && data_type == b.0, "incompatible data types for binop");

        let left = self.get_temp(a);
        let right = self.get_temp(b);
        let result = binop(left, right);
        self.set_temp(target, result);
    }

    /// Do a unary operation on temporaries.
    fn do_unop<F>(&mut self, target: Temporary, a: Temporary, unop: F)
    where F: FnOnce(Integer) -> Integer {
        let operand = self.get_temp(a);
        let result = unop(operand);
        self.set_temp(target, result);
    }

    /// Move a value from a location to another location.
    fn do_move(&mut self, dest: Location, src: Location) {
        assert_eq!(dest.data_type(), src.data_type(), "incompatible data types for move");
        let value = self.read_location(src);
        self.write_location(dest, value);
    }

    /// Evaluate a condition.
    fn evaluate_condition(&self, condition: Condition) -> bool {
        use {Condition::*, Flag::*};
        match condition {
            Always => true,
            Equal => self.get_flag(Zero),
            Less => self.get_flag(Sign) != self.get_flag(Overflow),
            Greater => !self.get_flag(Zero) && (self.get_flag(Sign) == self.get_flag(Overflow)),
        }
    }

    /// Retrieve data from a location.
    fn read_location(&self, src: Location) -> Integer {
        match src {
            Location::Temp(temp) => self.get_temp(temp),
            Location::Direct(data_type, space, addr) => {
                self.spaces[space].read_int(addr, data_type)
            },
            Location::Indirect(data_type, space, temp) => {
                let addr = self.get_temp(temp);
                assert_eq!(addr.0, DataType::U64, "address has to be u64");
                self.read_location(Location::Direct(data_type, space, addr.1))
            }
        }
    }

    /// Write data to a location.
    fn write_location(&mut self, dest: Location, value: Integer) {
        match dest {
            Location::Temp(temp) => self.set_temp(temp, value),
            Location::Direct(data_type, space, addr) => {
                self.spaces[space].write_int(addr, value);
            },
            Location::Indirect(data_type, space, temp) => {
                let addr = self.get_temp(temp);
                assert_eq!(addr.0, DataType::U64, "address has to be u64");
                self.write_location(Location::Direct(data_type, space, addr.1), value)
            }
        }
    }

    /// Jump to an address.
    fn jump(&mut self, addr: u64) {
        self.ip = addr;
        self.set_reg(Register::IP, ptr(addr));
    }

    /// Return the integer stored in the temporary.
    fn get_temp(&self, temp: Temporary) -> Integer {
        let integer = self.temporaries[&temp.1];
        assert_eq!(integer.0, temp.0, "wrong load type for temporary");
        integer
    }

    /// Set the temporary to a new value.
    fn set_temp(&mut self, temp: Temporary, value: Integer) {
        assert_eq!(temp.0, value.0, "wrong write type for temporary");
        self.temporaries.insert(temp.1, value);
    }

    /// Get a value from a register.
    fn get_reg(&self, reg: Register, signed: bool) -> Integer {
        let data_type = DataType::from_width(reg.width(), signed);
        self.spaces[1].read_int(reg.address(), data_type)
    }

    /// Set a register to a value.
    fn set_reg(&mut self, reg: Register, value: Integer) {
        self.spaces[1].write_int(reg.address(), value);
    }

    /// Get the truth value of a flag.
    fn get_flag(&self, flag: Flag) -> bool {
        self.spaces[1][flag.address()] != 0
    }

    /// Set the truth value of a flag.
    fn set_flag(&mut self, flag: Flag, value: bool) {
        self.spaces[1][flag.address()] = value as u8;
    }
}

/// Events occuring during execution.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Event {
    Jump(u64),
    Exit,
    Break,
}

/// Memory space that can be read, written and executed.
#[derive(Debug)]
pub struct MemorySpace {
    bounds: (u64, u64),
    map: HashMap<u64, u8>,
}

impl MemorySpace {
    /// Create a new zeroed memory space with the given inclusive bounds.
    pub fn new(bounds: (u64, u64)) -> MemorySpace {
        MemorySpace { bounds, map: HashMap::new() }
    }

    /// Read a number of bytes at an address.
    pub fn read(&self, addr: u64, num: u64) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(num as usize);
        for address in addr .. addr + num {
            bytes.push(self[address]);
        }
        bytes
    }

    /// Write bytes at an address.
    pub fn write(&mut self, addr: u64, bytes: &[u8]) {
        for (address, &byte) in (addr ..).zip(bytes) {
            self[address] = byte;
        }
    }

    /// Read an integer from an address.
    pub fn read_int(&self, addr: u64, data_type: DataType) -> Integer {
        let bytes = self.read(addr, data_type.width().bytes());
        Integer::from_bytes(data_type, &bytes)
    }

    /// Write an integer at an address.
    pub fn write_int(&mut self, addr: u64, value: Integer) {
        self.write(addr, &value.to_bytes());
    }

    /// Zero the entire memory.
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Check whether the given address lies in the bounds of this memory space.
    pub fn bounded(&self, address: u64) -> bool {
        self.bounds.0 <= address && address <= self.bounds.1
    }
}

impl Index<u64> for MemorySpace {
    type Output = u8;

    fn index(&self, address: u64) -> &u8 {
        if self.bounded(address) {
            self.map.get(&address).unwrap_or(&0)
        } else {
            panic!("memory (get) access out of bounds: {:#x}", address);
        }
    }
}

impl IndexMut<u64> for MemorySpace {
    fn index_mut(&mut self, address: u64) -> &mut u8 {
        if self.bounded(address) {
            self.map.entry(address).or_insert(0)
        } else {
            panic!("memory (get mut) access out of bounds: {:#x}", address);
        }
    }
}

/// Create an U64 integer.
fn ptr(addr: u64) -> Integer {
    Integer(DataType::U64, addr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use crate::elf::*;

    fn test(file: &str, variable: u32) {
        // Read the text section from the binary file.
        let bin = std::fs::read(file).unwrap();
        let mut file = ElfFile::from_slice(&bin).unwrap();
        let text = file.get_section(".text").unwrap();

        // Disassemble the whole code and print it.
        let code = Code::new(text.header.addr, &text.data);
        let program = code.disassemble_program().unwrap();

        // Run a simulation.
        let base = 0x400000;
        let mut simulator = Simulator::new();
        simulator.load(base, base + file.header.entry, program);
        simulator.add_break_point(base + 0x345);
        simulator.run();

        // Check that the variable `c` has the correct value.
        let addr = simulator.get_reg(Register::RBP, false).1 - 0xc;
        let c = simulator.spaces[0].read_int(addr, DataType::U32).1 as u32;
        assert_eq!(simulator.ip, 0x400345);
        assert_eq!(c, variable);
    }

    #[test]
    fn simulate() {
        test("test/block-first", 0xdeadbeef);
    }
}
