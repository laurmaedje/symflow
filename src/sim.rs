//! Simulation of micro code.

use std::collections::{HashMap, HashSet};
use std::ops::{Index, IndexMut, Not};

use crate::Program;
use crate::amd64::Register;
use crate::ir::{MicroOperation, Location, Condition, Temporary, MemoryMapped};
use crate::num::{DataType, Integer, Flags};


/// Simulates an execution.
#[derive(Debug)]
pub struct Simulator<'p> {
    base: u64,
    ip: u64,
    program: &'p Program,
    spaces: [MemorySpace; 2],
    flags: Flags,
    temporaries: HashMap<usize, Integer>,
    break_points: HashSet<u64>,
}

impl<'p> Simulator<'p> {
    /// Create a new simulator.
    pub fn new(base: u64, entry_point: u64, program: &'p Program) -> Simulator<'p> {
        let main_memory = MemorySpace::new((0x0, 0x8000_0000_0000_0000 - 1));
        let registers = MemorySpace::new((0x0, 0x200));

        let mut sim = Simulator {
            base,
            ip: entry_point,
            program,
            spaces: [main_memory, registers],
            flags: Flags::default(),
            temporaries: HashMap::new(),
            break_points: HashSet::new(),
        };

        // Set up the program counter and stack.
        let stack_start = ptr(0x7fff_ffff_0000_0000);
        sim.set_reg(Register::RIP, ptr(entry_point));
        sim.set_reg(Register::RSP, stack_start);
        sim.set_reg(Register::RBP, stack_start);

        sim
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
        let (len, _, microcode) = self.program.mapping[&(self.ip - self.base)].clone();

        // Adjust the instruction pointer already.
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
    pub fn run(&mut self) {
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
            Op::Cast { target, new, signed } => {
                let new_value = self.get_temp(target).cast(new, signed);
                self.set_temp(Temporary(new, target.1), new_value);
            },

            Op::Add { sum, a, b, flags } => self.do_binop(sum, a, b, flags, Integer::flagged_add),
            Op::Sub { diff, a, b, flags } => self.do_binop(diff, a, b, flags, Integer::flagged_sub),
            Op::Mul { prod, a, b, flags } => self.do_binop(prod, a, b, flags, Integer::flagged_mul),

            Op::And { and, a, b, flags } => self.do_binop(and, a, b, flags, Integer::flagged_and),
            Op::Or { or, a, b, flags } => self.do_binop(or, a, b, flags, Integer::flagged_or),
            Op::Not { not, a } => self.do_unop(not, a, Not::not),

            Op::Set { target, condition } => {
                let bit = self.evaluate_condition(condition);
                self.set_temp(target, Integer(DataType::N8, bit as u64));
            },
            Op::Jump { target, condition, relative } => {
                if self.evaluate_condition(condition) {
                    let value = self.get_temp(target);
                    assert_eq!(value.0, DataType::N64, "jump address has to be 64-bit");

                    return Some(Event::Jump(if relative {
                        self.ip.wrapping_add(value.1)
                    } else {
                        value.1
                    }))
                }
            },

            Op::Syscall => {
                let value = self.get_reg(Register::RAX).1;
                match value {
                    // Exit syscall
                    0x3c => return Some(Event::Exit),
                    _ => panic!("unimplemented syscall: {:#x}", value),
                }
            },
        }

        None
    }

    /// Do a binary operation on temporaries.
    fn do_binop<F>(&mut self, target: Temporary, a: Temporary, b: Temporary, flags: bool, binop: F)
     where F: FnOnce(Integer, Integer) -> (Integer, Flags) {
        let data_type = target.0;
        assert!(data_type == a.0 && data_type == b.0,
            "do_binop: incompatible data types for binop");

        let left = self.get_temp(a);
        let right = self.get_temp(b);
        let (result, result_flags) = binop(left, right);
        self.set_temp(target, result);

        // Set the flags if requested.
        if flags {
            self.flags = result_flags;
        }
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
        assert_eq!(dest.data_type(), src.data_type(),
            "do_move: incompatible data types for move");
        let value = self.read_location(src);

        self.write_location(dest, value);
    }

    /// Evaluate a condition.
    fn evaluate_condition(&self, condition: Condition) -> bool {
        match condition {
            Condition::Always => true,
            Condition::Equal => self.flags.zero,
            Condition::Less => self.flags.sign != self.flags.overflow,
            Condition::Greater => !self.flags.zero && (self.flags.sign == self.flags.overflow),
        }
    }

    /// Jump to an address.
    fn jump(&mut self, addr: u64) {
        self.ip = addr;
        self.set_reg(Register::RIP, ptr(addr));
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
                assert_eq!(addr.0, DataType::N64, "read_location: address has to be 64-bit");
                self.read_location(Location::Direct(data_type, space, addr.1))
            }
        }
    }

    /// Write data to a location.
    fn write_location(&mut self, dest: Location, value: Integer) {
        assert_eq!(dest.data_type(), value.0,
            "write_location: incompatible data types for write");

        match dest {
            Location::Temp(temp) => self.set_temp(temp, value),
            Location::Direct(_, space, addr) => {
                self.spaces[space].write_int(addr, value);
            },
            Location::Indirect(data_type, space, temp) => {
                let addr = self.get_temp(temp);
                assert_eq!(addr.0, DataType::N64, "write_location: address has to be 64-bit");
                self.write_location(Location::Direct(data_type, space, addr.1), value)
            }
        }
    }

    /// Return the integer stored in the temporary.
    fn get_temp(&self, temp: Temporary) -> Integer {
        let integer = self.temporaries[&temp.1];
        assert_eq!(integer.0, temp.0, "get_temp: incompatible data types");
        integer
    }

    /// Set the temporary to a new value.
    fn set_temp(&mut self, temp: Temporary, value: Integer) {
        assert_eq!(temp.0, value.0, "set_temp: incompatible data types");
        self.temporaries.insert(temp.1, value);
    }

    /// Get a value from a register.
    fn get_reg(&self, reg: Register) -> Integer {
        self.spaces[1].read_int(reg.address(), reg.data_type())
    }

    /// Set a register to a value.
    fn set_reg(&mut self, reg: Register, value: Integer) {
        self.spaces[1].write_int(reg.address(), value);
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
        let bytes = self.read(addr, data_type.bytes());
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
            panic!("memory access out of bounds: {:#x}", address);
        }
    }
}

impl IndexMut<u64> for MemorySpace {
    fn index_mut(&mut self, address: u64) -> &mut u8 {
        if self.bounded(address) {
            self.map.entry(address).or_insert(0)
        } else {
            panic!("mutable memory access out of bounds: {:#x}", address);
        }
    }
}

/// Create an 64-bit integer.
fn ptr(addr: u64) -> Integer {
    Integer(DataType::N64, addr)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::elf::*;

    fn test(file: &str, variable: u32) {
        // Read the text section from the binary file.
        let mut file = ElfFile::new(std::fs::File::open(file).unwrap()).unwrap();
        let text = file.get_section(".text").unwrap();
        let program = Program::new(&text).unwrap();

        // Run a simulation.
        let base = 0x400000;
        let mut simulator = Simulator::new(base, base + file.header.entry, &program);
        simulator.add_break_point(base + 0x345);
        simulator.run();

        // Check that the variable `c` has the correct value.
        let addr = simulator.get_reg(Register::RBP).1 - 0xc;
        let c = simulator.spaces[0].read_int(addr, DataType::N32).1 as u32;
        assert_eq!(simulator.ip, 0x400345);
        assert_eq!(c, variable);
    }

    #[test]
    fn simulate() {
        test("test/block-first", 0xdeadbeef);
        test("test/block-second", 0xbeefdead);
    }
}
