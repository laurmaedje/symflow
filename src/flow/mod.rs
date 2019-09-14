//! Control and value flow models.

use std::fmt::{self, Display, Formatter};
use crate::math::{Integer, DataType};
use crate::x86_64::{Register, Operand};

mod control;
mod alias;
mod value;
mod visualize;

pub use control::*;
pub use alias::*;
pub use value::*;


/// A storage location within the context in which it is valid.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct AbstractLocation {
    /// The address at which the location is valid.
    pub addr: u64,
    /// The call trace in which the location is valid.
    pub trace: Vec<u64>,
    /// The storage location.
    pub storage: StorageLocation,
}

impl AbstractLocation {
    pub fn new(addr: u64, trace: Vec<u64>, storage: StorageLocation) -> AbstractLocation {
        AbstractLocation { addr, trace, storage }
    }
}

/// A location in a real execution.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum StorageLocation {
    /// Directly the register.
    Direct(Register),
    /// The value at the address stored in base plus:
    /// - optionally a scaled offset of register * scale.
    /// - optionally an immediate displacement.
    Indirect {
        data_type: DataType,
        base: Register,
        scaled_offset: Option<(Register, u8)>,
        displacement: Option<i64>,
    },
}

impl StorageLocation {
    /// Create a storage location with just a simple indirect register.
    pub fn indirect_reg(data_type: DataType, base: Register) -> StorageLocation {
        StorageLocation::Indirect {
            data_type,
            base,
            scaled_offset: None,
            displacement: None,
        }
    }

    /// Create a storage location from an operand if it is a memory operand.
    pub fn from_operand(operand: Operand) -> Option<StorageLocation> {
        match operand {
            Operand::Direct(reg) => Some(StorageLocation::Direct(reg)),
            Operand::Indirect { data_type, base, scaled_offset, displacement } => Some(
                StorageLocation::Indirect { data_type, base, scaled_offset, displacement }
            ),
            _ => None,
        }
    }

    /// Change to base 64-bit registers for direct storage.
    ///
    /// This can be useful for comparing storage locations with registers
    /// that are not equal but share the same memory (like RAX and EAX).
    pub fn normalized(self) -> StorageLocation {
        match self {
            StorageLocation::Direct(reg) => StorageLocation::Direct(reg.base()),
            s => s,
        }
    }

    /// Whether this in an indirect access.
    pub fn accesses_memory(&self) -> bool {
        match self {
            StorageLocation::Direct(_) => false,
            StorageLocation::Indirect { .. } => true,
        }
    }

    /// The underlying data type of the value at the location.
    pub fn data_type(&self) -> DataType {
        match *self {
            StorageLocation::Direct(reg) => reg.data_type(),
            StorageLocation::Indirect { data_type, .. } => data_type,
        }
    }
}

/// Where a value comes from.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum ValueSource {
    Storage(StorageLocation),
    Const(Integer),
}

impl Display for AbstractLocation {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{} at {:x}", self.storage, self.addr)?;
        if !self.trace.is_empty() {
            write!(f, " by ")?;
        }
        let mut first = true;
        for &addr in &self.trace {
            if !first { write!(f, " -> ")?; } first = false;
            write!(f, "{:x}", addr)?;
        }
        Ok(())
    }
}

impl Display for StorageLocation {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use StorageLocation::*;
        use crate::helper::write_signed_hex;

        match *self {
            Direct(reg) => write!(f, "{}", reg),
            Indirect { data_type, base, scaled_offset, displacement } => {
                write!(f, "[{}", base)?;
                if let Some((index, scale)) = scaled_offset {
                    write!(f, "+{}*{}", index, scale)?;
                }
                if let Some(disp) = displacement {
                    write_signed_hex(f, disp)?;
                }
                write!(f, ":{}]", data_type)
            },
        }
    }
}

impl Display for ValueSource {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            ValueSource::Storage(storage) => write!(f, "{}", storage),
            ValueSource::Const(int) => write!(f, "{}", int),
        }
    }
}
