//! Control and value flow models.

use std::fmt::{self, Display, Formatter};
use crate::x86_64::{Register, Operand};
use crate::math::DataType;

mod control;
mod alias;
mod value;
mod visualize;

pub use control::*;
pub use alias::*;
pub use value::*;


/// A CPU location within the context in which it is valid.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct AbstractLocation {
    /// The address at which the location is valid.
    pub addr: u64,
    /// The call trace in which the location is valid.
    pub trace: Vec<u64>,
    /// The storage
    pub location: StorageLocation,
}

/// A location in a real execution.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
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
    pub fn indirect_reg(data_type: DataType, reg: Register) -> StorageLocation {
        StorageLocation::Indirect {
            data_type,
            base: reg,
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

    /// The underlying data type of the value at the location.
    pub fn data_type(&self) -> DataType {
        match *self {
            StorageLocation::Direct(reg) => reg.data_type(),
            StorageLocation::Indirect { data_type, .. } => data_type,
        }
    }
}

impl Display for AbstractLocation {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{} at {:x} by ", self.location, self.addr)?;
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
