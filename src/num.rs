//! Generic sized machine integer.

use std::fmt::{self, Debug, Display, Formatter};
use std::ops::{Add, Sub, Mul, Div, BitAnd, BitOr, Not};
use byteorder::{ByteOrder, LittleEndian};


/// Basic numeric types.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum DataType {
    I8, U8, I16, U16, I32, U32, I64, U64,
}

impl DataType {
    /// The data type for the specified bit width. (signed or unsigned).
    pub fn from_width(width: DataWidth, signed: bool) -> DataType {
        use DataType::*;
        match width {
            DataWidth::Bits8  => [U8, I8][signed as usize],
            DataWidth::Bits16 => [U16, I16][signed as usize],
            DataWidth::Bits32 => [U32, I32][signed as usize],
            DataWidth::Bits64 => [U64, I64][signed as usize],
        }
    }

    /// The width of the data type.
    pub fn width(&self) -> DataWidth {
        use DataType::*;
        match self {
            U8 | I8 => DataWidth::Bits8 ,
            U16 | I16 => DataWidth::Bits16,
            U32 | I32 => DataWidth::Bits32,
            U64 | I64 => DataWidth::Bits64,
        }
    }
}

impl Display for DataType {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", format!("{:?}", self).to_lowercase())
    }
}

/// Width of data in bits.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum DataWidth {
    Bits8 = 0,
    Bits16 = 1,
    Bits32 = 2,
    Bits64 = 3,
}

impl DataWidth {
    /// The number of bytes this width equals.
    pub fn bytes(&self) -> u64 {
        match self {
            DataWidth::Bits8 => 1,
            DataWidth::Bits16 => 2,
            DataWidth::Bits32 => 4,
            DataWidth::Bits64 => 8,
        }
    }
}

impl Display for DataWidth {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            DataWidth::Bits8 => write!(f, "byte"),
            DataWidth::Bits16 => write!(f, "word"),
            DataWidth::Bits32 => write!(f, "dword"),
            DataWidth::Bits64 => write!(f, "qword"),
        }
    }
}

/// Variable data type integer with machine semantics.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Integer(pub DataType, pub u64);

impl Integer {
    /// Read an integer of a specific type from bytes.
    pub fn from_bytes(data_type: DataType, bytes: &[u8]) -> Integer {
        use DataType::*;
        Integer(data_type, match data_type {
            I8  => bytes[0] as i8 as u64,
            U8  => bytes[0] as u64,
            I16 => LittleEndian::read_i16(bytes) as u64,
            U16 => LittleEndian::read_u16(bytes) as u64,
            I32 => LittleEndian::read_i32(bytes) as u64,
            U32 => LittleEndian::read_u32(bytes) as u64,
            I64 => LittleEndian::read_i64(bytes) as u64,
            U64 => LittleEndian::read_u64(bytes) as u64,
        })
    }

    /// Convert this integer into bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        use DataType::*;
        let mut buf = vec![0; self.0.width().bytes() as usize];
        match self.0 {
            I8  => buf.push(self.1 as i8 as u8),
            U8  => buf.push(self.1 as u8),
            I16 => LittleEndian::write_i16(&mut buf, self.1 as i16),
            U16 => LittleEndian::write_u16(&mut buf, self.1 as u16),
            I32 => LittleEndian::write_i32(&mut buf, self.1 as i32),
            U32 => LittleEndian::write_u32(&mut buf, self.1 as u32),
            I64 => LittleEndian::write_i64(&mut buf, self.1 as i64),
            U64 => LittleEndian::write_u64(&mut buf, self.1 as u64),
        }
        buf
    }

    /// Cast the integer to another type.
    pub fn cast(self, new_type: DataType) -> Integer {
        use DataType::*;
        Integer(new_type, match new_type {
            I8  => (self.1 as i8 ) as u64, U8  => (self.1 as u8 ) as u64,
            I16 => (self.1 as i16) as u64, U16 => (self.1 as u16) as u64,
            I32 => (self.1 as i32) as u64, U32 => (self.1 as u32) as u64,
            I64 => (self.1 as i64) as u64, U64 => (self.1 as u64) as u64,
        })
    }
}

macro_rules! int_operation {
    ($trait:ident, $func:ident, $op:tt) => {
        impl $trait for Integer {
            type Output = Integer;

            fn $func(self, other: Integer) -> Integer {
                assert_eq!(self.0, other.0,
                    "incompatible data types for integer operation");

                use DataType::*;
                Integer(self.0, match self.0 {
                    I8  => ((self.1 as i8 ).$op(other.1 as i8 )) as u64,
                    U8  => ((self.1 as u8 ).$op(other.1 as u8 )) as u64,
                    I16 => ((self.1 as i16).$op(other.1 as i16)) as u64,
                    U16 => ((self.1 as u16).$op(other.1 as u16)) as u64,
                    I32 => ((self.1 as i32).$op(other.1 as i32)) as u64,
                    U32 => ((self.1 as u32).$op(other.1 as u32)) as u64,
                    I64 => ((self.1 as i64).$op(other.1 as i64)) as u64,
                    U64 => ((self.1 as u64).$op(other.1 as u64)) as u64,
                })
            }
        }

    };
}

int_operation!(Add, add, wrapping_add);
int_operation!(Sub, sub, wrapping_sub);
int_operation!(Mul, mul, wrapping_mul);
int_operation!(Div, div, wrapping_div);
int_operation!(BitAnd, bitand, bitand);
int_operation!(BitOr, bitor, bitor);

impl Not for Integer {
    type Output = Integer;

    fn not(self) -> Integer {
        use DataType::*;
        Integer(self.0, match self.0 {
            I8  => !(self.1 as i8 ) as u64, U8  => !(self.1 as u8 ) as u64,
            I16 => !(self.1 as i16) as u64, U16 => !(self.1 as u16) as u64,
            I32 => !(self.1 as i32) as u64, U32 => !(self.1 as u32) as u64,
            I64 => !(self.1 as i64) as u64, U64 => !(self.1 as u64) as u64,
        })
    }
}

impl Display for Integer {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use DataType::*;
        match self.0 {
            I8 | I16 | I32 | I64 => if (self.1 as i64) >= 0 {
                write!(f, "{:#x}", self.1)
            } else {
                write!(f, "-{:#x}", -(self.1 as i64))
            }
            U8 | U16 | U32 | U64 => write!(f, "{:#x}", self.1),
        }?;
        write!(f, ":{}", self.0)
    }
}
