//! Integers with machine semantics.

use std::cmp::{Ord, PartialOrd, Eq, PartialEq, Ordering};
use std::fmt::{self, Display, Formatter};
use std::ops::{Add, Sub, Mul, BitAnd, BitOr, Not};
use byteorder::{ByteOrder, LittleEndian};

use DataType::*;


/// Variable data type integer with machine semantics.
#[derive(Debug, Copy, Clone, Hash)]
pub struct Integer(pub DataType, pub u64);

/// Different width numeric types.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum DataType {
    N8,
    N16,
    N32,
    N64,
}

/// Replicates code for all types.
macro_rules! typed {
    ($caster:ident => $data_type:expr, $signed:expr, $code:block) => {
        match ($data_type, $signed) {
            (N8 , false) => { let $caster = |n| n as u8; $code  }
            (N16, false) => { let $caster = |n| n as u16; $code }
            (N32, false) => { let $caster = |n| n as u32; $code }
            (N64, false) => { let $caster = |n| n as u64; $code }
            (N8 , true)  => { let $caster = |n| n as i8; $code  }
            (N16, true)  => { let $caster = |n| n as i16; $code }
            (N32, true)  => { let $caster = |n| n as i32; $code }
            (N64, true)  => { let $caster = |n| n as i64; $code }
        }
    };
}

/// Default arithmetic flags.
macro_rules! flags {
    ($target:expr) => {
        Flags {
            zero: $target == 0,
            sign: $target.leading_zeros() == 0,
            overflow: false,
        }
    };
}

/// Make sure operations only happen on same integers.
fn check_compatible(a: DataType, b: DataType, operation: &str) {
    assert_eq!(a, b, "incompatible data types for integer {}", operation);
}

/// Arithmetic operation with flags.
macro_rules! flagged {
    ($name:ident, $target:ident, $a:ident, $b:ident => $op:ident, $flags:expr) => {
        pub fn $name(self, other: Integer) -> (Integer, Flags) {
            check_compatible(self.0, other.0, "operation");
            typed!(cast => self.0, true, {
                let $a = cast(self.1);
                let $b = cast(other.1);
                let $target = $a.$op($b);
                (Integer(self.0, $target as u64), $flags)
            })
        }
    };
}

impl Integer {
    /// Create a pointer-sized integer.
    pub fn from_ptr(value: u64) -> Integer {
        Integer(N64, value)
    }

    /// Create a boolean-based integer.
    pub fn from_bool(value: bool, data_type: DataType) -> Integer {
        Integer(data_type, value as u64)
    }

    /// Read an integer of a specific type from bytes.
    pub fn from_bytes(bytes: &[u8], data_type: DataType) -> Integer {
        Integer(data_type, match data_type {
            N8  => bytes[0] as u64,
            N16 => LittleEndian::read_u16(bytes) as u64,
            N32 => LittleEndian::read_u32(bytes) as u64,
            N64 => LittleEndian::read_u64(bytes) as u64,
        })
    }

    /// Convert this integer into bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = vec![0; self.0.bytes()];
        match self.0 {
            N8  => buf[0] = self.1 as u8,
            N16 => LittleEndian::write_u16(&mut buf, self.1 as u16),
            N32 => LittleEndian::write_u32(&mut buf, self.1 as u32),
            N64 => LittleEndian::write_u64(&mut buf, self.1 as u64),
        }
        buf
    }

    /// Cast the integer to another type.
    /// - If the target type is smaller, it will get truncated.
    /// - If the target type is bigger, if signed is true the value will be
    ///   sign-extended and otherwise zero-extended.
    pub fn cast(self, new_type: DataType, signed: bool) -> Integer {
        Integer(new_type, typed!(cast => self.0, signed, {
            let src = cast(self.1) as u64;
            typed!(cast2 => new_type, false, {
                cast2(src) as u64
            })
        }))
    }

    // Operations with CPU flags.
    flagged!(flagged_add, sum, a, b => wrapping_add, Flags {
        overflow: a.overflowing_add(b).1, .. flags!(sum)
    });
    flagged!(flagged_sub, diff, a, b => wrapping_sub, Flags {
        overflow: a.overflowing_sub(b).1, .. flags!(diff)
    });
    flagged!(flagged_mul, product, a, b => wrapping_mul, Flags {
        zero: false, sign: false, overflow: a.overflowing_mul(b).1
    });
    flagged!(flagged_and, and, a, b => bitand, Flags { overflow: false, .. flags!(and) });
    flagged!(flagged_or, or, a, b => bitor, Flags { overflow: false, .. flags!(or) });
}

impl Display for Integer {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{:#x}:{}", self.1, self.0)
    }
}

macro_rules! binary_operation {
    ($trait:ident, $func:ident, $op:tt) => {
        impl $trait for Integer {
            type Output = Integer;

            fn $func(self, other: Integer) -> Integer {
                check_compatible(self.0, other.0, "operation");
                Integer(self.0, typed!(cast => self.0, false, {
                    (cast(self.1).$op(cast(other.1))) as u64
                }))
            }
        }
    };
}

binary_operation!(Add, add, wrapping_add);
binary_operation!(Sub, sub, wrapping_sub);
binary_operation!(Mul, mul, wrapping_mul);
binary_operation!(BitAnd, bitand, bitand);
binary_operation!(BitOr, bitor, bitor);

impl Ord for Integer {
    fn cmp(&self, other: &Integer) -> Ordering {
        check_compatible(self.0, other.0, "comparison");
        typed!(cast => self.0, false, {
            (cast(self.1).cmp(&cast(other.1)))
        })
    }
}

impl PartialOrd for Integer {
    fn partial_cmp(&self, other: &Integer) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for Integer {}
impl PartialEq for Integer {
    fn eq(&self, other: &Integer) -> bool {
        check_compatible(self.0, other.0, "comparison");
        self.1 == other.1
    }
}

impl Not for Integer {
    type Output = Integer;

    fn not(self) -> Integer {
        Integer(self.0, typed!(cast => self.0, false, { !cast(self.1) as u64 }))
    }
}

impl DataType {
    /// Word representation of the data type.
    pub fn name(&self) -> &'static str {
        match self {
            N8 => "byte",
            N16 => "word",
            N32 => "dword",
            N64 => "qword",
        }
    }

    /// Number of bytes this data type needs to be stored.
    pub fn bytes(&self) -> usize {
        match self {
            N8 => 1,
            N16 => 2,
            N32 => 4,
            N64 => 8,
        }
    }

    /// Number of bits this data types needs to be stored.
    pub fn bits(&self) -> usize {
        self.bytes() * 8
    }
}

impl Display for DataType {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            N8 => "n8",
            N16 => "n16",
            N32 => "n32",
            N64 => "n64",
        })
    }
}

/// Arithemtic operation flags returned by some functions on integers.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
pub struct Flags {
    pub zero: bool,
    pub sign: bool,
    pub overflow: bool,
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags() {
        assert_eq!(Integer(N8, 150).flagged_add(Integer(N8, 100)).1,
            Flags { zero: false, sign: true, overflow: false });

        assert_eq!(Integer(N8, -56i8 as u64).flagged_add(Integer(N8, 56)).1,
            Flags { zero: true, sign: false, overflow: false });

        assert_eq!(Integer(N8, 100).flagged_add(Integer(N8, 100)).1,
            Flags { zero: false, sign: true, overflow: true });

        assert_eq!(Integer(N8, 20).flagged_add(Integer(N8, 40)).1,
            Flags { zero: false, sign: false, overflow: false });

        assert_eq!(Integer(N32, 3).flagged_sub(Integer(N32, 4)).1,
            Flags { zero: false, sign: true, overflow: false });

        assert_eq!(Integer(N8, 130).flagged_sub(Integer(N8, 10)).1,
            Flags { zero: false, sign: false, overflow: true });
    }

    #[test]
    fn bytes() {
        assert_eq!(Integer(N8, 1).to_bytes(), vec![1]);
        assert_eq!(Integer(N16, 0xabef).to_bytes(), vec![0xef, 0xab]);
    }
}
