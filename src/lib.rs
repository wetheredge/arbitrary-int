#![cfg_attr(not(feature = "std"), no_std)]

mod lib {
    pub mod core {
        #[cfg(not(feature = "std"))]
        pub use core::*;
        #[cfg(feature = "std")]
        pub use std::*;
    }
}

use lib::core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl,
    ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};

struct CompileTimeAssert<const A: usize, const B: usize> {}

impl<const A: usize, const B: usize> CompileTimeAssert<A, B> {
    pub const SMALLER_OR_EQUAL: () = {
        assert!(A <= B);
    };
    pub const SMALLER_THAN: () = {
        assert!(A <= B);
    };
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Default, Ord, PartialOrd)]
pub struct UInt<T, const NUM_BITS: usize>(T);

impl<T, const NUM_BITS: usize> UInt<T, NUM_BITS>
where
    T: Copy
        + BitAnd<T, Output = T>
        + Sub<T, Output = T>
        + Shl<usize, Output = T>
        + Shr<usize, Output = T>
        + From<u8>,
{
    pub const fn value(&self) -> T {
        self.0
    }

    pub const unsafe fn new_unchecked(value: T) -> Self {
        Self(value)
    }

    fn mask() -> T {
        // It would be great if we could make this function const, but generic traits aren't compatible with
        // const fn
        // Also note that we have to use T::from(1) (as opposed to doing that at the end), as we
        // only require From(u8) in the generic constraints.
        let one = T::from(1);
        (one << NUM_BITS) - one
    }
}

// Next are specific implementations for u8, u16, u32, u64 and u128. A couple notes:
// - The existence of MAX also serves as a neat bounds-check for NUM_BITS: If NUM_BITS is too large,
//   the subtraction overflows which will fail to compile. This simplifies things a lot.
//   However, that only works if every constructor also uses MAX somehow (doing let _ = MAX is enough)

macro_rules! uint_impl {
    ($($type:ident),+) => {
        $(
            impl<const NUM_BITS: usize> UInt<$type, NUM_BITS> {
                /// Minimum value that can be represented by this type
                pub const MIN: Self = Self(0);

                /// Maximum value that can be represented by this type
                /// Note that the existence of MAX also serves as a bounds check: If NUM_BITS is > available bits,
                /// we will get a compiler error right here
                pub const MAX: Self = Self($type::MAX >> ($type::BITS as usize - NUM_BITS));

                /// Creates an instance. Panics if the given value is outside of the valid range
                pub const fn new(value: $type) -> Self {
                    assert!(value <= Self::MAX.0);

                    Self(value)
                }

                /// Extracts bits from a given value. The extract is equivalent to: `new((value >> start_bit) & MASK)`
                /// Unlike new, extract doesn't perform range-checking so it is slightly more efficient
                pub const fn extract(value: $type, start_bit: usize) -> Self {
                    assert!(start_bit + NUM_BITS <= $type::BITS as usize);
                    // Query MAX to ensure that we get a compiler error if the current definition is bogus (e.g. <u8, 9>)
                    let _ = Self::MAX;

                    Self((value >> start_bit) & Self::MAX.0)
                }

                /// Returns a UInt with a wider bit depth but with the same base data type
                pub const fn widen<const NUM_BITS_RESULT: usize>(
                    &self,
                ) -> UInt<$type, NUM_BITS_RESULT> {
                    let _ = CompileTimeAssert::<NUM_BITS, NUM_BITS_RESULT>::SMALLER_THAN;
                    // Query MAX of the result to ensure we get a compiler error if the current definition is bogus (e.g. <u8, 9>)
                    let _ = UInt::<$type, NUM_BITS_RESULT>::MAX;
                    UInt::<$type, NUM_BITS_RESULT>(self.0)
                }
            }
        )+
    };
}

uint_impl!(u8, u16, u32, u64, u128);

// Arithmetic implementations
impl<T, const NUM_BITS: usize> Add for UInt<T, NUM_BITS>
where
    T: PartialEq
        + Eq
        + Copy
        + BitAnd<T, Output = T>
        + Not<Output = T>
        + Add<T, Output = T>
        + Sub<T, Output = T>
        + Shr<usize, Output = T>
        + Shl<usize, Output = T>
        + From<u8>,
{
    type Output = UInt<T, NUM_BITS>;

    fn add(self, rhs: Self) -> Self::Output {
        let sum = self.0 + rhs.0;

        #[cfg(debug_assertions)]
        if (sum & !Self::mask()) != T::from(0) {
            panic!("attempt to add with overflow");
        }

        Self(sum & Self::mask())
    }
}

impl<T, const NUM_BITS: usize> AddAssign for UInt<T, NUM_BITS>
where
    T: PartialEq
        + Eq
        + Not<Output = T>
        + Copy
        + AddAssign<T>
        + BitAnd<T, Output = T>
        + BitAndAssign<T>
        + Sub<T, Output = T>
        + Shr<usize, Output = T>
        + Shl<usize, Output = T>
        + From<u8>,
{
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;

        #[cfg(debug_assertions)]
        if (self.0 & !Self::mask()) != T::from(0) {
            panic!("attempt to add with overflow");
        }

        self.0 &= Self::mask();
    }
}

impl<T, const NUM_BITS: usize> Sub for UInt<T, NUM_BITS>
where
    T: Copy
        + BitAnd<T, Output = T>
        + Sub<T, Output = T>
        + Shl<usize, Output = T>
        + Shr<usize, Output = T>
        + From<u8>,
{
    type Output = UInt<T, NUM_BITS>;

    fn sub(self, rhs: Self) -> Self::Output {
        // No need for extra overflow checking as the regular minus operator already handles it for us
        Self((self.0 - rhs.0) & Self::mask())
    }
}

impl<T, const NUM_BITS: usize> SubAssign for UInt<T, NUM_BITS>
where
    T: Copy
        + SubAssign<T>
        + BitAnd<T, Output = T>
        + BitAndAssign<T>
        + Sub<T, Output = T>
        + Shl<usize, Output = T>
        + Shr<usize, Output = T>
        + From<u8>,
{
    fn sub_assign(&mut self, rhs: Self) {
        // No need for extra overflow checking as the regular minus operator already handles it for us
        self.0 -= rhs.0;
        self.0 &= Self::mask();
    }
}

impl<T, const NUM_BITS: usize> BitAnd for UInt<T, NUM_BITS>
where
    T: Copy
        + BitAnd<T, Output = T>
        + Sub<T, Output = T>
        + Shl<usize, Output = T>
        + Shr<usize, Output = T>
        + From<u8>,
{
    type Output = UInt<T, NUM_BITS>;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl<T, const NUM_BITS: usize> BitAndAssign for UInt<T, NUM_BITS>
where
    T: Copy + BitAndAssign<T> + Sub<T, Output = T> + Shl<usize, Output = T> + From<u8>,
{
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl<T, const NUM_BITS: usize> BitOr for UInt<T, NUM_BITS>
where
    T: Copy + BitOr<T, Output = T> + Sub<T, Output = T> + Shl<usize, Output = T> + From<u8>,
{
    type Output = UInt<T, NUM_BITS>;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl<T, const NUM_BITS: usize> BitOrAssign for UInt<T, NUM_BITS>
where
    T: Copy + BitOrAssign<T> + Sub<T, Output = T> + Shl<usize, Output = T> + From<u8>,
{
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl<T, const NUM_BITS: usize> BitXor for UInt<T, NUM_BITS>
where
    T: Copy + BitXor<T, Output = T> + Sub<T, Output = T> + Shl<usize, Output = T> + From<u8>,
{
    type Output = UInt<T, NUM_BITS>;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl<T, const NUM_BITS: usize> BitXorAssign for UInt<T, NUM_BITS>
where
    T: Copy + BitXorAssign<T> + Sub<T, Output = T> + Shl<usize, Output = T> + From<u8>,
{
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl<T, const NUM_BITS: usize> Not for UInt<T, NUM_BITS>
where
    T: Copy
        + BitAnd<T, Output = T>
        + BitXor<T, Output = T>
        + Sub<T, Output = T>
        + Shl<usize, Output = T>
        + Shr<usize, Output = T>
        + From<u8>,
{
    type Output = UInt<T, NUM_BITS>;

    fn not(self) -> Self::Output {
        Self(self.0 ^ Self::mask())
    }
}

impl<T, TSHIFTBITS, const NUM_BITS: usize> Shl<TSHIFTBITS> for UInt<T, NUM_BITS>
where
    T: Copy
        + BitAnd<T, Output = T>
        + Shl<TSHIFTBITS, Output = T>
        + Sub<T, Output = T>
        + Shl<usize, Output = T>
        + Shr<usize, Output = T>
        + From<u8>,
{
    type Output = UInt<T, NUM_BITS>;

    fn shl(self, rhs: TSHIFTBITS) -> Self::Output {
        Self((self.0 << rhs) & Self::mask())
    }
}

impl<T, TSHIFTBITS, const NUM_BITS: usize> ShlAssign<TSHIFTBITS> for UInt<T, NUM_BITS>
where
    T: Copy
        + BitAnd<T, Output = T>
        + BitAndAssign<T>
        + ShlAssign<TSHIFTBITS>
        + Sub<T, Output = T>
        + Shr<usize, Output = T>
        + Shl<usize, Output = T>
        + From<u8>,
{
    fn shl_assign(&mut self, rhs: TSHIFTBITS) {
        self.0 <<= rhs;
        self.0 &= Self::mask();
    }
}

impl<T, TSHIFTBITS, const NUM_BITS: usize> Shr<TSHIFTBITS> for UInt<T, NUM_BITS>
where
    T: Copy + Shr<TSHIFTBITS, Output = T> + Sub<T, Output = T> + Shl<usize, Output = T> + From<u8>,
{
    type Output = UInt<T, NUM_BITS>;

    fn shr(self, rhs: TSHIFTBITS) -> Self::Output {
        Self(self.0 >> rhs)
    }
}

impl<T, TSHIFTBITS, const NUM_BITS: usize> ShrAssign<TSHIFTBITS> for UInt<T, NUM_BITS>
where
    T: Copy + ShrAssign<TSHIFTBITS> + Sub<T, Output = T> + Shl<usize, Output = T> + From<u8>,
{
    fn shr_assign(&mut self, rhs: TSHIFTBITS) {
        self.0 >>= rhs;
    }
}

// Conversions
macro_rules! from_impl {
    ($from:ty, [$($into:ty),+]) => {
        $(
            impl<const NUM_BITS: usize, const NUM_BITS_FROM: usize> From<UInt<$from, NUM_BITS_FROM>>
                for UInt<$into, NUM_BITS>
            {
                fn from(item: UInt<$from, NUM_BITS_FROM>) -> Self {
                    let _ = CompileTimeAssert::<NUM_BITS_FROM, NUM_BITS>::SMALLER_OR_EQUAL;
                    Self(item.0 as $into)
                }
            }
        )+
    };
}

from_impl!(u8, [u16, u32, u64, u128]);
from_impl!(u16, [u8, u32, u64, u128]);
from_impl!(u32, [u8, u16, u64, u128]);
from_impl!(u64, [u8, u16, u32, u128]);
from_impl!(u128, [u8, u32, u64, u16]);

// Define type aliases like u1, u63 and u80 using the smallest possible underlying data type.
// These are for convenience only - UInt<u32, 15> is still legal
macro_rules! type_alias {
    ($storage:ty, $(($name:ident, $bits:expr)),+) => {
        $( pub type $name = crate::UInt<$storage, $bits>; )+
    }
}

pub use aliases::*;

#[allow(non_camel_case_types)]
#[rustfmt::skip]
mod aliases {
    type_alias!(u8, (u1, 1), (u2, 2), (u3, 3), (u4, 4), (u5, 5), (u6, 6), (u7, 7));
    type_alias!(u16, (u9, 9), (u10, 10), (u11, 11), (u12, 12), (u13, 13), (u14, 14), (u15, 15));
    type_alias!(u32, (u17, 17), (u18, 18), (u19, 19), (u20, 20), (u21, 21), (u22, 22), (u23, 23), (u24, 24), (u25, 25), (u26, 26), (u27, 27), (u28, 28), (u29, 29), (u30, 30), (u31, 31));
    type_alias!(u64, (u33, 33), (u34, 34), (u35, 35), (u36, 36), (u37, 37), (u38, 38), (u39, 39), (u40, 40), (u41, 41), (u42, 42), (u43, 43), (u44, 44), (u45, 45), (u46, 46), (u47, 47), (u48, 48), (u49, 49), (u50, 50), (u51, 51), (u52, 52), (u53, 53), (u54, 54), (u55, 55), (u56, 56), (u57, 57), (u58, 58), (u59, 59), (u60, 60), (u61, 61), (u62, 62), (u63, 63));
    type_alias!(u128, (u65, 65), (u66, 66), (u67, 67), (u68, 68), (u69, 69), (u70, 70), (u71, 71), (u72, 72), (u73, 73), (u74, 74), (u75, 75), (u76, 76), (u77, 77), (u78, 78), (u79, 79), (u80, 80), (u81, 81), (u82, 82), (u83, 83), (u84, 84), (u85, 85), (u86, 86), (u87, 87), (u88, 88), (u89, 89), (u90, 90), (u91, 91), (u92, 92), (u93, 93), (u94, 94), (u95, 95), (u96, 96), (u97, 97), (u98, 98), (u99, 99), (u100, 100), (u101, 101), (u102, 102), (u103, 103), (u104, 104), (u105, 105), (u106, 106), (u107, 107), (u108, 108), (u109, 109), (u110, 110), (u111, 111), (u112, 112), (u113, 113), (u114, 114), (u115, 115), (u116, 116), (u117, 117), (u118, 118), (u119, 119), (u120, 120), (u121, 121), (u122, 122), (u123, 123), (u124, 124), (u125, 125), (u126, 126), (u127, 127));
}
