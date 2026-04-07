//! Eight boolean flags packed into one byte for a compact wire representation.

/// Eight flags read from one byte: `bit0` is the least significant bit (same ordering as the wire format).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PackedBools {
    pub bit0: bool,
    pub bit1: bool,
    pub bit2: bool,
    pub bit3: bool,
    pub bit4: bool,
    pub bit5: bool,
    pub bit6: bool,
    pub bit7: bool,
}

impl PackedBools {
    /// Decodes a byte from the wire into eight flags.
    #[inline]
    pub fn from_byte(byte: u8) -> Self {
        Self {
            bit0: (byte & 1) != 0,
            bit1: (byte & 2) != 0,
            bit2: (byte & 4) != 0,
            bit3: (byte & 8) != 0,
            bit4: (byte & 0x10) != 0,
            bit5: (byte & 0x20) != 0,
            bit6: (byte & 0x40) != 0,
            bit7: (byte & 0x80) != 0,
        }
    }

    /// First two flags.
    #[inline]
    pub fn two(self) -> (bool, bool) {
        (self.bit0, self.bit1)
    }

    /// First three flags.
    #[inline]
    pub fn three(self) -> (bool, bool, bool) {
        (self.bit0, self.bit1, self.bit2)
    }

    /// First four flags.
    #[inline]
    pub fn four(self) -> (bool, bool, bool, bool) {
        (self.bit0, self.bit1, self.bit2, self.bit3)
    }

    /// First five flags.
    #[inline]
    pub fn five(self) -> (bool, bool, bool, bool, bool) {
        (self.bit0, self.bit1, self.bit2, self.bit3, self.bit4)
    }

    /// First six flags.
    #[inline]
    pub fn six(self) -> (bool, bool, bool, bool, bool, bool) {
        (
            self.bit0, self.bit1, self.bit2, self.bit3, self.bit4, self.bit5,
        )
    }

    /// First seven flags.
    #[inline]
    pub fn seven(self) -> (bool, bool, bool, bool, bool, bool, bool) {
        (
            self.bit0, self.bit1, self.bit2, self.bit3, self.bit4, self.bit5, self.bit6,
        )
    }

    /// All eight flags.
    #[inline]
    pub fn eight(self) -> (bool, bool, bool, bool, bool, bool, bool, bool) {
        (
            self.bit0, self.bit1, self.bit2, self.bit3, self.bit4, self.bit5, self.bit6, self.bit7,
        )
    }
}
