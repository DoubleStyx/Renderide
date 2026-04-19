//! Enumerations serialized as `i32` list elements, matching the host’s enum value lists.

/// Enum-like type stored on the wire as a signed 32-bit integer per entry.
pub trait EnumRepr: Copy {
    /// Underlying discriminant for this variant.
    fn as_i32(self) -> i32;

    /// Reconstructs a value from wire discriminant (caller defines invalid mapping behavior).
    fn from_i32(i: i32) -> Self;
}
