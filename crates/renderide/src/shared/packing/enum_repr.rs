//! Trait for enums with integer representation, used for list serialization.

/// Types that can be serialized as their underlying integer representation.
/// Used for enum value lists where each element is written as i32.
pub trait EnumRepr: Copy {
    /// Returns the integer representation of this value.
    fn as_i32(self) -> i32;

    /// Constructs a value from its integer representation.
    fn from_i32(i: i32) -> Self;
}
