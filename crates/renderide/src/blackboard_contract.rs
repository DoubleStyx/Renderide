//! Typed blackboard slot contracts shared by graph inputs, passes, and render-graph validation.

/// Marker trait for blackboard slot keys.
///
/// Implement this on a zero-sized type to define a typed slot. The associated `Value` type is
/// what is stored and retrieved. All values must be `Send + 'static` to be safe across threads
/// and frames.
pub trait BlackboardSlot: 'static {
    /// Type stored under this slot.
    type Value: Send + Sync + 'static;
}

/// Defines a zero-sized blackboard slot marker and its [`BlackboardSlot`] value type.
macro_rules! blackboard_slot {
    (
        $(#[$attr:meta])*
        $vis:vis $name:ident => $value:ty $(,)?
    ) => {
        $(#[$attr])*
        $vis struct $name;

        $(#[$attr])*
        impl $crate::blackboard_contract::BlackboardSlot for $name {
            type Value = $value;
        }
    };
}

pub(crate) use blackboard_slot;
