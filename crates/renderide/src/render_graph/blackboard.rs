//! Typed key-value scratch for cross-pass data sharing within one frame.
//!
//! A [`Blackboard`] allows graph passes to share structured CPU-side state without storing it
//! inside the pass itself or routing it through a shared mutable "god bag". Passes insert values
//! under a typed slot key and retrieve them by the same key type.
//!
//! Slot keys are zero-sized types (ZSTs) that implement [`BlackboardSlot`]. The key's `type Value`
//! determines what is inserted and retrieved. [`std::any::TypeId`] is used as the runtime key so
//! lookup is a single [`hashbrown::HashMap`] probe.
//!
//! ## Scoping
//!
//! Two independent instances are used per tick:
//! - **Frame blackboard** — shared across all views; populated by [`super::pass::PassPhase::FrameGlobal`] passes.
//! - **View blackboard** — one instance per [`super::compiled::FrameView`]; populated and consumed by
//!   [`super::pass::PassPhase::PerView`] passes for that view.

use std::any::{Any, TypeId};

use hashbrown::HashMap;

use super::resources::ImportedTextureHandle;

/// Marker trait for blackboard slot keys.
///
/// Implement this on a ZST to define a typed slot. The associated `Value` type is what is
/// stored and retrieved. All values must be `Send + 'static` to be safe across threads and
/// frames.
///
/// # Example
///
/// ```ignore
/// struct MyDataSlot;
/// impl BlackboardSlot for MyDataSlot {
///     type Value = MyData;
/// }
/// ```
pub trait BlackboardSlot: 'static {
    /// Type stored under this slot.
    type Value: Send + 'static;
}

/// Typed key-value store for one frame scope.
///
/// Values are boxed as `dyn Any + Send` and retrieved by downcasting from the [`TypeId`] of the
/// slot key type. Insertion replaces any existing value for the same slot.
#[derive(Default)]
pub struct Blackboard {
    slots: HashMap<TypeId, Box<dyn Any + Send>>,
}

impl Blackboard {
    /// Creates an empty blackboard.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts `value` under slot `S`, replacing any previous value.
    pub fn insert<S: BlackboardSlot>(&mut self, value: S::Value) {
        self.slots.insert(TypeId::of::<S>(), Box::new(value));
    }

    /// Returns a shared reference to the value stored under slot `S`, or [`None`] if absent.
    pub fn get<S: BlackboardSlot>(&self) -> Option<&S::Value> {
        self.slots
            .get(&TypeId::of::<S>())
            .and_then(|v| v.downcast_ref::<S::Value>())
    }

    /// Returns a mutable reference to the value stored under slot `S`, or [`None`] if absent.
    pub fn get_mut<S: BlackboardSlot>(&mut self) -> Option<&mut S::Value> {
        self.slots
            .get_mut(&TypeId::of::<S>())
            .and_then(|v| v.downcast_mut::<S::Value>())
    }

    /// Removes and returns the value stored under slot `S`, or [`None`] if absent.
    pub fn take<S: BlackboardSlot>(&mut self) -> Option<S::Value> {
        self.slots
            .remove(&TypeId::of::<S>())
            .and_then(|v| v.downcast::<S::Value>().ok().map(|b| *b))
    }

    /// Returns `true` when slot `S` has a stored value.
    pub fn contains<S: BlackboardSlot>(&self) -> bool {
        self.slots.contains_key(&TypeId::of::<S>())
    }

    /// Removes all stored values.
    pub fn clear(&mut self) {
        self.slots.clear();
    }

    /// Number of stored slots.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether the blackboard has no stored slots.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

/// Blackboard slot reserving the per-view screen-space motion-vector texture for temporal
/// techniques (TAA, motion blur, temporal denoising).
///
/// **No pass produces this slot today.** The slot is declared so downstream work can land a
/// velocity prepass without coordinating a new blackboard key across the codebase in the same
/// change. `Value` is the [`ImportedTextureHandle`] of the `Rg16Float` velocity target; the
/// consumer resolves the actual `wgpu::TextureView` via the graph-resources context at encode
/// time.
///
/// Lives on the per-view blackboard because motion vectors are screen-space and view-specific.
pub struct FrameMotionVectorsSlot;
impl BlackboardSlot for FrameMotionVectorsSlot {
    type Value = ImportedTextureHandle;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FooSlot;
    impl BlackboardSlot for FooSlot {
        type Value = u32;
    }

    struct BarSlot;
    impl BlackboardSlot for BarSlot {
        type Value = String;
    }

    #[test]
    fn insert_and_get_typed_slot() {
        let mut bb = Blackboard::new();
        bb.insert::<FooSlot>(42u32);
        assert_eq!(bb.get::<FooSlot>(), Some(&42u32));
    }

    #[test]
    fn missing_slot_returns_none() {
        let bb = Blackboard::new();
        assert_eq!(bb.get::<FooSlot>(), None);
    }

    #[test]
    fn different_slot_types_are_independent() {
        let mut bb = Blackboard::new();
        bb.insert::<FooSlot>(7u32);
        bb.insert::<BarSlot>("hello".to_string());
        assert_eq!(bb.get::<FooSlot>(), Some(&7u32));
        assert_eq!(bb.get::<BarSlot>(), Some(&"hello".to_string()));
    }

    #[test]
    fn insert_replaces_previous_value() {
        let mut bb = Blackboard::new();
        bb.insert::<FooSlot>(1u32);
        bb.insert::<FooSlot>(99u32);
        assert_eq!(bb.get::<FooSlot>(), Some(&99u32));
    }

    #[test]
    fn take_removes_value() {
        let mut bb = Blackboard::new();
        bb.insert::<FooSlot>(55u32);
        let taken = bb.take::<FooSlot>();
        assert_eq!(taken, Some(55u32));
        assert_eq!(bb.get::<FooSlot>(), None);
    }

    #[test]
    fn take_returns_none_when_absent() {
        let mut bb = Blackboard::new();
        assert_eq!(bb.take::<FooSlot>(), None);
    }

    #[test]
    fn get_mut_allows_mutation() {
        let mut bb = Blackboard::new();
        bb.insert::<FooSlot>(10u32);
        *bb.get_mut::<FooSlot>().unwrap() = 20u32;
        assert_eq!(bb.get::<FooSlot>(), Some(&20u32));
    }

    #[test]
    fn contains_reflects_presence() {
        let mut bb = Blackboard::new();
        assert!(!bb.contains::<FooSlot>());
        bb.insert::<FooSlot>(0);
        assert!(bb.contains::<FooSlot>());
        bb.take::<FooSlot>();
        assert!(!bb.contains::<FooSlot>());
    }

    #[test]
    fn clear_empties_all_slots() {
        let mut bb = Blackboard::new();
        bb.insert::<FooSlot>(1);
        bb.insert::<BarSlot>("x".into());
        bb.clear();
        assert!(bb.is_empty());
    }

    #[test]
    fn frame_motion_vectors_slot_type_is_insertable() {
        // Scaffolding-only: confirm the slot key + value type compile and roundtrip.
        // No producer writes this today; this test exists so the declaration doesn't bit-rot
        // before a velocity-prepass consumer lands.
        let mut bb = Blackboard::new();
        let handle = ImportedTextureHandle(0);
        bb.insert::<FrameMotionVectorsSlot>(handle);
        assert_eq!(bb.get::<FrameMotionVectorsSlot>().copied(), Some(handle));
    }
}
