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
//! - **Frame blackboard** -- shared across all views; populated by [`super::pass::PassPhase::FrameGlobal`] passes.
//! - **View blackboard** -- one instance per [`super::compiled::FrameView`]; populated and consumed by
//!   [`super::pass::PassPhase::PerView`] passes for that view.

use std::any::{Any, TypeId};

use hashbrown::HashMap;

#[cfg(test)]
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

/// Defines a zero-sized blackboard slot marker and its [`BlackboardSlot`] value type.
macro_rules! blackboard_slot {
    (
        $(#[$attr:meta])*
        $vis:vis $name:ident => $value:ty $(,)?
    ) => {
        $(#[$attr])*
        $vis struct $name;

        $(#[$attr])*
        impl $crate::render_graph::blackboard::BlackboardSlot for $name {
            type Value = $value;
        }
    };
}

pub(crate) use blackboard_slot;

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

    /// Moves all slots from `other` into this blackboard.
    ///
    /// Slots from `other` replace existing values with the same slot key.
    pub fn extend(&mut self, other: Self) {
        self.slots.extend(other.slots);
    }

    /// Returns a shared reference to the value stored under slot `S`, or [`None`] if absent.
    pub fn get<S: BlackboardSlot>(&self) -> Option<&S::Value> {
        self.slots
            .get(&TypeId::of::<S>())
            .and_then(|v| v.downcast_ref::<S::Value>())
    }

    /// Returns `true` when a value exists for the raw slot type id.
    pub(crate) fn contains_type_id(&self, type_id: TypeId) -> bool {
        self.slots.contains_key(&type_id)
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
    #[cfg(test)]
    pub fn contains<S: BlackboardSlot>(&self) -> bool {
        self.slots.contains_key(&TypeId::of::<S>())
    }

    /// Removes all stored values.
    #[cfg(test)]
    pub fn clear(&mut self) {
        self.slots.clear();
    }

    /// Whether the blackboard has no stored slots.
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

/// Generic command-count diagnostics captured by pass families during one graph scope.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphCommandStats {
    /// Logical draw items represented by the recorded work.
    pub draw_items: usize,
    /// Batched instance groups emitted by the recorded work.
    pub instance_batches: usize,
    /// Pipeline-specific draw submissions after material or pass expansion.
    pub pipeline_pass_submits: usize,
    /// Runtime graph passes skipped because their `should_record` predicate returned false.
    pub skipped_passes: usize,
    /// Logical raster passes that recorded draw work.
    pub recorded_raster_passes: usize,
    /// Logical compute passes that recorded dispatch or copy work.
    pub recorded_compute_passes: usize,
    /// Logical encoder passes that recorded mixed command work.
    pub recorded_encoder_passes: usize,
    /// WGPU render-pass encoders opened by graph-managed or helper full-screen work.
    pub opened_render_passes: usize,
    /// Explicit texture copies recorded by graph passes.
    pub copy_count: usize,
    /// Manual or attachment resolves recorded by graph passes.
    pub resolve_count: usize,
    /// Runtime estimate of attachment, copy, and resolve bandwidth in bytes.
    pub estimated_bandwidth_bytes: u64,
}

impl GraphCommandStats {
    /// Adds another command-count sample into this one, saturating each field.
    pub fn add(&mut self, other: Self) {
        self.draw_items = self.draw_items.saturating_add(other.draw_items);
        self.instance_batches = self.instance_batches.saturating_add(other.instance_batches);
        self.pipeline_pass_submits = self
            .pipeline_pass_submits
            .saturating_add(other.pipeline_pass_submits);
        self.skipped_passes = self.skipped_passes.saturating_add(other.skipped_passes);
        self.recorded_raster_passes = self
            .recorded_raster_passes
            .saturating_add(other.recorded_raster_passes);
        self.recorded_compute_passes = self
            .recorded_compute_passes
            .saturating_add(other.recorded_compute_passes);
        self.recorded_encoder_passes = self
            .recorded_encoder_passes
            .saturating_add(other.recorded_encoder_passes);
        self.opened_render_passes = self
            .opened_render_passes
            .saturating_add(other.opened_render_passes);
        self.copy_count = self.copy_count.saturating_add(other.copy_count);
        self.resolve_count = self.resolve_count.saturating_add(other.resolve_count);
        self.estimated_bandwidth_bytes = self
            .estimated_bandwidth_bytes
            .saturating_add(other.estimated_bandwidth_bytes);
    }

    /// Adds one runtime-skipped graph pass.
    pub fn record_skipped_pass(&mut self) {
        self.skipped_passes = self.skipped_passes.saturating_add(1);
    }

    /// Adds one recorded logical raster pass.
    pub fn record_raster_pass(&mut self) {
        self.recorded_raster_passes = self.recorded_raster_passes.saturating_add(1);
    }

    /// Adds one recorded logical compute pass.
    pub fn record_compute_pass(&mut self) {
        self.recorded_compute_passes = self.recorded_compute_passes.saturating_add(1);
    }

    /// Adds one recorded logical encoder pass.
    pub fn record_encoder_pass(&mut self) {
        self.recorded_encoder_passes = self.recorded_encoder_passes.saturating_add(1);
    }

    /// Adds one opened render pass.
    pub fn record_opened_render_pass(&mut self) {
        self.opened_render_passes = self.opened_render_passes.saturating_add(1);
    }

    /// Adds one explicit copy operation.
    pub fn record_copy(&mut self) {
        self.copy_count = self.copy_count.saturating_add(1);
    }

    /// Adds one explicit or manual resolve operation.
    pub fn record_resolve(&mut self) {
        self.resolve_count = self.resolve_count.saturating_add(1);
    }
}

blackboard_slot! {
    /// Blackboard slot for generic command-count diagnostics produced by pass families.
    pub GraphCommandStatsSlot => GraphCommandStats,
}

blackboard_slot! {
    #[cfg(test)]
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
    pub FrameMotionVectorsSlot => ImportedTextureHandle,
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

    blackboard_slot! {
        /// Macro-defined slot used to test the declarative slot helper.
        TestMacroSlot => u16,
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
    fn extend_moves_slots_and_replaces_collisions() {
        let mut bb = Blackboard::new();
        bb.insert::<FooSlot>(7u32);
        bb.insert::<BarSlot>("old".to_string());

        let mut other = Blackboard::new();
        other.insert::<BarSlot>("new".to_string());

        bb.extend(other);

        assert_eq!(bb.get::<FooSlot>(), Some(&7u32));
        assert_eq!(bb.get::<BarSlot>(), Some(&"new".to_string()));
    }

    #[test]
    fn insert_replaces_previous_value() {
        let mut bb = Blackboard::new();
        bb.insert::<FooSlot>(1u32);
        bb.insert::<FooSlot>(99u32);
        assert_eq!(bb.get::<FooSlot>(), Some(&99u32));
    }

    #[test]
    fn macro_defined_slot_uses_typed_insert_get_and_take() {
        let mut bb = Blackboard::new();
        bb.insert::<TestMacroSlot>(7);

        assert_eq!(bb.get::<TestMacroSlot>(), Some(&7));
        assert_eq!(bb.take::<TestMacroSlot>(), Some(7));
        assert_eq!(bb.get::<TestMacroSlot>(), None);
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
