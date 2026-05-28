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

use std::any::{Any, TypeId, type_name};
use std::sync::Arc;

use hashbrown::HashMap;
use parking_lot::Mutex;

use super::pass::BlackboardAccessDecl;
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
pub struct Blackboard {
    slots: HashMap<TypeId, Arc<dyn Any + Send + Sync>>,
    access_validation: Mutex<Option<BlackboardAccessValidation>>,
}

impl Blackboard {
    /// Creates an empty blackboard.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts `value` under slot `S`, replacing any previous value.
    pub fn insert<S: BlackboardSlot>(&mut self, value: S::Value) {
        self.record_access::<S>(BlackboardRuntimeAccessKind::Write);
        self.insert_untracked::<S>(value);
    }

    /// Inserts `value` without recording a pass-facing blackboard access.
    pub(crate) fn insert_untracked<S: BlackboardSlot>(&mut self, value: S::Value) {
        self.slots.insert(TypeId::of::<S>(), Arc::new(value));
    }

    /// Moves all slots from `other` into this blackboard.
    ///
    /// Slots from `other` replace existing values with the same slot key.
    pub fn extend(&mut self, other: Self) {
        self.slots.extend(other.slots);
    }

    /// Creates a shallow read-only snapshot of the current slots.
    ///
    /// Slot payloads are reference-counted so parallel recording workers can read values without
    /// cloning the underlying payload. Mutable access to shared values returns [`None`], which is
    /// paired with runtime access validation to keep undeclared writes visible.
    pub(crate) fn clone_read_only(&self) -> Self {
        Self {
            slots: self.slots.clone(),
            access_validation: Mutex::new(None),
        }
    }

    /// Returns a shared reference to the value stored under slot `S`, or [`None`] if absent.
    pub fn get<S: BlackboardSlot>(&self) -> Option<&S::Value> {
        self.record_access::<S>(BlackboardRuntimeAccessKind::Read);
        self.get_untracked::<S>()
    }

    /// Returns a shared reference without recording a pass-facing blackboard access.
    pub(crate) fn get_untracked<S: BlackboardSlot>(&self) -> Option<&S::Value> {
        self.slots
            .get(&TypeId::of::<S>())
            .and_then(|v| v.as_ref().downcast_ref::<S::Value>())
    }

    /// Returns `true` when a value exists for the raw slot type id.
    pub(crate) fn contains_type_id(&self, type_id: TypeId) -> bool {
        self.slots.contains_key(&type_id)
    }

    /// Returns a mutable reference to the value stored under slot `S`, or [`None`] if absent.
    pub fn get_mut<S: BlackboardSlot>(&mut self) -> Option<&mut S::Value> {
        self.record_access::<S>(BlackboardRuntimeAccessKind::ReadWrite);
        self.get_mut_untracked::<S>()
    }

    /// Returns a mutable reference without recording a pass-facing blackboard access.
    pub(crate) fn get_mut_untracked<S: BlackboardSlot>(&mut self) -> Option<&mut S::Value> {
        self.slots
            .get_mut(&TypeId::of::<S>())
            .and_then(Arc::get_mut)
            .and_then(|v| v.downcast_mut::<S::Value>())
    }

    /// Removes and returns the value stored under slot `S`, or [`None`] if absent.
    pub fn take<S: BlackboardSlot>(&mut self) -> Option<S::Value> {
        self.record_access::<S>(BlackboardRuntimeAccessKind::ReadWrite);
        self.slots
            .remove(&TypeId::of::<S>())
            .and_then(|v| Arc::downcast::<S::Value>(v).ok())
            .and_then(|v| Arc::try_unwrap(v).ok())
    }

    /// Starts collecting pass-facing blackboard accesses for one graph pass.
    pub(crate) fn begin_access_validation(
        &self,
        pass_name: &str,
        declared_accesses: &[BlackboardAccessDecl],
    ) {
        let mut allowed = HashMap::new();
        for access in declared_accesses {
            let entry = allowed
                .entry(access.slot.type_id)
                .or_insert(AllowedBlackboardAccess {
                    type_name: access.slot.type_name,
                    reads: false,
                    writes: false,
                });
            entry.reads |= access.kind.reads();
            entry.writes |= access.kind.writes();
        }
        *self.access_validation.lock() = Some(BlackboardAccessValidation {
            pass_name: pass_name.to_owned(),
            allowed,
            violations: Vec::new(),
        });
    }

    /// Finishes collection and returns undeclared accesses observed since
    /// [`Self::begin_access_validation`].
    pub(crate) fn finish_access_validation(&self) -> Vec<BlackboardRuntimeAccessViolation> {
        self.access_validation
            .lock()
            .take()
            .map_or_else(Vec::new, |validation| validation.violations)
    }

    fn record_access<S: BlackboardSlot>(&self, access: BlackboardRuntimeAccessKind) {
        Self::record_access_in_validation::<S>(&mut self.access_validation.lock(), access);
    }

    /// Records one access while the validation state is already locked.
    fn record_access_in_validation<S: BlackboardSlot>(
        validation: &mut Option<BlackboardAccessValidation>,
        access: BlackboardRuntimeAccessKind,
    ) {
        let Some(validation) = validation.as_mut() else {
            return;
        };
        let type_id = TypeId::of::<S>();
        let allowed = validation.allowed.get(&type_id);
        let allowed_reads = allowed.is_some_and(|allowed| allowed.reads);
        let allowed_writes = allowed.is_some_and(|allowed| allowed.writes);
        let access_allowed = match access {
            BlackboardRuntimeAccessKind::Read => allowed_reads,
            BlackboardRuntimeAccessKind::Write => allowed_writes,
            BlackboardRuntimeAccessKind::ReadWrite => allowed_reads && allowed_writes,
        };
        if access_allowed {
            return;
        }
        let slot = allowed.map_or_else(|| type_name::<S>(), |allowed| allowed.type_name);
        if validation
            .violations
            .iter()
            .any(|violation| violation.slot == slot && violation.access == access)
        {
            return;
        }
        validation
            .violations
            .push(BlackboardRuntimeAccessViolation {
                pass: validation.pass_name.clone(),
                slot,
                access,
            });
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

impl Default for Blackboard {
    fn default() -> Self {
        Self {
            slots: HashMap::new(),
            access_validation: Mutex::new(None),
        }
    }
}

/// Runtime blackboard access category observed while pass code records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BlackboardRuntimeAccessKind {
    /// Shared read through [`Blackboard::get`].
    Read,
    /// Replacement write through [`Blackboard::insert`].
    Write,
    /// Mutable access or take, which both read and write slot state.
    ReadWrite,
}

impl BlackboardRuntimeAccessKind {
    /// Human-readable label for diagnostics.
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::ReadWrite => "read/write",
        }
    }
}

/// One pass-facing blackboard access that was not declared during setup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BlackboardRuntimeAccessViolation {
    /// Pass that touched the slot.
    pub(crate) pass: String,
    /// Slot type name.
    pub(crate) slot: &'static str,
    /// Access kind observed at runtime.
    pub(crate) access: BlackboardRuntimeAccessKind,
}

struct AllowedBlackboardAccess {
    type_name: &'static str,
    reads: bool,
    writes: bool,
}

struct BlackboardAccessValidation {
    pass_name: String,
    allowed: HashMap<TypeId, AllowedBlackboardAccess>,
    violations: Vec<BlackboardRuntimeAccessViolation>,
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
    /// Explicit texture copies skipped because they were no-ops for this frame.
    pub skipped_copy_count: usize,
    /// Manual or attachment resolves recorded by graph passes.
    pub resolve_count: usize,
    /// Manual or attachment resolves skipped because they were no-ops for this frame.
    pub skipped_resolve_count: usize,
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
        self.skipped_copy_count = self
            .skipped_copy_count
            .saturating_add(other.skipped_copy_count);
        self.resolve_count = self.resolve_count.saturating_add(other.resolve_count);
        self.skipped_resolve_count = self
            .skipped_resolve_count
            .saturating_add(other.skipped_resolve_count);
        self.estimated_bandwidth_bytes = self
            .estimated_bandwidth_bytes
            .saturating_add(other.estimated_bandwidth_bytes);
    }

    /// Returns whether command recording emitted GPU-visible work.
    pub fn has_recorded_work(&self) -> bool {
        self.recorded_raster_passes > 0
            || self.recorded_compute_passes > 0
            || self.recorded_encoder_passes > 0
            || self.opened_render_passes > 0
            || self.copy_count > 0
            || self.resolve_count > 0
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

    /// Adds one skipped explicit copy operation.
    pub fn record_skipped_copy(&mut self) {
        self.skipped_copy_count = self.skipped_copy_count.saturating_add(1);
    }

    /// Records whether a candidate explicit copy emitted GPU commands.
    pub fn record_copy_result(&mut self, recorded: bool) {
        if recorded {
            self.record_copy();
        } else {
            self.record_skipped_copy();
        }
    }

    /// Adds one explicit or manual resolve operation.
    pub fn record_resolve(&mut self) {
        self.resolve_count = self.resolve_count.saturating_add(1);
    }

    /// Adds one skipped explicit or manual resolve operation.
    pub fn record_skipped_resolve(&mut self) {
        self.skipped_resolve_count = self.skipped_resolve_count.saturating_add(1);
    }

    /// Records whether a candidate explicit or manual resolve emitted GPU commands.
    pub fn record_resolve_result(&mut self, recorded: bool) {
        if recorded {
            self.record_resolve();
        } else {
            self.record_skipped_resolve();
        }
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

    #[test]
    fn graph_command_stats_tracks_recorded_and_skipped_copy_resolve_results() {
        let mut stats = GraphCommandStats::default();

        stats.record_copy_result(true);
        stats.record_copy_result(false);
        stats.record_resolve_result(true);
        stats.record_resolve_result(false);

        assert_eq!(stats.copy_count, 1);
        assert_eq!(stats.skipped_copy_count, 1);
        assert_eq!(stats.resolve_count, 1);
        assert_eq!(stats.skipped_resolve_count, 1);
        assert!(stats.has_recorded_work());
    }

    #[test]
    fn graph_command_stats_skipped_commands_are_not_recorded_work() {
        let mut stats = GraphCommandStats::default();

        stats.record_skipped_pass();
        stats.record_copy_result(false);
        stats.record_resolve_result(false);

        assert!(!stats.has_recorded_work());
    }

    #[test]
    fn runtime_access_validation_reports_undeclared_read() {
        let bb = Blackboard::new();
        bb.begin_access_validation("read-pass", &[]);

        assert_eq!(bb.get::<FooSlot>(), None);

        assert_eq!(
            bb.finish_access_validation(),
            vec![BlackboardRuntimeAccessViolation {
                pass: "read-pass".to_owned(),
                slot: type_name::<FooSlot>(),
                access: BlackboardRuntimeAccessKind::Read,
            }]
        );
    }

    #[test]
    fn runtime_access_validation_accepts_declared_mutation() {
        let mut bb = Blackboard::new();
        bb.insert_untracked::<FooSlot>(1);
        let declared = [
            BlackboardAccessDecl::new::<FooSlot>(
                crate::render_graph::pass::params::BlackboardAccessKind::RequiredRead,
            ),
            BlackboardAccessDecl::new::<FooSlot>(
                crate::render_graph::pass::params::BlackboardAccessKind::Write,
            ),
        ];

        bb.begin_access_validation("mutate-pass", &declared);
        *bb.get_mut::<FooSlot>().expect("declared slot") = 2;

        assert!(bb.finish_access_validation().is_empty());
        assert_eq!(bb.get_untracked::<FooSlot>(), Some(&2));
    }

    #[test]
    fn read_only_clone_shares_reads_and_blocks_mutable_aliases() {
        let mut bb = Blackboard::new();
        bb.insert::<FooSlot>(7);

        let mut clone = bb.clone_read_only();

        assert_eq!(clone.get::<FooSlot>(), Some(&7));
        assert!(clone.get_mut::<FooSlot>().is_none());
        assert_eq!(bb.get::<FooSlot>(), Some(&7));
    }
}
