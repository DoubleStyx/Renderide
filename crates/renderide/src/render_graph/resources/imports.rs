//! Imported render-graph resource declarations and history-slot identifiers.

use super::access::{BufferAccess, TextureAccess};

/// Frame target role resolved from the current [`super::super::FrameView`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FrameTargetRole {
    /// Frame color target (swapchain, XR array, or offscreen RT color).
    ColorAttachment,
    /// Frame depth target.
    DepthAttachment,
}

/// Stable identifier for a persistent graph history slot.
///
/// A **history slot** is a ping-pong pair of GPU resources (textures or buffers) that survive
/// across frames. [`ImportSource::PingPong`] and [`BufferImportSource::PingPong`] reference a
/// slot by this id; a [`crate::backend::HistoryRegistry`] owns the concrete resources.
///
/// Slots are identified by a stable `&'static str` id so subsystems can register their own slot
/// names without editing a centralized enum. Use [`HistorySlotId::new`] to declare new ids; the
/// associated constants here cover slots that already ship.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HistorySlotId(&'static str);

impl HistorySlotId {
    /// Hi-Z pyramid for a view — the previous-frame depth pyramid used by GPU-side occlusion.
    pub const HI_Z: Self = Self("hi_z");

    /// Declares a new history slot id with a stable name. The name must be unique across
    /// subsystems and stable across frames (it is the hash key of the backing resources).
    pub const fn new(name: &'static str) -> Self {
        Self(name)
    }

    /// Returns the stable string name of this slot.
    pub const fn name(self) -> &'static str {
        self.0
    }
}

/// Texture import source.
///
/// The [`Self::PingPong`] variant carries a [`HistorySlotId`] ([`&'static str`] newtype) so slot
/// names stay readable in logs and registry errors. The size-difference lint is allowed because
/// the alternative — an interned `u32` id — loses the debug name without meaningful payoff for a
/// type instantiated a handful of times at graph build.
#[expect(
    variant_size_differences,
    reason = "trade enum payload uniformity for debug-readable history slot names"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ImportSource {
    /// Resolved from the frame target at execute time.
    FrameTarget(FrameTargetRole),
    /// Externally owned texture view.
    External,
    /// Ping-pong history slot owned by backend history.
    PingPong(HistorySlotId),
}

/// Imported texture declaration.
#[derive(Clone, Debug, PartialEq)]
pub struct ImportedTextureDecl {
    /// Debug label.
    pub label: &'static str,
    /// Import source.
    pub source: ImportSource,
    /// Expected starting access.
    pub initial_access: TextureAccess,
    /// Expected final access.
    pub final_access: TextureAccess,
}

/// Known backend [`FrameResourceManager`](crate::backend::FrameResourceManager) buffers wired into the render graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackendFrameBufferKind {
    /// Packed lights storage for clustered forward.
    Lights,
    /// Per-tile light counts (clustered forward).
    ClusterLightCounts,
    /// Per-tile light index lists (clustered forward).
    ClusterLightIndices,
    /// Per-draw uniform slab (`@group(2)`).
    PerDrawSlab,
    /// Per-frame uniform buffer (`@group(0)`).
    FrameUniforms,
}

impl BackendFrameBufferKind {
    /// Debug label matching [`ImportedBufferDecl::label`] for this kind.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Lights => "lights",
            Self::ClusterLightCounts => "cluster_light_counts",
            Self::ClusterLightIndices => "cluster_light_indices",
            Self::PerDrawSlab => "per_draw_slab",
            Self::FrameUniforms => "frame_uniforms",
        }
    }
}

/// Buffer import source.
///
/// See [`ImportSource`] for the rationale behind the size-difference allow — the
/// [`HistorySlotId`] carries a debug-readable name over an opaque id on purpose.
#[expect(
    variant_size_differences,
    reason = "trade enum payload uniformity for debug-readable history slot names"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BufferImportSource {
    /// Backend frame resource buffer resolved at execute time.
    BackendFrameResource(BackendFrameBufferKind),
    /// Externally owned buffer.
    External,
    /// Ping-pong history slot.
    PingPong(HistorySlotId),
}

/// Imported buffer declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportedBufferDecl {
    /// Debug label.
    pub label: &'static str,
    /// Import source.
    pub source: BufferImportSource,
    /// Expected starting access.
    pub initial_access: BufferAccess,
    /// Expected final access.
    pub final_access: BufferAccess,
}
