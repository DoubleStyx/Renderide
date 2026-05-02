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
/// across frames. [`ImportSourceKind::PingPong`] references a slot by this id; a
/// [`crate::backend::HistoryRegistry`] owns the concrete resources.
///
/// Slots are identified by a stable `&'static str` id so subsystems can register their own slot
/// names without editing a centralized enum. Use [`HistorySlotId::new`] to declare new ids; the
/// associated constants here cover slots that already ship.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HistorySlotId(&'static str);

impl HistorySlotId {
    /// Hi-Z pyramid for a view -- the previous-frame depth pyramid used by GPU-side occlusion.
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

/// Generic import source shared by texture and buffer imports.
///
/// `F` carries the frame-resolved kind: [`FrameTargetRole`] for textures (swapchain / depth)
/// and [`BackendFrameBufferKind`] for buffers (lights, cluster tables, per-draw slab,
/// frame uniforms). The [`Self::PingPong`] variant carries a [`HistorySlotId`]
/// ([`&'static str`] newtype) so slot names stay readable in logs and registry errors. The
/// alternative -- an interned `u32` id -- would lose the debug name without meaningful payoff for
/// a type instantiated a handful of times at graph build.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ImportSourceKind<F> {
    /// Resolved from the frame target / backend frame resource at execute time.
    Frame(F),
    /// Externally owned resource view.
    External,
    /// Ping-pong history slot owned by backend history.
    PingPong(HistorySlotId),
}

/// Texture import source.
///
/// See [`ImportSourceKind`] for the underlying generic enum.
pub type ImportSource = ImportSourceKind<FrameTargetRole>;

/// Buffer import source.
///
/// See [`ImportSourceKind`] for the underlying generic enum.
pub type BufferImportSource = ImportSourceKind<BackendFrameBufferKind>;

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
