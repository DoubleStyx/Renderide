//! Internal types backing [`super::GraphBuilder`] declarations.

use super::super::ids::{GroupId, PassId};
use super::super::pass::{GroupScope, PassNode, PassSetup};
use super::super::resources::{
    TransientArrayLayers, TransientExtent, TransientSampleCount, TransientTextureFormat,
};

/// One pass entry in the builder's declaration list.
pub(crate) struct PassEntry {
    /// Group this pass belongs to.
    pub(crate) group: GroupId,
    /// The pass node (owns the pass trait object).
    pub(crate) pass: PassNode,
}

/// Internal group declaration.
#[derive(Clone, Debug)]
pub(crate) struct GroupEntry {
    /// Stable label for logging and debug HUD.
    pub(crate) name: &'static str,
    /// Whether passes in this group run once per frame or once per view.
    pub(crate) scope: GroupScope,
    /// This group must execute after these groups.
    pub(crate) after: Vec<GroupId>,
}

/// Compiled setup data for one pass indexed by its declaration position.
pub(super) struct SetupEntry {
    /// Pass id (position in declaration order).
    pub(super) id: PassId,
    /// Group id for this pass.
    pub(super) group: GroupId,
    /// Pass name (owned for error messages).
    pub(super) name: String,
    /// Compiled setup data from the pass's `setup()` call.
    pub(super) setup: PassSetup,
}

/// Aliasing key for transient textures: two handles can share a physical slot when their keys
/// match and their lifetimes are disjoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct TextureAliasKey {
    pub(super) format: TransientTextureFormat,
    pub(super) extent: TransientExtent,
    pub(super) mip_levels: u32,
    pub(super) sample_count: TransientSampleCount,
    pub(super) dimension: wgpu::TextureDimension,
    pub(super) array_layers: TransientArrayLayers,
    pub(super) usage_bits: u64,
}

/// Aliasing key for transient buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct BufferAliasKey {
    pub(super) size_policy: super::super::resources::BufferSizePolicy,
    pub(super) usage_bits: u64,
}
