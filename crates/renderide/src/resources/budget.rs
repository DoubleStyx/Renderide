//! VRAM accounting and streaming policy hooks (mesh + texture residency).

/// Kind of GPU resource for sub-budgets inside [`VramAccounting`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VramResourceKind {
    /// Triangle mesh buffers.
    Mesh,
    /// 2D texture (future: partial mips still counted approximately).
    Texture,
}

/// Running tally of GPU bytes tied to pooled resources.
#[derive(Debug, Default, Clone)]
pub struct VramAccounting {
    total_resident_bytes: u64,
    mesh_resident_bytes: u64,
    texture_resident_bytes: u64,
}

impl VramAccounting {
    /// Adds `bytes` when a resource becomes resident.
    pub fn on_resident_added(&mut self, kind: VramResourceKind, bytes: u64) {
        self.total_resident_bytes = self.total_resident_bytes.saturating_add(bytes);
        match kind {
            VramResourceKind::Mesh => {
                self.mesh_resident_bytes = self.mesh_resident_bytes.saturating_add(bytes);
            }
            VramResourceKind::Texture => {
                self.texture_resident_bytes = self.texture_resident_bytes.saturating_add(bytes);
            }
        }
    }

    /// Subtracts `bytes` when a resource is freed or evicted.
    pub fn on_resident_removed(&mut self, kind: VramResourceKind, bytes: u64) {
        self.total_resident_bytes = self.total_resident_bytes.saturating_sub(bytes);
        match kind {
            VramResourceKind::Mesh => {
                self.mesh_resident_bytes = self.mesh_resident_bytes.saturating_sub(bytes);
            }
            VramResourceKind::Texture => {
                self.texture_resident_bytes = self.texture_resident_bytes.saturating_sub(bytes);
            }
        }
    }

    /// Combined resident size (meshes + textures + future kinds).
    pub fn total_resident_bytes(&self) -> u64 {
        self.total_resident_bytes
    }

    /// Resident bytes for meshes only.
    pub fn mesh_resident_bytes(&self) -> u64 {
        self.mesh_resident_bytes
    }

    /// Resident bytes for textures only.
    pub fn texture_resident_bytes(&self) -> u64 {
        self.texture_resident_bytes
    }
}

/// Future **LRU / priority / budget clamp** / mipmap residency: suggest IDs to drop under pressure.
///
/// Default implementation is a no-op. Replace with a policy that tracks last frame touched,
/// material importance, or host hints when implementing streaming.
pub trait StreamingPolicy: Send {
    /// Called when a draw or upload touches a mesh (for future LRU).
    fn note_mesh_access(&mut self, _asset_id: i32) {}

    /// Called when a texture is sampled or uploaded (for future LRU / residency tiers).
    fn note_texture_access(&mut self, _asset_id: i32) {}

    /// Under memory pressure, return mesh asset IDs to evict (highest priority first).
    fn suggest_mesh_evictions(&self, _budget: &VramAccounting) -> Vec<i32> {
        Vec::new()
    }

    /// Under memory pressure, return texture id + **minimum mip level to keep resident**; mips
    /// finer than the returned level may be dropped or re-streamed later.
    ///
    /// Example: `(asset_id, 2)` means keep mips 2..N, evict 0–1.
    fn suggest_texture_mip_evictions(&self, _budget: &VramAccounting) -> Vec<(i32, u8)> {
        Vec::new()
    }
}

/// No-op policy until streaming is implemented.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopStreamingPolicy;

impl StreamingPolicy for NoopStreamingPolicy {}

/// Extension hook: classify resources for future tiered residency (`Hot`, `Streaming`, ...).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ResidencyTier {
    /// Always try to keep resident (hero assets, bound materials).
    #[default]
    Hot,
    /// May be evicted when over budget (background LODs).
    Streaming,
    /// Not required to stay resident across frames.
    Volatile,
}

/// Host-driven hints for future texture mip streaming (see [`StreamingPolicy::suggest_texture_mip_evictions`]).
#[derive(Clone, Debug)]
pub struct TextureResidencyMeta {
    /// Retention priority for streaming decisions.
    pub tier: ResidencyTier,
    /// From host `apply_immediatelly` / integration priority (best-effort).
    pub integration_urgent: bool,
    /// Mipmap bias from [`SetTexture2DProperties`](crate::shared::SetTexture2DProperties) (inform policy).
    pub mipmap_bias: f32,
}

impl Default for TextureResidencyMeta {
    fn default() -> Self {
        Self {
            tier: ResidencyTier::Hot,
            integration_urgent: false,
            mipmap_bias: 0.0,
        }
    }
}

impl TextureResidencyMeta {
    /// Builds meta from host texture properties (partial: `high_priority` may be unset in IPC v1).
    pub fn from_texture_props(props: &crate::shared::SetTexture2DProperties) -> Self {
        Self {
            tier: if props.apply_immediatelly || props.high_priority {
                ResidencyTier::Hot
            } else {
                ResidencyTier::Streaming
            },
            integration_urgent: props.apply_immediatelly,
            mipmap_bias: props.mipmap_bias,
        }
    }
}

/// Metadata for future mesh eviction (not enforced yet).
#[derive(Clone, Debug)]
pub struct MeshResidencyMeta {
    /// Retention priority for future mesh eviction.
    pub tier: ResidencyTier,
}

impl Default for MeshResidencyMeta {
    fn default() -> Self {
        Self {
            tier: ResidencyTier::Hot,
        }
    }
}
