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
/// `Sync` is required so [`crate::gpu_pools::MeshPool`] can be shared across rayon threads during read-only draw prep.
pub trait StreamingPolicy: Send + Sync {
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

    /// Builds meta from host [`crate::shared::SetTexture3DProperties`].
    pub fn from_texture3d_props(props: &crate::shared::SetTexture3DProperties) -> Self {
        Self {
            tier: if props.apply_immediatelly || props.high_priority {
                ResidencyTier::Hot
            } else {
                ResidencyTier::Streaming
            },
            integration_urgent: props.apply_immediatelly,
            mipmap_bias: 0.0,
        }
    }

    /// Builds meta from host [`crate::shared::SetCubemapProperties`].
    pub fn from_cubemap_props(props: &crate::shared::SetCubemapProperties) -> Self {
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

#[cfg(test)]
mod tests {
    //! Unit tests for [`VramAccounting`] and [`TextureResidencyMeta`] host mapping.

    use super::{ResidencyTier, TextureResidencyMeta, VramAccounting, VramResourceKind};
    use crate::shared::{SetCubemapProperties, SetTexture2DProperties, SetTexture3DProperties};

    #[test]
    fn vram_accounting_tracks_mesh_and_texture_subtotals() {
        let mut a = VramAccounting::default();
        a.on_resident_added(VramResourceKind::Mesh, 100);
        a.on_resident_added(VramResourceKind::Texture, 200);
        assert_eq!(a.mesh_resident_bytes(), 100);
        assert_eq!(a.texture_resident_bytes(), 200);
        assert_eq!(a.total_resident_bytes(), 300);

        a.on_resident_removed(VramResourceKind::Mesh, 40);
        assert_eq!(a.mesh_resident_bytes(), 60);
        assert_eq!(a.texture_resident_bytes(), 200);
        assert_eq!(a.total_resident_bytes(), 260);
    }

    #[test]
    fn vram_accounting_saturates_total_on_add_overflow() {
        let mut a = VramAccounting::default();
        a.on_resident_added(VramResourceKind::Mesh, u64::MAX);
        a.on_resident_added(VramResourceKind::Texture, 1);
        assert_eq!(a.total_resident_bytes(), u64::MAX);
    }

    #[test]
    fn vram_accounting_saturates_subtract_at_zero() {
        let mut a = VramAccounting::default();
        a.on_resident_added(VramResourceKind::Mesh, 10);
        a.on_resident_removed(VramResourceKind::Mesh, 100);
        assert_eq!(a.mesh_resident_bytes(), 0);
        assert_eq!(a.total_resident_bytes(), 0);
    }

    #[test]
    fn texture_meta_from_2d_hot_when_urgent_or_high_priority() {
        let p = SetTexture2DProperties {
            apply_immediatelly: true,
            high_priority: false,
            mipmap_bias: -0.5,
            ..Default::default()
        };
        let m = TextureResidencyMeta::from_texture_props(&p);
        assert_eq!(m.tier, ResidencyTier::Hot);
        assert!(m.integration_urgent);
        assert_eq!(m.mipmap_bias, -0.5);

        let p2 = SetTexture2DProperties {
            apply_immediatelly: false,
            high_priority: true,
            ..Default::default()
        };
        let m2 = TextureResidencyMeta::from_texture_props(&p2);
        assert_eq!(m2.tier, ResidencyTier::Hot);
        assert!(!m2.integration_urgent);
    }

    #[test]
    fn texture_meta_from_2d_streaming_when_neither_urgent_nor_high_priority() {
        let p = SetTexture2DProperties {
            apply_immediatelly: false,
            high_priority: false,
            ..Default::default()
        };
        let m = TextureResidencyMeta::from_texture_props(&p);
        assert_eq!(m.tier, ResidencyTier::Streaming);
        assert!(!m.integration_urgent);
    }

    #[test]
    fn texture_meta_from_3d_matches_2d_tier_rules_and_zero_mipmap_bias() {
        let p = SetTexture3DProperties {
            apply_immediatelly: false,
            high_priority: true,
            ..Default::default()
        };
        let m = TextureResidencyMeta::from_texture3d_props(&p);
        assert_eq!(m.tier, ResidencyTier::Hot);
        assert!(!m.integration_urgent);
        assert_eq!(m.mipmap_bias, 0.0);

        let p2 = SetTexture3DProperties {
            apply_immediatelly: true,
            ..Default::default()
        };
        let m2 = TextureResidencyMeta::from_texture3d_props(&p2);
        assert_eq!(m2.tier, ResidencyTier::Hot);
        assert!(m2.integration_urgent);
    }

    #[test]
    fn texture_meta_from_cubemap_carries_mipmap_bias() {
        let p = SetCubemapProperties {
            apply_immediatelly: false,
            high_priority: false,
            mipmap_bias: 1.25,
            ..Default::default()
        };
        let m = TextureResidencyMeta::from_cubemap_props(&p);
        assert_eq!(m.tier, ResidencyTier::Streaming);
        assert_eq!(m.mipmap_bias, 1.25);
    }

    /// Defaults for streaming metadata must bias new resources toward [`ResidencyTier::Hot`] so
    /// assets are retained until an eviction policy opts them out.
    #[test]
    fn default_tiers_are_hot() {
        assert_eq!(ResidencyTier::default(), ResidencyTier::Hot);
        let meta = TextureResidencyMeta::default();
        assert_eq!(meta.tier, ResidencyTier::Hot);
        assert!(!meta.integration_urgent);
        assert_eq!(meta.mipmap_bias, 0.0);
        assert_eq!(super::MeshResidencyMeta::default().tier, ResidencyTier::Hot);
    }

    /// Subtracting from only one kind must not let another kind's subtotal underflow; per-kind
    /// tallies stay independent from the combined saturating total.
    #[test]
    fn per_kind_subtotals_are_independent_under_mixed_operations() {
        let mut a = VramAccounting::default();
        a.on_resident_added(VramResourceKind::Mesh, 50);
        a.on_resident_added(VramResourceKind::Texture, 50);
        a.on_resident_removed(VramResourceKind::Texture, 200);
        assert_eq!(a.mesh_resident_bytes(), 50);
        assert_eq!(a.texture_resident_bytes(), 0);
        assert_eq!(a.total_resident_bytes(), 0);
    }
}
