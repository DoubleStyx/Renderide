//! Frame-scoped GPU services owned behind the backend facade.
//!
//! This groups the resources that are advanced or consumed once per frame: frame bind groups,
//! mesh deformation compute state, the skin cache, and optional MSAA depth resolve helpers.

use std::sync::Arc;

use crate::gpu::{GpuLimits, MsaaDepthResolveResources};
use crate::mesh_deform::{GpuSkinCache, MeshDeformScratch, MeshPreprocessPipelines};
use crate::scene::RenderSpaceId;

use super::super::{FrameGpuBindingsError, FrameResourceManager};

/// Frame-global GPU resources and per-frame preprocess services.
pub(super) struct BackendFrameServices {
    /// Per-frame bind groups, light staging, and debug draw slab.
    pub(super) frame_resources: FrameResourceManager,
    /// Optional mesh skinning / blendshape compute pipelines after attach.
    mesh_preprocess: Option<MeshPreprocessPipelines>,
    /// Scratch buffers for mesh deformation compute after attach.
    mesh_deform_scratch: Option<MeshDeformScratch>,
    /// Arena-backed deformed vertex streams after attach.
    skin_cache: Option<GpuSkinCache>,
    /// MSAA depth -> R32F -> single-sample depth resolve resources when supported.
    msaa_depth_resolve: Option<Arc<MsaaDepthResolveResources>>,
}

impl BackendFrameServices {
    /// Creates frame services with no GPU resources attached.
    pub(super) fn new() -> Self {
        Self {
            frame_resources: FrameResourceManager::new(),
            mesh_preprocess: None,
            mesh_deform_scratch: None,
            skin_cache: None,
            msaa_depth_resolve: None,
        }
    }

    /// Allocates frame services that depend on the device.
    pub(super) fn attach(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gpu_limits: Arc<GpuLimits>,
    ) -> Result<(), FrameGpuBindingsError> {
        // Build a complete replacement before publishing it. Reusing the existing manager would
        // retain per-view bind groups and buffers created by the previous device.
        let mut replacement = Self::new();
        let max_buffer_size = gpu_limits.max_buffer_size();
        replacement.mesh_deform_scratch = Some(MeshDeformScratch::new(device, max_buffer_size));
        replacement
            .frame_resources
            .attach(device, queue, gpu_limits)?;
        replacement.skin_cache = Some(GpuSkinCache::new(device, max_buffer_size));
        match MeshPreprocessPipelines::new(device) {
            Ok(pipelines) => replacement.mesh_preprocess = Some(pipelines),
            Err(error) => {
                logger::warn!("mesh preprocess compute pipelines not created: {error}");
                replacement.mesh_preprocess = None;
            }
        }
        replacement.msaa_depth_resolve = MsaaDepthResolveResources::try_new(device).map(Arc::new);
        *self = replacement;
        Ok(())
    }

    /// Returns the canonical immutable-geometry store allocated with frame GPU resources.
    pub(super) fn shared_static_geometry_store(
        &self,
    ) -> Option<crate::graph_inputs::SharedStaticGeometryStore> {
        self.frame_resources.shared_static_geometry_store()
    }

    /// Resets per-tick coalescing, advances the skin-cache frame counter, and decays idle arenas.
    pub(super) fn reset_for_tick(&mut self, device: Option<&wgpu::Device>) {
        self.frame_resources.reset_light_prep_for_tick();
        let arenas_reset = if let Some(cache) = self.skin_cache.as_mut() {
            cache.advance_frame();
            device
                .map(|device| cache.maintain_capacity(device))
                .unwrap_or(false)
        } else {
            false
        };
        // Recreated arenas can alias stale deform bind groups, so drop those caches.
        if arenas_reset && let Some(scratch) = self.mesh_deform_scratch.as_mut() {
            scratch.invalidate_after_arena_reset();
        }
    }

    /// Removes mesh-deform skin-cache entries for closed render spaces.
    pub(super) fn purge_skin_cache_spaces(&mut self, spaces: &[RenderSpaceId]) -> usize {
        let Some(cache) = self.skin_cache.as_mut() else {
            return 0;
        };
        let mut removed = 0usize;
        for &space_id in spaces {
            removed = removed.saturating_add(cache.remove_space(space_id));
        }
        removed
    }

    /// Optional MSAA depth resolve resources.
    pub(super) fn msaa_depth_resolve(&self) -> Option<Arc<MsaaDepthResolveResources>> {
        self.msaa_depth_resolve.clone()
    }

    /// Disjoint frame-service slices required while assembling graph access.
    pub(super) fn graph_access_slices(
        &mut self,
    ) -> (
        &mut FrameResourceManager,
        Option<&MeshPreprocessPipelines>,
        Option<&mut MeshDeformScratch>,
        Option<&mut GpuSkinCache>,
    ) {
        (
            &mut self.frame_resources,
            self.mesh_preprocess.as_ref(),
            self.mesh_deform_scratch.as_mut(),
            self.skin_cache.as_mut(),
        )
    }

    /// `true` when mesh preprocess compute pipelines are available.
    pub(super) fn mesh_preprocess_enabled(&self) -> bool {
        self.mesh_preprocess.is_some()
    }

    /// `true` when MSAA depth resolve resources are available.
    pub(super) fn msaa_depth_resolve_enabled(&self) -> bool {
        self.msaa_depth_resolve.is_some()
    }
}
