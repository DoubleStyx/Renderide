use std::sync::Arc;

use hashbrown::HashSet;

use crate::backend::AssetTransferQueue;
use crate::backend::frame_gpu::{
    GpuReflectionProbeMetadata, REFLECTION_PROBE_ATLAS_FORMAT, ReflectionProbeSpecularResources,
};
use crate::gpu::GpuContext;
use crate::scene::{RenderSpaceId, SceneCoordinator};
use crate::skybox::ibl_cache::{
    SkyboxIblCache, SkyboxIblKey, build_key, clamp_face_size, mip_extent, mip_levels_for_edge,
};
use crate::{profiling, reflection_probes::ReflectionProbeSh2System};

use super::atlas::{AtlasCopyJob, ReflectionProbeAtlas, max_atlas_slots};
use super::captures::{
    RuntimeReflectionProbeCapture, RuntimeReflectionProbeCaptureKey,
    RuntimeReflectionProbeCaptureStore,
};
use super::selection::{ReflectionProbeFrameSelection, SpatialProbe};
use super::source::{metadata_for_spatial, resolve_probe_source, spatial_probe_for_state};

/// Default destination face size for reflection-probe IBL bakes.
const DEFAULT_REFLECTION_PROBE_FACE_SIZE: u32 = 256;
/// First atlas slot is reserved as a non-sampled black fallback.
const FIRST_PROBE_ATLAS_SLOT: u16 = 1;

/// Inputs for advancing specular reflection-probe IBL and selection state.
pub(crate) struct ReflectionProbeSpecularMaintainParams<'a> {
    /// GPU context used for IBL jobs and atlas writes.
    pub(crate) gpu: &'a mut GpuContext,
    /// Scene snapshot containing render spaces and reflection-probe entries.
    pub(crate) scene: &'a SceneCoordinator,
    /// Asset queues and pools used to resolve uploaded cubemaps.
    pub(crate) assets: &'a AssetTransferQueue,
    /// Render context used for reflection-probe world transform lookup.
    pub(crate) render_context: crate::shared::RenderingContext,
    /// SH2 projection service used when reflection-probe diffuse SH is enabled.
    pub(crate) sh2_system: &'a mut ReflectionProbeSh2System,
    /// Whether reflection probes should contribute SH2 indirect diffuse lighting.
    pub(crate) reflection_probe_sh2_enabled: bool,
}

/// Specular reflection-probe bake/cache/selection system.
pub struct ReflectionProbeSpecularSystem {
    ibl_cache: SkyboxIblCache,
    atlas: Option<ReflectionProbeAtlas>,
    resources: Option<ReflectionProbeSpecularResources>,
    selection: ReflectionProbeFrameSelection,
    captures: RuntimeReflectionProbeCaptureStore,
    version: u64,
}

impl Default for ReflectionProbeSpecularSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl ReflectionProbeSpecularSystem {
    /// Creates an empty reflection-probe specular system.
    #[must_use]
    pub fn new() -> Self {
        Self {
            ibl_cache: SkyboxIblCache::new(),
            atlas: None,
            resources: None,
            selection: ReflectionProbeFrameSelection::default(),
            captures: RuntimeReflectionProbeCaptureStore::default(),
            version: 1,
        }
    }

    /// Registers a completed runtime cubemap capture for an OnChanges reflection probe.
    pub(crate) fn register_runtime_capture(&mut self, capture: RuntimeReflectionProbeCapture) {
        self.captures.insert(capture);
    }

    /// Runtime OnChanges capture store used by SH2 task resolution.
    #[must_use]
    pub(crate) fn capture_store(&self) -> &RuntimeReflectionProbeCaptureStore {
        &self.captures
    }

    /// Purges reflection-probe GPU resources tied to closed render spaces.
    pub(crate) fn purge_render_space_resources(
        &mut self,
        spaces: &HashSet<RenderSpaceId>,
    ) -> usize {
        if spaces.is_empty() {
            return 0;
        }
        profiling::scope!("reflection_probes::specular::purge_render_space_resources");
        let captures = self.captures.purge_spaces(spaces);
        let ibl = self
            .ibl_cache
            .purge_where(|key| specular_ibl_key_matches_closed_spaces(key, spaces));
        let removed = captures.saturating_add(ibl);
        if removed > 0 {
            self.version = self.version.wrapping_add(1);
        }
        removed
    }

    /// Advances GPU bakes, updates the atlas, and rebuilds the CPU selection index.
    pub(crate) fn maintain(&mut self, mut params: ReflectionProbeSpecularMaintainParams<'_>) {
        profiling::scope!("reflection_probes::specular::maintain");
        self.ibl_cache.maintain_completed_jobs(params.gpu.device());
        let face_size = clamp_face_size(DEFAULT_REFLECTION_PROBE_FACE_SIZE, params.gpu.limits());
        let mut collected = CollectedProbeResources::default();

        self.collect_probe_resources(&mut params, face_size, &mut collected);
        self.captures.retain_active(&collected.active_capture_keys);
        self.ibl_cache
            .prune_completed_except(&collected.active_keys);
        collected.ready.sort_unstable_by_key(|probe| {
            (probe.identity.space_id.0, probe.identity.renderable_index)
        });
        self.sync_atlas_and_selection(params.gpu, face_size, collected.ready);
    }

    fn collect_probe_resources(
        &mut self,
        params: &mut ReflectionProbeSpecularMaintainParams<'_>,
        face_size: u32,
        collected: &mut CollectedProbeResources,
    ) {
        for space_id in params.scene.render_space_ids() {
            let Some(space) = params.scene.space(space_id) else {
                continue;
            };
            if !space.is_active() {
                continue;
            }
            for probe in space.reflection_probes() {
                let identity = ProbeIdentity {
                    space_id,
                    renderable_index: probe.renderable_index,
                };
                if probe.state.r#type == crate::shared::ReflectionProbeType::OnChanges {
                    collected
                        .active_capture_keys
                        .insert(RuntimeReflectionProbeCaptureKey {
                            space_id,
                            renderable_index: probe.renderable_index,
                        });
                }
                let Some(source) =
                    resolve_probe_source(space_id, probe, params.assets, &self.captures)
                else {
                    continue;
                };
                let key = build_key(&source, face_size);
                collected.active_keys.insert(key.clone());
                let sh2 = params
                    .reflection_probe_sh2_enabled
                    .then(|| params.sh2_system.ensure_ibl_source(space_id.0, &source))
                    .flatten();
                self.ibl_cache
                    .ensure_source(params.gpu, key.clone(), source);
                let Some(cube) = self.ibl_cache.completed_cube(&key) else {
                    continue;
                };
                if params.reflection_probe_sh2_enabled && sh2.is_none() {
                    continue;
                }
                let Some(spatial) = spatial_probe_for_state(
                    params.scene,
                    space_id,
                    probe,
                    params.render_context,
                    0,
                ) else {
                    continue;
                };
                let mut metadata = metadata_for_spatial(&spatial, probe.state, sh2.as_ref());
                metadata.params[1] = cube.mip_levels.saturating_sub(1) as f32;
                collected.ready.push(ReadyProbe {
                    identity,
                    key,
                    texture: cube.texture.clone(),
                    mip_levels: cube.mip_levels,
                    metadata,
                    spatial,
                });
            }
        }
    }

    /// Current frame-global GPU resources, if allocated.
    #[must_use]
    pub fn resources(&self) -> Option<ReflectionProbeSpecularResources> {
        self.resources.clone()
    }

    /// CPU selection snapshot used by draw collection.
    #[must_use]
    pub fn selection(&self) -> &ReflectionProbeFrameSelection {
        &self.selection
    }

    fn sync_atlas_and_selection(
        &mut self,
        gpu: &mut GpuContext,
        face_size: u32,
        mut ready: Vec<ReadyProbe>,
    ) {
        let max_slots = max_atlas_slots(gpu.limits());
        if max_slots <= 1 {
            self.selection.rebuild_spatial(Vec::new());
            return;
        }
        let usable_slots = usize::from(max_slots.saturating_sub(FIRST_PROBE_ATLAS_SLOT));
        if ready.len() > usable_slots {
            logger::warn!(
                "reflection probes: {} ready probes exceed atlas capacity {}; truncating",
                ready.len(),
                usable_slots
            );
            ready.truncate(usable_slots);
        }
        let used_slots = ready.len();
        let required_slots = (used_slots + usize::from(FIRST_PROBE_ATLAS_SLOT)).max(1);
        self.ensure_atlas(gpu.device(), face_size, required_slots as u16);

        let Some(atlas) = self.atlas.as_mut() else {
            self.selection.rebuild_spatial(Vec::new());
            return;
        };
        let mip_levels = atlas.mip_levels;
        let mut metadata = vec![GpuReflectionProbeMetadata::default(); atlas.capacity as usize];
        let mut copy_jobs = Vec::new();
        let mut selectable = Vec::with_capacity(ready.len());
        for (i, mut probe) in ready.into_iter().enumerate() {
            let slot = FIRST_PROBE_ATLAS_SLOT + i as u16;
            if atlas.keys[slot as usize].as_ref() != Some(&probe.key) {
                atlas.keys[slot as usize] = Some(probe.key.clone());
                copy_jobs.push(AtlasCopyJob {
                    slot,
                    texture: probe.texture.clone(),
                    mip_levels: probe.mip_levels.min(mip_levels),
                });
            }
            probe.spatial.atlas_index = slot;
            metadata[slot as usize] = probe.metadata;
            selectable.push((probe.identity.space_id, probe.spatial));
        }
        self.write_metadata(gpu.queue(), &metadata);
        self.encode_atlas_copies(gpu, face_size, mip_levels, copy_jobs);
        self.selection.rebuild_spatial(selectable);
    }

    fn ensure_atlas(&mut self, device: &wgpu::Device, face_size: u32, required_slots: u16) {
        let needs_new = self
            .atlas
            .as_ref()
            .is_none_or(|atlas| atlas.face_size != face_size || atlas.capacity < required_slots);
        if !needs_new {
            return;
        }
        let capacity = required_slots.max(2);
        let mip_levels = mip_levels_for_edge(face_size);
        let texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("reflection_probe_specular_atlas"),
            size: wgpu::Extent3d {
                width: face_size,
                height: face_size,
                depth_or_array_layers: u32::from(capacity) * 6,
            },
            mip_level_count: mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: REFLECTION_PROBE_ATLAS_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));
        let view = Arc::new(texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("reflection_probe_specular_atlas_view"),
            format: Some(REFLECTION_PROBE_ATLAS_FORMAT),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(mip_levels),
            base_array_layer: 0,
            array_layer_count: Some(u32::from(capacity) * 6),
        }));
        crate::profiling::note_resource_churn!(
            TextureView,
            "reflection_probes::specular_atlas_view"
        );
        let sampler = Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("reflection_probe_specular_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: mip_levels.saturating_sub(1) as f32,
            ..Default::default()
        }));
        let metadata_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reflection_probe_specular_metadata"),
            size: (usize::from(capacity) * size_of::<GpuReflectionProbeMetadata>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        crate::profiling::note_resource_churn!(
            Buffer,
            "reflection_probes::specular_metadata_buffer"
        );
        self.version = self.version.wrapping_add(1).max(1);
        self.resources = Some(ReflectionProbeSpecularResources {
            array_view: view,
            sampler,
            metadata_buffer,
            version: self.version,
        });
        self.atlas = Some(ReflectionProbeAtlas {
            texture,
            face_size,
            mip_levels,
            capacity,
            keys: vec![None; usize::from(capacity)],
        });
    }

    fn write_metadata(&self, queue: &wgpu::Queue, metadata: &[GpuReflectionProbeMetadata]) {
        let Some(resources) = &self.resources else {
            return;
        };
        queue.write_buffer(
            resources.metadata_buffer.as_ref(),
            0,
            bytemuck::cast_slice(metadata),
        );
    }

    fn encode_atlas_copies(
        &self,
        gpu: &mut GpuContext,
        face_size: u32,
        atlas_mips: u32,
        copy_jobs: Vec<AtlasCopyJob>,
    ) {
        profiling::scope!("reflection_probes::specular::atlas_copies");
        if copy_jobs.is_empty() {
            return;
        }
        let Some(atlas) = &self.atlas else {
            return;
        };
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("reflection_probe_atlas_copy"),
            });
        let mut profiler = gpu.take_gpu_profiler();
        let copy_query = profiler
            .as_ref()
            .map(|p| p.begin_query("reflection_probe_specular::atlas_copies", &mut encoder));
        for job in copy_jobs {
            let mips = job.mip_levels.min(atlas_mips);
            for mip in 0..mips {
                let extent = mip_extent(face_size, mip);
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: job.texture.as_ref(),
                        mip_level: mip,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: atlas.texture.as_ref(),
                        mip_level: mip,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: u32::from(job.slot) * 6,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: extent,
                        height: extent,
                        depth_or_array_layers: 6,
                    },
                );
            }
        }
        if let (Some(profiler), Some(query)) = (profiler.as_mut(), copy_query) {
            profiler.end_query(&mut encoder, query);
            profiler.resolve_queries(&mut encoder);
        }
        let command_buffer = {
            profiling::scope!("CommandEncoder::finish::reflection_probe_atlas_copy");
            encoder.finish()
        };
        gpu.restore_gpu_profiler(profiler);
        gpu.submit_frame_batch(vec![command_buffer], None, None);
    }
}

fn specular_ibl_key_matches_closed_spaces(
    key: &SkyboxIblKey,
    spaces: &HashSet<RenderSpaceId>,
) -> bool {
    match key {
        SkyboxIblKey::Cubemap { .. } | SkyboxIblKey::SolidColor { .. } => false,
        SkyboxIblKey::RuntimeCubemap {
            render_space_id, ..
        } => spaces.contains(&RenderSpaceId(*render_space_id)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_space_filter_matches_runtime_cubemap_space() {
        let mut spaces = HashSet::new();
        spaces.insert(RenderSpaceId(7));

        assert!(specular_ibl_key_matches_closed_spaces(
            &SkyboxIblKey::RuntimeCubemap {
                render_space_id: 7,
                renderable_index: 0,
                generation: 1,
                mip_levels: 1,
                storage_v_inverted: true,
                face_size: 128,
            },
            &spaces,
        ));
    }

    #[test]
    fn closed_space_filter_does_not_match_uploaded_asset_keys() {
        let mut spaces = HashSet::new();
        spaces.insert(RenderSpaceId(7));

        assert!(!specular_ibl_key_matches_closed_spaces(
            &SkyboxIblKey::Cubemap {
                material_asset_id: 21,
                material_generation: 1,
                route_hash: 99,
                asset_id: 7,
                allocation_generation: 1,
                mip_levels_resident: 1,
                content_generation: 1,
                storage_v_inverted: false,
                face_size: 128,
            },
            &spaces,
        ));
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct ProbeIdentity {
    space_id: RenderSpaceId,
    renderable_index: i32,
}

struct ReadyProbe {
    identity: ProbeIdentity,
    key: SkyboxIblKey,
    texture: Arc<wgpu::Texture>,
    mip_levels: u32,
    metadata: GpuReflectionProbeMetadata,
    spatial: SpatialProbe,
}

#[derive(Default)]
struct CollectedProbeResources {
    active_keys: HashSet<SkyboxIblKey>,
    active_capture_keys: HashSet<RuntimeReflectionProbeCaptureKey>,
    ready: Vec<ReadyProbe>,
}
