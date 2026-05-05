//! Unified IBL bake cache for the active skybox specular source.
//!
//! Owns one in-flight bake job tracker, three lazily-built mip-0 producer pipelines (analytic
//! procedural / gradient skies, host cubemaps, and Projection360 equirect Texture2Ds), and one
//! GGX convolve pipeline. For each new active skybox source the cache:
//!
//! 1. Allocates a fresh Rgba16Float cubemap with a full mip chain (`STORAGE_BINDING |
//!    TEXTURE_BINDING | COPY_SRC`).
//! 2. Records a mip-0 producer compute pass that converts the source into the cube's mip 0.
//! 3. Records one GGX convolve compute pass per mip in `1..N`, sampling the cube's mip 0 and
//!    writing each higher-roughness mip via solid-angle source-mip selection.
//! 4. Submits the encoder through [`GpuSubmitJobTracker`] and parks the cube in `pending` until
//!    the submit-completion callback promotes it to `completed`.
//!
//! The completed prefiltered cube is exposed as a
//! [`SkyboxSpecularEnvironmentSource::Cubemap`] for the frame-global skybox specular binding,
//! mirroring how Unity BiRP and Filament's `IBLPrefilterContext` unify all skybox source types
//! through a single GGX-prefiltered cube.

use std::sync::Arc;

use hashbrown::HashMap;
use thiserror::Error;

use crate::assets::asset_transfer_queue::AssetTransferQueue;
use crate::backend::frame_gpu::{SkyboxSpecularCubemapSource, SkyboxSpecularEnvironmentSource};
use crate::backend::gpu_jobs::{GpuJobResources, GpuSubmitJobTracker, SubmittedGpuJob};
use crate::gpu::{GpuContext, GpuLimits};
use crate::materials::MaterialSystem;
use crate::profiling::GpuProfilerHandle;
use crate::scene::SceneCoordinator;
use crate::skybox::specular::{SkyboxIblSource, resolve_active_main_skybox_ibl_source};

mod encode;
mod key;
mod pipeline;
mod resources;

use encode::{
    AnalyticEncodeContext, ConvolveEncodeContext, CubeEncodeContext, EquirectEncodeContext,
    encode_analytic_mip0, encode_convolve_mips, encode_cube_mip0, encode_equirect_mip0,
};
use key::build_key;
pub(crate) use key::{SkyboxIblKey, mip_levels_for_edge};
#[cfg(test)]
use key::{convolve_sample_count, hash_float4};
use pipeline::{
    ComputePipeline, analytic_layout_entries, ensure_pipeline, mip0_input_layout_entries,
};
use resources::{
    PendingBake, PendingBakeResources, PrefilteredCube, create_ibl_cube,
    create_mip0_cube_sample_view, prefiltered_sampler_state,
};

/// Maximum concurrent in-flight bakes; matches the analytic-only ceiling we used previously.
const MAX_IN_FLIGHT_IBL_BAKES: usize = 2;
/// Tick budget after which a missing submit-completion callback is treated as lost.
const MAX_PENDING_IBL_BAKE_AGE_FRAMES: u32 = 120;
/// Default destination cube face edge in texels (clamped to portable device limits).
const DEFAULT_IBL_FACE_SIZE: u32 = 256;

/// Clamps the configured cube face size against the device texture limit.
pub(crate) fn clamp_face_size(face_size: u32, limits: &GpuLimits) -> u32 {
    face_size.min(limits.max_texture_dimension_2d()).max(1)
}

/// Errors returned while preparing an IBL bake.
#[derive(Debug, Error)]
enum SkyboxIblBakeError {
    /// Embedded WGSL source was not available at compose time.
    #[error("embedded shader {0} not found")]
    MissingShader(&'static str),
}

/// Owns IBL bakes and serves the active prefiltered skybox specular cubemap.
pub(crate) struct SkyboxIblCache {
    /// Submit-completion tracker for in-flight bakes.
    jobs: GpuSubmitJobTracker<SkyboxIblKey>,
    /// In-flight prefiltered cubes retained until their submit callback fires.
    pending: HashMap<SkyboxIblKey, PendingBake>,
    /// Completed prefiltered cubes for the active skybox key.
    completed: HashMap<SkyboxIblKey, PrefilteredCube>,
    /// Lazily-built analytic mip-0 pipeline (re-uses the existing `skybox_bake_params` shader).
    analytic_pipeline: Option<ComputePipeline>,
    /// Lazily-built cube mip-0 pipeline.
    cube_pipeline: Option<ComputePipeline>,
    /// Lazily-built equirect mip-0 pipeline.
    equirect_pipeline: Option<ComputePipeline>,
    /// Lazily-built GGX convolve pipeline (cube -> cube via solid-angle source mip selection).
    convolve_pipeline: Option<ComputePipeline>,
    /// Cached input sampler used by all producers and the convolve pass.
    input_sampler: Option<Arc<wgpu::Sampler>>,
}

impl Default for SkyboxIblCache {
    fn default() -> Self {
        Self::new()
    }
}

impl SkyboxIblCache {
    /// Creates an empty IBL cache.
    pub(crate) fn new() -> Self {
        Self {
            jobs: GpuSubmitJobTracker::new(MAX_PENDING_IBL_BAKE_AGE_FRAMES),
            pending: HashMap::new(),
            completed: HashMap::new(),
            analytic_pipeline: None,
            cube_pipeline: None,
            equirect_pipeline: None,
            convolve_pipeline: None,
            input_sampler: None,
        }
    }

    /// Drains submit completions, prunes stale entries, and schedules a new bake when needed.
    pub(crate) fn maintain(
        &mut self,
        gpu: &mut GpuContext,
        scene: &SceneCoordinator,
        materials: &MaterialSystem,
        assets: &AssetTransferQueue,
    ) {
        profiling::scope!("skybox_ibl::maintain");
        let _ = gpu.device().poll(wgpu::PollType::Poll);
        {
            profiling::scope!("skybox_ibl::drain_completed_jobs");
            self.drain_completed_jobs();
        }
        let active = {
            profiling::scope!("skybox_ibl::resolve_active_source");
            resolve_active_main_skybox_ibl_source(scene, materials, assets)
        };
        let active_key = active
            .as_ref()
            .map(|source| build_key(source, clamp_face_size(DEFAULT_IBL_FACE_SIZE, gpu.limits())));
        self.prune_completed(active_key.as_ref());
        let (Some(source), Some(key)) = (active, active_key) else {
            return;
        };
        if self.completed.contains_key(&key)
            || self.pending.contains_key(&key)
            || self.jobs.contains_key(&key)
            || self.jobs.len() >= MAX_IN_FLIGHT_IBL_BAKES
        {
            return;
        }
        match self.schedule_bake(gpu, key, source) {
            Ok(()) => {}
            Err(e) => logger::warn!("skybox_ibl: bake failed: {e}"),
        }
    }

    /// Returns the prefiltered cube source for the active skybox, when ready.
    pub(crate) fn active_specular_source(
        &self,
        scene: &SceneCoordinator,
        materials: &MaterialSystem,
        assets: &AssetTransferQueue,
        limits: &GpuLimits,
    ) -> Option<SkyboxSpecularEnvironmentSource> {
        let source = resolve_active_main_skybox_ibl_source(scene, materials, assets)?;
        let key = build_key(&source, clamp_face_size(DEFAULT_IBL_FACE_SIZE, limits));
        let cube = self.completed.get(&key)?;
        Some(SkyboxSpecularEnvironmentSource::Cubemap(
            SkyboxSpecularCubemapSource {
                key_hash: key.source_hash(),
                view: cube.view.clone(),
                sampler: cube.sampler.clone(),
                mip_levels_resident: cube.mip_levels,
            },
        ))
    }

    /// Promotes submit-completed bakes into the completed cache.
    fn drain_completed_jobs(&mut self) {
        let outcomes = self.jobs.maintain();
        for key in outcomes.completed {
            if let Some(pending) = self.pending.remove(&key) {
                self.completed.insert(key, pending.cube);
            }
        }
        for key in outcomes.failed {
            self.pending.remove(&key);
            logger::warn!("skybox_ibl: bake expired before submit completion (key {key:?})");
        }
    }

    /// Drops completed cubes that no longer match the active skybox key.
    fn prune_completed(&mut self, active: Option<&SkyboxIblKey>) {
        self.completed
            .retain(|key, _| active.is_some_and(|active_key| active_key == key));
    }

    /// Encodes one IBL bake (mip-0 producer + per-mip GGX convolves) and submits it.
    fn schedule_bake(
        &mut self,
        gpu: &mut GpuContext,
        key: SkyboxIblKey,
        source: SkyboxIblSource,
    ) -> Result<(), SkyboxIblBakeError> {
        profiling::scope!("skybox_ibl::schedule_bake");
        let mut profiler = gpu.take_gpu_profiler();
        let result = self.schedule_bake_with_profiler(gpu, key, source, profiler.as_mut());
        gpu.restore_gpu_profiler(profiler);
        result
    }

    fn schedule_bake_with_profiler(
        &mut self,
        gpu: &GpuContext,
        key: SkyboxIblKey,
        source: SkyboxIblSource,
        mut profiler: Option<&mut GpuProfilerHandle>,
    ) -> Result<(), SkyboxIblBakeError> {
        self.ensure_pipelines(gpu.device())?;
        let input_sampler = self.ensure_input_sampler(gpu.device()).clone();
        let face_size = key.face_size();
        let mip_levels = mip_levels_for_edge(face_size);
        let cube = create_ibl_cube(gpu.device(), face_size, mip_levels);
        let mut resources = PendingBakeResources::default();
        let dst_sample_view = Arc::new(create_mip0_cube_sample_view(&cube.texture));
        resources.dst_sample_view = Some(dst_sample_view.clone());
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("skybox_ibl bake encoder"),
            });
        match source {
            SkyboxIblSource::Analytic(src) => {
                let pipeline = self.analytic_pipeline()?;
                encode_analytic_mip0(
                    AnalyticEncodeContext {
                        device: gpu.device(),
                        encoder: &mut encoder,
                        pipeline,
                        texture: cube.texture.as_ref(),
                        face_size,
                        params: &src.params,
                        profiler: profiler.as_deref(),
                    },
                    &mut resources,
                );
            }
            SkyboxIblSource::Cubemap(src) => {
                let pipeline = self.cube_pipeline()?;
                encode_cube_mip0(
                    CubeEncodeContext {
                        device: gpu.device(),
                        encoder: &mut encoder,
                        pipeline,
                        texture: cube.texture.as_ref(),
                        face_size,
                        src,
                        sampler: input_sampler.as_ref(),
                        profiler: profiler.as_deref(),
                    },
                    &mut resources,
                );
            }
            SkyboxIblSource::Equirect(src) => {
                let pipeline = self.equirect_pipeline()?;
                encode_equirect_mip0(
                    EquirectEncodeContext {
                        device: gpu.device(),
                        encoder: &mut encoder,
                        pipeline,
                        texture: cube.texture.as_ref(),
                        face_size,
                        src,
                        sampler: input_sampler.as_ref(),
                        profiler: profiler.as_deref(),
                    },
                    &mut resources,
                );
            }
        }
        let convolve_pipeline = self.convolve_pipeline()?;
        encode_convolve_mips(
            ConvolveEncodeContext {
                device: gpu.device(),
                encoder: &mut encoder,
                pipeline: convolve_pipeline,
                texture: cube.texture.as_ref(),
                src_view: dst_sample_view.as_ref(),
                sampler: input_sampler.as_ref(),
                face_size,
                mip_levels,
                profiler: profiler.as_deref(),
            },
            &mut resources,
        );
        if let Some(profiler) = profiler.as_mut() {
            profiling::scope!("skybox_ibl::resolve_profiler_queries");
            profiler.resolve_queries(&mut encoder);
        }
        let pending = PendingBake {
            cube: PrefilteredCube {
                _texture: cube.texture,
                view: cube.full_view,
                sampler: prefiltered_sampler_state(),
                mip_levels,
            },
            _resources: resources,
        };
        self.submit_pending_bake(gpu, key, encoder, pending);
        Ok(())
    }

    /// Ensures every compute pipeline used by IBL bakes is resident.
    fn ensure_pipelines(&mut self, device: &wgpu::Device) -> Result<(), SkyboxIblBakeError> {
        profiling::scope!("skybox_ibl::ensure_pipelines");
        let _ = ensure_pipeline(
            &mut self.analytic_pipeline,
            device,
            "skybox_bake_params",
            &analytic_layout_entries(),
        )?;
        let _ = ensure_pipeline(
            &mut self.cube_pipeline,
            device,
            "skybox_mip0_cube_params",
            &mip0_input_layout_entries(wgpu::TextureViewDimension::Cube),
        )?;
        let _ = ensure_pipeline(
            &mut self.equirect_pipeline,
            device,
            "skybox_mip0_equirect_params",
            &mip0_input_layout_entries(wgpu::TextureViewDimension::D2),
        )?;
        let _ = ensure_pipeline(
            &mut self.convolve_pipeline,
            device,
            "skybox_ibl_convolve_params",
            &mip0_input_layout_entries(wgpu::TextureViewDimension::Cube),
        )?;
        Ok(())
    }

    fn analytic_pipeline(&self) -> Result<&ComputePipeline, SkyboxIblBakeError> {
        self.analytic_pipeline
            .as_ref()
            .ok_or(SkyboxIblBakeError::MissingShader("skybox_bake_params"))
    }

    fn cube_pipeline(&self) -> Result<&ComputePipeline, SkyboxIblBakeError> {
        self.cube_pipeline
            .as_ref()
            .ok_or(SkyboxIblBakeError::MissingShader("skybox_mip0_cube_params"))
    }

    fn equirect_pipeline(&self) -> Result<&ComputePipeline, SkyboxIblBakeError> {
        self.equirect_pipeline
            .as_ref()
            .ok_or(SkyboxIblBakeError::MissingShader(
                "skybox_mip0_equirect_params",
            ))
    }

    fn convolve_pipeline(&self) -> Result<&ComputePipeline, SkyboxIblBakeError> {
        self.convolve_pipeline
            .as_ref()
            .ok_or(SkyboxIblBakeError::MissingShader(
                "skybox_ibl_convolve_params",
            ))
    }

    /// Returns a cached linear/clamp sampler used for all source/destination cube reads.
    fn ensure_input_sampler(&mut self, device: &wgpu::Device) -> &Arc<wgpu::Sampler> {
        self.input_sampler.get_or_insert_with(|| {
            Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("skybox_ibl_input_sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            }))
        })
    }

    /// Tracks and submits an encoded bake, retaining transient resources until completion.
    fn submit_pending_bake(
        &mut self,
        gpu: &GpuContext,
        key: SkyboxIblKey,
        encoder: wgpu::CommandEncoder,
        pending: PendingBake,
    ) {
        profiling::scope!("skybox_ibl::submit_bake");
        let tx = self.jobs.submit_done_sender();
        let callback_key = key.clone();
        self.jobs.insert(
            key.clone(),
            SubmittedGpuJob {
                resources: GpuJobResources::new(),
            },
        );
        self.pending.insert(key, pending);
        let command_buffer = {
            profiling::scope!("CommandEncoder::finish::skybox_ibl");
            encoder.finish()
        };
        gpu.submit_frame_batch_with_callbacks(
            vec![command_buffer],
            None,
            None,
            vec![Box::new(move || {
                let _ = tx.send(callback_key);
            })],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip: applying the runtime parabolic LOD then the inverse returns the input.
    #[test]
    fn roughness_lod_round_trip() {
        for i in 0..=20u32 {
            let r = i as f32 / 20.0;
            let lod = r * (2.0 - r);
            let r_back = 1.0 - (1.0 - lod).max(0.0).sqrt();
            assert!((r - r_back).abs() < 1e-6, "r={r} r_back={r_back}");
        }
    }

    /// Mip count includes mip 0 through the one-texel mip.
    #[test]
    fn mip_levels_for_edge_includes_tail_mip() {
        assert_eq!(mip_levels_for_edge(1), 1);
        assert_eq!(mip_levels_for_edge(2), 2);
        assert_eq!(mip_levels_for_edge(128), 8);
        assert_eq!(mip_levels_for_edge(256), 9);
    }

    /// Per-mip sample count clamps to the documented base/cap envelope.
    #[test]
    fn convolve_sample_count_envelope() {
        assert_eq!(convolve_sample_count(0), 1);
        assert_eq!(convolve_sample_count(1), 64);
        assert_eq!(convolve_sample_count(2), 128);
        assert_eq!(convolve_sample_count(3), 256);
        assert_eq!(convolve_sample_count(4), 512);
        assert_eq!(convolve_sample_count(5), 1024);
        assert_eq!(convolve_sample_count(8), 1024);
    }

    /// Analytic key invariants: identity bits change the source hash.
    #[test]
    fn analytic_key_hash_changes_with_identity_fields() {
        let a = SkyboxIblKey::Analytic {
            material_asset_id: 1,
            material_generation: 2,
            route_hash: 3,
            face_size: 256,
        };
        let b = SkyboxIblKey::Analytic {
            material_asset_id: 1,
            material_generation: 2,
            route_hash: 3,
            face_size: 128,
        };
        let c = SkyboxIblKey::Analytic {
            material_asset_id: 1,
            material_generation: 9,
            route_hash: 3,
            face_size: 256,
        };
        assert_ne!(a.source_hash(), b.source_hash());
        assert_ne!(a.source_hash(), c.source_hash());
    }

    /// Cubemap key invariants: residency growth and face size resize both invalidate.
    #[test]
    fn cubemap_key_invalidates_on_residency_or_face_change() {
        let a = SkyboxIblKey::Cubemap {
            asset_id: 7,
            mip_levels_resident: 1,
            storage_v_inverted: false,
            face_size: 256,
        };
        let b = SkyboxIblKey::Cubemap {
            asset_id: 7,
            mip_levels_resident: 4,
            storage_v_inverted: false,
            face_size: 256,
        };
        let c = SkyboxIblKey::Cubemap {
            asset_id: 7,
            mip_levels_resident: 1,
            storage_v_inverted: false,
            face_size: 128,
        };
        assert_ne!(a, b);
        assert_ne!(a, c);
    }

    /// Equirect key invariants: FOV / ST hash inputs invalidate the bake.
    #[test]
    fn equirect_key_invalidates_on_param_changes() {
        let base = SkyboxIblKey::Equirect {
            asset_id: 9,
            mip_levels_resident: 3,
            storage_v_inverted: false,
            fov_hash: hash_float4(&[1.0, 1.0, 0.0, 0.0]),
            st_hash: hash_float4(&[1.0, 1.0, 0.0, 0.0]),
            face_size: 256,
        };
        let altered_fov = SkyboxIblKey::Equirect {
            asset_id: 9,
            mip_levels_resident: 3,
            storage_v_inverted: false,
            fov_hash: hash_float4(&[2.0, 1.0, 0.0, 0.0]),
            st_hash: hash_float4(&[1.0, 1.0, 0.0, 0.0]),
            face_size: 256,
        };
        let altered_st = SkyboxIblKey::Equirect {
            asset_id: 9,
            mip_levels_resident: 3,
            storage_v_inverted: false,
            fov_hash: hash_float4(&[1.0, 1.0, 0.0, 0.0]),
            st_hash: hash_float4(&[2.0, 1.0, 0.0, 0.0]),
            face_size: 256,
        };
        assert_ne!(base, altered_fov);
        assert_ne!(base, altered_st);
    }
}
