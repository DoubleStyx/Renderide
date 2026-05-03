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

use std::borrow::Cow;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use hashbrown::HashMap;
use thiserror::Error;
use wgpu::util::DeviceExt;

use crate::assets::asset_transfer_queue::AssetTransferQueue;
use crate::backend::frame_gpu::{SkyboxSpecularCubemapSource, SkyboxSpecularEnvironmentSource};
use crate::backend::gpu_jobs::{GpuJobResources, GpuSubmitJobTracker, SubmittedGpuJob};
use crate::embedded_shaders;
use crate::gpu::{GpuContext, GpuLimits};
use crate::gpu_pools::SamplerState;
use crate::materials::MaterialSystem;
use crate::profiling::{GpuProfilerHandle, compute_pass_timestamp_writes};
use crate::scene::SceneCoordinator;
use crate::shared::{TextureFilterMode, TextureWrapMode};
use crate::skybox::params::SkyboxEvaluatorParams;
use crate::skybox::specular::{
    CubemapIblSource, EquirectIblSource, SkyboxIblSource, resolve_active_main_skybox_ibl_source,
};

/// Maximum concurrent in-flight bakes; matches the analytic-only ceiling we used previously.
const MAX_IN_FLIGHT_IBL_BAKES: usize = 2;
/// Tick budget after which a missing submit-completion callback is treated as lost.
const MAX_PENDING_IBL_BAKE_AGE_FRAMES: u32 = 120;
/// Default destination cube face edge in texels (clamped to portable device limits).
const DEFAULT_IBL_FACE_SIZE: u32 = 256;
/// IBL cubemap format. Matches the analytic skybox bake; supports STORAGE_BINDING.
const IBL_CUBE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Compute workgroup edge used by every mip-0 producer and the GGX convolve.
const IBL_WORKGROUP_EDGE: u32 = 8;
/// Base GGX importance sample count for mip 1; doubles per mip up to [`IBL_MAX_SAMPLES`].
const IBL_BASE_SAMPLE_COUNT: u32 = 64;
/// Cap on GGX importance sample count for the highest-roughness mips.
const IBL_MAX_SAMPLES: u32 = 1024;

/// Errors returned while preparing an IBL bake.
#[derive(Debug, Error)]
enum SkyboxIblBakeError {
    /// Embedded WGSL source was not available at compose time.
    #[error("embedded shader {0} not found")]
    MissingShader(&'static str),
}

/// Identity for one IBL bake.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) enum SkyboxIblKey {
    /// Analytic procedural / gradient skybox material identity.
    Analytic {
        /// Active skybox material asset id.
        material_asset_id: i32,
        /// Material property generation; invalidates when host edits material props.
        material_generation: u64,
        /// Stable hash of the shader route stem.
        route_hash: u64,
        /// Destination cube face edge (clamped to device limits).
        face_size: u32,
    },
    /// Host-uploaded cubemap material identity.
    Cubemap {
        /// Source cubemap asset id.
        asset_id: i32,
        /// Source resident mip count; growth re-bakes once more mips arrive.
        mip_levels_resident: u32,
        /// Storage V-flip flag for the source cube.
        storage_v_inverted: bool,
        /// Destination cube face edge.
        face_size: u32,
    },
    /// Host-uploaded equirect Texture2D material identity.
    Equirect {
        /// Source Texture2D asset id.
        asset_id: i32,
        /// Source resident mip count.
        mip_levels_resident: u32,
        /// Storage V-flip flag for the source texture.
        storage_v_inverted: bool,
        /// Bit-stable hash of `_FOV` material parameters.
        fov_hash: u64,
        /// Bit-stable hash of `_MainTex_ST` material parameters.
        st_hash: u64,
        /// Destination cube face edge.
        face_size: u32,
    },
}

impl SkyboxIblKey {
    /// Returns the destination face size for this bake.
    fn face_size(&self) -> u32 {
        match *self {
            Self::Analytic { face_size, .. }
            | Self::Cubemap { face_size, .. }
            | Self::Equirect { face_size, .. } => face_size,
        }
    }

    /// Returns a stable renderer-side identity hash for the frame-global binding key.
    fn source_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

/// Completed prefiltered cubemap that the frame-global binding owns.
struct PrefilteredCube {
    /// Texture backing [`Self::view`]. Held to keep the storage alive while the view is bound.
    _texture: Arc<wgpu::Texture>,
    /// Full mip-chain cube view.
    view: Arc<wgpu::TextureView>,
    /// Sampler state used when binding for material sampling.
    sampler: SamplerState,
    /// Mip count of [`Self::view`].
    mip_levels: u32,
}

/// Pending bake retained until the submit callback fires.
struct PendingBake {
    /// Completed cube that becomes visible after submit completion.
    cube: PrefilteredCube,
    /// Transient resources retained until the queued commands complete.
    _resources: PendingBakeResources,
}

/// Transient command resources that must survive until submit completion.
#[derive(Default)]
struct PendingBakeResources {
    /// Uniform and transient buffers retained until the queued commands complete.
    buffers: Vec<wgpu::Buffer>,
    /// Bind groups retained until the queued commands complete.
    bind_groups: Vec<wgpu::BindGroup>,
    /// Per-mip texture views retained until the queued commands complete.
    texture_views: Vec<wgpu::TextureView>,
    /// Source asset views/textures retained for the duration of the bake.
    source_views: Vec<Arc<wgpu::TextureView>>,
    /// Cube sampling view of the destination retained for the convolve passes.
    dst_sample_view: Option<Arc<wgpu::TextureView>>,
}

/// Compute pipeline + bind-group layout pair built lazily from an embedded shader stem.
struct ComputePipeline {
    /// Compute pipeline.
    pipeline: wgpu::ComputePipeline,
    /// Bind-group layout for this pipeline.
    layout: wgpu::BindGroupLayout,
}

/// Uniform payload shared by the cubemap and convolve mip-0 producers.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Mip0CubeParams {
    dst_size: u32,
    src_face_size: u32,
    storage_v_inverted: u32,
    _pad0: u32,
}

/// Uniform payload for the equirect mip-0 producer.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Mip0EquirectParams {
    dst_size: u32,
    storage_v_inverted: u32,
    _pad0: u32,
    _pad1: u32,
    fov: [f32; 4],
    st: [f32; 4],
}

/// Uniform payload for one GGX convolve mip dispatch.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ConvolveParams {
    dst_size: u32,
    mip_index: u32,
    mip_count: u32,
    sample_count: u32,
    src_face_size: u32,
    src_max_lod: f32,
    _pad0: u32,
    _pad1: u32,
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
            .map(|source| build_key(source, gpu.limits()));
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
        let key = build_key(&source, limits);
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

/// Builds a cache key for an active source, clamping the destination face size to device limits.
fn build_key(source: &SkyboxIblSource, limits: &GpuLimits) -> SkyboxIblKey {
    let face_size = clamp_face_size(DEFAULT_IBL_FACE_SIZE, limits);
    match source {
        SkyboxIblSource::Analytic(src) => SkyboxIblKey::Analytic {
            material_asset_id: src.material_asset_id,
            material_generation: src.material_generation,
            route_hash: src.route_hash,
            face_size,
        },
        SkyboxIblSource::Cubemap(src) => SkyboxIblKey::Cubemap {
            asset_id: src.asset_id,
            mip_levels_resident: src.mip_levels_resident,
            storage_v_inverted: src.storage_v_inverted,
            face_size,
        },
        SkyboxIblSource::Equirect(src) => SkyboxIblKey::Equirect {
            asset_id: src.asset_id,
            mip_levels_resident: src.mip_levels_resident,
            storage_v_inverted: src.storage_v_inverted,
            fov_hash: hash_float4(&src.equirect_fov),
            st_hash: hash_float4(&src.equirect_st),
            face_size,
        },
    }
}

/// Hashes four `f32`s by their bit patterns.
fn hash_float4(values: &[f32; 4]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for v in values {
        v.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

/// Clamps the configured cube face size against the device texture limit.
pub(crate) fn clamp_face_size(face_size: u32, limits: &GpuLimits) -> u32 {
    face_size.min(limits.max_texture_dimension_2d()).max(1)
}

/// Returns the full mip count for a cube face edge.
pub(crate) fn mip_levels_for_edge(edge: u32) -> u32 {
    u32::BITS - edge.max(1).leading_zeros()
}

/// Returns the dispatch group count along one 8x8 compute dimension.
fn dispatch_groups(size: u32) -> u32 {
    size.max(1).div_ceil(IBL_WORKGROUP_EDGE)
}

/// Returns a mip edge clamped to one texel.
fn mip_extent(base: u32, mip: u32) -> u32 {
    (base >> mip).max(1)
}

/// Returns the GGX importance sample count for the given convolve mip.
fn convolve_sample_count(mip_index: u32) -> u32 {
    if mip_index == 0 {
        return 1;
    }
    let exponent = (mip_index - 1).min(4);
    (IBL_BASE_SAMPLE_COUNT << exponent).min(IBL_MAX_SAMPLES)
}

/// Sampler state used when the prefiltered cube is bound for material sampling.
fn prefiltered_sampler_state() -> SamplerState {
    SamplerState {
        filter_mode: TextureFilterMode::Trilinear,
        aniso_level: 1,
        wrap_u: TextureWrapMode::Clamp,
        wrap_v: TextureWrapMode::Clamp,
        wrap_w: TextureWrapMode::default(),
        mipmap_bias: 0.0,
    }
}

/// IBL cube texture handles produced by [`create_ibl_cube`].
struct IblCubeTexture {
    /// Texture backing the destination cubemap.
    texture: Arc<wgpu::Texture>,
    /// Full mip-chain cube view bound at runtime.
    full_view: Arc<wgpu::TextureView>,
}

/// Allocates the destination Rgba16Float cube and its full sampling view.
fn create_ibl_cube(device: &wgpu::Device, face_size: u32, mip_levels: u32) -> IblCubeTexture {
    let texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
        label: Some("skybox_ibl_cube"),
        size: wgpu::Extent3d {
            width: face_size,
            height: face_size,
            depth_or_array_layers: 6,
        },
        mip_level_count: mip_levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: IBL_CUBE_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    }));
    let full_view = Arc::new(texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("skybox_ibl_cube_view"),
        format: Some(IBL_CUBE_FORMAT),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(mip_levels),
        base_array_layer: 0,
        array_layer_count: Some(6),
    }));
    IblCubeTexture { texture, full_view }
}

/// Creates a cube-dimension sampling view of mip 0 only, used as the convolve input source.
///
/// The view must not overlap any storage-bound mip in the same compute dispatch -- wgpu treats
/// overlapping subresources as a usage conflict between `RESOURCE` and `STORAGE_WRITE_ONLY`.
/// A mip-0-only view is non-overlapping with every per-mip storage view (mip >= 1).
fn create_mip0_cube_sample_view(texture: &wgpu::Texture) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("skybox_ibl_cube_mip0_sample_view"),
        format: Some(IBL_CUBE_FORMAT),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(1),
        base_array_layer: 0,
        array_layer_count: Some(6),
    })
}

/// Creates a per-mip storage view for one face-array of the destination cube.
fn create_mip_storage_view(texture: &wgpu::Texture, mip: u32) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("skybox_ibl_mip_storage_view"),
        format: Some(IBL_CUBE_FORMAT),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: mip,
        mip_level_count: Some(1),
        base_array_layer: 0,
        array_layer_count: Some(6),
    })
}

/// Storage texture entry for one cubemap mip.
fn storage_texture_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: IBL_CUBE_FORMAT,
            view_dimension: wgpu::TextureViewDimension::D2Array,
        },
        count: None,
    }
}

/// Bind-group layout entries for the analytic mip-0 producer.
fn analytic_layout_entries() -> [wgpu::BindGroupLayoutEntry; 2] {
    [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        storage_texture_layout_entry(1),
    ]
}

/// Bind-group layout entries for the cube/equirect/convolve passes that read a sampled texture.
fn mip0_input_layout_entries(
    input_dim: wgpu::TextureViewDimension,
) -> [wgpu::BindGroupLayoutEntry; 4] {
    [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: input_dim,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
        storage_texture_layout_entry(3),
    ]
}

/// Lazily creates and caches a compute pipeline from an embedded shader stem.
fn ensure_pipeline<'a>(
    slot: &'a mut Option<ComputePipeline>,
    device: &wgpu::Device,
    stem: &'static str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> Result<&'a ComputePipeline, SkyboxIblBakeError> {
    if slot.is_none() {
        profiling::scope!("skybox_ibl::create_pipeline", stem);
        let source = embedded_shaders::embedded_target_wgsl(stem)
            .ok_or(SkyboxIblBakeError::MissingShader(stem))?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(stem),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{stem} bind group layout")),
            entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{stem} pipeline layout")),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(stem),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        *slot = Some(ComputePipeline { pipeline, layout });
    }
    slot.as_ref().ok_or(SkyboxIblBakeError::MissingShader(stem))
}

/// Inputs for [`encode_analytic_mip0`].
struct AnalyticEncodeContext<'a> {
    device: &'a wgpu::Device,
    encoder: &'a mut wgpu::CommandEncoder,
    pipeline: &'a ComputePipeline,
    texture: &'a wgpu::Texture,
    face_size: u32,
    params: &'a SkyboxEvaluatorParams,
    profiler: Option<&'a GpuProfilerHandle>,
}

/// Encodes mip 0 from analytic procedural / gradient sky parameters.
fn encode_analytic_mip0(ctx: AnalyticEncodeContext<'_>, resources: &mut PendingBakeResources) {
    profiling::scope!("skybox_ibl::encode_mip0_analytic");
    let mut params = *ctx.params;
    params = params.with_sample_size(ctx.face_size);
    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("skybox_ibl analytic params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let mip0_storage = create_mip_storage_view(ctx.texture, 0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("skybox_ibl analytic bind group"),
        layout: &ctx.pipeline.layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&mip0_storage),
            },
        ],
    });
    let pass_query = ctx
        .profiler
        .map(|profiler| profiler.begin_pass_query("skybox_ibl::mip0_analytic", ctx.encoder));
    {
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("skybox_ibl analytic mip0"),
                timestamp_writes: compute_pass_timestamp_writes(pass_query.as_ref()),
            });
        pass.set_pipeline(&ctx.pipeline.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            dispatch_groups(ctx.face_size),
            dispatch_groups(ctx.face_size),
            6,
        );
    };
    if let (Some(profiler), Some(query)) = (ctx.profiler, pass_query) {
        profiler.end_query(ctx.encoder, query);
    }
    resources.buffers.push(params_buffer);
    resources.bind_groups.push(bind_group);
    resources.texture_views.push(mip0_storage);
}

/// Inputs for [`encode_cube_mip0`].
struct CubeEncodeContext<'a> {
    device: &'a wgpu::Device,
    encoder: &'a mut wgpu::CommandEncoder,
    pipeline: &'a ComputePipeline,
    texture: &'a wgpu::Texture,
    face_size: u32,
    src: CubemapIblSource,
    sampler: &'a wgpu::Sampler,
    profiler: Option<&'a GpuProfilerHandle>,
}

/// Encodes mip 0 by resampling a host cubemap source.
fn encode_cube_mip0(ctx: CubeEncodeContext<'_>, resources: &mut PendingBakeResources) {
    profiling::scope!("skybox_ibl::encode_mip0_cube");
    let params = Mip0CubeParams {
        dst_size: ctx.face_size,
        src_face_size: ctx.src.face_size,
        storage_v_inverted: u32::from(ctx.src.storage_v_inverted),
        _pad0: 0,
    };
    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("skybox_ibl cube mip0 params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let mip0_storage = create_mip_storage_view(ctx.texture, 0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("skybox_ibl cube mip0 bind group"),
        layout: &ctx.pipeline.layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(ctx.src.view.as_ref()),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(ctx.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&mip0_storage),
            },
        ],
    });
    let pass_query = ctx
        .profiler
        .map(|profiler| profiler.begin_pass_query("skybox_ibl::mip0_cube", ctx.encoder));
    {
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("skybox_ibl cube mip0"),
                timestamp_writes: compute_pass_timestamp_writes(pass_query.as_ref()),
            });
        pass.set_pipeline(&ctx.pipeline.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            dispatch_groups(ctx.face_size),
            dispatch_groups(ctx.face_size),
            6,
        );
    };
    if let (Some(profiler), Some(query)) = (ctx.profiler, pass_query) {
        profiler.end_query(ctx.encoder, query);
    }
    resources.buffers.push(params_buffer);
    resources.bind_groups.push(bind_group);
    resources.texture_views.push(mip0_storage);
    resources.source_views.push(ctx.src.view);
}

/// Inputs for [`encode_equirect_mip0`].
struct EquirectEncodeContext<'a> {
    device: &'a wgpu::Device,
    encoder: &'a mut wgpu::CommandEncoder,
    pipeline: &'a ComputePipeline,
    texture: &'a wgpu::Texture,
    face_size: u32,
    src: EquirectIblSource,
    sampler: &'a wgpu::Sampler,
    profiler: Option<&'a GpuProfilerHandle>,
}

/// Encodes mip 0 by resampling an equirect Texture2D source.
fn encode_equirect_mip0(ctx: EquirectEncodeContext<'_>, resources: &mut PendingBakeResources) {
    profiling::scope!("skybox_ibl::encode_mip0_equirect");
    let params = Mip0EquirectParams {
        dst_size: ctx.face_size,
        storage_v_inverted: u32::from(ctx.src.storage_v_inverted),
        _pad0: 0,
        _pad1: 0,
        fov: ctx.src.equirect_fov,
        st: ctx.src.equirect_st,
    };
    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("skybox_ibl equirect mip0 params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let mip0_storage = create_mip_storage_view(ctx.texture, 0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("skybox_ibl equirect mip0 bind group"),
        layout: &ctx.pipeline.layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(ctx.src.view.as_ref()),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(ctx.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&mip0_storage),
            },
        ],
    });
    let pass_query = ctx
        .profiler
        .map(|profiler| profiler.begin_pass_query("skybox_ibl::mip0_equirect", ctx.encoder));
    {
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("skybox_ibl equirect mip0"),
                timestamp_writes: compute_pass_timestamp_writes(pass_query.as_ref()),
            });
        pass.set_pipeline(&ctx.pipeline.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            dispatch_groups(ctx.face_size),
            dispatch_groups(ctx.face_size),
            6,
        );
    };
    if let (Some(profiler), Some(query)) = (ctx.profiler, pass_query) {
        profiler.end_query(ctx.encoder, query);
    }
    resources.buffers.push(params_buffer);
    resources.bind_groups.push(bind_group);
    resources.texture_views.push(mip0_storage);
    resources.source_views.push(ctx.src.view);
}

/// Inputs for [`encode_convolve_mips`].
struct ConvolveEncodeContext<'a> {
    device: &'a wgpu::Device,
    encoder: &'a mut wgpu::CommandEncoder,
    pipeline: &'a ComputePipeline,
    texture: &'a wgpu::Texture,
    src_view: &'a wgpu::TextureView,
    sampler: &'a wgpu::Sampler,
    face_size: u32,
    mip_levels: u32,
    profiler: Option<&'a GpuProfilerHandle>,
}

/// Encodes the GGX convolve passes for mips `1..mip_levels` of the destination cube.
fn encode_convolve_mips(ctx: ConvolveEncodeContext<'_>, resources: &mut PendingBakeResources) {
    profiling::scope!("skybox_ibl::encode_convolve_mips");
    if ctx.mip_levels <= 1 {
        return;
    }
    // Source view is mip 0 only (see create_mip0_cube_sample_view) -- clamp source LOD to zero so
    // the shader's solid-angle source-mip selection collapses to a plain mip-0 sample.
    let src_max_lod = 0.0_f32;
    for mip in 1..ctx.mip_levels {
        profiling::scope!("skybox_ibl::encode_convolve_mip");
        let dst_size = mip_extent(ctx.face_size, mip);
        let params = ConvolveParams {
            dst_size,
            mip_index: mip,
            mip_count: ctx.mip_levels,
            sample_count: convolve_sample_count(mip),
            src_face_size: ctx.face_size,
            src_max_lod,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("skybox_ibl convolve params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let dst_view = create_mip_storage_view(ctx.texture, mip);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skybox_ibl convolve bind group"),
            layout: &ctx.pipeline.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(ctx.src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(ctx.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&dst_view),
                },
            ],
        });
        let pass_query = ctx.profiler.map(|profiler| {
            profiler.begin_pass_query(format!("skybox_ibl::convolve_mip{mip}"), ctx.encoder)
        });
        {
            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("skybox_ibl convolve mip"),
                    timestamp_writes: compute_pass_timestamp_writes(pass_query.as_ref()),
                });
            pass.set_pipeline(&ctx.pipeline.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_groups(dst_size), dispatch_groups(dst_size), 6);
        };
        if let (Some(profiler), Some(query)) = (ctx.profiler, pass_query) {
            profiler.end_query(ctx.encoder, query);
        }
        resources.buffers.push(params_buffer);
        resources.bind_groups.push(bind_group);
        resources.texture_views.push(dst_view);
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
