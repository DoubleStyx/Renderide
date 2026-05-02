//! Compute + fullscreen depth blit used to resolve multisampled depth to the single-sample forward depth target
//! (no storage writes on depth in core WebGPU).
//!
//! WGSL is sourced from the build-time embedded shader registry. The blit shader lives in a
//! single source with `#ifdef MULTIVIEW` and is composed into `depth_blit_r32_to_depth_default`
//! (2D source) and `depth_blit_r32_to_depth_multiview` (2D array source + `@builtin(view_index)`);
//! the compute shader is single-variant.
//!
//! Submodules:
//! - [`targets`] -- view bundles consumed by the encoders
//!   ([`MsaaDepthResolveMonoTargets`], [`MsaaDepthResolveStereoTargets`]).
//! - [`pipelines`] -- init-time pipeline construction (compute + per-format blit pipelines).
//! - [`encode`] -- per-frame pass recording (compute resolve + depth blit).

mod encode;
mod pipelines;
mod targets;

use crate::embedded_shaders::{
    DEPTH_BLIT_R32_TO_DEPTH_DEFAULT_WGSL, MSAA_DEPTH_RESOLVE_TO_R32_WGSL,
};
use crate::gpu::bind_layout::{storage_texture_layout_entry, texture_layout_entry};
use crate::gpu::limits::GpuLimits;
use crate::profiling::GpuProfilerHandle;

use pipelines::{create_desktop_blit_pipelines, create_stereo_multiview_blit_pipelines};

pub use targets::{MsaaDepthResolveMonoTargets, MsaaDepthResolveStereoTargets};

/// Pipelines and layouts for MSAA depth -> R32F compute -> depth blit.
///
/// Exposes both the desktop (`D2`) path via [`Self::encode_resolve`] and the stereo (OpenXR 2-layer
/// `D2Array`) path via [`Self::encode_resolve_stereo`]. The stereo path reuses the same compute
/// pipeline by dispatching once per eye on single-layer `D2` views, then runs one **multiview**
/// blit pass (`multiview_mask = 0b11`) that writes both depth layers via `@builtin(view_index)`.
///
/// This indirection exists because WGSL `texture_depth_multisampled_2d_array` is not yet available
/// in current `wgpu`, so we keep the compute shader as `texture_depth_multisampled_2d` and issue
/// two dispatches from per-layer views produced by the stereo MSAA depth attachment.
pub struct MsaaDepthResolveResources {
    compute_pipeline: wgpu::ComputePipeline,
    blit_pipeline_depth32: wgpu::RenderPipeline,
    blit_pipeline_depth24_stencil8: wgpu::RenderPipeline,
    blit_pipeline_depth32_stencil8: Option<wgpu::RenderPipeline>,
    compute_bgl: wgpu::BindGroupLayout,
    blit_bgl: wgpu::BindGroupLayout,
    /// Multiview blit pipeline for the stereo path; `None` when `MULTIVIEW` is unavailable
    /// (disables stereo MSAA depth resolve but allows the rest of the engine to run).
    blit_stereo_pipeline_depth32: Option<wgpu::RenderPipeline>,
    blit_stereo_pipeline_depth24_stencil8: Option<wgpu::RenderPipeline>,
    blit_stereo_pipeline_depth32_stencil8: Option<wgpu::RenderPipeline>,
    /// Bind-group layout for the stereo blit (`texture_2d_array<f32>` source).
    blit_stereo_bgl: Option<wgpu::BindGroupLayout>,
}

impl MsaaDepthResolveResources {
    /// Builds compute and blit pipelines; returns [`None`] if shader creation fails.
    pub fn try_new(device: &wgpu::Device) -> Option<Self> {
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("msaa_depth_resolve_cs"),
            source: wgpu::ShaderSource::Wgsl(MSAA_DEPTH_RESOLVE_TO_R32_WGSL.into()),
        });
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("msaa_depth_resolve_blit"),
            source: wgpu::ShaderSource::Wgsl(DEPTH_BLIT_R32_TO_DEPTH_DEFAULT_WGSL.into()),
        });

        let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("msaa_depth_resolve_compute_bgl"),
            entries: &[
                texture_layout_entry(
                    0,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::TextureSampleType::Depth,
                    wgpu::TextureViewDimension::D2,
                    true,
                ),
                storage_texture_layout_entry(
                    1,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::StorageTextureAccess::WriteOnly,
                    wgpu::TextureFormat::R32Float,
                    wgpu::TextureViewDimension::D2,
                ),
            ],
        });

        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("msaa_depth_blit_bgl"),
            entries: &[texture_layout_entry(
                0,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::TextureSampleType::Float { filterable: false },
                wgpu::TextureViewDimension::D2,
                false,
            )],
        });

        let compute_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("msaa_depth_resolve_compute_pl"),
            bind_group_layouts: &[Some(&compute_bgl)],
            ..Default::default()
        });

        let blit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("msaa_depth_blit_pl"),
            bind_group_layouts: &[Some(&blit_bgl)],
            ..Default::default()
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("msaa_depth_resolve_compute"),
            layout: Some(&compute_layout),
            module: &compute_shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let desktop = create_desktop_blit_pipelines(device, &blit_shader, &blit_layout);
        let stereo = create_stereo_multiview_blit_pipelines(device);

        Some(Self {
            compute_pipeline,
            blit_pipeline_depth32: desktop.depth32,
            blit_pipeline_depth24_stencil8: desktop.depth24_stencil8,
            blit_pipeline_depth32_stencil8: desktop.depth32_stencil8,
            compute_bgl,
            blit_bgl,
            blit_stereo_pipeline_depth32: stereo.depth32,
            blit_stereo_pipeline_depth24_stencil8: stereo.depth24_stencil8,
            blit_stereo_pipeline_depth32_stencil8: stereo.depth32_stencil8,
            blit_stereo_bgl: stereo.bgl,
        })
    }

    /// Resolves `targets.msaa_depth_view` into `targets.dst_depth_view` via R32F intermediate `targets.r32_view`.
    ///
    /// When the 8x8-tiled compute dispatch would exceed [`GpuLimits::compute_dispatch_fits`], logs a
    /// warning and skips compute and blit (degraded depth for intersection / Hi-Z vs invalid GPU work).
    ///
    /// `profiler` opens a pass-level GPU timestamp query around each of the two passes (compute
    /// resolve and depth blit) so they appear individually on the Tracy GPU timeline. Pass [`None`]
    /// when the GPU profiler is unavailable.
    pub fn encode_resolve(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        extent: (u32, u32),
        targets: MsaaDepthResolveMonoTargets<'_>,
        limits: &GpuLimits,
        profiler: Option<&GpuProfilerHandle>,
    ) {
        encode::encode_resolve(self, device, encoder, extent, targets, limits, profiler);
    }

    /// Stereo (OpenXR multiview) MSAA depth resolve.
    ///
    /// See [`MsaaDepthResolveStereoTargets`] for the view layout. Issues two compute dispatches
    /// (one per eye) because WGSL lacks `texture_depth_multisampled_2d_array` today, then one
    /// multiview blit pass (`multiview_mask = 0b11`) that writes both depth layers via
    /// `@builtin(view_index)`.
    ///
    /// Does nothing when [`wgpu::Features::MULTIVIEW`] was unavailable at construction
    /// (stereo MSAA is implicitly off in that case via the feature mask in the XR bootstrap).
    ///
    /// See [`Self::encode_resolve`] for compute dispatch limit handling.
    ///
    /// `profiler` opens pass-level GPU timestamp queries around each per-eye compute pass and the
    /// final multiview blit pass.
    pub fn encode_resolve_stereo(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        extent: (u32, u32),
        targets: MsaaDepthResolveStereoTargets<'_>,
        limits: &GpuLimits,
        profiler: Option<&GpuProfilerHandle>,
    ) {
        encode::encode_resolve_stereo(self, device, encoder, extent, targets, limits, profiler);
    }

    fn blit_pipeline_for_format(
        &self,
        format: wgpu::TextureFormat,
    ) -> Option<&wgpu::RenderPipeline> {
        match format {
            wgpu::TextureFormat::Depth24PlusStencil8 => Some(&self.blit_pipeline_depth24_stencil8),
            wgpu::TextureFormat::Depth32FloatStencil8 => {
                self.blit_pipeline_depth32_stencil8.as_ref()
            }
            _ => Some(&self.blit_pipeline_depth32),
        }
    }

    fn stereo_blit_pipeline_for_format(
        &self,
        format: wgpu::TextureFormat,
    ) -> Option<&wgpu::RenderPipeline> {
        match format {
            wgpu::TextureFormat::Depth24PlusStencil8 => {
                self.blit_stereo_pipeline_depth24_stencil8.as_ref()
            }
            wgpu::TextureFormat::Depth32FloatStencil8 => {
                self.blit_stereo_pipeline_depth32_stencil8.as_ref()
            }
            _ => self.blit_stereo_pipeline_depth32.as_ref(),
        }
    }

    fn compute_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.compute_pipeline
    }

    fn compute_bgl(&self) -> &wgpu::BindGroupLayout {
        &self.compute_bgl
    }

    fn blit_bgl(&self) -> &wgpu::BindGroupLayout {
        &self.blit_bgl
    }

    fn blit_stereo_bgl(&self) -> Option<&wgpu::BindGroupLayout> {
        self.blit_stereo_bgl.as_ref()
    }
}
