//! Cached compute pipelines and bind group layouts for Hi-Z pyramid construction.
//!
//! WGSL is sourced from the build-time embedded shader registry. The mip0 shader lives in a
//! single source with `#ifdef MULTIVIEW` and is composed into `hi_z_mip0_default` (2D source)
//! and `hi_z_mip0_multiview` (2D array source + per-dispatch layer uniform).

use std::num::NonZeroU64;
use std::sync::OnceLock;

use crate::embedded_shaders::{
    HI_Z_DOWNSAMPLE_MAX_WGSL, HI_Z_MIP0_DEFAULT_WGSL, HI_Z_MIP0_MULTIVIEW_WGSL,
};
use crate::gpu::bind_layout::{
    storage_texture_layout_entry, texture_layout_entry, uniform_buffer_layout_entry,
};

pub(crate) struct HiZPipelines {
    pub mip0_desktop: wgpu::ComputePipeline,
    pub mip0_stereo: wgpu::ComputePipeline,
    pub downsample: wgpu::ComputePipeline,
    pub bgl_mip0_desktop: wgpu::BindGroupLayout,
    pub bgl_mip0_stereo: wgpu::BindGroupLayout,
    pub bgl_downsample: wgpu::BindGroupLayout,
}

/// Bind group layout: depth `D2` sample + mip0 R32F storage write.
fn hi_z_create_bgl_mip0_desktop(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hi_z_mip0_desktop"),
        entries: &[
            texture_layout_entry(
                0,
                wgpu::ShaderStages::COMPUTE,
                wgpu::TextureSampleType::Depth,
                wgpu::TextureViewDimension::D2,
                false,
            ),
            storage_texture_layout_entry(
                1,
                wgpu::ShaderStages::COMPUTE,
                wgpu::StorageTextureAccess::WriteOnly,
                wgpu::TextureFormat::R32Float,
                wgpu::TextureViewDimension::D2,
            ),
        ],
    })
}

/// Bind group layout: depth array sample + layer uniform + mip0 R32F storage write.
fn hi_z_create_bgl_mip0_stereo(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hi_z_mip0_stereo"),
        entries: &[
            texture_layout_entry(
                0,
                wgpu::ShaderStages::COMPUTE,
                wgpu::TextureSampleType::Depth,
                wgpu::TextureViewDimension::D2Array,
                false,
            ),
            uniform_buffer_layout_entry(1, wgpu::ShaderStages::COMPUTE, NonZeroU64::new(16)),
            storage_texture_layout_entry(
                2,
                wgpu::ShaderStages::COMPUTE,
                wgpu::StorageTextureAccess::WriteOnly,
                wgpu::TextureFormat::R32Float,
                wgpu::TextureViewDimension::D2,
            ),
        ],
    })
}

/// Bind group layout: adjacent pyramid mips + downsample dimensions uniform.
fn hi_z_create_bgl_downsample(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hi_z_downsample"),
        entries: &[
            storage_texture_layout_entry(
                0,
                wgpu::ShaderStages::COMPUTE,
                wgpu::StorageTextureAccess::ReadOnly,
                wgpu::TextureFormat::R32Float,
                wgpu::TextureViewDimension::D2,
            ),
            storage_texture_layout_entry(
                1,
                wgpu::ShaderStages::COMPUTE,
                wgpu::StorageTextureAccess::WriteOnly,
                wgpu::TextureFormat::R32Float,
                wgpu::TextureViewDimension::D2,
            ),
            uniform_buffer_layout_entry(2, wgpu::ShaderStages::COMPUTE, NonZeroU64::new(16)),
        ],
    })
}

fn hi_z_create_compute_pipeline(
    device: &wgpu::Device,
    label: &'static str,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module,
        entry_point: Some("cs_main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

impl HiZPipelines {
    pub(crate) fn get(device: &wgpu::Device) -> &'static Self {
        static CACHE: OnceLock<HiZPipelines> = OnceLock::new();
        CACHE.get_or_init(|| Self::new(device))
    }

    fn new(device: &wgpu::Device) -> Self {
        let bgl_mip0_desktop = hi_z_create_bgl_mip0_desktop(device);
        let bgl_mip0_stereo = hi_z_create_bgl_mip0_stereo(device);
        let bgl_downsample = hi_z_create_bgl_downsample(device);

        let layout_mip0_d = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hi_z_mip0_desktop_layout"),
            bind_group_layouts: &[Some(&bgl_mip0_desktop)],
            immediate_size: 0,
        });
        let layout_mip0_s = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hi_z_mip0_stereo_layout"),
            bind_group_layouts: &[Some(&bgl_mip0_stereo)],
            immediate_size: 0,
        });
        let layout_ds = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hi_z_downsample_layout"),
            bind_group_layouts: &[Some(&bgl_downsample)],
            immediate_size: 0,
        });

        let shader_m0d = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hi_z_mip0_desktop"),
            source: wgpu::ShaderSource::Wgsl(HI_Z_MIP0_DEFAULT_WGSL.into()),
        });
        let shader_m0s = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hi_z_mip0_stereo"),
            source: wgpu::ShaderSource::Wgsl(HI_Z_MIP0_MULTIVIEW_WGSL.into()),
        });
        let shader_ds = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hi_z_downsample"),
            source: wgpu::ShaderSource::Wgsl(HI_Z_DOWNSAMPLE_MAX_WGSL.into()),
        });

        let mip0_desktop =
            hi_z_create_compute_pipeline(device, "hi_z_mip0_desktop", &layout_mip0_d, &shader_m0d);
        let mip0_stereo =
            hi_z_create_compute_pipeline(device, "hi_z_mip0_stereo", &layout_mip0_s, &shader_m0s);
        let downsample =
            hi_z_create_compute_pipeline(device, "hi_z_downsample", &layout_ds, &shader_ds);

        Self {
            mip0_desktop,
            mip0_stereo,
            downsample,
            bgl_mip0_desktop,
            bgl_mip0_stereo,
            bgl_downsample,
        }
    }
}
