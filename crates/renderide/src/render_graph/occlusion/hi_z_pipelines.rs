//! Cached compute pipelines and bind group layouts for Hi-Z pyramid construction.
//!
//! WGSL is sourced from the build-time embedded shader registry
//! ([`crate::embedded_shaders::embedded_target_wgsl`]). The mip0 shader lives in a single
//! source with `#ifdef MULTIVIEW` and is composed into `hi_z_mip0_default` (2D source) and
//! `hi_z_mip0_multiview` (2D array source + per-dispatch layer uniform).

use std::num::NonZeroU64;
use std::sync::OnceLock;

use crate::embedded_shaders::embedded_target_wgsl;

/// Embedded shader stem for the desktop (non-multiview) mip0 compute.
const MIP0_STEM_DESKTOP: &str = "hi_z_mip0_default";
/// Embedded shader stem for the stereo (multiview) mip0 compute.
const MIP0_STEM_STEREO: &str = "hi_z_mip0_multiview";
/// Embedded shader stem for the pyramid downsample compute (single-variant).
const DOWNSAMPLE_STEM: &str = "hi_z_downsample_max";

fn load_embedded(stem: &str) -> &'static str {
    #[expect(
        clippy::expect_used,
        reason = "embedded shader is required; absence is a build script regression"
    )]
    embedded_target_wgsl(stem)
        .expect("hi_z_pipelines: embedded shader missing (build script regression)")
}

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
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::R32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    })
}

/// Bind group layout: depth array sample + layer uniform + mip0 R32F storage write.
fn hi_z_create_bgl_mip0_stereo(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hi_z_mip0_stereo"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(16),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::R32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    })
}

/// Bind group layout: adjacent pyramid mips + downsample dimensions uniform.
fn hi_z_create_bgl_downsample(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hi_z_downsample"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::ReadOnly,
                    format: wgpu::TextureFormat::R32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::R32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(16),
                },
                count: None,
            },
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
            source: wgpu::ShaderSource::Wgsl(load_embedded(MIP0_STEM_DESKTOP).into()),
        });
        let shader_m0s = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hi_z_mip0_stereo"),
            source: wgpu::ShaderSource::Wgsl(load_embedded(MIP0_STEM_STEREO).into()),
        });
        let shader_ds = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hi_z_downsample"),
            source: wgpu::ShaderSource::Wgsl(load_embedded(DOWNSAMPLE_STEM).into()),
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
