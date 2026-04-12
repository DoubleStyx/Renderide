//! Cached compute pipelines and bind group layouts for Hi-Z pyramid construction.

use std::num::NonZeroU64;
use std::sync::OnceLock;

const MIP0_DESKTOP_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/source/compute/hi_z_mip0_desktop.wgsl"
));
const MIP0_STEREO_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/source/compute/hi_z_mip0_stereo.wgsl"
));
const DOWNSAMPLE_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/source/compute/hi_z_downsample_max.wgsl"
));

pub(crate) struct HiZPipelines {
    pub mip0_desktop: wgpu::ComputePipeline,
    pub mip0_stereo: wgpu::ComputePipeline,
    pub downsample: wgpu::ComputePipeline,
    pub bgl_mip0_desktop: wgpu::BindGroupLayout,
    pub bgl_mip0_stereo: wgpu::BindGroupLayout,
    pub bgl_downsample: wgpu::BindGroupLayout,
}

impl HiZPipelines {
    pub(crate) fn get(device: &wgpu::Device) -> &'static Self {
        static CACHE: OnceLock<HiZPipelines> = OnceLock::new();
        CACHE.get_or_init(|| Self::new(device))
    }

    fn new(device: &wgpu::Device) -> Self {
        let bgl_mip0_desktop = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        });

        let bgl_mip0_stereo = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        });

        let bgl_downsample = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        });

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
            source: wgpu::ShaderSource::Wgsl(MIP0_DESKTOP_SRC.into()),
        });
        let shader_m0s = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hi_z_mip0_stereo"),
            source: wgpu::ShaderSource::Wgsl(MIP0_STEREO_SRC.into()),
        });
        let shader_ds = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hi_z_downsample"),
            source: wgpu::ShaderSource::Wgsl(DOWNSAMPLE_SRC.into()),
        });

        let mip0_desktop = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hi_z_mip0_desktop"),
            layout: Some(&layout_mip0_d),
            module: &shader_m0d,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let mip0_stereo = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hi_z_mip0_stereo"),
            layout: Some(&layout_mip0_s),
            module: &shader_m0s,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let downsample = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hi_z_downsample"),
            layout: Some(&layout_ds),
            module: &shader_ds,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });

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
