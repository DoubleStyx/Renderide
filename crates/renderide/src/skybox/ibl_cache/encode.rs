//! Command encoding for IBL mip-0 producers and GGX convolve passes.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::profiling::{GpuProfilerHandle, compute_pass_timestamp_writes};
use crate::skybox::params::SkyboxEvaluatorParams;
use crate::skybox::specular::{CubemapIblSource, EquirectIblSource};

use super::key::{convolve_sample_count, dispatch_groups, mip_extent};
use super::pipeline::ComputePipeline;
use super::resources::{PendingBakeResources, create_mip_storage_view};

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

/// Inputs for [`encode_analytic_mip0`].
pub(super) struct AnalyticEncodeContext<'a> {
    pub(super) device: &'a wgpu::Device,
    pub(super) encoder: &'a mut wgpu::CommandEncoder,
    pub(super) pipeline: &'a ComputePipeline,
    pub(super) texture: &'a wgpu::Texture,
    pub(super) face_size: u32,
    pub(super) params: &'a SkyboxEvaluatorParams,
    pub(super) profiler: Option<&'a GpuProfilerHandle>,
}

/// Encodes mip 0 from analytic procedural / gradient sky parameters.
pub(super) fn encode_analytic_mip0(
    ctx: AnalyticEncodeContext<'_>,
    resources: &mut PendingBakeResources,
) {
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
pub(super) struct CubeEncodeContext<'a> {
    pub(super) device: &'a wgpu::Device,
    pub(super) encoder: &'a mut wgpu::CommandEncoder,
    pub(super) pipeline: &'a ComputePipeline,
    pub(super) texture: &'a wgpu::Texture,
    pub(super) face_size: u32,
    pub(super) src: CubemapIblSource,
    pub(super) sampler: &'a wgpu::Sampler,
    pub(super) profiler: Option<&'a GpuProfilerHandle>,
}

/// Encodes mip 0 by resampling a host cubemap source.
pub(super) fn encode_cube_mip0(ctx: CubeEncodeContext<'_>, resources: &mut PendingBakeResources) {
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
pub(super) struct EquirectEncodeContext<'a> {
    pub(super) device: &'a wgpu::Device,
    pub(super) encoder: &'a mut wgpu::CommandEncoder,
    pub(super) pipeline: &'a ComputePipeline,
    pub(super) texture: &'a wgpu::Texture,
    pub(super) face_size: u32,
    pub(super) src: EquirectIblSource,
    pub(super) sampler: &'a wgpu::Sampler,
    pub(super) profiler: Option<&'a GpuProfilerHandle>,
}

/// Encodes mip 0 by resampling an equirect Texture2D source.
pub(super) fn encode_equirect_mip0(
    ctx: EquirectEncodeContext<'_>,
    resources: &mut PendingBakeResources,
) {
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
pub(super) struct ConvolveEncodeContext<'a> {
    pub(super) device: &'a wgpu::Device,
    pub(super) encoder: &'a mut wgpu::CommandEncoder,
    pub(super) pipeline: &'a ComputePipeline,
    pub(super) texture: &'a wgpu::Texture,
    pub(super) src_view: &'a wgpu::TextureView,
    pub(super) sampler: &'a wgpu::Sampler,
    pub(super) face_size: u32,
    pub(super) mip_levels: u32,
    pub(super) profiler: Option<&'a GpuProfilerHandle>,
}

/// Encodes the GGX convolve passes for mips `1..mip_levels` of the destination cube.
pub(super) fn encode_convolve_mips(
    ctx: ConvolveEncodeContext<'_>,
    resources: &mut PendingBakeResources,
) {
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
