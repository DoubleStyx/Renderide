//! VR desktop mirror: copy one HMD eye into a staging texture, then blit to the window surface.
//!
//! The surface blit uses **cover** (fill) mapping: the window is filled with a uniform scale of the
//! staging texture; aspect mismatch is resolved by cropping the center (no letterboxing).
//!
//! Used instead of a second full world render when OpenXR multiview has already drawn the scene.

use std::sync::OnceLock;

use bytemuck::{Pod, Zeroable};
use winit::window::Window;

use crate::gpu::GpuContext;
use crate::present::{acquire_surface_outcome, PresentClearError, SurfaceFrameOutcome};
use crate::xr::XR_COLOR_FORMAT;

const EYE_TO_STAGING_WGSL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/source/present/vr_mirror_eye_to_staging.wgsl"
));
const SURFACE_BLIT_WGSL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/source/present/vr_mirror_surface.wgsl"
));

/// OpenXR `PRIMARY_STEREO` layer index used for the desktop mirror (left eye).
pub const VR_MIRROR_EYE_LAYER: u32 = 0;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SurfaceBlitUniform {
    uv_scale: [f32; 2],
    uv_offset: [f32; 2],
}

/// UV transform for [`cover_uv_params`].
///
/// Implements CSS **object-fit: cover**: the staging texture is scaled uniformly so the window is
/// fully covered; excess is cropped from the center (no black bars).
///
/// Compare window aspect `W_s/H_s` to staging aspect `W_t/H_t`: if the window is wider (larger
/// aspect ratio), crop top/bottom in texture space; if the window is taller (smaller aspect ratio),
/// crop left/right.
///
/// Shader: `tuv = uv * uv_scale + uv_offset` maps screen `uv` in [0, 1]² into texture UV in a centered
/// sub-rectangle of [0, 1]².
fn cover_uv_params(eye_w: u32, eye_h: u32, surf_w: u32, surf_h: u32) -> SurfaceBlitUniform {
    let ew = eye_w.max(1) as f32;
    let eh = eye_h.max(1) as f32;
    let sw = surf_w.max(1) as f32;
    let sh = surf_h.max(1) as f32;
    let eye_aspect = ew / eh;
    let surf_aspect = sw / sh;
    if surf_aspect > eye_aspect {
        // Window is wider than the staging texture aspect (R_s > R_t): cover crops top/bottom.
        let frac = eye_aspect / surf_aspect;
        SurfaceBlitUniform {
            uv_scale: [1.0, frac],
            uv_offset: [0.0, (1.0 - frac) * 0.5],
        }
    } else {
        // Window is taller or narrower (R_s <= R_t): cover crops left/right.
        let frac = surf_aspect / eye_aspect;
        SurfaceBlitUniform {
            uv_scale: [frac, 1.0],
            uv_offset: [(1.0 - frac) * 0.5, 0.0],
        }
    }
}

fn eye_pipeline(device: &wgpu::Device) -> &'static wgpu::RenderPipeline {
    static PIPE: OnceLock<wgpu::RenderPipeline> = OnceLock::new();
    PIPE.get_or_init(|| {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vr_mirror_eye_to_staging"),
            source: wgpu::ShaderSource::Wgsl(EYE_TO_STAGING_WGSL.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vr_mirror_eye_to_staging"),
            bind_group_layouts: &[Some(eye_bind_group_layout(device))],
            immediate_size: 0,
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("vr_mirror_eye_to_staging"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: XR_COLOR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        })
    })
}

fn eye_bind_group_layout(device: &wgpu::Device) -> &'static wgpu::BindGroupLayout {
    static LAYOUT: OnceLock<wgpu::BindGroupLayout> = OnceLock::new();
    LAYOUT.get_or_init(|| {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vr_mirror_eye_to_staging"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    })
}

fn surface_bind_group_layout(device: &wgpu::Device) -> &'static wgpu::BindGroupLayout {
    static LAYOUT: OnceLock<wgpu::BindGroupLayout> = OnceLock::new();
    LAYOUT.get_or_init(|| {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vr_mirror_surface"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(16),
                    },
                    count: None,
                },
            ],
        })
    })
}

fn surface_pipeline(device: &wgpu::Device, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("vr_mirror_surface"),
        source: wgpu::ShaderSource::Wgsl(SURFACE_BLIT_WGSL.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("vr_mirror_surface"),
        bind_group_layouts: &[Some(surface_bind_group_layout(device))],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("vr_mirror_surface"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn linear_sampler(device: &wgpu::Device) -> &'static wgpu::Sampler {
    static SAMPLER: OnceLock<wgpu::Sampler> = OnceLock::new();
    SAMPLER.get_or_init(|| {
        device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("vr_mirror_linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        })
    })
}

/// GPU resources for VR mirror blit (staging texture + pipelines).
pub struct VrMirrorBlitResources {
    staging_texture: Option<wgpu::Texture>,
    staging_extent: (u32, u32),
    /// `true` after a successful eye→staging copy this session.
    pub staging_valid: bool,
    surface_uniform: Option<wgpu::Buffer>,
    surface_pipeline: Option<(wgpu::TextureFormat, wgpu::RenderPipeline)>,
}

impl Default for VrMirrorBlitResources {
    fn default() -> Self {
        Self::new()
    }
}

impl VrMirrorBlitResources {
    /// Empty resources; staging is allocated on first successful HMD frame.
    pub fn new() -> Self {
        Self {
            staging_texture: None,
            staging_extent: (0, 0),
            staging_valid: false,
            surface_uniform: None,
            surface_pipeline: None,
        }
    }

    fn ensure_staging(&mut self, device: &wgpu::Device, extent: (u32, u32)) {
        if self.staging_extent == extent && self.staging_texture.is_some() {
            return;
        }
        let w = extent.0.max(1);
        let h = extent.1.max(1);
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vr_mirror_staging"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: XR_COLOR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.staging_texture = Some(tex);
        self.staging_extent = extent;
    }

    fn ensure_surface_uniform(&mut self, device: &wgpu::Device) {
        if self.surface_uniform.is_some() {
            return;
        }
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vr_mirror_surface_uv"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.surface_uniform = Some(buf);
    }

    fn surface_pipeline_for_format(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> &wgpu::RenderPipeline {
        let entry = self
            .surface_pipeline
            .get_or_insert_with(|| (format, surface_pipeline(device, format)));
        if entry.0 != format {
            *entry = (format, surface_pipeline(device, format));
        }
        &entry.1
    }

    /// Copies the acquired swapchain eye layer into the staging texture and submits GPU work.
    ///
    /// Call after the multiview render graph submit, before [`openxr::Swapchain::release_image`].
    pub fn submit_eye_to_staging(
        &mut self,
        gpu: &GpuContext,
        eye_extent: (u32, u32),
        source_layer_view: &wgpu::TextureView,
    ) {
        let device = gpu.device().as_ref();
        self.ensure_staging(device, eye_extent);
        self.ensure_surface_uniform(device);

        let Some(staging_tex) = self.staging_texture.as_ref() else {
            return;
        };
        let staging_view = staging_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = linear_sampler(device);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vr_mirror_eye_to_staging"),
            layout: eye_bind_group_layout(device),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_layer_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vr_mirror_eye_to_staging"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vr_mirror_eye_to_staging"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &staging_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_pipeline(eye_pipeline(device));
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        gpu.submit_tracked_frame_commands(encoder.finish());
        self.staging_valid = true;
    }

    /// Blits staging to the window with **cover** mapping (fills the window, crops staging as needed).
    /// No-op when [`Self::staging_valid`] is false
    /// (caller may [`crate::present::present_clear_frame`] instead).
    pub fn present_staging_to_surface(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
    ) -> Result<(), PresentClearError> {
        if !self.staging_valid {
            return Ok(());
        }
        if self.staging_texture.is_none() {
            return Ok(());
        }

        let frame = match acquire_surface_outcome(gpu, window)? {
            SurfaceFrameOutcome::Skip | SurfaceFrameOutcome::Reconfigured => return Ok(()),
            SurfaceFrameOutcome::Acquired(f) => f,
        };

        let surface_format = gpu.config_format();
        let (sw, sh) = gpu.surface_extent_px();
        let sw = sw.max(1);
        let sh = sh.max(1);
        let (ew, eh) = (self.staging_extent.0.max(1), self.staging_extent.1.max(1));

        let u = cover_uv_params(ew, eh, sw, sh);
        let uniform_bytes = bytemuck::bytes_of(&u);
        let device = gpu.device().as_ref();
        self.ensure_surface_uniform(device);
        let Some(uniform_buf) = self.surface_uniform.as_ref() else {
            logger::warn!(
                "vr_mirror_blit: surface uniform buffer missing after ensure_surface_uniform"
            );
            return Ok(());
        };
        {
            let q = gpu.queue().lock().unwrap_or_else(|e| e.into_inner());
            q.write_buffer(uniform_buf, 0, uniform_bytes);
        }

        let Some(staging_tex) = self.staging_texture.as_ref() else {
            return Ok(());
        };
        let staging_view = staging_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let surface_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = linear_sampler(device);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vr_mirror_surface"),
            layout: surface_bind_group_layout(device),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&staging_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        let pipeline = self.surface_pipeline_for_format(device, surface_format);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vr_mirror_surface"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vr_mirror_surface"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        gpu.submit_tracked_frame_commands(encoder.finish());
        frame.present();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::cover_uv_params;

    #[test]
    fn cover_uv_center_crop_when_surface_wider_than_eye() {
        // 2:1 window, 1:1 eye (R_s > R_t) → crop texture top/bottom (cover).
        let u = cover_uv_params(100, 100, 200, 100);
        assert!((u.uv_scale[0] - 1.0).abs() < 1e-5);
        assert!((u.uv_scale[1] - 0.5).abs() < 1e-5);
        assert!((u.uv_offset[0] - 0.0).abs() < 1e-5);
        assert!((u.uv_offset[1] - 0.25).abs() < 1e-5);
    }

    #[test]
    fn cover_uv_center_crop_when_surface_taller_than_eye() {
        // 1:2 window, 1:1 eye (R_s < R_t) → crop texture left/right (cover).
        let u = cover_uv_params(100, 100, 100, 200);
        assert!((u.uv_scale[0] - 0.5).abs() < 1e-5);
        assert!((u.uv_scale[1] - 1.0).abs() < 1e-5);
        assert!((u.uv_offset[0] - 0.25).abs() < 1e-5);
        assert!((u.uv_offset[1] - 0.0).abs() < 1e-5);
    }
}
