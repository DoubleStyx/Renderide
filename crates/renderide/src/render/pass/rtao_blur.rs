//! RTAO denoiser: 3-pass À-Trous wavelet filter.
//!
//! Replaces the old single 5×5 cross-bilateral pass with three compute dispatches
//! at step sizes 1 → 2 → 4.  Each pass applies a 5×5 B3-spline kernel with
//! depth and normal edge-stopping weights, ping-ponging between the raw-AO and
//! final-AO textures so no extra GPU allocation is needed.
//!
//! ## Bypass for experiments
//!
//! [`RTAO_ATROUS_BLUR_ENABLED`] gates the À-Trous passes.  When `false`, this pass
//! only copies [`ResourceSlot::AoRaw`] into [`ResourceSlot::Ao`] so
//! [`super::composite::CompositePass`] sees sparse per-pixel RTAO with no spatial
//! smoothing (useful for A/B testing or when evaluating a separate denoiser path).
//!
//! ## Why À-Trous?
//!
//! A single 5×5 kernel can only smooth noise within a 5-pixel radius.  The RTAO
//! compute pass uses a small fixed ray count per pixel so the raw signal is still
//! noisy; a small
//! kernel leaves visible grain, and a large kernel blurs detail.  À-Trous doubles
//! the effective radius each pass (1 → 2 → 4 pixel strides) reaching an 8-pixel
//! support with only three passes, while the Gaussian weights and edge-stopping
//! preserve contacts and silhouettes.
//!
//! ## Ping-pong layout
//!
//! | pass | step | reads     | writes    |
//! |------|------|-----------|-----------|
//! |  0   |  1   | `ao_raw`  | `ao`      |
//! |  1   |  2   | `ao`      | `ao_raw`  |
//! |  2   |  4   | `ao_raw`  | `ao`      |
//!
//! Both textures were created with `STORAGE_BINDING | TEXTURE_BINDING` so each can
//! appear in either role.  The final result always lands in `ao`, which is what
//! [`super::composite::CompositePass`] samples.
//!
//! ## Edge stopping
//!
//! - **Spatial**: B3-spline 5-tap weights `[1/16, 4/16, 6/16, 4/16, 1/16]` (separable).
//! - **Depth**: `exp(-|Δdepth| × 200 / step_size)` — tighter at small steps.
//! - **Normal**: `max(0, dot(n_s, n_c))^8` — very sharp at geometric edges, prevents
//!   dark plant/foliage AO from bleeding onto surrounding floors and walls.
//! - Background pixels (depth ≈ 0 in reverse-Z, or zero-length normal) are
//!   passed through unchanged so the sky stays at full visibility (AO = 1).

use super::{PassResources, RenderPass, RenderPassError, ResourceSlot};

const TILE_SIZE: u32 = 8;

/// When `true`, runs the three-pass À-Trous bilateral filter from raw AO into `ao`.
///
/// When `false`, copies raw AO into `ao` unchanged so composite uses unfiltered RTAO.
pub const RTAO_ATROUS_BLUR_ENABLED: bool = true;

/// WGSL compute shader that copies `ao_raw` into `ao` (same format, no filtering).
const RTAO_AO_RAW_COPY_SHADER_SRC: &str = r#"
@group(0) @binding(0) var ao_input:  texture_2d<f32>;
@group(0) @binding(1) var ao_output: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(ao_input);
    if id.x >= dims.x || id.y >= dims.y {
        return;
    }
    let p = vec2i(id.xy);
    let v = textureLoad(ao_input, p, 0);
    textureStore(ao_output, p, v);
}
"#;

/// WGSL shader for one À-Trous pass.
///
/// Binding 4 carries a 16-byte uniform with `step_size` (i32).  The caller
/// creates three pre-initialised uniform buffers (step = 1, 2, 4) and selects
/// the right one per dispatch; the shader itself is compiled once.
const RTAO_ATROUS_SHADER_SRC: &str = r#"
struct BlurUniforms {
    step_size: i32,
    _pad0:     i32,
    _pad1:     i32,
    _pad2:     i32,
}

@group(0) @binding(0) var ao_input:  texture_2d<f32>;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var ao_output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<uniform> uniforms: BlurUniforms;

// B3-spline 5-tap kernel: [1/16, 4/16, 6/16, 4/16, 1/16].
// Applied separably in X and Y; the 2-D weight is the product of the two 1-D values.
const KERNEL: array<f32, 5> = array<f32, 5>(
    0.0625, 0.25, 0.375, 0.25, 0.0625
);

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(ao_input);
    if id.x >= dims.x || id.y >= dims.y { return; }

    let cp   = vec2i(id.xy);
    let step = uniforms.step_size;

    let center_depth  = textureLoad(depth_tex,  cp, 0);
    let center_normal = textureLoad(normal_tex,  cp, 0).xyz;
    let cn_len        = length(center_normal);

    // Background / sky: depth == 0.0 in reverse-Z means far plane — nothing was
    // rendered.  Zero normal also indicates an empty pixel.  Pass AO through
    // unchanged so the sky stays at full visibility.
    if center_depth == 0.0 || cn_len < 0.1 {
        let passthru = textureLoad(ao_input, cp, 0).r;
        textureStore(ao_output, cp, vec4f(passthru, 0.0, 0.0, 1.0));
        return;
    }
    let cn = center_normal / cn_len;    // safe normalise

    var sum        = 0.0;
    var weight_sum = 0.0;

    for (var ky = 0; ky < 5; ky++) {
        for (var kx = 0; kx < 5; kx++) {
            let sp = cp + vec2i((kx - 2) * step, (ky - 2) * step);

            // Clamp to texture border — skip out-of-bounds samples entirely.
            if sp.x < 0 || sp.y < 0 || sp.x >= i32(dims.x) || sp.y >= i32(dims.y) {
                continue;
            }

            let ao_s     = textureLoad(ao_input,  sp, 0).r;
            let depth_s  = textureLoad(depth_tex, sp, 0);
            let normal_s = textureLoad(normal_tex, sp, 0).xyz;
            let ns_len   = length(normal_s);

            // Skip background / degenerate samples — they must not bleed into geometry.
            if depth_s == 0.0 || ns_len < 0.1 { continue; }
            let ns = normal_s / ns_len;

            // ── Spatial weight (B3-spline, separable) ─────────────────────
            let w_spatial = KERNEL[kx] * KERNEL[ky];

            // ── Depth edge-stopping ────────────────────────────────────────
            // 200/step: tighter than before so AO cannot bleed across depth
            // discontinuities (e.g. plant leaves vs. floor).
            let depth_diff = abs(depth_s - center_depth);
            let w_depth    = exp(-depth_diff * (200.0 / f32(step)));

            // ── Normal edge-stopping (^8 = very sharp, stops foliage bleed) ─
            let n_dot    = max(0.0, dot(ns, cn));
            let n2       = n_dot * n_dot;
            let n4       = n2 * n2;
            let w_normal = n4 * n4;   // n_dot^8

            let w = w_spatial * w_depth * w_normal;
            sum        += ao_s * w;
            weight_sum += w;
        }
    }

    // Guard against weight collapse (all neighbours were background / edge-rejected).
    let ao_center = textureLoad(ao_input, cp, 0).r;
    let final_ao  = select(ao_center, sum / weight_sum, weight_sum > 1e-6);
    textureStore(ao_output, cp, vec4f(final_ao, 0.0, 0.0, 1.0));
}
"#;

// ─── Step-size uniform buffer layout ─────────────────────────────────────────
// [step_size: i32, _pad × 3] → 16 bytes, 16-byte aligned.
const STEP_UNIFORM_SIZE: u64 = 16;

/// Creates a small pre-initialised uniform buffer holding a single `step_size` value.
///
/// Uses `mapped_at_creation` so no queue write is needed; the buffer is immutable
/// after creation (`UNIFORM` usage only — no `COPY_DST`).
fn create_step_buffer(device: &wgpu::Device, step: i32) -> wgpu::Buffer {
    let data: [i32; 4] = [step, 0, 0, 0];
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("RTAO atrous step uniform"),
        size: STEP_UNIFORM_SIZE,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: true,
    });
    {
        let mut view = buf.slice(..).get_mapped_range_mut();
        view.copy_from_slice(bytemuck::cast_slice(&data));
    }
    buf.unmap();
    buf
}

/// RTAO À-Trous denoiser pass.
///
/// When [`RTAO_ATROUS_BLUR_ENABLED`] is `true`, runs three compute dispatches
/// (step = 1, 2, 4) on the À-Trous shader, each reading from one AO texture and
/// writing to the other.  When `false`, copies [`ResourceSlot::AoRaw`] into
/// [`ResourceSlot::Ao`] with a single dispatch.
///
/// The result always lands in `ao` (the texture read by
/// [`super::composite::CompositePass`]).  Pipelines are created lazily on first
/// use and reused every frame.  Skips cleanly when MRT views are unavailable.
pub struct RtaoBlurPass {
    pipeline: Option<wgpu::ComputePipeline>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// Pre-initialised uniform buffers for step sizes [1, 2, 4].
    step_buffers: Option<[wgpu::Buffer; 3]>,
    /// Raw-AO copy path when [`RTAO_ATROUS_BLUR_ENABLED`] is `false`.
    raw_copy_pipeline: Option<wgpu::ComputePipeline>,
    raw_copy_bind_group_layout: Option<wgpu::BindGroupLayout>,
}

impl RtaoBlurPass {
    /// Creates a new RTAO blur pass. Pipeline is built lazily on first use.
    pub fn new() -> Self {
        Self {
            pipeline: None,
            bind_group_layout: None,
            step_buffers: None,
            raw_copy_pipeline: None,
            raw_copy_bind_group_layout: None,
        }
    }

    fn ensure_raw_copy_pipeline(
        &mut self,
        device: &wgpu::Device,
    ) -> Option<(&wgpu::ComputePipeline, &wgpu::BindGroupLayout)> {
        if self.raw_copy_pipeline.is_none() {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("RTAO ao_raw copy shader"),
                source: wgpu::ShaderSource::Wgsl(RTAO_AO_RAW_COPY_SHADER_SRC.into()),
            });
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("RTAO ao_raw copy BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("RTAO ao_raw copy pipeline layout"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });
            self.raw_copy_pipeline = Some(device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("RTAO ao_raw copy pipeline"),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: None,
                    compilation_options: Default::default(),
                    cache: None,
                },
            ));
            self.raw_copy_bind_group_layout = Some(bgl);
        }
        match (
            self.raw_copy_pipeline.as_ref(),
            self.raw_copy_bind_group_layout.as_ref(),
        ) {
            (Some(p), Some(b)) => Some((p, b)),
            _ => None,
        }
    }

    fn ensure_pipeline(
        &mut self,
        device: &wgpu::Device,
    ) -> Option<(
        &wgpu::ComputePipeline,
        &wgpu::BindGroupLayout,
        &[wgpu::Buffer; 3],
    )> {
        if self.pipeline.is_none() {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("RTAO atrous blur shader"),
                source: wgpu::ShaderSource::Wgsl(RTAO_ATROUS_SHADER_SRC.into()),
            });

            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("RTAO atrous blur BGL"),
                entries: &[
                    // 0: ao_input (texture_2d<f32>)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // 1: depth_tex (texture_depth_2d)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // 2: normal_tex (texture_2d<f32>)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // 3: ao_output (storage texture, write-only)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // 4: uniforms (step_size, 16 bytes)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: std::num::NonZeroU64::new(STEP_UNIFORM_SIZE),
                        },
                        count: None,
                    },
                ],
            });

            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("RTAO atrous blur pipeline layout"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });

            self.pipeline = Some(device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("RTAO atrous blur pipeline"),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: None,
                    compilation_options: Default::default(),
                    cache: None,
                },
            ));
            self.bind_group_layout = Some(bgl);

            // Pre-create one uniform buffer per À-Trous step size.
            self.step_buffers = Some([
                create_step_buffer(device, 1),
                create_step_buffer(device, 2),
                create_step_buffer(device, 4),
            ]);
        }

        match (
            self.pipeline.as_ref(),
            self.bind_group_layout.as_ref(),
            self.step_buffers.as_ref(),
        ) {
            (Some(p), Some(b), Some(s)) => Some((p, b, s)),
            _ => None,
        }
    }
}

impl RenderPass for RtaoBlurPass {
    fn name(&self) -> &str {
        "rtao_blur"
    }

    /// Declares G-buffer inputs used by the À-Trous shader.
    ///
    /// When [`RTAO_ATROUS_BLUR_ENABLED`] is `false`, [`Self::execute`] only copies
    /// raw AO and does not sample depth or normals; those slots stay declared so
    /// render-target wiring and pass ordering match the filtered path.
    ///
    /// [`ResourceSlot::Normal`] is required so per-pass [`super::RenderTargetViews`]
    /// wiring fills [`super::RenderTargetViews::mrt_normal_view`].
    /// Both [`ResourceSlot::AoRaw`] and [`ResourceSlot::Ao`] must be accessible
    /// so the ping-pong dispatches can swap input and output each pass.
    fn resources(&self) -> PassResources {
        PassResources {
            reads: vec![
                ResourceSlot::AoRaw,
                ResourceSlot::Depth,
                ResourceSlot::Normal,
            ],
            writes: vec![ResourceSlot::Ao],
        }
    }

    fn execute(&mut self, ctx: &mut super::RenderPassContext) -> Result<(), RenderPassError> {
        // Both AO views are needed for ping-pong; both will be Some because:
        //  - mrt_ao_raw_view: AoRaw is in reads.
        //  - mrt_ao_view:     Ao    is in writes.
        let ao_raw_view = match ctx.render_target.mrt_ao_raw_view {
            Some(v) => v,
            None => return Ok(()),
        };
        let ao_view = match ctx.render_target.mrt_ao_view {
            Some(v) => v,
            None => return Ok(()),
        };

        if !RTAO_ATROUS_BLUR_ENABLED {
            let (copy_pipeline, copy_bgl) = match self.ensure_raw_copy_pipeline(&ctx.gpu.device) {
                Some(x) => x,
                None => return Ok(()),
            };
            let bind_group = ctx
                .gpu
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("RTAO ao_raw copy bind group"),
                    layout: copy_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(ao_raw_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(ao_view),
                        },
                    ],
                });
            let (width, height) = ctx.viewport;
            let wg_x = width.div_ceil(TILE_SIZE);
            let wg_y = height.div_ceil(TILE_SIZE);
            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("RTAO ao_raw copy pass"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(copy_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
            return Ok(());
        }

        let norm_view = match ctx.render_target.mrt_normal_view {
            Some(v) => v,
            None => return Ok(()),
        };
        let depth_tex = match &ctx.gpu.depth_texture {
            Some(t) => t,
            None => return Ok(()),
        };

        // Depth24PlusStencil8 texture; bind depth aspect only.
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("RTAO atrous depth-only view"),
            format: None,
            dimension: None,
            aspect: wgpu::TextureAspect::DepthOnly,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
        });

        let (pipeline, bgl, step_bufs) = match self.ensure_pipeline(&ctx.gpu.device) {
            Some(x) => x,
            None => return Ok(()),
        };

        let (width, height) = ctx.viewport;
        let wg_x = width.div_ceil(TILE_SIZE);
        let wg_y = height.div_ceil(TILE_SIZE);

        // ── 3 À-Trous passes ─────────────────────────────────────────────────
        //
        // Pass 0 (step=1): ao_raw → ao
        // Pass 1 (step=2): ao    → ao_raw
        // Pass 2 (step=4): ao_raw → ao   ← composite reads `ao`
        //
        // Each pass is a *separate* compute pass so wgpu/the driver inserts the
        // necessary memory barrier between the write of one texture and the read
        // of the same texture in the next dispatch.

        for (pass_idx, step_buf) in step_bufs.iter().enumerate().take(3) {
            // Ping-pong: even passes read ao_raw / write ao; odd passes swap.
            let (input_view, output_view): (&wgpu::TextureView, &wgpu::TextureView) =
                if pass_idx % 2 == 0 {
                    (ao_raw_view, ao_view)
                } else {
                    (ao_view, ao_raw_view)
                };

            let bind_group = ctx
                .gpu
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("RTAO atrous blur bind group"),
                    layout: bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(input_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(norm_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(output_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: step_buf.as_entire_binding(),
                        },
                    ],
                });

            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("RTAO atrous blur pass"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
            // `pass` drops here → ends the compute pass, issuing a full UAV barrier.
        }

        Ok(())
    }
}

impl Default for RtaoBlurPass {
    fn default() -> Self {
        Self::new()
    }
}
