//! RTAO blur pass: cross-bilateral filter for edge-aware spatial blur.
//!
//! Blurs the raw RTAO output while respecting depth and normal edges.
//! Uses a 5×5 kernel with depth and normal similarity weights to avoid bleeding across geometry.

use super::{PassResources, RenderPass, RenderPassError, ResourceSlot};

const TILE_SIZE: u32 = 8;

const RTAO_BLUR_SHADER_SRC: &str = r#"
@group(0) @binding(0) var ao_input: texture_2d<f32>;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var ao_output: texture_storage_2d<rgba8unorm, write>;

fn depth_normal_similarity(center_depth: f32, center_normal: vec3f, sample_pos: vec2i) -> f32 {
    let d = textureLoad(depth_tex, sample_pos, 0);
    let n = normalize(textureLoad(normal_tex, sample_pos, 0).xyz);
    let depth_weight = exp(-abs(d - center_depth) * 30.0);
    let normal_weight = max(0.0, dot(n, center_normal)) * 0.5 + 0.5;
    return depth_weight * normal_weight;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(ao_input);
    if id.x >= dims.x || id.y >= dims.y { return; }

    let center_pos = vec2i(id.xy);
    let center_depth = textureLoad(depth_tex, center_pos, 0);
    let center_normal = normalize(textureLoad(normal_tex, center_pos, 0).xyz);
    let center_ao = textureLoad(ao_input, center_pos, 0).r;

    var sum = center_ao;
    var weight_sum = 1.0;

    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            if x == 0 && y == 0 { continue; }
            let offset = vec2i(x, y);
            let sample_pos = center_pos + offset;

            if sample_pos.x < 0 || sample_pos.y < 0 || sample_pos.x >= i32(dims.x) || sample_pos.y >= i32(dims.y) {
                continue;
            }

            let sample_ao = textureLoad(ao_input, sample_pos, 0).r;
            let w = depth_normal_similarity(center_depth, center_normal, sample_pos);
            sum += sample_ao * w;
            weight_sum += w;
        }
    }

    let final_ao = sum / weight_sum;
    textureStore(ao_output, center_pos, vec4f(final_ao, 0.0, 0.0, 1.0));
}
"#;

/// RTAO blur pass: cross-bilateral filter that blurs raw AO while keeping edges sharp.
///
/// Reads raw AO, depth, and normal; writes blurred AO to the final texture.
/// Dispatches (width/8, height/8, 1) workgroups. Skips when MRT views or depth are unavailable.
pub struct RtaoBlurPass {
    pipeline: Option<wgpu::ComputePipeline>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
}

impl RtaoBlurPass {
    /// Creates a new RTAO blur pass. Pipeline is built lazily when first used.
    pub fn new() -> Self {
        Self {
            pipeline: None,
            bind_group_layout: None,
        }
    }

    fn ensure_pipeline(
        &mut self,
        device: &wgpu::Device,
    ) -> Option<(&wgpu::ComputePipeline, &wgpu::BindGroupLayout)> {
        if self.pipeline.is_none() {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("RTAO blur shader"),
                source: wgpu::ShaderSource::Wgsl(RTAO_BLUR_SHADER_SRC.into()),
            });
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("RTAO blur bind group layout"),
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
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
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
                ],
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("RTAO blur pipeline layout"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });
            self.pipeline = Some(device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("RTAO blur pipeline"),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: None,
                    compilation_options: Default::default(),
                    cache: None,
                },
            ));
            self.bind_group_layout = Some(bgl);
        }
        self.pipeline.as_ref().zip(self.bind_group_layout.as_ref())
    }
}

impl RenderPass for RtaoBlurPass {
    fn name(&self) -> &str {
        "rtao_blur"
    }

    fn resources(&self) -> PassResources {
        PassResources {
            reads: vec![ResourceSlot::AoRaw, ResourceSlot::Depth],
            writes: vec![ResourceSlot::Ao],
        }
    }

    fn execute(&mut self, ctx: &mut super::RenderPassContext) -> Result<(), RenderPassError> {
        let ao_raw_view = match ctx.render_target.mrt_ao_raw_view {
            Some(v) => v,
            None => return Ok(()),
        };
        let ao_view = match ctx.render_target.mrt_ao_view {
            Some(v) => v,
            None => return Ok(()),
        };
        let norm_view = match ctx.render_target.mrt_normal_view {
            Some(v) => v,
            None => return Ok(()),
        };
        let depth_tex = match &ctx.gpu.depth_texture {
            Some(t) => t,
            None => return Ok(()),
        };
        // Depth24PlusStencil8 has both aspects; bound views cannot have both. Use depth-only.
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("RTAO blur depth-only view"),
            format: None,
            dimension: None,
            aspect: wgpu::TextureAspect::DepthOnly,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
        });

        let (pipeline, bgl) = match self.ensure_pipeline(&ctx.gpu.device) {
            Some((p, b)) => (p, b),
            None => return Ok(()),
        };

        let bind_group = ctx
            .gpu
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("RTAO blur bind group"),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(ao_raw_view),
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
                        resource: wgpu::BindingResource::TextureView(ao_view),
                    },
                ],
            });

        let (width, height) = ctx.viewport;
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RTAO blur pass"),
                timestamp_writes: None,
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(width.div_ceil(TILE_SIZE), height.div_ceil(TILE_SIZE), 1);

        Ok(())
    }
}

impl Default for RtaoBlurPass {
    fn default() -> Self {
        Self::new()
    }
}
