//! Cached static-caster depth for shadow atlas layers.
//!
//! A shadow layer used to be all-or-nothing: either its atlas depth was still exactly right and the
//! layer was skipped, or every visible caster was redrawn. One skinned caster anywhere in the layer
//! forced the second path, so in a session with four avatars all thirteen layers redrew the entire
//! static room every frame. That was 81% of all draw submissions in the capture.
//!
//! This store holds the rigid half of each layer's depth in a parallel depth array. Layers that mix
//! static and dynamic casters restore their static depth with one fullscreen draw and then draw only
//! the dynamic casters over it. Depth testing makes that identical to the combined pass, so it is an
//! exact reproduction rather than an approximation.
//!
//! The store is 1:1 with the atlas: store layer N backs atlas layer N. That costs a second atlas
//! worth of VRAM, so allocation is capped by [`STATIC_STORE_VRAM_BUDGET_BYTES`] and simply declines
//! when the atlas is too big, leaving the old full-redraw behavior in place.

use std::sync::Arc;

use crate::embedded_shaders::embedded_wgsl;

/// Largest static store worth holding, in bytes.
///
/// A 2048 atlas over 13 layers of Depth32Float is already 218 MB, and the win does not justify
/// pushing a VR user into VRAM eviction. Past this the store declines and layers redraw in full.
const STATIC_STORE_VRAM_BUDGET_BYTES: u64 = 192 * 1024 * 1024;

/// Per-layer cached static depth, 1:1 with the shadow atlas layers.
pub(super) struct ShadowStaticStore {
    #[expect(dead_code, reason = "kept alive for the views borrowed from it")]
    texture: Arc<wgpu::Texture>,
    /// Per-layer depth views, used both as render target and as restore source.
    layer_views: Vec<Arc<wgpu::TextureView>>,
    /// Restore bind group per layer, holding that layer's view as a 2D depth source.
    layer_bind_groups: Vec<Arc<wgpu::BindGroup>>,
    resolution: u32,
    layers: u32,
}

impl ShadowStaticStore {
    /// Allocates a store matching the atlas, or [`None`] when it would not fit the budget.
    pub(super) fn new(
        device: &wgpu::Device,
        resolution: u32,
        layers: u32,
        format: wgpu::TextureFormat,
    ) -> Option<Self> {
        if resolution == 0 || layers == 0 || !static_store_fits_budget(resolution, layers, format) {
            return None;
        }
        let texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow_static_store"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }));
        

        let layout = restore_bind_group_layout(device);
        let mut layer_views = Vec::with_capacity(layers as usize);
        let mut layer_bind_groups = Vec::with_capacity(layers as usize);
        for layer in 0..layers {
            let view = Arc::new(texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("shadow_static_store_layer"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            }));
            let bind_group = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shadow_static_restore"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view.as_ref()),
                }],
            }));
            crate::profiling::note_resource_churn!(TextureView, "shadows::static_store_layer");
            crate::profiling::note_resource_churn!(BindGroup, "shadows::static_store_layer");
            layer_views.push(view);
            layer_bind_groups.push(bind_group);
        }
        Some(Self {
            texture,
            layer_views,
            layer_bind_groups,
            resolution,
            layers,
        })
    }

    /// Whether this store still matches the atlas it backs.
    pub(super) const fn matches(&self, resolution: u32, layers: u32) -> bool {
        self.resolution == resolution && self.layers == layers
    }

    /// Render target for one layer's static depth.
    pub(super) fn layer_view(&self, layer: u32) -> Option<&wgpu::TextureView> {
        self.layer_views.get(layer as usize).map(Arc::as_ref)
    }

    /// Restore source bind group for one layer.
    pub(super) fn layer_bind_group(&self, layer: u32) -> Option<&wgpu::BindGroup> {
        self.layer_bind_groups.get(layer as usize).map(Arc::as_ref)
    }

    /// Bytes of VRAM this store occupies.
    pub(super) const fn vram_bytes(&self, format: wgpu::TextureFormat) -> u64 {
        static_store_bytes(self.resolution, self.layers, format)
    }
}

/// Bytes a store of this shape would occupy.
const fn static_store_bytes(resolution: u32, layers: u32, format: wgpu::TextureFormat) -> u64 {
    let texel = match format {
        wgpu::TextureFormat::Depth16Unorm => 2u64,
        // Depth24Plus is implementation-defined but never narrower than 3 bytes; assume 4 so the
        // budget is not quietly overshot on drivers that pad it.
        _ => 4u64,
    };
    (resolution as u64)
        .saturating_mul(resolution as u64)
        .saturating_mul(layers as u64)
        .saturating_mul(texel)
}

/// Whether a store of this shape is worth its VRAM.
fn static_store_fits_budget(resolution: u32, layers: u32, format: wgpu::TextureFormat) -> bool {
    static_store_bytes(resolution, layers, format) <= STATIC_STORE_VRAM_BUDGET_BYTES
}

/// Bind group layout for the restore pass: one 2D depth source.
fn restore_bind_group_layout(device: &wgpu::Device) -> &'static wgpu::BindGroupLayout {
    static LAYOUT: std::sync::OnceLock<wgpu::BindGroupLayout> = std::sync::OnceLock::new();
    LAYOUT.get_or_init(|| {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_static_restore"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        })
    })
}

/// Builds the fullscreen depth-restore pipeline for the atlas format.
pub(super) fn create_restore_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("shadow_static_restore"),
        source: wgpu::ShaderSource::Wgsl(embedded_wgsl!("shadow_static_restore").into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("shadow_static_restore"),
        bind_group_layouts: &[Some(restore_bind_group_layout(device))],
        ..Default::default()
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("shadow_static_restore"),
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
            targets: &[],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format,
            depth_write_enabled: Some(true),
            // Always, not LessEqual: this writes the stored depth verbatim, it does not merge.
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    crate::profiling::note_resource_churn!(RenderPipeline, "shadows::static_restore_pipeline");
    pipeline
}

#[cfg(test)]
mod tests {
    use super::{STATIC_STORE_VRAM_BUDGET_BYTES, static_store_bytes, static_store_fits_budget};

    #[test]
    fn depth32_store_bytes_match_the_atlas_footprint() {
        // 2048 square, 13 layers, 4 bytes per texel is the shape measured in the 4-user capture.
        assert_eq!(
            static_store_bytes(2048, 13, wgpu::TextureFormat::Depth32Float),
            2048 * 2048 * 13 * 4
        );
    }

    #[test]
    fn depth16_store_is_half_the_size_of_depth32() {
        let wide = static_store_bytes(1024, 8, wgpu::TextureFormat::Depth32Float);
        let narrow = static_store_bytes(1024, 8, wgpu::TextureFormat::Depth16Unorm);

        assert_eq!(narrow * 2, wide);
    }

    #[test]
    fn oversized_atlases_decline_the_store_instead_of_eating_vram() {
        // A 4096 atlas over 16 layers is 1 GB. Redrawing casters is cheaper than that.
        assert!(!static_store_fits_budget(
            4096,
            16,
            wgpu::TextureFormat::Depth32Float
        ));
        assert!(static_store_fits_budget(
            1024,
            13,
            wgpu::TextureFormat::Depth32Float
        ));
    }

    #[test]
    fn depth24_plus_is_costed_as_four_bytes_so_the_budget_is_not_overshot() {
        assert_eq!(
            static_store_bytes(512, 4, wgpu::TextureFormat::Depth24Plus),
            static_store_bytes(512, 4, wgpu::TextureFormat::Depth32Float)
        );
    }

    #[test]
    fn budget_is_stated_in_whole_megabytes() {
        assert_eq!(STATIC_STORE_VRAM_BUDGET_BYTES % (1024 * 1024), 0);
    }
}
