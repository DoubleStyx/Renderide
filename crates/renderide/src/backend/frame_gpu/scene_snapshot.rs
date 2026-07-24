//! Scene depth/color snapshots sampled through `@group(0)`.

use crate::gpu::GpuLimits;

/// Default scene-color snapshot format used before any grab pass has run.
pub(super) const DEFAULT_SCENE_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Scene-depth snapshot storage format; a blit target, not a depth copy.
pub(super) const SCENE_DEPTH_SNAPSHOT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// Snapshot texture family.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SceneSnapshotKind {
    /// Depth snapshot used by `_CameraDepthTexture` style material sampling.
    Depth,
    /// Per-object color snapshot used by unnamed grab-pass style material sampling.
    Color,
    /// Shared color snapshot used by named `_BackgroundTexture` grab passes.
    NamedColor,
}

/// Snapshot texture layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SceneSnapshotLayout {
    /// Single-view `texture_2d`.
    Mono2d,
    /// Two-layer `texture_2d_array`.
    StereoArray,
}

impl SceneSnapshotLayout {
    /// Selects the layout for a multiview flag.
    pub(super) fn from_multiview(multiview: bool) -> Self {
        if multiview {
            Self::StereoArray
        } else {
            Self::Mono2d
        }
    }

    /// Number of copied layers for this layout.
    fn layer_count(self) -> u32 {
        match self {
            Self::Mono2d => 1,
            Self::StereoArray => 2,
        }
    }

    /// View dimension for bind-group layout and texture views.
    fn view_dimension(self) -> wgpu::TextureViewDimension {
        match self {
            Self::Mono2d => wgpu::TextureViewDimension::D2,
            Self::StereoArray => wgpu::TextureViewDimension::D2Array,
        }
    }

    /// Stable label suffix for GPU object labels.
    fn label_suffix(self) -> &'static str {
        match self {
            Self::Mono2d => "2d",
            Self::StereoArray => "array",
        }
    }
}

/// Sampled scene depth/color snapshot views and sampler for `@group(0)` bindings 4-8.
pub struct FrameSceneSnapshotTextureViews<'a> {
    /// Single-view depth snapshot at binding 4.
    pub scene_depth_2d: &'a wgpu::TextureView,
    /// Multiview depth snapshot at binding 5.
    pub scene_depth_array: &'a wgpu::TextureView,
    /// Single-view color snapshot at binding 6.
    pub scene_color_2d: &'a wgpu::TextureView,
    /// Multiview color snapshot at binding 7.
    pub scene_color_array: &'a wgpu::TextureView,
    /// Shared sampler for scene color at binding 8.
    pub scene_color_sampler: &'a wgpu::Sampler,
}

/// One allocated snapshot texture/view pair.
struct SceneSnapshotTexture {
    /// Backing GPU texture.
    texture: wgpu::Texture,
    /// Default sampled view for the texture.
    view: wgpu::TextureView,
    /// Per-layer render-target views for the depth blit; empty for color snapshots.
    layer_views: Vec<wgpu::TextureView>,
    /// Allocated extent in pixels, clamped to at least `1x1`.
    extent_px: (u32, u32),
    /// Texture format used by the allocation.
    format: wgpu::TextureFormat,
}

impl SceneSnapshotTexture {
    /// Creates a snapshot texture and its bindable view.
    fn new(
        device: &wgpu::Device,
        kind: SceneSnapshotKind,
        layout: SceneSnapshotLayout,
        extent_px: (u32, u32),
        format: wgpu::TextureFormat,
    ) -> Self {
        let extent_px = clamp_snapshot_extent(extent_px);
        let kind_label = kind.label_prefix();
        let layout_label = layout.label_suffix();
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("frame_scene_{kind_label}_{layout_label}")),
            size: wgpu::Extent3d {
                width: extent_px.0,
                height: extent_px.1,
                depth_or_array_layers: layout.layer_count(),
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: kind.texture_usage(),
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&format!("frame_scene_{kind_label}_{layout_label}_view")),
            dimension: Some(layout.view_dimension()),
            array_layer_count: (layout == SceneSnapshotLayout::StereoArray).then_some(2),
            aspect: wgpu::TextureAspect::All,
            ..Default::default()
        });
        crate::profiling::note_resource_churn!(TextureView, "backend::scene_snapshot_view");
        let layer_views = if kind == SceneSnapshotKind::Depth {
            (0..layout.layer_count())
                .map(|layer| {
                    texture.create_view(&wgpu::TextureViewDescriptor {
                        label: Some(&format!(
                            "frame_scene_{kind_label}_{layout_label}_target_l{layer}"
                        )),
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        base_array_layer: layer,
                        array_layer_count: Some(1),
                        ..Default::default()
                    })
                })
                .collect()
        } else {
            Vec::new()
        };
        Self {
            texture,
            view,
            layer_views,
            extent_px,
            format,
        }
    }

    /// Returns true when this allocation already satisfies the requested shape.
    fn matches(&self, extent_px: (u32, u32), format: wgpu::TextureFormat) -> bool {
        self.extent_px == clamp_snapshot_extent(extent_px) && self.format == format
    }

    /// Retains this snapshot's GPU handles until driver submit.
    fn retain_submit_resources(&self, resources: &mut crate::gpu::GpuRetainedResources) {
        resources.retain_texture(self.texture.clone());
        resources.retain_texture_view(self.view.clone());
        resources.retain_texture_views(self.layer_views.iter().cloned());
    }
}

impl SceneSnapshotKind {
    /// Stable label prefix for GPU object labels.
    fn label_prefix(self) -> &'static str {
        match self {
            Self::Depth => "depth",
            Self::Color => "color",
            Self::NamedColor => "named_color",
        }
    }

    /// Texture usage for this snapshot family; depth renders via blit, colors copy.
    fn texture_usage(self) -> wgpu::TextureUsages {
        match self {
            Self::Depth => {
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
            }
            Self::Color | Self::NamedColor => {
                wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING
            }
        }
    }

    /// Storage format for this snapshot family given the source attachment format.
    pub(super) fn snapshot_format(self, source_format: wgpu::TextureFormat) -> wgpu::TextureFormat {
        match self {
            Self::Depth => SCENE_DEPTH_SNAPSHOT_FORMAT,
            Self::Color | Self::NamedColor => source_format,
        }
    }
}

/// Depth and color snapshots for one texture layout.
struct SceneSnapshotLayoutTargets {
    /// Depth snapshot for this layout.
    depth: SceneSnapshotTexture,
    /// Per-object color snapshot for this layout.
    color: SceneSnapshotTexture,
    /// Named `_BackgroundTexture` color snapshot for this layout.
    named_color: SceneSnapshotTexture,
}

impl SceneSnapshotLayoutTargets {
    /// Allocates the depth and color snapshots for `layout`.
    fn new(
        device: &wgpu::Device,
        layout: SceneSnapshotLayout,
        depth_format: wgpu::TextureFormat,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            depth: SceneSnapshotTexture::new(
                device,
                SceneSnapshotKind::Depth,
                layout,
                (1, 1),
                SceneSnapshotKind::Depth.snapshot_format(depth_format),
            ),
            color: SceneSnapshotTexture::new(
                device,
                SceneSnapshotKind::Color,
                layout,
                (1, 1),
                color_format,
            ),
            named_color: SceneSnapshotTexture::new(
                device,
                SceneSnapshotKind::NamedColor,
                layout,
                (1, 1),
                color_format,
            ),
        }
    }

    /// Returns the target for `kind`.
    fn target(&self, kind: SceneSnapshotKind) -> &SceneSnapshotTexture {
        match kind {
            SceneSnapshotKind::Depth => &self.depth,
            SceneSnapshotKind::Color => &self.color,
            SceneSnapshotKind::NamedColor => &self.named_color,
        }
    }

    /// Returns the mutable target for `kind`.
    fn target_mut(&mut self, kind: SceneSnapshotKind) -> &mut SceneSnapshotTexture {
        match kind {
            SceneSnapshotKind::Depth => &mut self.depth,
            SceneSnapshotKind::Color => &mut self.color,
            SceneSnapshotKind::NamedColor => &mut self.named_color,
        }
    }

    /// Retains every snapshot target in this layout until driver submit.
    fn retain_submit_resources(&self, resources: &mut crate::gpu::GpuRetainedResources) {
        self.depth.retain_submit_resources(resources);
        self.color.retain_submit_resources(resources);
        self.named_color.retain_submit_resources(resources);
    }
}

/// Depth-to-R32Float blit pipeline shared by both snapshot layouts.
struct SceneDepthBlitter {
    /// Fullscreen blit pipeline writing an R32Float color target.
    pipeline: wgpu::RenderPipeline,
    /// Single-entry layout binding the source depth attachment.
    bgl: wgpu::BindGroupLayout,
}

impl SceneDepthBlitter {
    /// Returns [`None`] when the blit shader is missing from the shader package.
    fn try_new(device: &wgpu::Device) -> Option<Self> {
        let source = crate::embedded_shaders::embedded_wgsl!("scene_depth_to_r32_blit");
        if source.is_empty() {
            return None;
        }
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene_depth_to_r32_blit"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene_depth_to_r32_blit_bgl"),
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
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene_depth_to_r32_blit_pl"),
            bind_group_layouts: &[Some(&bgl)],
            ..Default::default()
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scene_depth_to_r32_blit"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: SCENE_DEPTH_SNAPSHOT_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        crate::profiling::note_resource_churn!(RenderPipeline, "backend::scene_depth_blit");
        Some(Self { pipeline, bgl })
    }
}

/// Owns mono/stereo depth and color snapshots plus their shared color sampler.
pub(super) struct SceneSnapshotSet {
    /// Single-view depth and color snapshots.
    mono: SceneSnapshotLayoutTargets,
    /// Stereo-array depth and color snapshots.
    stereo: SceneSnapshotLayoutTargets,
    /// Shared color sampler.
    color_sampler: wgpu::Sampler,
    /// Depth-to-R32Float snapshot blit pipeline; absent when the package lacks the blit shader.
    depth_blitter: Option<SceneDepthBlitter>,
}

impl SceneSnapshotSet {
    /// Allocates the initial `1x1` snapshot set.
    pub(super) fn new(
        device: &wgpu::Device,
        depth_format: wgpu::TextureFormat,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let color_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("frame_scene_color_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        Self {
            mono: SceneSnapshotLayoutTargets::new(
                device,
                SceneSnapshotLayout::Mono2d,
                depth_format,
                color_format,
            ),
            stereo: SceneSnapshotLayoutTargets::new(
                device,
                SceneSnapshotLayout::StereoArray,
                depth_format,
                color_format,
            ),
            color_sampler,
            depth_blitter: SceneDepthBlitter::try_new(device),
        }
    }

    /// Returns the four snapshot views and color sampler used by `@group(0)`.
    pub(super) fn views(&self) -> FrameSceneSnapshotTextureViews<'_> {
        FrameSceneSnapshotTextureViews {
            scene_depth_2d: &self.mono.depth.view,
            scene_depth_array: &self.stereo.depth.view,
            scene_color_2d: &self.mono.color.view,
            scene_color_array: &self.stereo.color.view,
            scene_color_sampler: &self.color_sampler,
        }
    }

    /// Returns snapshot views with scene-color bindings pointing at the named grab target.
    pub(super) fn named_color_views(&self) -> FrameSceneSnapshotTextureViews<'_> {
        FrameSceneSnapshotTextureViews {
            scene_depth_2d: &self.mono.depth.view,
            scene_depth_array: &self.stereo.depth.view,
            scene_color_2d: &self.mono.named_color.view,
            scene_color_array: &self.stereo.named_color.view,
            scene_color_sampler: &self.color_sampler,
        }
    }

    /// Ensures one snapshot target exists for the requested shape.
    pub(super) fn ensure(
        &mut self,
        device: &wgpu::Device,
        limits: &GpuLimits,
        kind: SceneSnapshotKind,
        layout: SceneSnapshotLayout,
        extent_px: (u32, u32),
        format: wgpu::TextureFormat,
    ) -> bool {
        let format = kind.snapshot_format(format);
        let want = clamp_snapshot_extent(extent_px);
        let max_dim = limits.max_texture_dimension_2d();
        if want.0 > max_dim || want.1 > max_dim {
            logger::warn!(
                "scene {} {} snapshot: extent {}x{} exceeds max_texture_dimension_2d ({max_dim}); keeping previous texture",
                kind.label_prefix(),
                layout.label_suffix(),
                want.0,
                want.1
            );
            return false;
        }
        let target = self.targets_mut(layout).target_mut(kind);
        if target.matches(want, format) {
            return false;
        }
        *target = SceneSnapshotTexture::new(device, kind, layout, want, format);
        true
    }

    /// Encodes a copy into a pre-synchronized color snapshot target.
    pub(super) fn encode_copy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Texture,
        kind: SceneSnapshotKind,
        layout: SceneSnapshotLayout,
        viewport: (u32, u32),
    ) -> bool {
        debug_assert!(
            kind != SceneSnapshotKind::Depth,
            "depth snapshots blit via encode_depth_blit"
        );
        let width = viewport.0.max(1);
        let height = viewport.1.max(1);
        let format = source.format();
        let target = self.targets(layout).target(kind);
        if !target.matches((width, height), format) {
            logger::warn!(
                "scene {} snapshot copy: {} target not pre-synced for {}x{} {:?}; skipping copy",
                kind.label_prefix(),
                layout.label_suffix(),
                width,
                height,
                format
            );
            return false;
        }
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: source,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &target.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: layout.layer_count(),
            },
        );
        true
    }

    /// Blits the single-sample depth attachment into the R32Float depth snapshot.
    pub(super) fn encode_depth_blit(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        source_depth: &wgpu::Texture,
        layout: SceneSnapshotLayout,
        viewport: (u32, u32),
    ) -> bool {
        let Some(blitter) = self.depth_blitter.as_ref() else {
            logger::warn!("scene depth snapshot blit: blit pipeline unavailable; skipping");
            return false;
        };
        let width = viewport.0.max(1);
        let height = viewport.1.max(1);
        let target = self.targets(layout).target(SceneSnapshotKind::Depth);
        if !target.matches((width, height), SCENE_DEPTH_SNAPSHOT_FORMAT) {
            logger::warn!(
                "scene depth snapshot blit: {} target not pre-synced for {}x{}; skipping blit",
                layout.label_suffix(),
                width,
                height
            );
            return false;
        }
        for layer in 0..layout.layer_count() {
            let Some(target_view) = target.layer_views.get(layer as usize) else {
                logger::warn!("scene depth snapshot blit: missing target view for layer {layer}");
                return false;
            };
            let source_view = source_depth.create_view(&wgpu::TextureViewDescriptor {
                label: Some("scene_depth_blit_src"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::DepthOnly,
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scene_depth_blit_bg"),
                layout: &blitter.bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&source_view),
                }],
            });
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene_depth_snapshot_blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            rpass.set_pipeline(&blitter.pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }
        true
    }

    /// Retains every snapshot texture, view, and sampler until driver submit.
    pub(super) fn retain_submit_resources(&self, resources: &mut crate::gpu::GpuRetainedResources) {
        self.mono.retain_submit_resources(resources);
        self.stereo.retain_submit_resources(resources);
        resources.retain_sampler(self.color_sampler.clone());
    }

    /// Returns immutable targets for `layout`.
    fn targets(&self, layout: SceneSnapshotLayout) -> &SceneSnapshotLayoutTargets {
        match layout {
            SceneSnapshotLayout::Mono2d => &self.mono,
            SceneSnapshotLayout::StereoArray => &self.stereo,
        }
    }

    /// Returns mutable targets for `layout`.
    fn targets_mut(&mut self, layout: SceneSnapshotLayout) -> &mut SceneSnapshotLayoutTargets {
        match layout {
            SceneSnapshotLayout::Mono2d => &mut self.mono,
            SceneSnapshotLayout::StereoArray => &mut self.stereo,
        }
    }
}

/// Clamps zero-sized snapshot extents to the smallest valid texture size.
fn clamp_snapshot_extent(extent_px: (u32, u32)) -> (u32, u32) {
    (extent_px.0.max(1), extent_px.1.max(1))
}

#[cfg(test)]
mod tests {
    use super::{SceneSnapshotKind, SceneSnapshotLayout, clamp_snapshot_extent};

    /// Zero viewport dimensions clamp to a valid texture extent.
    #[test]
    fn snapshot_extent_clamps_to_one_pixel() {
        assert_eq!(clamp_snapshot_extent((0, 0)), (1, 1));
        assert_eq!(clamp_snapshot_extent((640, 0)), (640, 1));
    }

    /// Mono and stereo layouts select the expected copy layer counts.
    #[test]
    fn snapshot_layout_layer_counts_match_target_shape() {
        assert_eq!(SceneSnapshotLayout::from_multiview(false).layer_count(), 1);
        assert_eq!(SceneSnapshotLayout::from_multiview(true).layer_count(), 2);
    }

    /// Mono and stereo layouts bind through the matching texture view dimensions.
    #[test]
    fn snapshot_layout_view_dimensions_match_target_shape() {
        assert_eq!(
            SceneSnapshotLayout::from_multiview(false).view_dimension(),
            wgpu::TextureViewDimension::D2
        );
        assert_eq!(
            SceneSnapshotLayout::from_multiview(true).view_dimension(),
            wgpu::TextureViewDimension::D2Array
        );
    }

    /// Depth snapshots store R32Float regardless of source depth format; colors keep the source.
    #[test]
    fn snapshot_kind_formats_map_depth_to_r32() {
        assert_eq!(
            SceneSnapshotKind::Depth.snapshot_format(wgpu::TextureFormat::Depth32Float),
            wgpu::TextureFormat::R32Float
        );
        assert_eq!(
            SceneSnapshotKind::Depth.snapshot_format(wgpu::TextureFormat::Depth24PlusStencil8),
            wgpu::TextureFormat::R32Float
        );
        assert_eq!(
            SceneSnapshotKind::Color.snapshot_format(wgpu::TextureFormat::Rgba16Float),
            wgpu::TextureFormat::Rgba16Float
        );
        assert_eq!(
            SceneSnapshotKind::NamedColor.snapshot_format(wgpu::TextureFormat::Rgba16Float),
            wgpu::TextureFormat::Rgba16Float
        );
    }
}
