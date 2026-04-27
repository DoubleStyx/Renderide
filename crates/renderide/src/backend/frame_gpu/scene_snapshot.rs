//! Scene depth/color snapshots sampled through `@group(0)`.

use crate::gpu::GpuLimits;

/// Default scene-color snapshot format used before any grab pass has run.
pub(super) const DEFAULT_SCENE_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Snapshot texture family.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SceneSnapshotKind {
    /// Depth snapshot used by `_CameraDepthTexture` style material sampling.
    Depth,
    /// Color snapshot used by grab-pass style material sampling.
    Color,
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
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&format!("frame_scene_{kind_label}_{layout_label}_view")),
            dimension: Some(layout.view_dimension()),
            array_layer_count: (layout == SceneSnapshotLayout::StereoArray).then_some(2),
            aspect: kind.view_aspect(),
            ..Default::default()
        });
        Self {
            texture,
            view,
            extent_px,
            format,
        }
    }

    /// Returns true when this allocation already satisfies the requested shape.
    fn matches(&self, extent_px: (u32, u32), format: wgpu::TextureFormat) -> bool {
        self.extent_px == clamp_snapshot_extent(extent_px) && self.format == format
    }
}

impl SceneSnapshotKind {
    /// Stable label prefix for GPU object labels.
    fn label_prefix(self) -> &'static str {
        match self {
            Self::Depth => "depth",
            Self::Color => "color",
        }
    }

    /// Texture view aspect for the bindable snapshot view.
    fn view_aspect(self) -> wgpu::TextureAspect {
        match self {
            Self::Depth => wgpu::TextureAspect::DepthOnly,
            Self::Color => wgpu::TextureAspect::All,
        }
    }

    /// Copy aspect for `copy_texture_to_texture`.
    fn copy_aspect(self, source_format: wgpu::TextureFormat) -> wgpu::TextureAspect {
        match self {
            Self::Depth if source_format.has_stencil_aspect() => wgpu::TextureAspect::All,
            Self::Depth => wgpu::TextureAspect::DepthOnly,
            Self::Color => wgpu::TextureAspect::All,
        }
    }
}

/// Owns mono/stereo depth and color snapshots plus their shared color sampler.
pub(super) struct SceneSnapshotSet {
    /// Single-view depth snapshot.
    depth_2d: SceneSnapshotTexture,
    /// Stereo-array depth snapshot.
    depth_array: SceneSnapshotTexture,
    /// Single-view color snapshot.
    color_2d: SceneSnapshotTexture,
    /// Stereo-array color snapshot.
    color_array: SceneSnapshotTexture,
    /// Shared color sampler.
    color_sampler: wgpu::Sampler,
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
            depth_2d: SceneSnapshotTexture::new(
                device,
                SceneSnapshotKind::Depth,
                SceneSnapshotLayout::Mono2d,
                (1, 1),
                depth_format,
            ),
            depth_array: SceneSnapshotTexture::new(
                device,
                SceneSnapshotKind::Depth,
                SceneSnapshotLayout::StereoArray,
                (1, 1),
                depth_format,
            ),
            color_2d: SceneSnapshotTexture::new(
                device,
                SceneSnapshotKind::Color,
                SceneSnapshotLayout::Mono2d,
                (1, 1),
                color_format,
            ),
            color_array: SceneSnapshotTexture::new(
                device,
                SceneSnapshotKind::Color,
                SceneSnapshotLayout::StereoArray,
                (1, 1),
                color_format,
            ),
            color_sampler,
        }
    }

    /// Returns the four snapshot views and color sampler used by `@group(0)`.
    pub(super) fn views(&self) -> FrameSceneSnapshotTextureViews<'_> {
        FrameSceneSnapshotTextureViews {
            scene_depth_2d: &self.depth_2d.view,
            scene_depth_array: &self.depth_array.view,
            scene_color_2d: &self.color_2d.view,
            scene_color_array: &self.color_array.view,
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
        let target = self.target_mut(kind, layout);
        if target.matches(want, format) {
            return false;
        }
        *target = SceneSnapshotTexture::new(device, kind, layout, want, format);
        true
    }

    /// Encodes a copy into a pre-synchronized snapshot target.
    pub(super) fn encode_copy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Texture,
        kind: SceneSnapshotKind,
        layout: SceneSnapshotLayout,
        viewport: (u32, u32),
    ) {
        let width = viewport.0.max(1);
        let height = viewport.1.max(1);
        let format = source.format();
        let target = self.target(kind, layout);
        if !target.matches((width, height), format) {
            logger::warn!(
                "scene {} snapshot copy: {} target not pre-synced for {}x{} {:?}; skipping copy",
                kind.label_prefix(),
                layout.label_suffix(),
                width,
                height,
                format
            );
            return;
        }
        let aspect = kind.copy_aspect(format);
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: source,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &target.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: layout.layer_count(),
            },
        );
    }

    /// Returns an immutable target reference.
    fn target(
        &self,
        kind: SceneSnapshotKind,
        layout: SceneSnapshotLayout,
    ) -> &SceneSnapshotTexture {
        match (kind, layout) {
            (SceneSnapshotKind::Depth, SceneSnapshotLayout::Mono2d) => &self.depth_2d,
            (SceneSnapshotKind::Depth, SceneSnapshotLayout::StereoArray) => &self.depth_array,
            (SceneSnapshotKind::Color, SceneSnapshotLayout::Mono2d) => &self.color_2d,
            (SceneSnapshotKind::Color, SceneSnapshotLayout::StereoArray) => &self.color_array,
        }
    }

    /// Returns a mutable target reference.
    fn target_mut(
        &mut self,
        kind: SceneSnapshotKind,
        layout: SceneSnapshotLayout,
    ) -> &mut SceneSnapshotTexture {
        match (kind, layout) {
            (SceneSnapshotKind::Depth, SceneSnapshotLayout::Mono2d) => &mut self.depth_2d,
            (SceneSnapshotKind::Depth, SceneSnapshotLayout::StereoArray) => &mut self.depth_array,
            (SceneSnapshotKind::Color, SceneSnapshotLayout::Mono2d) => &mut self.color_2d,
            (SceneSnapshotKind::Color, SceneSnapshotLayout::StereoArray) => &mut self.color_array,
        }
    }
}

/// Clamps zero-sized snapshot extents to the smallest valid texture size.
fn clamp_snapshot_extent(extent_px: (u32, u32)) -> (u32, u32) {
    (extent_px.0.max(1), extent_px.1.max(1))
}

#[cfg(test)]
mod tests {
    use super::{clamp_snapshot_extent, SceneSnapshotKind, SceneSnapshotLayout};

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

    /// Depth copies use depth-only aspects except combined depth-stencil sources, while color
    /// copies always copy all aspects.
    #[test]
    fn snapshot_kind_copy_aspect_matches_source_format() {
        assert_eq!(
            SceneSnapshotKind::Depth.copy_aspect(wgpu::TextureFormat::Depth32Float),
            wgpu::TextureAspect::DepthOnly
        );
        assert_eq!(
            SceneSnapshotKind::Depth.copy_aspect(wgpu::TextureFormat::Depth24PlusStencil8),
            wgpu::TextureAspect::All
        );
        assert_eq!(
            SceneSnapshotKind::Color.copy_aspect(wgpu::TextureFormat::Rgba16Float),
            wgpu::TextureAspect::All
        );
    }
}
