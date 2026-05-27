//! Frame-global light-cookie atlases and GPU blit support.

use std::borrow::Cow;
use std::sync::Arc;

use hashbrown::HashMap;
use parking_lot::Mutex;

use crate::assets::texture::{HostTextureAssetKind, unpack_host_texture_packed};
use crate::backend::light_gpu::{
    LIGHT_COOKIE_KIND_DIRECTIONAL_2D, LIGHT_COOKIE_KIND_POINT_CUBE, LIGHT_COOKIE_KIND_SPOT_2D,
    LightCookieBinding,
};
use crate::gpu::{
    GpuLimits, LIGHT_COOKIE_WRAP_MODE_CLAMP, LIGHT_COOKIE_WRAP_MODE_MASK,
    LIGHT_COOKIE_WRAP_MODE_MIRROR, LIGHT_COOKIE_WRAP_MODE_MIRROR_ONCE,
    LIGHT_COOKIE_WRAP_MODE_REPEAT, LIGHT_COOKIE_WRAP_U_SHIFT, LIGHT_COOKIE_WRAP_V_SHIFT,
};
use crate::gpu_pools::{GpuCubemap, SamplerState};
use crate::render_graph::GraphAssetResources;
use crate::shared::{LightType, TextureFormat, TextureWrapMode};

/// Edge length of each light-cookie atlas layer.
const LIGHT_COOKIE_ATLAS_EDGE: u32 = 256;
/// Maximum 2D cookie layers including the fallback layer.
const COOKIE_2D_LAYER_CAP: u32 = 64;
/// Maximum resident point-light cookie cubemaps.
const POINT_COOKIE_CUBEMAP_CAP: u32 = 16;
/// Cubemap face count.
const POINT_COOKIE_FACE_COUNT: u32 = 6;
/// Embedded WGSL target for copying 2D source cookies into atlas layers.
const LIGHT_COOKIE_BLIT_2D_STEM: &str = "light_cookie_blit_2d";
/// Source WGSL used only if embedded shader metadata is unexpectedly missing.
const LIGHT_COOKIE_BLIT_2D_SOURCE: &str =
    include_str!("../../../shaders/passes/backend/light_cookie_blit_2d.wgsl");

/// Scalar storage format used for light-cookie atlas layers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LightCookieAtlasFormat {
    /// Signed half-float scalar storage.
    R16Float,
    /// Signed half-float RGBA storage used when scalar render targets are unavailable.
    Rgba16Float,
    /// Unsigned normalized scalar fallback.
    R8Unorm,
}

impl LightCookieAtlasFormat {
    /// Returns the wgpu texture format.
    const fn wgpu(self) -> wgpu::TextureFormat {
        match self {
            Self::R16Float => wgpu::TextureFormat::R16Float,
            Self::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
            Self::R8Unorm => wgpu::TextureFormat::R8Unorm,
        }
    }

    /// Returns bytes per texel for CPU fallback-layer writes.
    const fn bytes_per_texel(self) -> u32 {
        match self {
            Self::R16Float => 2,
            Self::Rgba16Float => 8,
            Self::R8Unorm => 1,
        }
    }
}

/// Channel read from a source texture into the scalar cookie atlas.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LightCookieSourceChannel {
    /// Source red channel.
    Red,
    /// Source alpha channel.
    Alpha,
}

/// Sampler/layout mode used for source cookie blits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LightCookieSourceSampling {
    /// Filterable source texture and filtering sampler.
    Filtering,
    /// Unfilterable float source texture and non-filtering sampler.
    NonFiltering,
}

/// Resolved source texture view plus sampling policy.
#[derive(Clone, Copy)]
struct LightCookieSource<'a> {
    /// Source texture view.
    view: &'a wgpu::TextureView,
    /// Source channel copied into the scalar atlas.
    channel: LightCookieSourceChannel,
    /// Source sampler/layout mode.
    sampling: LightCookieSourceSampling,
}

/// Resolved point-cookie source cubemap plus sampling policy.
#[derive(Clone, Copy)]
struct LightCookiePointSource<'a> {
    /// Source cubemap resource.
    cubemap: &'a GpuCubemap,
    /// Source channel copied into the scalar atlas.
    channel: LightCookieSourceChannel,
    /// Source sampler/layout mode.
    sampling: LightCookieSourceSampling,
}

/// One requested light-cookie source assigned to an atlas layer.
#[derive(Clone, Copy, Debug)]
struct LightCookieRequest {
    /// Packed host texture handle.
    packed_id: i32,
    /// Unpacked host asset id.
    asset_id: i32,
    /// Unpacked host texture kind.
    kind: HostTextureAssetKind,
    /// 2D atlas layer or first point face layer.
    layer: u32,
}

/// One unpacked light-cookie handle ready for atlas assignment.
#[derive(Clone, Copy, Debug)]
struct LightCookieAssignment {
    /// Host light type requesting the cookie.
    light_type: LightType,
    /// Packed host texture handle.
    packed_id: i32,
    /// Unpacked host asset id.
    asset_id: i32,
    /// Unpacked host texture kind.
    kind: HostTextureAssetKind,
    /// Packed 2D cookie wrap modes.
    wrap_bits: u32,
}

/// Atlas slot state for a packed host texture handle.
#[derive(Clone, Copy, Debug)]
struct LightCookieSlot {
    /// Atlas layer assigned to this packed handle.
    layer: u32,
    /// Whether this slot is referenced by the current frame's packed lights.
    requested_this_frame: bool,
}

/// Mutable cookie assignment state shared by light packing and atlas encoding.
#[derive(Debug)]
struct LightCookieAtlasState {
    /// Persistent 2D-cookie slots keyed by packed texture handle.
    two_d_slots: HashMap<i32, LightCookieSlot>,
    /// Persistent point-cookie slots keyed by packed texture handle.
    point_slots: HashMap<i32, LightCookieSlot>,
    /// Unique 2D-cookie requests for the current frame.
    two_d_requests: Vec<LightCookieRequest>,
    /// Unique point-cookie requests for the current frame.
    point_requests: Vec<LightCookieRequest>,
    /// One-shot guard for 2D-cookie atlas overflow.
    two_d_overflow_logged: bool,
    /// One-shot guard for point-cookie atlas overflow.
    point_overflow_logged: bool,
}

impl LightCookieAtlasState {
    /// Creates an empty assignment table.
    fn new() -> Self {
        Self {
            two_d_slots: HashMap::new(),
            point_slots: HashMap::new(),
            two_d_requests: Vec::new(),
            point_requests: Vec::new(),
            two_d_overflow_logged: false,
            point_overflow_logged: false,
        }
    }

    /// Marks all slots unrequested and clears current-frame request lists.
    fn begin_frame(&mut self) {
        for slot in self.two_d_slots.values_mut() {
            slot.requested_this_frame = false;
        }
        for slot in self.point_slots.values_mut() {
            slot.requested_this_frame = false;
        }
        self.two_d_requests.clear();
        self.point_requests.clear();
    }

    /// Assigns a cookie atlas binding for one resolved light.
    fn assign(
        &mut self,
        assignment: LightCookieAssignment,
        two_d_layers: u32,
        point_layers: u32,
    ) -> LightCookieBinding {
        match (assignment.light_type, assignment.kind) {
            (
                LightType::Spot,
                HostTextureAssetKind::Texture2D
                | HostTextureAssetKind::RenderTexture
                | HostTextureAssetKind::VideoTexture,
            ) => self.assign_2d(
                assignment.packed_id,
                assignment.asset_id,
                assignment.kind,
                two_d_layers,
                LIGHT_COOKIE_KIND_SPOT_2D,
                assignment.wrap_bits,
            ),
            (
                LightType::Directional,
                HostTextureAssetKind::Texture2D
                | HostTextureAssetKind::RenderTexture
                | HostTextureAssetKind::VideoTexture,
            ) => self.assign_2d(
                assignment.packed_id,
                assignment.asset_id,
                assignment.kind,
                two_d_layers,
                LIGHT_COOKIE_KIND_DIRECTIONAL_2D,
                assignment.wrap_bits,
            ),
            (LightType::Point, HostTextureAssetKind::Cubemap) => self.assign_point(
                assignment.packed_id,
                assignment.asset_id,
                assignment.kind,
                point_layers,
            ),
            _ => LightCookieBinding::NONE,
        }
    }

    /// Assigns a 2D cookie layer.
    fn assign_2d(
        &mut self,
        packed_id: i32,
        asset_id: i32,
        kind: HostTextureAssetKind,
        layers: u32,
        cookie_kind: u32,
        wrap_bits: u32,
    ) -> LightCookieBinding {
        let Some(layer) = assign_cookie_layer(
            &mut self.two_d_slots,
            packed_id,
            1,
            layers,
            1,
            &mut self.two_d_overflow_logged,
            "2D",
        ) else {
            return LightCookieBinding::NONE;
        };
        if let Some(slot) = self.two_d_slots.get_mut(&packed_id)
            && !slot.requested_this_frame
        {
            slot.requested_this_frame = true;
            self.two_d_requests.push(LightCookieRequest {
                packed_id,
                asset_id,
                kind,
                layer,
            });
        }
        LightCookieBinding {
            kind: cookie_kind,
            layer,
            wrap_bits,
        }
    }

    /// Assigns six 2D-array layers for a point-light cubemap cookie.
    fn assign_point(
        &mut self,
        packed_id: i32,
        asset_id: i32,
        kind: HostTextureAssetKind,
        layers: u32,
    ) -> LightCookieBinding {
        let Some(layer) = assign_cookie_layer(
            &mut self.point_slots,
            packed_id,
            1,
            layers,
            POINT_COOKIE_FACE_COUNT,
            &mut self.point_overflow_logged,
            "point",
        ) else {
            return LightCookieBinding::NONE;
        };
        if let Some(slot) = self.point_slots.get_mut(&packed_id)
            && !slot.requested_this_frame
        {
            slot.requested_this_frame = true;
            self.point_requests.push(LightCookieRequest {
                packed_id,
                asset_id,
                kind,
                layer,
            });
        }
        LightCookieBinding {
            kind: LIGHT_COOKIE_KIND_POINT_CUBE,
            layer,
            wrap_bits: 0,
        }
    }

    /// Returns whether any current-frame request needs atlas synchronization.
    fn has_requests(&self) -> bool {
        !(self.two_d_requests.is_empty() && self.point_requests.is_empty())
    }

    /// Snapshot of requests for encoder recording without holding the state lock.
    fn requests(&self) -> (Vec<LightCookieRequest>, Vec<LightCookieRequest>) {
        (self.two_d_requests.clone(), self.point_requests.clone())
    }
}

/// Assigns or reuses one atlas layer block.
fn assign_cookie_layer(
    slots: &mut HashMap<i32, LightCookieSlot>,
    packed_id: i32,
    first_layer: u32,
    layer_count: u32,
    layer_stride: u32,
    overflow_logged: &mut bool,
    label: &str,
) -> Option<u32> {
    if let Some(slot) = slots.get(&packed_id) {
        return Some(slot.layer);
    }
    let last_start = layer_count.checked_sub(layer_stride)?;
    let mut layer = first_layer;
    while layer <= last_start {
        if !slots.values().any(|slot| slot.layer == layer) {
            slots.insert(
                packed_id,
                LightCookieSlot {
                    layer,
                    requested_this_frame: false,
                },
            );
            return Some(layer);
        }
        layer = layer.saturating_add(layer_stride);
    }
    let reusable = slots
        .iter()
        .find_map(|(&id, slot)| (!slot.requested_this_frame).then_some((id, slot.layer)));
    if let Some((old_id, layer)) = reusable {
        slots.remove(&old_id);
        slots.insert(
            packed_id,
            LightCookieSlot {
                layer,
                requested_this_frame: false,
            },
        );
        return Some(layer);
    }
    if !*overflow_logged {
        logger::warn!(
            "light-cookie {label} atlas full; additional {label} cookies will be ignored"
        );
        *overflow_logged = true;
    }
    None
}

/// Selects the scalar texture format used by light-cookie atlases.
fn select_light_cookie_atlas_format(limits: &GpuLimits) -> LightCookieAtlasFormat {
    if light_cookie_atlas_format_supported(limits, LightCookieAtlasFormat::R16Float) {
        return LightCookieAtlasFormat::R16Float;
    }
    if light_cookie_atlas_format_supported(limits, LightCookieAtlasFormat::Rgba16Float) {
        logger::warn!(
            "signed scalar light-cookie atlas format R16Float is unavailable; using Rgba16Float HDR cookie storage"
        );
        return LightCookieAtlasFormat::Rgba16Float;
    }
    logger::warn!(
        "HDR light-cookie atlas formats are unavailable; falling back to unsigned R8Unorm cookies"
    );
    LightCookieAtlasFormat::R8Unorm
}

/// Returns whether the device can use `format` for sampled scalar light-cookie atlases.
fn light_cookie_atlas_format_supported(limits: &GpuLimits, format: LightCookieAtlasFormat) -> bool {
    let features = limits.texture_format_features(format.wgpu());
    features.allowed_usages.contains(
        wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_DST,
    ) && features
        .flags
        .contains(wgpu::TextureFormatFeatureFlags::FILTERABLE)
}

/// Returns the source channel that carries the scalar cookie value.
fn source_channel_for_host_format(format: TextureFormat) -> LightCookieSourceChannel {
    match format {
        TextureFormat::ARGB32
        | TextureFormat::RGBA32
        | TextureFormat::BGRA32
        | TextureFormat::RGBAHalf
        | TextureFormat::ARGBHalf
        | TextureFormat::RGBAFloat
        | TextureFormat::ARGBFloat
        | TextureFormat::BC2
        | TextureFormat::BC3
        | TextureFormat::BC7
        | TextureFormat::ETC2RGBA1
        | TextureFormat::ETC2RGBA8
        | TextureFormat::ASTC4x4
        | TextureFormat::ASTC5x5
        | TextureFormat::ASTC6x6
        | TextureFormat::ASTC8x8
        | TextureFormat::ASTC10x10
        | TextureFormat::ASTC12x12 => LightCookieSourceChannel::Alpha,
        TextureFormat::Unknown
        | TextureFormat::Alpha8
        | TextureFormat::R8
        | TextureFormat::RGB24
        | TextureFormat::RGB565
        | TextureFormat::BGR565
        | TextureFormat::RHalf
        | TextureFormat::RGHalf
        | TextureFormat::RFloat
        | TextureFormat::RGFloat
        | TextureFormat::BC1
        | TextureFormat::BC4
        | TextureFormat::BC5
        | TextureFormat::BC6H
        | TextureFormat::ETC2RGB => LightCookieSourceChannel::Red,
    }
}

/// Packs U/V sampler wrap modes for 2D cookie shader addressing.
fn light_cookie_wrap_bits(sampler: &SamplerState) -> u32 {
    pack_wrap_mode(sampler.wrap_u, LIGHT_COOKIE_WRAP_U_SHIFT)
        | pack_wrap_mode(sampler.wrap_v, LIGHT_COOKIE_WRAP_V_SHIFT)
}

/// Packs one wrap mode into the shader metadata bitfield.
fn pack_wrap_mode(mode: TextureWrapMode, shift: u32) -> u32 {
    (wrap_mode_bits(mode) & LIGHT_COOKIE_WRAP_MODE_MASK) << shift
}

/// Converts a host texture wrap mode to the compact shader enum.
fn wrap_mode_bits(mode: TextureWrapMode) -> u32 {
    match mode {
        TextureWrapMode::Repeat => LIGHT_COOKIE_WRAP_MODE_REPEAT,
        TextureWrapMode::Clamp => LIGHT_COOKIE_WRAP_MODE_CLAMP,
        TextureWrapMode::Mirror => LIGHT_COOKIE_WRAP_MODE_MIRROR,
        TextureWrapMode::MirrorOnce => LIGHT_COOKIE_WRAP_MODE_MIRROR_ONCE,
    }
}

/// Layered atlas texture and one-layer render-target views.
struct LightCookieLayeredAtlas {
    /// Backing texture.
    _texture: Arc<wgpu::Texture>,
    /// Full array view bound by frame globals.
    view: Arc<wgpu::TextureView>,
    /// Single-layer views used as render-pass targets.
    layer_views: Vec<Arc<wgpu::TextureView>>,
    /// Array layer count.
    layers: u32,
}

impl LightCookieLayeredAtlas {
    /// Creates a light-cookie atlas with one-layer render-target views.
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &'static str,
        layers: u32,
        format: LightCookieAtlasFormat,
    ) -> Self {
        let wgpu_format = format.wgpu();
        let texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: LIGHT_COOKIE_ATLAS_EDGE,
                height: LIGHT_COOKIE_ATLAS_EDGE,
                depth_or_array_layers: layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));
        write_white_layer(queue, texture.as_ref(), 0, format);
        let view = Arc::new(texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&format!("{label}_view")),
            format: Some(wgpu_format),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(layers),
        }));
        crate::profiling::note_resource_churn!(TextureView, "backend::light_cookie_atlas_view");
        let layer_views = (0..layers)
            .map(|layer| {
                Arc::new(texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(&format!("{label}_layer_{layer}")),
                    format: Some(wgpu_format),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                }))
            })
            .collect::<Vec<_>>();
        crate::profiling::note_resource_churn!(TextureView, "backend::light_cookie_layer_views");
        Self {
            _texture: texture,
            view,
            layer_views,
            layers,
        }
    }

    /// Returns a single-layer render target view.
    fn layer_view(&self, layer: u32) -> Option<&wgpu::TextureView> {
        self.layer_views.get(layer as usize).map(Arc::as_ref)
    }
}

/// Writes a white fallback layer.
fn write_white_layer(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    layer: u32,
    format: LightCookieAtlasFormat,
) {
    let bytes = white_layer_bytes(format);
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: 0,
                y: 0,
                z: layer,
            },
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(LIGHT_COOKIE_ATLAS_EDGE * format.bytes_per_texel()),
            rows_per_image: Some(LIGHT_COOKIE_ATLAS_EDGE),
        },
        wgpu::Extent3d {
            width: LIGHT_COOKIE_ATLAS_EDGE,
            height: LIGHT_COOKIE_ATLAS_EDGE,
            depth_or_array_layers: 1,
        },
    );
}

/// Builds a CPU-side full-white fallback layer for `format`.
fn white_layer_bytes(format: LightCookieAtlasFormat) -> Vec<u8> {
    let texels = (LIGHT_COOKIE_ATLAS_EDGE * LIGHT_COOKIE_ATLAS_EDGE) as usize;
    match format {
        LightCookieAtlasFormat::R16Float => {
            let mut bytes = Vec::with_capacity(texels * 2);
            for _ in 0..texels {
                bytes.extend_from_slice(&0x3c00u16.to_le_bytes());
            }
            bytes
        }
        LightCookieAtlasFormat::Rgba16Float => {
            let mut bytes = Vec::with_capacity(texels * 8);
            for _ in 0..(texels * 4) {
                bytes.extend_from_slice(&0x3c00u16.to_le_bytes());
            }
            bytes
        }
        LightCookieAtlasFormat::R8Unorm => vec![255u8; texels],
    }
}

/// Pipelines and bind-group layouts used to copy source cookies into atlases.
struct LightCookieBlitPipelines {
    /// Filterable 2D texture source bind-group layout.
    source_filter_layout: wgpu::BindGroupLayout,
    /// Non-filterable 2D texture source bind-group layout.
    source_non_filter_layout: wgpu::BindGroupLayout,
    /// Alpha-channel filterable source blit pipeline.
    alpha_filter_pipeline: wgpu::RenderPipeline,
    /// Red-channel filterable source blit pipeline.
    red_filter_pipeline: wgpu::RenderPipeline,
    /// Alpha-channel non-filterable source blit pipeline.
    alpha_non_filter_pipeline: wgpu::RenderPipeline,
    /// Red-channel non-filterable source blit pipeline.
    red_non_filter_pipeline: wgpu::RenderPipeline,
    /// Nearest sampler used with non-filterable float source textures.
    source_nearest_sampler: wgpu::Sampler,
}

impl LightCookieBlitPipelines {
    /// Creates blit pipelines for light-cookie atlas updates.
    fn new(device: &wgpu::Device, atlas_format: LightCookieAtlasFormat) -> Self {
        let source_filter_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("light_cookie_source_2d_filter_bgl"),
                entries: &[
                    sampled_texture_entry(0, wgpu::TextureViewDimension::D2, true),
                    sampler_entry(1, wgpu::SamplerBindingType::Filtering),
                ],
            });
        let source_non_filter_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("light_cookie_source_2d_non_filter_bgl"),
                entries: &[
                    sampled_texture_entry(0, wgpu::TextureViewDimension::D2, false),
                    sampler_entry(1, wgpu::SamplerBindingType::NonFiltering),
                ],
            });
        let alpha_filter_pipeline = create_blit_pipeline(
            device,
            "light_cookie_blit_alpha_filter",
            light_cookie_blit_2d_wgsl(),
            &source_filter_layout,
            blit_fragment_entry(LightCookieSourceChannel::Alpha, atlas_format),
            atlas_format,
        );
        let red_filter_pipeline = create_blit_pipeline(
            device,
            "light_cookie_blit_red_filter",
            light_cookie_blit_2d_wgsl(),
            &source_filter_layout,
            blit_fragment_entry(LightCookieSourceChannel::Red, atlas_format),
            atlas_format,
        );
        let alpha_non_filter_pipeline = create_blit_pipeline(
            device,
            "light_cookie_blit_alpha_non_filter",
            light_cookie_blit_2d_wgsl(),
            &source_non_filter_layout,
            blit_fragment_entry(LightCookieSourceChannel::Alpha, atlas_format),
            atlas_format,
        );
        let red_non_filter_pipeline = create_blit_pipeline(
            device,
            "light_cookie_blit_red_non_filter",
            light_cookie_blit_2d_wgsl(),
            &source_non_filter_layout,
            blit_fragment_entry(LightCookieSourceChannel::Red, atlas_format),
            atlas_format,
        );
        let source_nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("light_cookie_source_nearest_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        Self {
            source_filter_layout,
            source_non_filter_layout,
            alpha_filter_pipeline,
            red_filter_pipeline,
            alpha_non_filter_pipeline,
            red_non_filter_pipeline,
            source_nearest_sampler,
        }
    }

    /// Returns the bind-group layout for `sampling`.
    fn layout(&self, sampling: LightCookieSourceSampling) -> &wgpu::BindGroupLayout {
        match sampling {
            LightCookieSourceSampling::Filtering => &self.source_filter_layout,
            LightCookieSourceSampling::NonFiltering => &self.source_non_filter_layout,
        }
    }

    /// Returns the render pipeline for `channel` and `sampling`.
    fn pipeline(
        &self,
        channel: LightCookieSourceChannel,
        sampling: LightCookieSourceSampling,
    ) -> &wgpu::RenderPipeline {
        match (channel, sampling) {
            (LightCookieSourceChannel::Alpha, LightCookieSourceSampling::Filtering) => {
                &self.alpha_filter_pipeline
            }
            (LightCookieSourceChannel::Red, LightCookieSourceSampling::Filtering) => {
                &self.red_filter_pipeline
            }
            (LightCookieSourceChannel::Alpha, LightCookieSourceSampling::NonFiltering) => {
                &self.alpha_non_filter_pipeline
            }
            (LightCookieSourceChannel::Red, LightCookieSourceSampling::NonFiltering) => {
                &self.red_non_filter_pipeline
            }
        }
    }

    /// Returns the sampler used for source blits.
    fn sampler<'a>(
        &'a self,
        sampling: LightCookieSourceSampling,
        filtering_sampler: &'a wgpu::Sampler,
    ) -> &'a wgpu::Sampler {
        match sampling {
            LightCookieSourceSampling::Filtering => filtering_sampler,
            LightCookieSourceSampling::NonFiltering => &self.source_nearest_sampler,
        }
    }
}

/// Returns the fragment entry point matching the atlas target channel count.
fn blit_fragment_entry(
    channel: LightCookieSourceChannel,
    atlas_format: LightCookieAtlasFormat,
) -> &'static str {
    match (channel, atlas_format) {
        (LightCookieSourceChannel::Alpha, LightCookieAtlasFormat::Rgba16Float) => "fs_alpha_rgba",
        (LightCookieSourceChannel::Red, LightCookieAtlasFormat::Rgba16Float) => "fs_red_rgba",
        (LightCookieSourceChannel::Alpha, _) => "fs_alpha_scalar",
        (LightCookieSourceChannel::Red, _) => "fs_red_scalar",
    }
}

/// Builds a source bind group for one blit.
fn create_source_bind_group(
    device: &wgpu::Device,
    blit: &LightCookieBlitPipelines,
    source: LightCookieSource<'_>,
    filtering_sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("light_cookie_source_bg"),
        layout: blit.layout(source.sampling),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(source.view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(
                    blit.sampler(source.sampling, filtering_sampler),
                ),
            },
        ],
    })
}

/// Returns the composed 2D light-cookie blit shader.
fn light_cookie_blit_2d_wgsl() -> &'static str {
    let Some(source) = crate::embedded_shaders::embedded_target_wgsl(LIGHT_COOKIE_BLIT_2D_STEM)
    else {
        logger::warn!(
            "embedded WGSL target `{LIGHT_COOKIE_BLIT_2D_STEM}` missing; using raw source fallback"
        );
        return LIGHT_COOKIE_BLIT_2D_SOURCE;
    };
    source
}

/// Builds a sampled texture binding layout entry.
fn sampled_texture_entry(
    binding: u32,
    view_dimension: wgpu::TextureViewDimension,
    filterable: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension,
            multisampled: false,
        },
        count: None,
    }
}

/// Builds a sampler binding layout entry.
fn sampler_entry(
    binding: u32,
    sampler_type: wgpu::SamplerBindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(sampler_type),
        count: None,
    }
}

/// Creates a fullscreen scalar-cookie render pipeline.
fn create_blit_pipeline(
    device: &wgpu::Device,
    label: &'static str,
    source: &'static str,
    bind_group_layout: &wgpu::BindGroupLayout,
    fragment_entry: &'static str,
    atlas_format: LightCookieAtlasFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label}_layout")),
        bind_group_layouts: &[Some(bind_group_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some(fragment_entry),
            targets: &[Some(wgpu::ColorTargetState {
                format: atlas_format.wgpu(),
                blend: None,
                write_mask: wgpu::ColorWrites::RED,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    crate::profiling::note_resource_churn!(RenderPipeline, "backend::light_cookie_blit_pipeline");
    pipeline
}

/// Returns the source sampling mode supported by `format`.
fn source_sampling_for_limits(
    limits: &GpuLimits,
    format: wgpu::TextureFormat,
) -> Option<LightCookieSourceSampling> {
    let features = limits.texture_format_features(format);
    if !features
        .allowed_usages
        .contains(wgpu::TextureUsages::TEXTURE_BINDING)
    {
        return None;
    }
    if features
        .flags
        .contains(wgpu::TextureFormatFeatureFlags::FILTERABLE)
    {
        return Some(LightCookieSourceSampling::Filtering);
    }
    Some(LightCookieSourceSampling::NonFiltering)
}

/// Frame-global light-cookie atlas resources.
pub(super) struct LightCookieAtlasResources {
    /// 2D cookie atlas shared by spot and directional lights.
    two_d: LightCookieLayeredAtlas,
    /// Point-light cubemap-face cookie atlas.
    point: LightCookieLayeredAtlas,
    /// Sampler used by material lighting shaders.
    sampler: Arc<wgpu::Sampler>,
    /// Source-to-atlas blit pipelines.
    blit: LightCookieBlitPipelines,
    /// Assignment state for current and recent frames.
    state: Mutex<LightCookieAtlasState>,
    /// GPU limits used for source-format validation.
    limits: Arc<GpuLimits>,
}

impl LightCookieAtlasResources {
    /// Creates frame-global light-cookie atlas resources.
    pub(super) fn new(device: &wgpu::Device, queue: &wgpu::Queue, limits: Arc<GpuLimits>) -> Self {
        let max_layers = limits.max_texture_array_layers().max(1);
        let two_d_layers = COOKIE_2D_LAYER_CAP.min(max_layers).max(1);
        let point_layers = (1 + POINT_COOKIE_CUBEMAP_CAP * POINT_COOKIE_FACE_COUNT)
            .min(max_layers)
            .max(1);
        let atlas_format = select_light_cookie_atlas_format(&limits);
        let two_d = LightCookieLayeredAtlas::new(
            device,
            queue,
            "frame_light_cookie_2d_atlas",
            two_d_layers,
            atlas_format,
        );
        let point = LightCookieLayeredAtlas::new(
            device,
            queue,
            "frame_light_cookie_point_atlas",
            point_layers,
            atlas_format,
        );
        let sampler = Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("frame_light_cookie_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        }));
        let blit = LightCookieBlitPipelines::new(device, atlas_format);
        Self {
            two_d,
            point,
            sampler,
            blit,
            state: Mutex::new(LightCookieAtlasState::new()),
            limits,
        }
    }

    /// Full 2D cookie atlas view for group-0 binding.
    pub(super) fn two_d_view(&self) -> &wgpu::TextureView {
        self.two_d.view.as_ref()
    }

    /// Full point-cookie atlas view for group-0 binding.
    pub(super) fn point_view(&self) -> &wgpu::TextureView {
        self.point.view.as_ref()
    }

    /// Cookie sampler for group-0 binding.
    pub(super) fn sampler(&self) -> &wgpu::Sampler {
        self.sampler.as_ref()
    }

    /// Starts a new light-cookie assignment frame.
    pub(super) fn begin_frame(&self) {
        self.state.lock().begin_frame();
    }

    /// Assigns a cookie atlas binding for one resolved light.
    pub(super) fn assign(
        &self,
        light_type: LightType,
        packed_id: i32,
        assets: Option<&dyn GraphAssetResources>,
    ) -> LightCookieBinding {
        let Some((asset_id, kind)) = unpack_host_texture_packed(packed_id) else {
            return LightCookieBinding::NONE;
        };
        let wrap_bits = self.source_wrap_bits(assets, asset_id, kind);
        let assignment = LightCookieAssignment {
            light_type,
            packed_id,
            asset_id,
            kind,
            wrap_bits,
        };
        self.state
            .lock()
            .assign(assignment, self.two_d.layers, self.point.layers)
    }

    /// Returns whether a frame-global atlas update pass has work.
    pub(super) fn has_requests(&self) -> bool {
        self.state.lock().has_requests()
    }

    /// Records all current-frame atlas clears and source blits.
    pub(super) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        assets: &dyn GraphAssetResources,
    ) {
        profiling::scope!("light_cookies::encode_atlas");
        let (two_d_requests, point_requests) = self.state.lock().requests();
        for request in two_d_requests {
            self.encode_2d_request(device, encoder, assets, request);
        }
        for request in point_requests {
            self.encode_point_request(device, encoder, assets, request);
        }
    }

    /// Records one 2D cookie atlas update.
    fn encode_2d_request(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        assets: &dyn GraphAssetResources,
        request: LightCookieRequest,
    ) {
        let Some(target) = self.two_d.layer_view(request.layer) else {
            return;
        };
        let Some(source) = self.resolve_2d_source(assets, request) else {
            clear_cookie_layer(encoder, target, "light_cookie_2d_clear");
            return;
        };
        let bind_group = create_source_bind_group(device, &self.blit, source, self.sampler());
        crate::profiling::note_resource_churn!(BindGroup, "backend::light_cookie_2d_source_bg");
        blit_cookie_layer(
            encoder,
            target,
            "light_cookie_2d_blit",
            self.blit.pipeline(source.channel, source.sampling),
            &bind_group,
        );
    }

    /// Records one point-light cubemap cookie atlas update.
    fn encode_point_request(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        assets: &dyn GraphAssetResources,
        request: LightCookieRequest,
    ) {
        let Some(source) = self.resolve_point_source(assets, request) else {
            for face in 0..POINT_COOKIE_FACE_COUNT {
                if let Some(target) = self.point.layer_view(request.layer + face) {
                    clear_cookie_layer(encoder, target, "light_cookie_point_clear");
                }
            }
            return;
        };
        for face in 0..POINT_COOKIE_FACE_COUNT {
            let Some(target) = self.point.layer_view(request.layer + face) else {
                continue;
            };
            let face_source = LightCookieSource {
                view: source.cubemap.face_views[face as usize].as_ref(),
                channel: source.channel,
                sampling: source.sampling,
            };
            let bind_group =
                create_source_bind_group(device, &self.blit, face_source, self.sampler());
            crate::profiling::note_resource_churn!(
                BindGroup,
                "backend::light_cookie_point_source_bg"
            );
            blit_cookie_layer(
                encoder,
                target,
                "light_cookie_point_blit",
                self.blit
                    .pipeline(face_source.channel, face_source.sampling),
                &bind_group,
            );
        }
    }

    /// Resolves a 2D source texture view and sampling policy.
    fn resolve_2d_source<'a>(
        &self,
        assets: &'a dyn GraphAssetResources,
        request: LightCookieRequest,
    ) -> Option<LightCookieSource<'a>> {
        match request.kind {
            HostTextureAssetKind::Texture2D => {
                let texture = assets.texture_pool().get(request.asset_id)?;
                if texture.mip_levels_resident == 0 {
                    return None;
                }
                Some(LightCookieSource {
                    view: texture.view.as_ref(),
                    channel: source_channel_for_host_format(texture.host_format),
                    sampling: self.source_sampling(texture.wgpu_format)?,
                })
            }
            HostTextureAssetKind::RenderTexture => {
                let texture = assets.render_texture_pool().get(request.asset_id)?;
                if !texture.is_sampleable() {
                    return None;
                }
                Some(LightCookieSource {
                    view: texture.color_view.as_ref(),
                    channel: LightCookieSourceChannel::Alpha,
                    sampling: self.source_sampling(texture.wgpu_color_format)?,
                })
            }
            HostTextureAssetKind::VideoTexture => {
                let texture = assets.video_texture_pool().get(request.asset_id)?;
                if !texture.is_sampleable() {
                    return None;
                }
                Some(LightCookieSource {
                    view: texture.view.as_ref(),
                    channel: LightCookieSourceChannel::Alpha,
                    sampling: LightCookieSourceSampling::Filtering,
                })
            }
            HostTextureAssetKind::Texture3D
            | HostTextureAssetKind::Cubemap
            | HostTextureAssetKind::Desktop => {
                logger::trace!(
                    "2D light cookie {} ignored unsupported source kind {:?}",
                    request.packed_id,
                    request.kind
                );
                None
            }
        }
    }

    /// Returns packed U/V wrap mode bits for a 2D cookie source.
    fn source_wrap_bits(
        &self,
        assets: Option<&dyn GraphAssetResources>,
        asset_id: i32,
        kind: HostTextureAssetKind,
    ) -> u32 {
        let Some(assets) = assets else {
            return 0;
        };
        match kind {
            HostTextureAssetKind::Texture2D => assets
                .texture_pool()
                .get(asset_id)
                .map_or(0, |texture| light_cookie_wrap_bits(&texture.sampler)),
            HostTextureAssetKind::RenderTexture => assets
                .render_texture_pool()
                .get(asset_id)
                .map_or(0, |texture| light_cookie_wrap_bits(&texture.sampler)),
            HostTextureAssetKind::VideoTexture => assets
                .video_texture_pool()
                .get(asset_id)
                .map_or(0, |texture| light_cookie_wrap_bits(&texture.sampler)),
            HostTextureAssetKind::Cubemap
            | HostTextureAssetKind::Texture3D
            | HostTextureAssetKind::Desktop => 0,
        }
    }

    /// Resolves a point-light cubemap source texture view.
    fn resolve_point_source<'a>(
        &self,
        assets: &'a dyn GraphAssetResources,
        request: LightCookieRequest,
    ) -> Option<LightCookiePointSource<'a>> {
        if request.kind != HostTextureAssetKind::Cubemap {
            return None;
        }
        let cubemap = assets.cubemap_pool().get(request.asset_id)?;
        if cubemap.mip_levels_resident == 0 {
            return None;
        }
        Some(LightCookiePointSource {
            cubemap,
            channel: source_channel_for_host_format(cubemap.host_format),
            sampling: self.source_sampling(cubemap.wgpu_format)?,
        })
    }

    /// Returns the source sampling mode supported by `format`.
    fn source_sampling(&self, format: wgpu::TextureFormat) -> Option<LightCookieSourceSampling> {
        source_sampling_for_limits(&self.limits, format)
    }
}

/// Clears a cookie layer to white.
fn clear_cookie_layer(
    encoder: &mut wgpu::CommandEncoder,
    target: &wgpu::TextureView,
    label: &'static str,
) {
    let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
}

/// Draws one fullscreen blit into a cookie layer.
fn blit_cookie_layer(
    encoder: &mut wgpu::CommandEncoder,
    target: &wgpu::TextureView,
    label: &'static str,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}

/// Encoder pass label for diagnostics.
pub(crate) const LIGHT_COOKIE_ATLAS_PASS_NAME: &str = "light_cookie_atlas";

/// Main-graph frame-global pass that updates light-cookie atlas layers.
pub(crate) struct LightCookieAtlasPass;

impl LightCookieAtlasPass {
    /// Creates the light-cookie atlas update pass.
    pub(crate) const fn new() -> Self {
        Self
    }
}

impl crate::render_graph::pass::EncoderPass for LightCookieAtlasPass {
    fn name(&self) -> &str {
        LIGHT_COOKIE_ATLAS_PASS_NAME
    }

    fn profiling_label(&self) -> Cow<'_, str> {
        Cow::Borrowed("light_cookies::atlas")
    }

    fn setup(
        &mut self,
        builder: &mut crate::render_graph::pass::PassBuilder<'_>,
    ) -> Result<(), crate::render_graph::error::SetupError> {
        builder.encoder();
        builder.cull_exempt();
        builder.never_parallel();
        Ok(())
    }

    fn should_record(
        &self,
        ctx: &crate::render_graph::context::EncoderPassCtx<'_, '_, '_>,
    ) -> Result<bool, crate::render_graph::error::RenderPassError> {
        Ok(ctx
            .pass_frame
            .shared
            .frame_resources
            .has_light_cookie_requests())
    }

    fn record(
        &self,
        ctx: &mut crate::render_graph::context::EncoderPassCtx<'_, '_, '_>,
    ) -> Result<(), crate::render_graph::error::RenderPassError> {
        ctx.pass_frame
            .shared
            .frame_resources
            .encode_light_cookie_atlas(
                ctx.device,
                ctx.encoder,
                ctx.pass_frame.shared.asset_resources,
            );
        Ok(())
    }

    fn phase(&self) -> crate::render_graph::pass::PassPhase {
        crate::render_graph::pass::PassPhase::FrameGlobal
    }
}

#[cfg(test)]
mod tests {
    use super::{
        LIGHT_COOKIE_ATLAS_EDGE, LIGHT_COOKIE_BLIT_2D_STEM, LightCookieAssignment,
        LightCookieAtlasFormat, LightCookieAtlasState, LightCookieSourceChannel,
        LightCookieSourceSampling, light_cookie_atlas_format_supported, light_cookie_wrap_bits,
        select_light_cookie_atlas_format, source_channel_for_host_format,
        source_sampling_for_limits, white_layer_bytes,
    };
    use crate::assets::texture::HostTextureAssetKind;
    use crate::gpu::{
        GpuLimits, LIGHT_COOKIE_KIND_DIRECTIONAL_2D, LIGHT_COOKIE_WRAP_MODE_CLAMP,
        LIGHT_COOKIE_WRAP_MODE_MIRROR_ONCE, LIGHT_COOKIE_WRAP_U_SHIFT, LIGHT_COOKIE_WRAP_V_SHIFT,
    };
    use crate::gpu_pools::SamplerState;
    use crate::shared::{LightType, TextureFormat, TextureWrapMode};

    use hashbrown::HashMap;

    #[test]
    fn blit_shader_stem_resolves_to_embedded_wgsl() {
        let wgsl = crate::embedded_shaders::embedded_target_wgsl(LIGHT_COOKIE_BLIT_2D_STEM);
        assert!(wgsl.is_some_and(|source| !source.trim().is_empty()));
    }

    #[test]
    fn atlas_format_prefers_signed_filterable_r16_float() {
        let limits = limits_with_format(
            wgpu::TextureFormat::R16Float,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_DST,
            wgpu::TextureFormatFeatureFlags::FILTERABLE,
        );

        assert!(light_cookie_atlas_format_supported(
            &limits,
            LightCookieAtlasFormat::R16Float
        ));
        assert_eq!(
            select_light_cookie_atlas_format(&limits),
            LightCookieAtlasFormat::R16Float
        );
    }

    #[test]
    fn atlas_format_falls_back_when_r16_float_is_not_filterable() {
        let limits = limits_with_format_features([
            (
                wgpu::TextureFormat::R16Float,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_DST,
                wgpu::TextureFormatFeatureFlags::empty(),
            ),
            (
                wgpu::TextureFormat::Rgba16Float,
                wgpu::TextureUsages::empty(),
                wgpu::TextureFormatFeatureFlags::empty(),
            ),
        ]);

        assert!(!light_cookie_atlas_format_supported(
            &limits,
            LightCookieAtlasFormat::R16Float
        ));
        assert_eq!(
            select_light_cookie_atlas_format(&limits),
            LightCookieAtlasFormat::R8Unorm
        );
    }

    #[test]
    fn atlas_format_uses_rgba16_float_when_scalar_hdr_is_unavailable() {
        let limits = limits_with_format_features([
            (
                wgpu::TextureFormat::R16Float,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_DST,
                wgpu::TextureFormatFeatureFlags::empty(),
            ),
            (
                wgpu::TextureFormat::Rgba16Float,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_DST,
                wgpu::TextureFormatFeatureFlags::FILTERABLE,
            ),
        ]);

        assert!(!light_cookie_atlas_format_supported(
            &limits,
            LightCookieAtlasFormat::R16Float
        ));
        assert!(light_cookie_atlas_format_supported(
            &limits,
            LightCookieAtlasFormat::Rgba16Float
        ));
        assert_eq!(
            select_light_cookie_atlas_format(&limits),
            LightCookieAtlasFormat::Rgba16Float
        );
    }

    #[test]
    fn source_channel_uses_alpha_for_alpha_capable_formats() {
        for format in [
            TextureFormat::ARGB32,
            TextureFormat::RGBA32,
            TextureFormat::BGRA32,
            TextureFormat::RGBAHalf,
            TextureFormat::ARGBHalf,
            TextureFormat::RGBAFloat,
            TextureFormat::ARGBFloat,
            TextureFormat::BC3,
            TextureFormat::BC7,
            TextureFormat::ETC2RGBA8,
            TextureFormat::ASTC4x4,
        ] {
            assert_eq!(
                source_channel_for_host_format(format),
                LightCookieSourceChannel::Alpha,
                "{format:?}"
            );
        }
    }

    #[test]
    fn source_channel_uses_red_for_scalar_and_no_alpha_formats() {
        for format in [
            TextureFormat::Alpha8,
            TextureFormat::R8,
            TextureFormat::RHalf,
            TextureFormat::RFloat,
            TextureFormat::RGFloat,
            TextureFormat::RGB24,
            TextureFormat::BC1,
            TextureFormat::BC4,
            TextureFormat::ETC2RGB,
        ] {
            assert_eq!(
                source_channel_for_host_format(format),
                LightCookieSourceChannel::Red,
                "{format:?}"
            );
        }
    }

    #[test]
    fn source_sampling_accepts_filtering_and_non_filtering_textures() {
        let filterable = limits_with_format(
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::TEXTURE_BINDING,
            wgpu::TextureFormatFeatureFlags::FILTERABLE,
        );
        let non_filterable = limits_with_format(
            wgpu::TextureFormat::Rgba32Float,
            wgpu::TextureUsages::TEXTURE_BINDING,
            wgpu::TextureFormatFeatureFlags::empty(),
        );
        let not_sampled = limits_with_format(
            wgpu::TextureFormat::Rgba32Float,
            wgpu::TextureUsages::COPY_DST,
            wgpu::TextureFormatFeatureFlags::empty(),
        );

        assert_eq!(
            source_sampling_for_limits(&filterable, wgpu::TextureFormat::Rgba16Float),
            Some(LightCookieSourceSampling::Filtering)
        );
        assert_eq!(
            source_sampling_for_limits(&non_filterable, wgpu::TextureFormat::Rgba32Float),
            Some(LightCookieSourceSampling::NonFiltering)
        );
        assert_eq!(
            source_sampling_for_limits(&not_sampled, wgpu::TextureFormat::Rgba32Float),
            None
        );
    }

    #[test]
    fn white_layer_bytes_match_atlas_formats() {
        let texels = (LIGHT_COOKIE_ATLAS_EDGE * LIGHT_COOKIE_ATLAS_EDGE) as usize;
        let r8 = white_layer_bytes(LightCookieAtlasFormat::R8Unorm);
        assert_eq!(r8.len(), texels);
        assert!(r8.iter().all(|&b| b == 255));

        let r16 = white_layer_bytes(LightCookieAtlasFormat::R16Float);
        assert_eq!(r16.len(), texels * 2);
        assert!(r16.chunks_exact(2).all(|bytes| bytes == [0x00, 0x3c]));

        let rgba16 = white_layer_bytes(LightCookieAtlasFormat::Rgba16Float);
        assert_eq!(rgba16.len(), texels * 8);
        assert!(rgba16.chunks_exact(2).all(|bytes| bytes == [0x00, 0x3c]));
    }

    #[test]
    fn light_cookie_wrap_bits_pack_u_and_v_modes() {
        let sampler = SamplerState {
            wrap_u: TextureWrapMode::Clamp,
            wrap_v: TextureWrapMode::MirrorOnce,
            ..Default::default()
        };
        let bits = light_cookie_wrap_bits(&sampler);

        assert_eq!(
            bits,
            (LIGHT_COOKIE_WRAP_MODE_CLAMP << LIGHT_COOKIE_WRAP_U_SHIFT)
                | (LIGHT_COOKIE_WRAP_MODE_MIRROR_ONCE << LIGHT_COOKIE_WRAP_V_SHIFT)
        );
    }

    #[test]
    fn assigns_directional_cookies_to_2d_atlas() {
        let mut state = LightCookieAtlasState::new();
        let binding = state.assign(
            LightCookieAssignment {
                light_type: LightType::Directional,
                packed_id: 42,
                asset_id: 7,
                kind: HostTextureAssetKind::Texture2D,
                wrap_bits: 0x0d,
            },
            8,
            8,
        );

        assert_eq!(binding.kind, LIGHT_COOKIE_KIND_DIRECTIONAL_2D);
        assert_eq!(binding.layer, 1);
        assert_eq!(binding.wrap_bits, 0x0d);
        let (two_d, point) = state.requests();
        assert_eq!(two_d.len(), 1);
        assert_eq!(two_d[0].asset_id, 7);
        assert!(point.is_empty());
    }

    fn limits_with_format(
        format: wgpu::TextureFormat,
        allowed_usages: wgpu::TextureUsages,
        flags: wgpu::TextureFormatFeatureFlags,
    ) -> GpuLimits {
        limits_with_format_features([(format, allowed_usages, flags)])
    }

    fn limits_with_format_features<const N: usize>(
        features: [(
            wgpu::TextureFormat,
            wgpu::TextureUsages,
            wgpu::TextureFormatFeatureFlags,
        ); N],
    ) -> GpuLimits {
        let mut format_features = HashMap::new();
        for (format, allowed_usages, flags) in features {
            format_features.insert(
                format,
                wgpu::TextureFormatFeatures {
                    allowed_usages,
                    flags,
                },
            );
        }
        GpuLimits::synthetic_for_tests(
            wgpu::Limits {
                max_texture_dimension_2d: 4096,
                max_texture_dimension_3d: 4096,
                max_texture_array_layers: 64,
                max_storage_buffer_binding_size: 256 * 1024,
                max_buffer_size: 256 * 1024,
                ..Default::default()
            },
            wgpu::Features::empty(),
            format_features,
        )
    }
}
