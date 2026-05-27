//! Frame-global light-cookie atlases and GPU blit support.

use std::borrow::Cow;
use std::sync::Arc;

use hashbrown::HashMap;
use parking_lot::Mutex;

use crate::assets::texture::{HostTextureAssetKind, unpack_host_texture_packed};
use crate::backend::light_gpu::{
    LIGHT_COOKIE_KIND_POINT_CUBE, LIGHT_COOKIE_KIND_SPOT_2D, LightCookieBinding,
};
use crate::gpu::GpuLimits;
use crate::gpu_pools::GpuCubemap;
use crate::render_graph::GraphAssetResources;
use crate::shared::LightType;

/// Edge length of each light-cookie atlas layer.
const LIGHT_COOKIE_ATLAS_EDGE: u32 = 256;
/// Maximum spotlight cookie layers including the fallback layer.
const SPOT_COOKIE_LAYER_CAP: u32 = 64;
/// Maximum resident point-light cookie cubemaps.
const POINT_COOKIE_CUBEMAP_CAP: u32 = 16;
/// Cubemap face count.
const POINT_COOKIE_FACE_COUNT: u32 = 6;
/// Atlas format used for scalar cookie alpha masks.
const LIGHT_COOKIE_ATLAS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// Embedded WGSL target for copying 2D source cookies into atlas layers.
const LIGHT_COOKIE_BLIT_2D_STEM: &str = "light_cookie_blit_2d";
/// Source WGSL used only if embedded shader metadata is unexpectedly missing.
const LIGHT_COOKIE_BLIT_2D_SOURCE: &str =
    include_str!("../../../shaders/passes/backend/light_cookie_blit_2d.wgsl");

/// One requested light-cookie source assigned to an atlas layer.
#[derive(Clone, Copy, Debug)]
struct LightCookieRequest {
    /// Packed host texture handle.
    packed_id: i32,
    /// Unpacked host asset id.
    asset_id: i32,
    /// Unpacked host texture kind.
    kind: HostTextureAssetKind,
    /// Spot atlas layer or first point face layer.
    layer: u32,
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
    /// Persistent spot-cookie slots keyed by packed texture handle.
    spot_slots: HashMap<i32, LightCookieSlot>,
    /// Persistent point-cookie slots keyed by packed texture handle.
    point_slots: HashMap<i32, LightCookieSlot>,
    /// Unique spot-cookie requests for the current frame.
    spot_requests: Vec<LightCookieRequest>,
    /// Unique point-cookie requests for the current frame.
    point_requests: Vec<LightCookieRequest>,
    /// One-shot guard for spot-cookie atlas overflow.
    spot_overflow_logged: bool,
    /// One-shot guard for point-cookie atlas overflow.
    point_overflow_logged: bool,
}

impl LightCookieAtlasState {
    /// Creates an empty assignment table.
    fn new() -> Self {
        Self {
            spot_slots: HashMap::new(),
            point_slots: HashMap::new(),
            spot_requests: Vec::new(),
            point_requests: Vec::new(),
            spot_overflow_logged: false,
            point_overflow_logged: false,
        }
    }

    /// Marks all slots unrequested and clears current-frame request lists.
    fn begin_frame(&mut self) {
        for slot in self.spot_slots.values_mut() {
            slot.requested_this_frame = false;
        }
        for slot in self.point_slots.values_mut() {
            slot.requested_this_frame = false;
        }
        self.spot_requests.clear();
        self.point_requests.clear();
    }

    /// Assigns a cookie atlas binding for one resolved light.
    fn assign(
        &mut self,
        light_type: LightType,
        packed_id: i32,
        spot_layers: u32,
        point_layers: u32,
    ) -> LightCookieBinding {
        let Some((asset_id, kind)) = unpack_host_texture_packed(packed_id) else {
            return LightCookieBinding::NONE;
        };
        match (light_type, kind) {
            (
                LightType::Spot,
                HostTextureAssetKind::Texture2D
                | HostTextureAssetKind::RenderTexture
                | HostTextureAssetKind::VideoTexture,
            ) => self.assign_spot(packed_id, asset_id, kind, spot_layers),
            (LightType::Point, HostTextureAssetKind::Cubemap) => {
                self.assign_point(packed_id, asset_id, kind, point_layers)
            }
            _ => LightCookieBinding::NONE,
        }
    }

    /// Assigns a 2D spotlight cookie layer.
    fn assign_spot(
        &mut self,
        packed_id: i32,
        asset_id: i32,
        kind: HostTextureAssetKind,
        layers: u32,
    ) -> LightCookieBinding {
        let Some(layer) = assign_cookie_layer(
            &mut self.spot_slots,
            packed_id,
            1,
            layers,
            1,
            &mut self.spot_overflow_logged,
            "spot",
        ) else {
            return LightCookieBinding::NONE;
        };
        if let Some(slot) = self.spot_slots.get_mut(&packed_id)
            && !slot.requested_this_frame
        {
            slot.requested_this_frame = true;
            self.spot_requests.push(LightCookieRequest {
                packed_id,
                asset_id,
                kind,
                layer,
            });
        }
        LightCookieBinding {
            kind: LIGHT_COOKIE_KIND_SPOT_2D,
            layer,
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
        }
    }

    /// Returns whether any current-frame request needs atlas synchronization.
    fn has_requests(&self) -> bool {
        !(self.spot_requests.is_empty() && self.point_requests.is_empty())
    }

    /// Snapshot of requests for encoder recording without holding the state lock.
    fn requests(&self) -> (Vec<LightCookieRequest>, Vec<LightCookieRequest>) {
        (self.spot_requests.clone(), self.point_requests.clone())
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
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, label: &'static str, layers: u32) -> Self {
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
            format: LIGHT_COOKIE_ATLAS_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));
        write_white_layer(queue, texture.as_ref(), 0);
        let view = Arc::new(texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&format!("{label}_view")),
            format: Some(LIGHT_COOKIE_ATLAS_FORMAT),
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
                    format: Some(LIGHT_COOKIE_ATLAS_FORMAT),
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
fn write_white_layer(queue: &wgpu::Queue, texture: &wgpu::Texture, layer: u32) {
    let bytes = vec![255u8; (LIGHT_COOKIE_ATLAS_EDGE * LIGHT_COOKIE_ATLAS_EDGE * 4) as usize];
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
            bytes_per_row: Some(LIGHT_COOKIE_ATLAS_EDGE * 4),
            rows_per_image: Some(LIGHT_COOKIE_ATLAS_EDGE),
        },
        wgpu::Extent3d {
            width: LIGHT_COOKIE_ATLAS_EDGE,
            height: LIGHT_COOKIE_ATLAS_EDGE,
            depth_or_array_layers: 1,
        },
    );
}

/// Pipelines and bind-group layouts used to copy source cookies into atlases.
struct LightCookieBlitPipelines {
    /// 2D texture source bind-group layout.
    source_2d_layout: wgpu::BindGroupLayout,
    /// 2D source blit pipeline.
    source_2d_pipeline: wgpu::RenderPipeline,
}

impl LightCookieBlitPipelines {
    /// Creates blit pipelines for light-cookie atlas updates.
    fn new(device: &wgpu::Device) -> Self {
        let source_2d_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("light_cookie_source_2d_bgl"),
            entries: &[
                sampled_texture_entry(0, wgpu::TextureViewDimension::D2),
                sampler_entry(1),
            ],
        });
        let source_2d_pipeline = create_blit_pipeline(
            device,
            "light_cookie_blit_2d",
            light_cookie_blit_2d_wgsl(),
            &source_2d_layout,
            "fs_main",
        );
        Self {
            source_2d_layout,
            source_2d_pipeline,
        }
    }
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
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension,
            multisampled: false,
        },
        count: None,
    }
}

/// Builds a filtering sampler binding layout entry.
fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

/// Creates a fullscreen alpha-copy render pipeline.
fn create_blit_pipeline(
    device: &wgpu::Device,
    label: &'static str,
    source: &'static str,
    bind_group_layout: &wgpu::BindGroupLayout,
    fragment_entry: &'static str,
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
                format: LIGHT_COOKIE_ATLAS_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
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

/// Frame-global light-cookie atlas resources.
pub(super) struct LightCookieAtlasResources {
    /// Spotlight 2D cookie atlas.
    spot: LightCookieLayeredAtlas,
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
        let spot_layers = SPOT_COOKIE_LAYER_CAP.min(max_layers).max(1);
        let point_layers = (1 + POINT_COOKIE_CUBEMAP_CAP * POINT_COOKIE_FACE_COUNT)
            .min(max_layers)
            .max(1);
        let spot = LightCookieLayeredAtlas::new(
            device,
            queue,
            "frame_light_cookie_spot_atlas",
            spot_layers,
        );
        let point = LightCookieLayeredAtlas::new(
            device,
            queue,
            "frame_light_cookie_point_atlas",
            point_layers,
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
        let blit = LightCookieBlitPipelines::new(device);
        Self {
            spot,
            point,
            sampler,
            blit,
            state: Mutex::new(LightCookieAtlasState::new()),
            limits,
        }
    }

    /// Full spotlight atlas view for group-0 binding.
    pub(super) fn spot_view(&self) -> &wgpu::TextureView {
        self.spot.view.as_ref()
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
    pub(super) fn assign(&self, light_type: LightType, packed_id: i32) -> LightCookieBinding {
        self.state
            .lock()
            .assign(light_type, packed_id, self.spot.layers, self.point.layers)
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
        let (spot_requests, point_requests) = self.state.lock().requests();
        for request in spot_requests {
            self.encode_spot_request(device, encoder, assets, request);
        }
        for request in point_requests {
            self.encode_point_request(device, encoder, assets, request);
        }
    }

    /// Records one spotlight cookie atlas update.
    fn encode_spot_request(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        assets: &dyn GraphAssetResources,
        request: LightCookieRequest,
    ) {
        let Some(target) = self.spot.layer_view(request.layer) else {
            return;
        };
        let Some(source) = self.resolve_spot_source(assets, request) else {
            clear_cookie_layer(encoder, target, "light_cookie_spot_clear");
            return;
        };
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("light_cookie_spot_source_bg"),
            layout: &self.blit.source_2d_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(self.sampler()),
                },
            ],
        });
        crate::profiling::note_resource_churn!(BindGroup, "backend::light_cookie_spot_source_bg");
        blit_cookie_layer(
            encoder,
            target,
            "light_cookie_spot_blit",
            &self.blit.source_2d_pipeline,
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
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("light_cookie_point_source_bg"),
                layout: &self.blit.source_2d_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            source.face_views[face as usize].as_ref(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(self.sampler()),
                    },
                ],
            });
            crate::profiling::note_resource_churn!(
                BindGroup,
                "backend::light_cookie_point_source_bg"
            );
            blit_cookie_layer(
                encoder,
                target,
                "light_cookie_point_blit",
                &self.blit.source_2d_pipeline,
                &bind_group,
            );
        }
    }

    /// Resolves a spotlight source texture view.
    fn resolve_spot_source<'a>(
        &self,
        assets: &'a dyn GraphAssetResources,
        request: LightCookieRequest,
    ) -> Option<&'a wgpu::TextureView> {
        match request.kind {
            HostTextureAssetKind::Texture2D => {
                let texture = assets.texture_pool().get(request.asset_id)?;
                if texture.mip_levels_resident == 0
                    || !self.source_format_filterable(texture.wgpu_format)
                {
                    return None;
                }
                Some(texture.view.as_ref())
            }
            HostTextureAssetKind::RenderTexture => {
                let texture = assets.render_texture_pool().get(request.asset_id)?;
                if !texture.is_sampleable()
                    || !self.source_format_filterable(texture.wgpu_color_format)
                {
                    return None;
                }
                Some(texture.color_view.as_ref())
            }
            HostTextureAssetKind::VideoTexture => {
                let texture = assets.video_texture_pool().get(request.asset_id)?;
                if !texture.is_sampleable() {
                    return None;
                }
                Some(texture.view.as_ref())
            }
            HostTextureAssetKind::Texture3D
            | HostTextureAssetKind::Cubemap
            | HostTextureAssetKind::Desktop => {
                logger::trace!(
                    "spotlight cookie {} ignored unsupported source kind {:?}",
                    request.packed_id,
                    request.kind
                );
                None
            }
        }
    }

    /// Resolves a point-light cubemap source texture view.
    fn resolve_point_source<'a>(
        &self,
        assets: &'a dyn GraphAssetResources,
        request: LightCookieRequest,
    ) -> Option<&'a GpuCubemap> {
        if request.kind != HostTextureAssetKind::Cubemap {
            return None;
        }
        let cubemap = assets.cubemap_pool().get(request.asset_id)?;
        if cubemap.mip_levels_resident == 0 || !self.source_format_filterable(cubemap.wgpu_format) {
            return None;
        }
        Some(cubemap)
    }

    /// Returns whether a source format can be sampled by the filtering blit pipeline.
    fn source_format_filterable(&self, format: wgpu::TextureFormat) -> bool {
        self.limits
            .texture_format_features(format)
            .flags
            .contains(wgpu::TextureFormatFeatureFlags::FILTERABLE)
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
    use super::LIGHT_COOKIE_BLIT_2D_STEM;

    #[test]
    fn blit_shader_stem_resolves_to_embedded_wgsl() {
        let wgsl = crate::embedded_shaders::embedded_target_wgsl(LIGHT_COOKIE_BLIT_2D_STEM);
        assert!(wgsl.is_some_and(|source| !source.trim().is_empty()));
    }
}
