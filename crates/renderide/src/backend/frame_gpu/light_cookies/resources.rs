use std::mem::size_of;
use std::sync::Arc;

use hashbrown::HashMap;
use parking_lot::Mutex;

use crate::assets::texture::{HostTextureAssetKind, unpack_host_texture_packed};
use crate::backend::light_gpu::LightCookieBinding;
use crate::frame_upload_batch::GraphUploadSink;
use crate::gpu::{GpuLightCookieRect, GpuLimits};
use crate::render_graph::GraphAssetResources;
use crate::shared::LightType;

use super::POINT_COOKIE_FACE_COUNT;
use super::assignment::{
    LIGHT_COOKIE_RECT_CAPACITY, LightCookieAssignment, LightCookieAtlasState, LightCookieRequest,
};
use super::atlas::LightCookiePackedAtlas;
use super::blit::{
    LightCookieBlitPipelines, blit_cookie_rect, clear_cookie_atlas, create_source_bind_group,
};
use super::format::{
    LightCookiePointSource, LightCookieSource, LightCookieSourceChannel, LightCookieSourceSampling,
    light_cookie_wrap_bits, select_light_cookie_atlas_format, source_channel_for_host_format,
    source_sampling_for_limits,
};
use super::packing::{
    LightCookieAtlasRect, LightCookiePackItem, LightCookiePackPlan, LightCookiePackedRect,
    pack_light_cookie_rects,
};

/// Render-pass blit rectangles for one packed atlas.
#[derive(Clone, Debug, Default)]
struct LightCookieAtlasBlitPlan {
    /// Rectangles keyed by light-cookie metadata row.
    rects: HashMap<u32, LightCookieAtlasRect>,
}

impl LightCookieAtlasBlitPlan {
    /// Returns the packed atlas rectangle for a metadata row.
    fn rect(&self, rect_index: u32) -> Option<LightCookieAtlasRect> {
        self.rects.get(&rect_index).copied()
    }
}

/// Current packed atlas state consumed by the atlas render pass.
#[derive(Clone, Debug, Default)]
struct LightCookieFramePlan {
    /// 2D cookie atlas blit rectangles.
    two_d: LightCookieAtlasBlitPlan,
    /// Point-cookie face atlas blit rectangles.
    point: LightCookieAtlasBlitPlan,
}

/// GPU-visible identity of one active point-cookie cubemap.
///
/// The cubemap allocation identifies all six face views. Content generation catches in-place
/// uploads that leave those views unchanged.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PointCookieSourceSignature {
    pub(super) packed_id: i32,
    pub(super) asset_id: i32,
    pub(super) layer: u32,
    pub(super) allocation_generation: u64,
    pub(super) content_generation: u64,
    pub(super) mip_levels_resident: u32,
    pub(super) size: u32,
    pub(super) channel: LightCookieSourceChannel,
    pub(super) sampling: LightCookieSourceSampling,
}

/// Exact point-atlas contents produced by one successful encode.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PointCookieAtlasSignature {
    extent: super::atlas::LightCookieAtlasExtent,
    rects: Vec<LightCookiePackedRect>,
    sources: Vec<PointCookieSourceSignature>,
}

impl PointCookieAtlasSignature {
    /// Canonicalizes unordered request snapshots so harmless traversal-order changes remain hits.
    pub(super) fn new(
        extent: super::atlas::LightCookieAtlasExtent,
        mut rects: Vec<LightCookiePackedRect>,
        mut sources: Vec<PointCookieSourceSignature>,
    ) -> Option<Self> {
        if rects.is_empty() {
            return None;
        }
        rects.sort_by_key(|packed| {
            (
                packed.rect_index,
                packed.rect.x,
                packed.rect.y,
                packed.rect.width,
                packed.rect.height,
            )
        });
        sources.sort_by_key(|source| (source.layer, source.packed_id, source.asset_id));
        Some(Self {
            extent,
            rects,
            sources,
        })
    }

    fn contains_source_layer(&self, layer: u32) -> bool {
        self.sources.iter().any(|source| source.layer == layer)
    }
}

/// Pending/committed atlas state used by graph pass admission.
#[derive(Default)]
pub(super) struct LightCookieEncodeState {
    has_2d_requests: bool,
    pending_point: Option<PointCookieAtlasSignature>,
    last_encoded_point: Option<PointCookieAtlasSignature>,
    point_needs_encode: bool,
}

impl LightCookieEncodeState {
    pub(super) fn update(
        &mut self,
        has_2d_requests: bool,
        pending_point: Option<PointCookieAtlasSignature>,
    ) {
        self.has_2d_requests = has_2d_requests;
        self.point_needs_encode =
            pending_point.is_some() && pending_point.as_ref() != self.last_encoded_point.as_ref();
        self.pending_point = pending_point;
    }

    pub(super) fn should_record(&self) -> bool {
        self.has_2d_requests || self.point_needs_encode
    }

    /// Commits only the exact signature that was read before encoding began.
    pub(super) fn commit_point(&mut self, encoded: &PointCookieAtlasSignature) {
        if self.pending_point.as_ref() != Some(encoded) {
            return;
        }
        self.last_encoded_point = Some(encoded.clone());
        self.point_needs_encode = false;
    }
}

/// Source binding retained in one fixed atlas metadata slot.
///
/// Assignment layers are bounded and persistent, so keying by the destination metadata row keeps
/// this cache bounded as cookie asset ids churn. Reassignment replaces the row only when the
/// source view or its filterability class actually changes.
struct CachedLightCookieSourceBindGroup {
    source_view: wgpu::TextureView,
    sampling: LightCookieSourceSampling,
    bind_group: wgpu::BindGroup,
}

pub(in crate::backend::frame_gpu) struct LightCookieAtlasResources {
    /// 2D cookie atlas shared by spot and directional lights.
    two_d: LightCookiePackedAtlas,
    /// Point-light cubemap-face cookie atlas.
    point: LightCookiePackedAtlas,
    /// Normalized atlas rectangles indexed by packed light cookie rows.
    metadata_buffer: Arc<wgpu::Buffer>,
    /// Monotonic version for atlas view changes.
    version: u64,
    /// Sampler used by material lighting shaders.
    sampler: Arc<wgpu::Sampler>,
    /// Source-to-atlas blit pipelines.
    blit: LightCookieBlitPipelines,
    /// Assignment state for current and recent frames.
    state: Mutex<LightCookieAtlasState>,
    /// Packed blit rectangles for the current frame-global atlas pass.
    plan: Mutex<LightCookieFramePlan>,
    /// Dirty state for the point atlas and unconditional 2D admission.
    encode_state: Mutex<LightCookieEncodeState>,
    /// Source bind groups keyed by their fixed atlas metadata row.
    source_bind_groups: Mutex<HashMap<u32, CachedLightCookieSourceBindGroup>>,
    /// GPU limits used for source-format validation and atlas sizing.
    limits: Arc<GpuLimits>,
    /// One-shot guard for 2D-cookie pack overflow.
    two_d_pack_overflow_logged: bool,
    /// One-shot guard for point-cookie pack overflow.
    point_pack_overflow_logged: bool,
}

impl LightCookieAtlasResources {
    /// Creates frame-global light-cookie atlas resources.
    pub(in crate::backend::frame_gpu) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        limits: Arc<GpuLimits>,
    ) -> Self {
        let atlas_format = select_light_cookie_atlas_format(&limits);
        let two_d =
            LightCookiePackedAtlas::new(device, queue, "frame_light_cookie_2d_atlas", atlas_format);
        let point = LightCookiePackedAtlas::new(
            device,
            queue,
            "frame_light_cookie_point_atlas",
            atlas_format,
        );
        let metadata_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame_light_cookie_rects"),
            size: (LIGHT_COOKIE_RECT_CAPACITY * size_of::<GpuLightCookieRect>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        crate::profiling::note_resource_churn!(Buffer, "backend::light_cookie_rects");
        let metadata = fallback_metadata_rows();
        queue.write_buffer(metadata_buffer.as_ref(), 0, bytemuck::cast_slice(&metadata));
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
            metadata_buffer,
            version: 0,
            sampler,
            blit,
            state: Mutex::new(LightCookieAtlasState::new()),
            plan: Mutex::new(LightCookieFramePlan::default()),
            encode_state: Mutex::new(LightCookieEncodeState::default()),
            source_bind_groups: Mutex::new(HashMap::new()),
            limits,
            two_d_pack_overflow_logged: false,
            point_pack_overflow_logged: false,
        }
    }

    /// Full 2D cookie atlas view for group-0 binding.
    pub(in crate::backend::frame_gpu) fn two_d_view(&self) -> &wgpu::TextureView {
        self.two_d.view()
    }

    /// Full point-cookie atlas view for group-0 binding.
    pub(in crate::backend::frame_gpu) fn point_view(&self) -> &wgpu::TextureView {
        self.point.view()
    }

    /// Cookie rect metadata for group-0 binding.
    pub(in crate::backend::frame_gpu) fn metadata_buffer(&self) -> &wgpu::Buffer {
        self.metadata_buffer.as_ref()
    }

    /// Cookie sampler for group-0 binding.
    pub(in crate::backend::frame_gpu) fn sampler(&self) -> &wgpu::Sampler {
        self.sampler.as_ref()
    }

    /// Current atlas bind-resource version.
    pub(in crate::backend::frame_gpu) fn version(&self) -> u64 {
        self.version
    }

    /// Retains atlas handles and sampler until driver submit.
    pub(in crate::backend::frame_gpu) fn retain_submit_resources(
        &self,
        resources: &mut crate::gpu::GpuRetainedResources,
    ) {
        self.two_d.retain_submit_resources(resources);
        self.point.retain_submit_resources(resources);
        resources.retain_buffer(self.metadata_buffer.as_ref().clone());
        resources.retain_sampler(self.sampler.as_ref().clone());
    }

    /// Starts a new light-cookie assignment frame.
    pub(in crate::backend::frame_gpu) fn begin_frame(&self) {
        self.state.lock().begin_frame();
    }

    /// Assigns a cookie atlas binding for one resolved light.
    pub(in crate::backend::frame_gpu) fn assign(
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
        self.state.lock().assign(assignment)
    }

    /// Returns whether a frame-global atlas update pass has work.
    pub(in crate::backend::frame_gpu) fn has_requests(&self) -> bool {
        self.encode_state.lock().should_record()
    }

    /// Synchronizes packed atlas textures and uploads current rect metadata before graph recording.
    pub(in crate::backend::frame_gpu) fn sync(
        &mut self,
        device: &wgpu::Device,
        uploads: GraphUploadSink<'_>,
        assets: &dyn GraphAssetResources,
    ) -> bool {
        profiling::scope!("light_cookies::sync_atlas");
        let (two_d_requests, point_requests) = {
            let state = self.state.lock();
            if !state.has_requests() {
                drop(state);
                *self.plan.lock() = LightCookieFramePlan::default();
                self.encode_state.lock().update(false, None);
                self.source_bind_groups.lock().clear();
                return false;
            }
            state.requests()
        };

        let max_extent = self.limits.max_texture_dimension_2d().max(1);
        let two_d_pack = self.pack_2d_requests(assets, &two_d_requests, max_extent);
        let point_pack = self.pack_point_requests(assets, &point_requests, max_extent);
        retain_active_source_bind_group_rows(
            &mut self.source_bind_groups.lock(),
            two_d_pack
                .rects
                .iter()
                .chain(&point_pack.rects)
                .map(|packed| packed.rect_index),
        );
        self.log_pack_overflow(two_d_pack.overflow_count, point_pack.overflow_count);

        let two_d_changed = self.two_d.sync(device, two_d_pack.extent);
        let point_changed = self.point.sync(device, point_pack.extent);
        let point_signature =
            self.point_atlas_signature(assets, &point_requests, &point_pack, self.point.extent());
        let (frame_plan, metadata) = build_frame_plan(
            &two_d_pack,
            self.two_d.extent(),
            &point_pack,
            self.point.extent(),
        );
        uploads.write_buffer(
            self.metadata_buffer.as_ref(),
            0,
            bytemuck::cast_slice(&metadata),
        );
        *self.plan.lock() = frame_plan;
        self.encode_state
            .lock()
            .update(!two_d_requests.is_empty(), point_signature);

        let changed = two_d_changed || point_changed;
        if changed {
            self.version = self.version.wrapping_add(1);
        }
        changed
    }

    /// Records all current-frame atlas clears and source blits.
    pub(in crate::backend::frame_gpu) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        assets: &dyn GraphAssetResources,
        profiler: Option<&crate::profiling::GpuProfilerHandle>,
    ) {
        profiling::scope!("light_cookies::encode_atlas");
        let (two_d_requests, point_requests) = self.state.lock().requests();
        let plan = self.plan.lock().clone();
        let pending_point = {
            let state = self.encode_state.lock();
            if state.point_needs_encode {
                state.pending_point.clone()
            } else {
                None
            }
        };

        if !two_d_requests.is_empty() {
            clear_cookie_atlas(
                encoder,
                self.two_d.view(),
                "light_cookie_2d_clear",
                profiler,
            );
            for request in two_d_requests {
                self.encode_2d_request(device, encoder, assets, profiler, request, &plan.two_d);
            }
        }

        if let Some(point_signature) = pending_point {
            clear_cookie_atlas(
                encoder,
                self.point.view(),
                "light_cookie_point_clear",
                profiler,
            );
            let mut encoded_sources = 0usize;
            for request in point_requests {
                if !point_signature.contains_source_layer(request.layer) {
                    continue;
                }
                if self.encode_point_request(
                    device,
                    encoder,
                    assets,
                    profiler,
                    request,
                    &plan.point,
                ) {
                    encoded_sources += 1;
                }
            }
            if encoded_sources == point_signature.sources.len() {
                self.encode_state.lock().commit_point(&point_signature);
            }
        }
    }

    /// Records one 2D cookie atlas update.
    fn encode_2d_request(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        assets: &dyn GraphAssetResources,
        profiler: Option<&crate::profiling::GpuProfilerHandle>,
        request: LightCookieRequest,
        plan: &LightCookieAtlasBlitPlan,
    ) {
        let Some(rect) = plan.rect(request.layer) else {
            return;
        };
        let Some(source) = self.resolve_2d_source(assets, request) else {
            return;
        };
        let (bind_group, created) = self.source_bind_group(device, request.layer, source);
        if created {
            crate::profiling::note_resource_churn!(BindGroup, "backend::light_cookie_2d_source_bg");
        }
        blit_cookie_rect(
            encoder,
            self.two_d.view(),
            rect,
            "light_cookie_2d_blit",
            self.blit.pipeline(source.channel, source.sampling),
            &bind_group,
            profiler,
        );
    }

    /// Records one point-light cubemap cookie atlas update.
    fn encode_point_request(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        assets: &dyn GraphAssetResources,
        profiler: Option<&crate::profiling::GpuProfilerHandle>,
        request: LightCookieRequest,
        plan: &LightCookieAtlasBlitPlan,
    ) -> bool {
        let Some(source) = self.resolve_point_source(assets, request) else {
            return false;
        };
        for face in 0..POINT_COOKIE_FACE_COUNT {
            let Some(rect) = plan.rect(request.layer + face) else {
                continue;
            };
            let face_source = LightCookieSource {
                view: source.cubemap.face_views[face as usize].as_ref(),
                channel: source.channel,
                sampling: source.sampling,
            };
            let (bind_group, created) =
                self.source_bind_group(device, request.layer + face, face_source);
            if created {
                crate::profiling::note_resource_churn!(
                    BindGroup,
                    "backend::light_cookie_point_source_bg"
                );
            }
            blit_cookie_rect(
                encoder,
                self.point.view(),
                rect,
                "light_cookie_point_blit",
                self.blit
                    .pipeline(face_source.channel, face_source.sampling),
                &bind_group,
                profiler,
            );
        }
        true
    }

    /// Returns the persistent source binding for one bounded atlas row.
    fn source_bind_group(
        &self,
        device: &wgpu::Device,
        rect_index: u32,
        source: LightCookieSource<'_>,
    ) -> (wgpu::BindGroup, bool) {
        let cached_bind_group = self
            .source_bind_groups
            .lock()
            .get(&rect_index)
            .filter(|cached| {
                &cached.source_view == source.view && cached.sampling == source.sampling
            })
            .map(|cached| cached.bind_group.clone());
        if let Some(bind_group) = cached_bind_group {
            return (bind_group, false);
        }
        let bind_group = create_source_bind_group(device, &self.blit, source, self.sampler());
        self.source_bind_groups.lock().insert(
            rect_index,
            CachedLightCookieSourceBindGroup {
                source_view: source.view.clone(),
                sampling: source.sampling,
                bind_group: bind_group.clone(),
            },
        );
        (bind_group, true)
    }

    /// Packs current 2D-cookie requests using resident source dimensions.
    fn pack_2d_requests(
        &self,
        assets: &dyn GraphAssetResources,
        requests: &[LightCookieRequest],
        max_extent: u32,
    ) -> LightCookiePackPlan {
        let items = requests
            .iter()
            .filter_map(|request| {
                let (width, height) = self.resolve_2d_source_dimensions(assets, *request)?;
                Some(LightCookiePackItem {
                    rect_index: request.layer,
                    width,
                    height,
                })
            })
            .collect::<Vec<_>>();
        pack_light_cookie_rects(&items, max_extent)
    }

    /// Packs current point-cookie face requests using resident cubemap dimensions.
    fn pack_point_requests(
        &self,
        assets: &dyn GraphAssetResources,
        requests: &[LightCookieRequest],
        max_extent: u32,
    ) -> LightCookiePackPlan {
        let mut items = Vec::with_capacity(requests.len() * POINT_COOKIE_FACE_COUNT as usize);
        for request in requests {
            let Some((width, height)) = self.resolve_point_source_dimensions(assets, *request)
            else {
                continue;
            };
            for face in 0..POINT_COOKIE_FACE_COUNT {
                items.push(LightCookiePackItem {
                    rect_index: request.layer + face,
                    width,
                    height,
                });
            }
        }
        pack_light_cookie_rects(&items, max_extent)
    }

    /// Builds the exact static-source identity for the point atlas produced by `pack`.
    fn point_atlas_signature(
        &self,
        assets: &dyn GraphAssetResources,
        requests: &[LightCookieRequest],
        pack: &LightCookiePackPlan,
        extent: super::atlas::LightCookieAtlasExtent,
    ) -> Option<PointCookieAtlasSignature> {
        let mut sources = Vec::with_capacity(requests.len());
        for request in requests {
            let face_end = request.layer.saturating_add(POINT_COOKIE_FACE_COUNT);
            if !pack
                .rects
                .iter()
                .any(|packed| packed.rect_index >= request.layer && packed.rect_index < face_end)
            {
                continue;
            }
            let Some(source) = self.resolve_point_source(assets, *request) else {
                continue;
            };
            sources.push(PointCookieSourceSignature {
                packed_id: request.packed_id,
                asset_id: request.asset_id,
                layer: request.layer,
                allocation_generation: source.cubemap.allocation_generation,
                content_generation: source.cubemap.content_generation,
                mip_levels_resident: source.cubemap.mip_levels_resident,
                size: source.cubemap.size,
                channel: source.channel,
                sampling: source.sampling,
            });
        }
        PointCookieAtlasSignature::new(extent, pack.rects.clone(), sources)
    }

    /// Logs one warning for cookie sources that exceed the packed atlas budget.
    fn log_pack_overflow(&mut self, two_d_overflow_count: usize, point_overflow_count: usize) {
        if two_d_overflow_count > 0 && !self.two_d_pack_overflow_logged {
            logger::warn!(
                "light-cookie 2D atlas cannot fit {two_d_overflow_count} active cookies; overflowing cookies use the white fallback"
            );
            self.two_d_pack_overflow_logged = true;
        }
        if point_overflow_count > 0 && !self.point_pack_overflow_logged {
            logger::warn!(
                "light-cookie point atlas cannot fit {point_overflow_count} active cubemap faces; overflowing faces use the white fallback"
            );
            self.point_pack_overflow_logged = true;
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

    /// Resolves the resident source extent for a 2D cookie.
    fn resolve_2d_source_dimensions(
        &self,
        assets: &dyn GraphAssetResources,
        request: LightCookieRequest,
    ) -> Option<(u32, u32)> {
        match request.kind {
            HostTextureAssetKind::Texture2D => {
                let texture = assets.texture_pool().get(request.asset_id)?;
                if texture.mip_levels_resident == 0 {
                    return None;
                }
                self.source_sampling(texture.wgpu_format)?;
                Some((texture.width.max(1), texture.height.max(1)))
            }
            HostTextureAssetKind::RenderTexture => {
                let texture = assets.render_texture_pool().get(request.asset_id)?;
                if !texture.is_sampleable() {
                    return None;
                }
                self.source_sampling(texture.wgpu_color_format)?;
                Some((texture.width.max(1), texture.height.max(1)))
            }
            HostTextureAssetKind::VideoTexture => {
                let texture = assets.video_texture_pool().get(request.asset_id)?;
                if !texture.is_sampleable() {
                    return None;
                }
                Some((texture.width.max(1), texture.height.max(1)))
            }
            HostTextureAssetKind::Texture3D
            | HostTextureAssetKind::Cubemap
            | HostTextureAssetKind::Desktop => None,
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

    /// Resolves the resident source extent for a point-light cubemap cookie face.
    fn resolve_point_source_dimensions(
        &self,
        assets: &dyn GraphAssetResources,
        request: LightCookieRequest,
    ) -> Option<(u32, u32)> {
        if request.kind != HostTextureAssetKind::Cubemap {
            return None;
        }
        let cubemap = assets.cubemap_pool().get(request.asset_id)?;
        if cubemap.mip_levels_resident == 0 {
            return None;
        }
        self.source_sampling(cubemap.wgpu_format)?;
        Some((cubemap.size.max(1), cubemap.size.max(1)))
    }

    /// Returns the source sampling mode supported by `format`.
    fn source_sampling(&self, format: wgpu::TextureFormat) -> Option<LightCookieSourceSampling> {
        source_sampling_for_limits(&self.limits, format)
    }
}

/// Drops bindings for atlas rows that are not backed by a currently packed resident source.
///
/// The metadata table is bounded, but retaining one old source view per row can otherwise pin
/// large texture allocations after lights stop referencing them. Packed rows are the exact set
/// that can be encoded this frame, so pruning them here preserves every useful cache hit.
pub(super) fn retain_active_source_bind_group_rows<T>(
    cache: &mut HashMap<u32, T>,
    active_rows: impl IntoIterator<Item = u32>,
) {
    let mut active = [false; LIGHT_COOKIE_RECT_CAPACITY];
    for row in active_rows {
        if let Some(slot) = active.get_mut(row as usize) {
            *slot = true;
        }
    }
    cache.retain(|row, _| active.get(*row as usize).copied().unwrap_or(false));
}

/// Builds frame-pass blit rectangles and shader metadata from packed atlas plans.
fn build_frame_plan(
    two_d_pack: &LightCookiePackPlan,
    two_d_extent: super::atlas::LightCookieAtlasExtent,
    point_pack: &LightCookiePackPlan,
    point_extent: super::atlas::LightCookieAtlasExtent,
) -> (LightCookieFramePlan, Vec<GpuLightCookieRect>) {
    let mut metadata = fallback_metadata_rows();
    let mut two_d = LightCookieAtlasBlitPlan::default();
    let mut point = LightCookieAtlasBlitPlan::default();

    for packed in &two_d_pack.rects {
        if let Some(row) = metadata.get_mut(packed.rect_index as usize) {
            *row = packed.rect.metadata(two_d_extent);
            two_d.rects.insert(packed.rect_index, packed.rect);
        }
    }
    for packed in &point_pack.rects {
        if let Some(row) = metadata.get_mut(packed.rect_index as usize) {
            *row = packed.rect.metadata(point_extent);
            point.rects.insert(packed.rect_index, packed.rect);
        }
    }

    (LightCookieFramePlan { two_d, point }, metadata)
}

/// Builds fallback rect metadata for all possible cookie rows.
fn fallback_metadata_rows() -> Vec<GpuLightCookieRect> {
    vec![GpuLightCookieRect::default(); LIGHT_COOKIE_RECT_CAPACITY]
}
