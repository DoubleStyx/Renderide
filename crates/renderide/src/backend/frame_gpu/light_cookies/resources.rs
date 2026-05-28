use std::sync::Arc;

use parking_lot::Mutex;

use crate::assets::texture::{HostTextureAssetKind, unpack_host_texture_packed};
use crate::backend::light_gpu::LightCookieBinding;
use crate::gpu::GpuLimits;
use crate::render_graph::GraphAssetResources;
use crate::shared::LightType;

use super::POINT_COOKIE_FACE_COUNT;
use super::assignment::{LightCookieAssignment, LightCookieAtlasState, LightCookieRequest};
use super::atlas::LightCookieLayeredAtlas;
use super::blit::{
    LightCookieBlitPipelines, blit_cookie_layer, clear_cookie_layer, create_source_bind_group,
};
use super::format::{
    LightCookiePointSource, LightCookieSource, LightCookieSourceChannel, LightCookieSourceSampling,
    light_cookie_wrap_bits, select_light_cookie_atlas_format, source_channel_for_host_format,
    source_sampling_for_limits,
};

/// Maximum 2D cookie layers including the fallback layer.
const COOKIE_2D_LAYER_CAP: u32 = 64;
/// Maximum resident point-light cookie cubemaps.
const POINT_COOKIE_CUBEMAP_CAP: u32 = 16;

pub(in crate::backend::frame_gpu) struct LightCookieAtlasResources {
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
    pub(in crate::backend::frame_gpu) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        limits: Arc<GpuLimits>,
    ) -> Self {
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
    pub(in crate::backend::frame_gpu) fn two_d_view(&self) -> &wgpu::TextureView {
        self.two_d.view.as_ref()
    }

    /// Full point-cookie atlas view for group-0 binding.
    pub(in crate::backend::frame_gpu) fn point_view(&self) -> &wgpu::TextureView {
        self.point.view.as_ref()
    }

    /// Cookie sampler for group-0 binding.
    pub(in crate::backend::frame_gpu) fn sampler(&self) -> &wgpu::Sampler {
        self.sampler.as_ref()
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
        self.state
            .lock()
            .assign(assignment, self.two_d.layers, self.point.layers)
    }

    /// Returns whether a frame-global atlas update pass has work.
    pub(in crate::backend::frame_gpu) fn has_requests(&self) -> bool {
        self.state.lock().has_requests()
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
        for request in two_d_requests {
            self.encode_2d_request(device, encoder, assets, profiler, request);
        }
        for request in point_requests {
            self.encode_point_request(device, encoder, assets, profiler, request);
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
    ) {
        let Some(target) = self.two_d.layer_view(request.layer) else {
            return;
        };
        let Some(source) = self.resolve_2d_source(assets, request) else {
            clear_cookie_layer(encoder, target, "light_cookie_2d_clear", profiler);
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
    ) {
        let Some(source) = self.resolve_point_source(assets, request) else {
            for face in 0..POINT_COOKIE_FACE_COUNT {
                if let Some(target) = self.point.layer_view(request.layer + face) {
                    clear_cookie_layer(encoder, target, "light_cookie_point_clear", profiler);
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
                profiler,
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
