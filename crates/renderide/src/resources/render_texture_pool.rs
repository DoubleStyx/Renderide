//! GPU render targets for host [`crate::shared::SetRenderTextureFormat`] (Unity `RenderTexture` assets).
//!
//! Color textures use `RENDER_ATTACHMENT | TEXTURE_BINDING` so the same asset can be sampled from
//! materials after the offscreen pass. Depth buffers are separate textures when `depth > 0`; depth
//! also includes `COPY_SRC` so [`crate::backend::frame_gpu::FrameGpuResources::copy_scene_depth_snapshot`]
//! can copy scene depth for intersection / frame bindings (same as main `renderide-depth`).
//!
//! ### Known orientation mismatch
//!
//! Host-uploaded textures arrive bottom-up (Resonite/Unity convention); material shaders apply a
//! V-flip via [`crate::shaders::source::modules::uv_utils`]'s `apply_st` / `flip_v` to compensate
//! for wgpu's top-down `textureSample` origin. Render textures produced by a camera pass into the
//! `color_texture` here are **top-down** (wgpu render attachments), so a material that samples a
//! [`GpuRenderTexture`] receives a vertically-inverted result through the same `apply_st` call.
//!
//! This is a pre-existing limitation of the engine's "store bottom-up + shader-side V-flip"
//! convention; it surfaces only for materials that sample render textures (secondary cameras,
//! reflection probes feeding `Reflection`, etc.). A correct fix requires either rendering with
//! Y-inverted clip space (and reversing cull winding) or plumbing a per-binding `skip_v_flip`
//! flag from [`crate::backend::embedded::material_bind`] into every material shader's
//! `apply_st` site. Tracked as future work; intentionally not addressed in the BC6H/BC7
//! flip-fix PR to keep that change small and contained.

use hashbrown::HashMap;
use std::sync::Arc;

use crate::gpu::GpuLimits;
use crate::resources::budget::{VramAccounting, VramResourceKind};
use crate::resources::{GpuResource, Texture2dSamplerState};
use crate::shared::SetRenderTextureFormat;

/// Host render texture mirrored as a wgpu color target + optional depth.
#[derive(Debug)]
pub struct GpuRenderTexture {
    /// Host render-texture asset id.
    pub asset_id: i32,
    /// Color target (`Rgba16Float`); sampleable after offscreen draws.
    pub color_texture: Arc<wgpu::Texture>,
    /// Default view over the full color mip.
    pub color_view: Arc<wgpu::TextureView>,
    /// Optional depth texture (always allocated for scene draws in [`Self::new_from_format`]).
    pub depth_texture: Option<Arc<wgpu::Texture>>,
    /// View over `depth_texture` when present.
    pub depth_view: Option<Arc<wgpu::TextureView>>,
    /// wgpu format of `color_texture`.
    pub wgpu_color_format: wgpu::TextureFormat,
    /// Pixel width of the render target.
    pub width: u32,
    /// Pixel height of the render target.
    pub height: u32,
    /// Estimated VRAM for color + depth (see [`estimate_texture_bytes`]).
    pub resident_bytes: u64,
    /// Sampler state mirrored from host format for material binds.
    pub sampler: Texture2dSamplerState,
}

impl GpuResource for GpuRenderTexture {
    fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    fn asset_id(&self) -> i32 {
        self.asset_id
    }
}

impl GpuRenderTexture {
    /// Creates GPU storage for a host [`SetRenderTextureFormat`].
    ///
    /// Color format: **`Rgba16Float`** when `hdr_color` (Unity `ARGBHalf` / HDR parity), else
    /// **`Rgba8Unorm`** for lower VRAM on typical LDR targets. Depth is always [`Depth32Float`].
    /// Size is clamped per edge via [`GpuLimits::clamp_render_texture_edge`] (Unity-style bounds).
    pub fn new_from_format(
        device: &wgpu::Device,
        limits: &GpuLimits,
        fmt: &SetRenderTextureFormat,
        hdr_color: bool,
    ) -> Option<Self> {
        let w = limits.clamp_render_texture_edge(fmt.size.x);
        let h = limits.clamp_render_texture_edge(fmt.size.y);
        if w == 0 || h == 0 {
            return None;
        }
        let max_dim = limits.max_texture_dimension_2d();
        if w > max_dim || h > max_dim {
            logger::warn!(
                "render texture {}: size {}×{} exceeds max_texture_dimension_2d ({max_dim})",
                fmt.asset_id,
                w,
                h
            );
            return None;
        }

        let wgpu_color_format = if hdr_color {
            wgpu::TextureFormat::Rgba16Float
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };
        let size = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };

        let color_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("RenderTexture {}", fmt.asset_id)),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_color_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }));
        let color_view =
            Arc::new(color_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        // Host `depth` is Unity depth-stencil bits; when zero the asset may still be used as a full
        // scene target — we always allocate a depth attachment so the forward pass can run.
        // `TEXTURE_BINDING` is required so Hi-Z build can bind the depth view for mip0 (`hi_z_mip0_d_bg`).
        let depth_format =
            crate::render_graph::main_forward_depth_stencil_format(device.features());
        let dt = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("RenderTextureDepth {}", fmt.asset_id)),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: depth_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }));
        let dv = Arc::new(dt.create_view(&wgpu::TextureViewDescriptor::default()));
        let depth_texture = Some(dt);
        let depth_view = Some(dv);

        let color_bytes = estimate_texture_bytes(wgpu_color_format, w, h, 1);
        let depth_bytes = estimate_texture_bytes(depth_format, w, h, 1);
        let resident_bytes = color_bytes.saturating_add(depth_bytes);

        let sampler = Texture2dSamplerState {
            filter_mode: fmt.filter_mode,
            aniso_level: fmt.aniso_level.max(0),
            wrap_u: fmt.wrap_u,
            wrap_v: fmt.wrap_v,
            mipmap_bias: 0.0,
        };

        Some(Self {
            asset_id: fmt.asset_id,
            color_texture,
            color_view,
            depth_texture,
            depth_view,
            wgpu_color_format,
            width: w,
            height: h,
            resident_bytes,
            sampler,
        })
    }

    /// `true` when the color target exists and can be sampled (always after successful creation).
    #[inline]
    pub fn is_sampleable(&self) -> bool {
        true
    }
}

fn estimate_texture_bytes(format: wgpu::TextureFormat, width: u32, height: u32, mips: u32) -> u64 {
    let bpp = match format {
        wgpu::TextureFormat::Rgba16Float => 8u64,
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => 4u64,
        wgpu::TextureFormat::Depth32Float => 4u64,
        wgpu::TextureFormat::Depth24PlusStencil8 => 4u64,
        wgpu::TextureFormat::Depth32FloatStencil8 => 8u64,
        _ => 4u64,
    };
    let mut total = 0u64;
    let mut w = width as u64;
    let mut h = height as u64;
    for _ in 0..mips {
        total = total.saturating_add(w.saturating_mul(h).saturating_mul(bpp));
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }
    total
}

/// Pool of [`GpuRenderTexture`] entries keyed by host asset id (per-type id; disambiguate with packed texture type in materials).
#[derive(Debug)]
pub struct RenderTexturePool {
    textures: HashMap<i32, GpuRenderTexture>,
    accounting: VramAccounting,
}

impl Default for RenderTexturePool {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderTexturePool {
    /// Empty pool.
    pub fn new() -> Self {
        Self {
            textures: HashMap::new(),
            accounting: VramAccounting::default(),
        }
    }

    /// VRAM accounting for resident render textures.
    pub fn accounting(&self) -> &VramAccounting {
        &self.accounting
    }

    /// Inserts or replaces a render texture; returns `true` if a previous entry was replaced.
    pub fn insert_texture(&mut self, tex: GpuRenderTexture) -> bool {
        let id = tex.asset_id;
        let bytes = tex.resident_bytes;
        let existed_before = self.textures.contains_key(&id);
        if let Some(old) = self.textures.insert(id, tex) {
            self.accounting
                .on_resident_removed(VramResourceKind::Texture, old.resident_bytes);
        }
        self.accounting
            .on_resident_added(VramResourceKind::Texture, bytes);
        existed_before
    }

    /// Removes by asset id; returns `true` if present.
    pub fn remove(&mut self, asset_id: i32) -> bool {
        if let Some(old) = self.textures.remove(&asset_id) {
            self.accounting
                .on_resident_removed(VramResourceKind::Texture, old.resident_bytes);
            return true;
        }
        false
    }

    /// Borrows a resident render texture by host asset id.
    #[inline]
    pub fn get(&self, asset_id: i32) -> Option<&GpuRenderTexture> {
        self.textures.get(&asset_id)
    }

    /// Mutably borrows a resident render texture for in-place updates.
    #[inline]
    pub fn get_mut(&mut self, asset_id: i32) -> Option<&mut GpuRenderTexture> {
        self.textures.get_mut(&asset_id)
    }

    /// Full map for diagnostics and iteration.
    #[inline]
    pub fn textures(&self) -> &HashMap<i32, GpuRenderTexture> {
        &self.textures
    }

    /// Number of host render-texture assets currently resident on the GPU.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.textures.len()
    }

    /// Whether the pool has no render textures.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.textures.is_empty()
    }
}
