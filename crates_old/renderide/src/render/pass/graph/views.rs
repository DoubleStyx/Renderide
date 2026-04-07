//! Slot-to-texture wiring and per-pass [`RenderTargetViews`].

use crate::render::target::RenderTarget;

use super::resources::{PassResources, ResourceSlot};

/// Maps resource slots to texture views for pass execution.
///
/// Built from target, MRT views, and depth. Slots may be `None` when RTAO is disabled.
pub(super) struct SlotMap<'a> {
    color: Option<&'a wgpu::TextureView>,
    position: Option<&'a wgpu::TextureView>,
    normal: Option<&'a wgpu::TextureView>,
    ao_raw: Option<&'a wgpu::TextureView>,
    ao: Option<&'a wgpu::TextureView>,
    surface: &'a wgpu::TextureView,
    depth: Option<&'a wgpu::TextureView>,
}

/// MRT (Multiple Render Target) views for RTAO pass.
///
/// When RTAO is enabled, the mesh pass renders to these instead of the surface.
pub struct MrtViews<'a> {
    /// Color attachment view (matches surface format for copy-back).
    pub color_view: &'a wgpu::TextureView,
    /// Color texture for copy to surface (same as color_view's texture).
    pub color_texture: &'a wgpu::Texture,
    /// Position G-buffer view (Rgba16Float, camera-relative `world - view_position`).
    pub position_view: &'a wgpu::TextureView,
    /// Position G-buffer texture (for [`wgpu::CommandEncoder::transition_resources`]).
    pub position_texture: &'a wgpu::Texture,
    /// Normal G-buffer view (Rgba16Float).
    pub normal_view: &'a wgpu::TextureView,
    /// Normal G-buffer texture.
    pub normal_texture: &'a wgpu::Texture,
    /// Raw AO view (Rgba8Unorm). Written by RTAO compute, read by blur pass.
    pub ao_raw_view: &'a wgpu::TextureView,
    /// Raw AO texture.
    pub ao_raw_texture: &'a wgpu::Texture,
    /// AO output view (Rgba8Unorm). Written by blur pass, read by composite.
    pub ao_view: &'a wgpu::TextureView,
    /// Blurred AO texture.
    pub ao_texture: &'a wgpu::Texture,
}

/// Builds a slot-to-view map from the render target, MRT views, and depth override.
pub(super) fn build_slot_map<'a>(
    target: &'a RenderTarget,
    mrt_views: Option<&'a MrtViews<'a>>,
    depth_view_override: Option<&'a wgpu::TextureView>,
) -> SlotMap<'a> {
    let surface = target.color_view();
    let depth = target.depth_view().or(depth_view_override);
    match mrt_views {
        Some(mrt) => SlotMap {
            color: Some(mrt.color_view),
            position: Some(mrt.position_view),
            normal: Some(mrt.normal_view),
            ao_raw: Some(mrt.ao_raw_view),
            ao: Some(mrt.ao_view),
            surface,
            depth,
        },
        None => SlotMap {
            color: None,
            position: None,
            normal: None,
            ao_raw: None,
            ao: None,
            surface,
            depth,
        },
    }
}

/// Computes [`RenderTargetViews`] for a pass from its resource declarations and the slot map.
pub(super) fn render_target_views_for_pass<'a>(
    slot_map: &SlotMap<'a>,
    resources: Option<&PassResources>,
) -> RenderTargetViews<'a> {
    let uses = |slot: ResourceSlot| {
        resources.is_some_and(|r| r.reads.contains(&slot) || r.writes.contains(&slot))
    };
    let writes = |slot: ResourceSlot| resources.is_some_and(|r| r.writes.contains(&slot));

    let color_view = if writes(ResourceSlot::Surface) {
        slot_map.surface
    } else {
        slot_map.color.unwrap_or(slot_map.surface)
    };

    RenderTargetViews {
        color_view,
        depth_view: if uses(ResourceSlot::Depth) {
            slot_map.depth
        } else {
            None
        },
        mrt_position_view: if uses(ResourceSlot::Position) {
            slot_map.position
        } else {
            None
        },
        mrt_normal_view: if uses(ResourceSlot::Normal) {
            slot_map.normal
        } else {
            None
        },
        mrt_ao_raw_view: if uses(ResourceSlot::AoRaw) {
            slot_map.ao_raw
        } else {
            None
        },
        mrt_ao_view: if uses(ResourceSlot::Ao) {
            slot_map.ao
        } else {
            None
        },
        mrt_color_input_view: if resources.is_some_and(|r| r.reads.contains(&ResourceSlot::Color)) {
            slot_map.color
        } else {
            None
        },
    }
}

/// Target [`wgpu::TextureUses`] when the current pass reads `slot` as input (after a prior write).
///
/// Used by unit tests documenting intended transition semantics if explicit barriers are reintroduced
/// (e.g. for multi-submit batching).
#[cfg(test)]
pub(crate) fn texture_read_target_uses(
    slot: ResourceSlot,
    curr: &PassResources,
) -> Option<wgpu::TextureUses> {
    match slot {
        ResourceSlot::Color
        | ResourceSlot::Position
        | ResourceSlot::Normal
        | ResourceSlot::AoRaw
        | ResourceSlot::Ao => Some(wgpu::TextureUses::RESOURCE),
        ResourceSlot::Depth => {
            if curr.writes.contains(&ResourceSlot::Surface)
                && !curr.writes.contains(&ResourceSlot::Depth)
            {
                Some(wgpu::TextureUses::DEPTH_STENCIL_WRITE)
            } else {
                Some(wgpu::TextureUses::DEPTH_STENCIL_READ)
            }
        }
        ResourceSlot::Surface => Some(wgpu::TextureUses::RESOURCE),
        ResourceSlot::ClusterBuffers | ResourceSlot::LightBuffer => None,
    }
}

/// Color and optional depth texture views for the current render pass.
pub struct RenderTargetViews<'a> {
    /// Color attachment view (output for this pass).
    pub color_view: &'a wgpu::TextureView,
    /// Optional depth attachment view.
    pub depth_view: Option<&'a wgpu::TextureView>,
    /// When RTAO is enabled, position G-buffer view for MRT mesh pass.
    pub mrt_position_view: Option<&'a wgpu::TextureView>,
    /// When RTAO is enabled, normal G-buffer view for MRT mesh pass.
    pub mrt_normal_view: Option<&'a wgpu::TextureView>,
    /// When RTAO is enabled, raw AO texture view (RTAO output, blur input).
    pub mrt_ao_raw_view: Option<&'a wgpu::TextureView>,
    /// When RTAO is enabled, AO texture view for blur output and composite.
    pub mrt_ao_view: Option<&'a wgpu::TextureView>,
    /// When RTAO is enabled, mesh color input for composite pass (MRT color texture).
    pub mrt_color_input_view: Option<&'a wgpu::TextureView>,
}
