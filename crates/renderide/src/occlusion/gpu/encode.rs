//! Hi-Z pyramid compute dispatch and copy-to-staging encoding.
//!
//! This module is the top-level coordinator for one `encode_hi_z_build` call. Per-stage work
//! (mip0 from depth, hierarchical downsample, copy-to-staging) lives in the submodules
//! [`mip0`], [`downsample`], and [`staging_copy`]; each takes a borrowed [`EncodeSession`] plus
//! per-call arguments.

mod downsample;
mod mip0;
mod staging_copy;

use crate::gpu::OutputDepthMode;
use crate::hi_z_cpu::pyramid::{hi_z_pyramid_dimensions, mip_levels_for_extent};
use crate::history_texture::HistoryTextureMipViews;

use self::mip0::DepthBinding;
use super::pipelines::HiZPipelines;
use super::scratch::{HIZ_MAX_MIPS, HiZGpuScratch};
use super::state::HiZGpuState;

/// GPU handles recorded into for one [`encode_hi_z_build`] call.
pub struct HiZBuildRecord<'a> {
    /// Device for pipeline cache and bind group creation.
    pub device: &'a wgpu::Device,
    /// Effective device caps used to validate scratch allocations and dispatches.
    pub limits: &'a crate::gpu::GpuLimits,
    /// Command encoder receiving the mip0, downsample, and staging copy commands.
    pub encoder: &'a mut wgpu::CommandEncoder,
}

/// Registry-owned Hi-Z pyramid selected for this view and ping-pong half.
pub struct HiZHistoryTarget<'a> {
    /// Backing history texture that receives mip writes and is copied to readback staging.
    pub texture: &'a wgpu::Texture,
    /// Per-layer/per-mip texture views for writing the current view's pyramid.
    pub mip_views: &'a HistoryTextureMipViews,
}

/// Which history texture layer the current mip0 + downsample call should target.
///
/// Controls which cache slots [`super::scratch::HiZBindGroupCache`] reuses or rebuilds.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum PyramidSide {
    /// Desktop (non-stereo) or stereo-left pyramid.
    DesktopOrLeft,
    /// Stereo-right layer in a stereo history pyramid.
    Right,
}

/// Per-layer mip chains selected from a registry-owned Hi-Z history texture.
struct HiZHistoryViews<'a> {
    /// Desktop or stereo-left mip chain.
    left: &'a [wgpu::TextureView],
    /// Stereo-right mip chain when the depth target is a two-layer array.
    right: Option<&'a [wgpu::TextureView]>,
}

/// Stable handles shared across every per-stage dispatch inside one [`encode_hi_z_build`] call.
///
/// Per-call differentiators such as `pyramid_views`, `depth_bind`, `side`, `history_texture`,
/// `ws`, `right_eye`, and `layer` are passed as function arguments to each stage; only the
/// invariant handles live here so the stereo loop can re-use the session across both sides.
pub(super) struct EncodeSession<'a> {
    /// Device for on-demand bind-group creation.
    pub(super) device: &'a wgpu::Device,
    /// Active command encoder receiving compute passes and staging copies.
    pub(super) encoder: &'a mut wgpu::CommandEncoder,
    /// Source depth view (sampled in the mip0 pass).
    pub(super) depth_view: &'a wgpu::TextureView,
    /// Scratch buffers and viewports (extent, mip count, uniforms) plus cached bind groups.
    pub(super) scratch: &'a mut HiZGpuScratch,
    /// Compiled Hi-Z pipelines (mip0 desktop/stereo + downsample).
    pub(super) pipes: &'a HiZPipelines,
    /// GPU profiler for per-dispatch pass-level timestamp queries; [`None`] when disabled.
    pub(super) profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
}

/// Records Hi-Z build + copy-to-staging into the state's current readback slot.
///
/// Claims the staging slot at encode time so two consecutive frames can never aim the
/// same buffer even if the prior frame's `on_submitted_work_done` callback has not yet fired.
///
/// The claimed slot is stored as a transient handoff for the main-thread submit path to bake into
/// a [`wgpu::Queue::on_submitted_work_done`] closure, so the slot travels with the closure by value
/// and a late-firing callback cannot consume a newer frame's slot.
///
/// Call [`HiZGpuState::drain_completed_map_async`] at the **start** of the next frame to drain
/// completed maps.
pub fn encode_hi_z_build(
    record: HiZBuildRecord<'_>,
    depth_view: &wgpu::TextureView,
    history: HiZHistoryTarget<'_>,
    extent: (u32, u32),
    mode: OutputDepthMode,
    state: &mut HiZGpuState,
    profiler: Option<&crate::profiling::GpuProfilerHandle>,
) {
    let HiZBuildRecord {
        device,
        limits,
        encoder,
    } = record;
    if !prepare_scratch(device, limits, extent, mode, state) {
        return;
    }

    let ws = state.next_write_slot();
    let Some(scratch) = state.scratch_mut() else {
        return;
    };
    let pipes = HiZPipelines::get(device);
    let Some(history_views) = resolve_history_views(history.mip_views, mode, scratch.mip_levels)
    else {
        return;
    };

    invalidate_caches_for_targets(scratch, depth_view, &history_views);

    let mut session = EncodeSession {
        device,
        encoder,
        depth_view,
        scratch,
        pipes,
        profiler,
    };
    let recorded = match mode {
        OutputDepthMode::DesktopSingle => {
            record_desktop_pyramid(&mut session, &history_views, history.texture, ws)
        }
        OutputDepthMode::StereoArray { .. } => {
            record_stereo_pyramids(&mut session, &history_views, history.texture, ws)
        }
    };

    if !recorded {
        return;
    }
    let claimed_ws = state.claim_encoded_slot();
    debug_assert_eq!(claimed_ws, ws);
}

/// Resets slot validity, invalidates cache, ensures [`HiZGpuScratch`] matches `extent` / stereo layout.
///
/// Returns `false` when encoding must abort (zero extent, missing scratch, or GPU not ready).
fn prepare_scratch(
    device: &wgpu::Device,
    limits: &crate::gpu::GpuLimits,
    extent: (u32, u32),
    mode: OutputDepthMode,
    state: &mut HiZGpuState,
) -> bool {
    state.clear_encoded_slot();
    state.invalidate_if_needed(extent, mode);

    let (full_w, full_h) = extent;
    if full_w == 0 || full_h == 0 {
        return false;
    }

    let (bw, bh) = hi_z_pyramid_dimensions(full_w, full_h);
    if bw == 0 || bh == 0 {
        return false;
    }

    let stereo = matches!(mode, OutputDepthMode::StereoArray { .. });
    let mip_levels = mip_levels_for_extent(bw, bh, HIZ_MAX_MIPS);
    let needs_new = match state.scratch() {
        Some(scratch) => {
            (scratch.extent, scratch.mip_levels) != ((bw, bh), mip_levels)
                || scratch.is_stereo() != stereo
        }
        None => true,
    };
    if needs_new {
        state.replace_scratch(HiZGpuScratch::new(device, limits, (bw, bh), stereo));
        state.set_secondary_readback_enabled(stereo);
    }
    state.set_secondary_readback_enabled(stereo);
    let Some(scratch_ref) = state.scratch() else {
        return false;
    };

    state.can_encode_hi_z(scratch_ref)
}

/// Drops cached bind groups whose source views (depth attachment / pyramid target) have changed.
fn invalidate_caches_for_targets(
    scratch: &mut HiZGpuScratch,
    depth_view: &wgpu::TextureView,
    history_views: &HiZHistoryViews<'_>,
) {
    scratch
        .bind_groups
        .invalidate_mip0_if_depth_changed(depth_view);
    scratch.bind_groups.invalidate_pyramid_if_target_changed(
        &history_views.left[0],
        history_views.right.map(|views| &views[0]),
    );
}

/// Resolves the history texture layer/mip chains required by the current depth mode.
fn resolve_history_views(
    mip_views: &HistoryTextureMipViews,
    mode: OutputDepthMode,
    required_mips: u32,
) -> Option<HiZHistoryViews<'_>> {
    let Some(left) = history_layer_mip_views(mip_views, 0, required_mips) else {
        logger::warn!("hi_z history texture missing layer 0 mip views; skipping encode");
        return None;
    };
    let right = match mode {
        OutputDepthMode::DesktopSingle => None,
        OutputDepthMode::StereoArray { .. } => {
            if let Some(views) = history_layer_mip_views(mip_views, 1, required_mips) {
                Some(views)
            } else {
                logger::warn!(
                    "hi_z stereo history texture missing layer 1 mip views; skipping encode"
                );
                return None;
            }
        }
    };
    Some(HiZHistoryViews { left, right })
}

/// Returns the mip view chain for `layer` when it covers every mip the current encode needs.
fn history_layer_mip_views(
    mip_views: &HistoryTextureMipViews,
    layer: u32,
    required_mips: u32,
) -> Option<&[wgpu::TextureView]> {
    let views = mip_views.layer_mip_views(layer)?;
    if views.len() < required_mips as usize {
        return None;
    }
    Some(&views[..required_mips as usize])
}

fn record_desktop_pyramid(
    session: &mut EncodeSession<'_>,
    history_views: &HiZHistoryViews<'_>,
    history_texture: &wgpu::Texture,
    ws: usize,
) -> bool {
    record_pyramid_side(
        session,
        history_views.left,
        DepthBinding::D2,
        PyramidSide::DesktopOrLeft,
    );
    staging_copy::copy_layer(session, history_texture, ws, 0, false);
    true
}

fn record_stereo_pyramids(
    session: &mut EncodeSession<'_>,
    history_views: &HiZHistoryViews<'_>,
    history_texture: &wgpu::Texture,
    ws: usize,
) -> bool {
    if !session.scratch.is_stereo() {
        return false;
    }
    let Some(views_right) = history_views.right else {
        return false;
    };

    record_pyramid_side(
        session,
        history_views.left,
        DepthBinding::D2Array { layer: 0 },
        PyramidSide::DesktopOrLeft,
    );
    record_pyramid_side(
        session,
        views_right,
        DepthBinding::D2Array { layer: 1 },
        PyramidSide::Right,
    );

    staging_copy::copy_layer(session, history_texture, ws, 0, false);
    staging_copy::copy_layer(session, history_texture, ws, 1, true);
    true
}

/// Records mip0 + downsample dispatches for one pyramid layer chain.
fn record_pyramid_side(
    session: &mut EncodeSession<'_>,
    pyramid_views: &[wgpu::TextureView],
    depth_bind: DepthBinding,
    side: PyramidSide,
) {
    mip0::dispatch(session, pyramid_views, depth_bind);
    downsample::dispatch(session, pyramid_views, side);
}

#[cfg(test)]
mod tests {
    #[test]
    fn hi_z_encode_avoids_deferred_uploads_for_dispatch_local_uniforms() {
        let mip0 = include_str!("encode/mip0.rs");
        let downsample = include_str!("encode/downsample.rs");

        assert!(!mip0.contains("GraphUploadSink"));
        assert!(!mip0.contains("uploads.write_buffer"));
        assert!(!downsample.contains("GraphUploadSink"));
        assert!(!downsample.contains("uploads.write_buffer"));
    }
}
