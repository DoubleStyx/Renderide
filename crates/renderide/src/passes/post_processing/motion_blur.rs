//! Screen-space motion blur post-processing effect.
//!
//! The effect contributes two graph passes: a conditional velocity pass that derives camera
//! motion vectors from depth and per-view camera history, followed by a fullscreen HDR blur
//! resolve. When the velocity pass is skipped for a view, the resolve pass copies input to output
//! so the post-processing chain remains well-defined.

mod pipeline;

use std::num::NonZeroU32;
use std::sync::{Arc, LazyLock};

use glam::Mat4;
use hashbrown::HashMap;
use parking_lot::Mutex;

use pipeline::{
    MotionBlurParamsGpu, MotionBlurPipelineCache, MotionBlurPipelineKind, MotionVectorParamsGpu,
};

use crate::camera::{HostCameraFrame, ViewId, WorldProjectionSet, world_to_view_pair_for_skybox};
use crate::config::{MotionBlurSettings, PostProcessingSettings};
use crate::passes::helpers::{
    color_attachment, missing_pass_resource, read_fragment_sampled_texture,
    transient_output_format_or,
};
use crate::render_graph::builder::GraphBuilder;
use crate::render_graph::context::RasterPassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::gpu_cache::{create_d2_array_view, raster_stereo_mask_override};
use crate::render_graph::pass::{PassBuilder, RasterPass, RenderPassTemplate};
use crate::render_graph::post_process_chain::{
    EffectPasses, PostProcessEffect, PostProcessEffectId,
};
use crate::render_graph::post_process_settings::{MotionBlurSettingsSlot, MotionBlurSettingsValue};
use crate::render_graph::resources::{
    ImportedTextureHandle, TextureAccess, TextureHandle, TransientArrayLayers, TransientExtent,
    TransientSampleCount, TransientTextureDesc, TransientTextureFormat,
};

/// Velocity texture format used by the motion blur pass.
const MOTION_VECTOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

/// Default output format used before graph resources are resolved.
const DEFAULT_OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Minimum absolute determinant accepted for matrix inversion.
const MATRIX_INVERSION_EPSILON: f32 = 1.0e-20;

/// Effect descriptor plugged into [`crate::render_graph::post_process_chain::PostProcessChain`].
pub struct MotionBlurEffect {
    /// Imported frame depth target used by the velocity pass.
    depth: ImportedTextureHandle,
    /// Shared per-view camera history and GPU uniform buffers.
    state_cache: Arc<MotionBlurStateCache>,
}

impl MotionBlurEffect {
    /// Creates a motion blur effect backed by a shared per-view state cache.
    pub(crate) fn new(
        depth: ImportedTextureHandle,
        state_cache: Arc<MotionBlurStateCache>,
    ) -> Self {
        Self { depth, state_cache }
    }
}

impl PostProcessEffect for MotionBlurEffect {
    fn id(&self) -> PostProcessEffectId {
        PostProcessEffectId::MotionBlur
    }

    fn is_enabled(&self, settings: &PostProcessingSettings) -> bool {
        settings.enabled && settings.motion_blur.is_effectively_enabled()
    }

    fn register(
        &self,
        builder: &mut GraphBuilder,
        input: TextureHandle,
        output: TextureHandle,
    ) -> EffectPasses {
        let velocity = builder.create_texture(TransientTextureDesc {
            label: "motion_vectors",
            format: TransientTextureFormat::Fixed(MOTION_VECTOR_FORMAT),
            extent: TransientExtent::Backbuffer,
            mip_levels: 1,
            sample_count: TransientSampleCount::Fixed(1),
            dimension: wgpu::TextureDimension::D2,
            array_layers: TransientArrayLayers::Frame,
            base_usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            alias: true,
        });
        let vectors = builder.add_raster_pass(Box::new(MotionVectorsPass::new(
            self.depth,
            velocity,
            Arc::clone(&self.state_cache),
        )));
        let blur = builder.add_raster_pass(Box::new(MotionBlurResolvePass::new(
            input,
            velocity,
            output,
            Arc::clone(&self.state_cache),
        )));
        builder.add_edge(vectors, blur);
        EffectPasses {
            first: vectors,
            last: blur,
        }
    }
}

/// Fullscreen velocity pass deriving camera motion from depth reprojection.
pub struct MotionVectorsPass {
    /// Imported frame depth target sampled by the velocity shader.
    depth: ImportedTextureHandle,
    /// Transient velocity render target written by this pass.
    velocity: TextureHandle,
    /// Per-view camera history and uniform buffer cache.
    state_cache: Arc<MotionBlurStateCache>,
    /// Shared render pipeline cache for motion blur shaders.
    pipelines: &'static MotionBlurPipelineCache,
}

impl MotionVectorsPass {
    /// Creates a velocity pass for the supplied depth and velocity targets.
    fn new(
        depth: ImportedTextureHandle,
        velocity: TextureHandle,
        state_cache: Arc<MotionBlurStateCache>,
    ) -> Self {
        Self {
            depth,
            velocity,
            state_cache,
            pipelines: motion_blur_pipelines(),
        }
    }
}

impl RasterPass for MotionVectorsPass {
    fn name(&self) -> &str {
        "MotionVectors"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.import_texture(
            self.depth,
            TextureAccess::Sampled {
                stages: wgpu::ShaderStages::FRAGMENT,
            },
        );
        color_attachment(b, self.velocity, wgpu::LoadOp::Clear(wgpu::Color::BLACK));
        Ok(())
    }

    fn multiview_mask_override(
        &self,
        ctx: &RasterPassCtx<'_, '_>,
        template: &RenderPassTemplate,
    ) -> Option<NonZeroU32> {
        raster_stereo_mask_override(ctx, template)
    }

    fn should_record(&self, ctx: &RasterPassCtx<'_, '_>) -> Result<bool, RenderPassError> {
        let frame = &*ctx.pass_frame;
        let settings = motion_blur_settings(ctx.blackboard);
        Ok(
            view_motion_blur_active(&frame.view, settings)
                && frame.view.depth_sample_view.is_some(),
        )
    }

    fn release_view_resources(&mut self, retired_views: &[ViewId]) {
        self.state_cache.retire_views(retired_views);
    }

    fn record(
        &self,
        ctx: &mut RasterPassCtx<'_, '_>,
        rpass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), RenderPassError> {
        profiling::scope!("post_processing::motion_vectors");
        let frame = &*ctx.pass_frame;
        let Some(depth_view) = frame.view.depth_sample_view.as_ref() else {
            return Ok(());
        };
        let params = self
            .state_cache
            .compute_motion_vector_params(ctx.device, frame);
        let state = self.state_cache.ensure(ctx.device, frame.view.view_id);
        ctx.write_buffer(
            &state.motion_vector_params_buffer,
            0,
            bytemuck::bytes_of(&params),
        );

        let output_format =
            transient_output_format_or(self.velocity, ctx.graph_resources, MOTION_VECTOR_FORMAT);
        let pipeline = self.pipelines.pipeline(
            ctx.device,
            MotionBlurPipelineKind::MotionVectors,
            output_format,
            frame.view.multiview_stereo,
        );
        let bind_group = self.pipelines.motion_vectors_bind_group(
            ctx.device,
            depth_view,
            &state.motion_vector_params_buffer,
            frame.view.multiview_stereo,
        );
        rpass.set_pipeline(pipeline.as_ref());
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
        Ok(())
    }
}

/// Fullscreen pass that resolves HDR motion blur from velocity.
pub struct MotionBlurResolvePass {
    /// HDR scene-color input from the previous chain stage.
    input: TextureHandle,
    /// Velocity texture produced by [`MotionVectorsPass`].
    velocity: TextureHandle,
    /// HDR scene-color output for the next chain stage.
    output: TextureHandle,
    /// Per-view uniform buffer cache shared with the velocity pass.
    state_cache: Arc<MotionBlurStateCache>,
    /// Shared render pipeline cache for motion blur shaders.
    pipelines: &'static MotionBlurPipelineCache,
}

impl MotionBlurResolvePass {
    /// Creates an HDR blur resolve pass between two chain textures.
    fn new(
        input: TextureHandle,
        velocity: TextureHandle,
        output: TextureHandle,
        state_cache: Arc<MotionBlurStateCache>,
    ) -> Self {
        Self {
            input,
            velocity,
            output,
            state_cache,
            pipelines: motion_blur_pipelines(),
        }
    }
}

impl RasterPass for MotionBlurResolvePass {
    fn name(&self) -> &str {
        "MotionBlur"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        read_fragment_sampled_texture(b, self.input);
        read_fragment_sampled_texture(b, self.velocity);
        color_attachment(b, self.output, wgpu::LoadOp::Clear(wgpu::Color::BLACK));
        Ok(())
    }

    fn multiview_mask_override(
        &self,
        ctx: &RasterPassCtx<'_, '_>,
        template: &RenderPassTemplate,
    ) -> Option<NonZeroU32> {
        raster_stereo_mask_override(ctx, template)
    }

    fn should_record(&self, ctx: &RasterPassCtx<'_, '_>) -> Result<bool, RenderPassError> {
        Ok(super::view_post_processing_enabled(&ctx.pass_frame.view))
    }

    fn release_view_resources(&mut self, retired_views: &[ViewId]) {
        self.state_cache.retire_views(retired_views);
    }

    fn record(
        &self,
        ctx: &mut RasterPassCtx<'_, '_>,
        rpass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), RenderPassError> {
        profiling::scope!("post_processing::motion_blur");
        let frame = &*ctx.pass_frame;
        let Some(input) = ctx.graph_resources.transient_texture(self.input) else {
            return Err(missing_pass_resource(
                self.name(),
                format_args!("missing transient input {:?}", self.input),
            ));
        };
        let Some(velocity) = ctx.graph_resources.transient_texture(self.velocity) else {
            return Err(missing_pass_resource(
                self.name(),
                format_args!("missing transient velocity {:?}", self.velocity),
            ));
        };
        let settings = motion_blur_settings(ctx.blackboard);
        let active = view_motion_blur_active(&frame.view, settings);
        let params = MotionBlurParamsGpu::from_settings(settings, frame.view.viewport_px, active);
        let state = self.state_cache.ensure(ctx.device, frame.view.view_id);
        ctx.write_buffer(&state.blur_params_buffer, 0, bytemuck::bytes_of(&params));

        let input_view = create_d2_array_view(
            &input.texture,
            "motion_blur_scene_color",
            frame.view.multiview_stereo,
        );
        let velocity_view = create_d2_array_view(
            &velocity.texture,
            "motion_blur_velocity",
            frame.view.multiview_stereo,
        );
        let output_format =
            transient_output_format_or(self.output, ctx.graph_resources, DEFAULT_OUTPUT_FORMAT);
        let pipeline = self.pipelines.pipeline(
            ctx.device,
            MotionBlurPipelineKind::BlurResolve,
            output_format,
            frame.view.multiview_stereo,
        );
        let bind_group = self.pipelines.blur_bind_group(
            ctx.device,
            &input_view,
            &velocity_view,
            &state.blur_params_buffer,
        );
        rpass.set_pipeline(pipeline.as_ref());
        rpass.set_bind_group(1, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
        Ok(())
    }
}

/// Per-view motion blur state retained while a view remains active.
pub(crate) struct MotionBlurViewState {
    /// Uniform buffer consumed by the velocity pass.
    motion_vector_params_buffer: wgpu::Buffer,
    /// Uniform buffer consumed by the blur resolve pass.
    blur_params_buffer: wgpu::Buffer,
    /// Last rendered camera matrices for this view.
    history: Mutex<Option<MotionBlurCameraHistory>>,
}

impl MotionBlurViewState {
    /// Allocates per-view motion blur uniform buffers.
    fn new(device: &wgpu::Device, pipelines: &MotionBlurPipelineCache) -> Self {
        Self {
            motion_vector_params_buffer: pipelines.create_motion_vector_params_buffer(device),
            blur_params_buffer: pipelines.create_blur_params_buffer(device),
            history: Mutex::new(None),
        }
    }
}

/// Per-view history and GPU state cache for motion blur.
#[derive(Default)]
pub(crate) struct MotionBlurStateCache {
    per_view: Mutex<HashMap<ViewId, Arc<MotionBlurViewState>>>,
}

impl MotionBlurStateCache {
    /// Returns the retained state for `view_id`, allocating it on first use.
    fn ensure(&self, device: &wgpu::Device, view_id: ViewId) -> Arc<MotionBlurViewState> {
        let mut per_view = self.per_view.lock();
        Arc::clone(
            per_view.entry(view_id).or_insert_with(|| {
                Arc::new(MotionBlurViewState::new(device, motion_blur_pipelines()))
            }),
        )
    }

    /// Computes velocity shader parameters and advances this view's camera history.
    fn compute_motion_vector_params(
        &self,
        device: &wgpu::Device,
        frame: &crate::render_graph::frame_params::GraphPassFrame<'_>,
    ) -> MotionVectorParamsGpu {
        let state = self.ensure(device, frame.view.view_id);
        let current = MotionBlurCameraHistory::from_frame(frame);
        let mut history = state.history.lock();
        let previous = history.as_ref();
        let valid = previous.is_some_and(|previous| previous.is_valid_previous_for(&current));
        let previous = previous.copied().unwrap_or(current);
        *history = Some(current);
        drop(history);
        MotionVectorParamsGpu::from_history(current, previous, valid)
    }

    /// Releases motion blur state for views that are no longer active.
    pub(crate) fn retire_views(&self, retired_views: &[ViewId]) {
        if retired_views.is_empty() {
            return;
        }
        let mut per_view = self.per_view.lock();
        for view_id in retired_views {
            per_view.remove(view_id);
        }
    }
}

/// Camera history sample retained between consecutive frames for one logical view.
#[derive(Clone, Copy)]
struct MotionBlurCameraHistory {
    /// Left-eye or mono view-projection matrix for the rendered frame.
    view_proj_left: Mat4,
    /// Right-eye view-projection matrix, or a duplicate of left for mono.
    view_proj_right: Mat4,
    /// Viewport size in pixels.
    viewport_px: (u32, u32),
    /// Whether the rendered frame used stereo multiview.
    multiview_stereo: bool,
    /// Host frame index associated with this history sample.
    frame_index: i32,
}

impl MotionBlurCameraHistory {
    /// Captures the camera matrices and view shape for a graph pass frame.
    fn from_frame(frame: &crate::render_graph::frame_params::GraphPassFrame<'_>) -> Self {
        let view = &frame.view;
        let (view_proj_left, view_proj_right) =
            view_proj_pair(frame.shared.scene, &view.host_camera, view.viewport_px);
        Self {
            view_proj_left,
            view_proj_right,
            viewport_px: view.viewport_px,
            multiview_stereo: view.multiview_stereo,
            frame_index: view.host_camera.frame_index,
        }
    }

    /// Returns `true` when this sample can be used as the previous frame for `current`.
    fn is_valid_previous_for(self, current: &Self) -> bool {
        self.frame_index >= 0
            && current.frame_index == self.frame_index.saturating_add(1)
            && self.viewport_px == current.viewport_px
            && self.multiview_stereo == current.multiview_stereo
    }
}

impl MotionVectorParamsGpu {
    /// Packs current-to-previous clip transforms for the velocity shader.
    fn from_history(
        current: MotionBlurCameraHistory,
        previous: MotionBlurCameraHistory,
        valid: bool,
    ) -> Self {
        Self {
            current_clip_to_prev_clip_left: current_to_previous_clip_matrix(
                current.view_proj_left,
                previous.view_proj_left,
                valid,
            )
            .to_cols_array_2d(),
            current_clip_to_prev_clip_right: current_to_previous_clip_matrix(
                current.view_proj_right,
                previous.view_proj_right,
                valid,
            )
            .to_cols_array_2d(),
            viewport_px: [current.viewport_px.0 as f32, current.viewport_px.1 as f32],
            history_valid: if valid { 1.0 } else { 0.0 },
            _pad0: 0.0,
        }
    }
}

impl MotionBlurParamsGpu {
    /// Packs clamped blur settings for one view.
    fn from_settings(settings: MotionBlurSettings, viewport_px: (u32, u32), active: bool) -> Self {
        Self {
            shutter_angle: settings.effective_shutter_angle(),
            max_velocity_pixels: settings.effective_max_velocity_pixels(),
            sample_count: settings.effective_sample_count(),
            enabled: u32::from(active),
            viewport_px: [viewport_px.0 as f32, viewport_px.1 as f32],
            _pad0: [0.0, 0.0],
        }
    }
}

/// Returns the process-wide motion blur pipeline cache.
fn motion_blur_pipelines() -> &'static MotionBlurPipelineCache {
    static CACHE: LazyLock<MotionBlurPipelineCache> =
        LazyLock::new(MotionBlurPipelineCache::default);
    &CACHE
}

/// Reads live motion blur settings from the graph blackboard.
fn motion_blur_settings(
    blackboard: &crate::render_graph::blackboard::Blackboard,
) -> MotionBlurSettings {
    blackboard
        .get::<MotionBlurSettingsSlot>()
        .map(|MotionBlurSettingsValue(settings)| *settings)
        .unwrap_or_default()
}

/// Returns whether motion blur work should run for the current view.
fn view_motion_blur_active(
    view: &crate::render_graph::frame_params::GraphPassFrameView<'_>,
    settings: MotionBlurSettings,
) -> bool {
    super::view_post_processing_enabled(view)
        && view.post_processing.motion_blur
        && settings.is_effectively_enabled()
        && (!view.multiview_stereo || settings.allow_vr)
}

/// Builds left/right view-projection matrices for reprojection.
fn view_proj_pair(
    scene: &crate::scene::SceneCoordinator,
    host_camera: &HostCameraFrame,
    viewport_px: (u32, u32),
) -> (Mat4, Mat4) {
    if let Some(stereo) = host_camera.active_stereo() {
        return stereo.view_proj_pair();
    }
    let projections = WorldProjectionSet::from_scene_host(scene, viewport_px, host_camera);
    let (view_left, view_right) = world_to_view_pair_for_skybox(scene, host_camera);
    (
        projections.world_proj * view_left,
        projections.world_proj * view_right,
    )
}

/// Returns a current-clip to previous-clip transform, falling back to identity when invalid.
fn current_to_previous_clip_matrix(current: Mat4, previous: Mat4, valid: bool) -> Mat4 {
    if !valid {
        return Mat4::IDENTITY;
    }
    let determinant = current.determinant();
    if !determinant.is_finite() || determinant.abs() < MATRIX_INVERSION_EPSILON {
        return Mat4::IDENTITY;
    }
    previous * current.inverse()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_requires_consecutive_frame_same_shape() {
        let previous = MotionBlurCameraHistory {
            view_proj_left: Mat4::IDENTITY,
            view_proj_right: Mat4::IDENTITY,
            viewport_px: (1280, 720),
            multiview_stereo: false,
            frame_index: 7,
        };
        let mut current = previous;
        current.frame_index = 8;
        assert!(previous.is_valid_previous_for(&current));

        current.frame_index = 10;
        assert!(!previous.is_valid_previous_for(&current));

        current.frame_index = 8;
        current.viewport_px = (1920, 1080);
        assert!(!previous.is_valid_previous_for(&current));
    }

    #[test]
    fn blur_params_copy_when_view_inactive() {
        let params =
            MotionBlurParamsGpu::from_settings(MotionBlurSettings::default(), (100, 50), false);

        assert_eq!(params.enabled, 0);
        assert_eq!(
            params.sample_count,
            MotionBlurSettings::default().sample_count
        );
    }

    #[test]
    fn non_invertible_current_matrix_falls_back_to_identity() {
        let matrix = current_to_previous_clip_matrix(
            Mat4::ZERO,
            Mat4::from_scale(glam::Vec3::splat(2.0)),
            true,
        );

        assert_eq!(matrix, Mat4::IDENTITY);
    }
}
