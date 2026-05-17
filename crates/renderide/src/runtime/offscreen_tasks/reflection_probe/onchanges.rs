//! Time-sliced OnChanges reflection-probe captures.

use std::sync::Arc;

use crate::backend::RenderBackend;
use crate::camera::{HostCameraFrame, ViewId};
use crate::gpu::GpuContext;
use crate::reflection_probes::specular::{
    RuntimeReflectionProbeCapture, RuntimeReflectionProbeCaptureKey,
};
use crate::scene::{
    ReflectionProbeOnChangesRenderRequest, RenderSpaceId, SceneCoordinator,
    changed_probe_completion,
};
use crate::shared::{
    ReflectionProbeClear, ReflectionProbeState, ReflectionProbeTimeSlicingMode, RenderingContext,
};

use super::{
    FrameViewPlan, FrameViewPlanTarget, ProbeCubeFace, ProbeTaskExtent, ProbeTaskTargets,
    ReflectionProbeBakeError, RendererRuntime, clear_from_reflection_probe_state,
    draw_filter_from_reflection_probe_state, host_camera_frame_for_probe_face,
    reflection_probe_bake_post_processing, render_reflection_probe_faces_offscreen,
};

/// Active OnChanges probe capture, retained across ticks when host time slicing requests it.
pub(crate) struct ActiveOnChangesReflectionProbeCapture {
    pub(in crate::runtime) request: ReflectionProbeOnChangesRenderRequest,
    generation: u64,
    extent: ProbeTaskExtent,
    targets: ProbeTaskTargets,
    rendered_faces: u8,
    pub(in crate::runtime) queued_unique_id: Option<i32>,
}

impl RendererRuntime {
    pub(in crate::runtime) fn drain_onchanges_reflection_probe_captures(
        &mut self,
        gpu: &mut GpuContext,
    ) {
        profiling::scope!("reflection_probe_onchanges::drain");
        self.start_pending_onchanges_reflection_probe_captures(gpu);
        if self
            .tick_state
            .active_onchanges_reflection_probe_captures
            .is_empty()
        {
            return;
        }

        let base_camera = &self.host_camera;
        let mut active =
            std::mem::take(&mut self.tick_state.active_onchanges_reflection_probe_captures);
        let mut still_active = Vec::with_capacity(active.len());
        let mut completed_results = Vec::new();
        for mut capture in active.drain(..) {
            match step_onchanges_reflection_probe_capture(OnChangesCaptureStepCtx {
                gpu,
                backend: &mut self.backend,
                scene: &self.scene,
                base_camera,
                capture: &mut capture,
            }) {
                Ok(OnChangesCaptureStep::Pending) => still_active.push(capture),
                Ok(OnChangesCaptureStep::Complete) => {
                    let completed_request = capture.request;
                    let follow_up = capture.queued_unique_id.take().map(|unique_id| {
                        ReflectionProbeOnChangesRenderRequest {
                            unique_id,
                            ..completed_request
                        }
                    });
                    self.backend
                        .register_runtime_reflection_probe_capture(capture.into_runtime_capture());
                    completed_results.push(changed_probe_completion(
                        completed_request.render_space_id,
                        completed_request.unique_id,
                        false,
                    ));
                    if let Some(request) = follow_up {
                        self.tick_state
                            .pending_onchanges_reflection_probe_requests
                            .push(request);
                    }
                }
                Err(error) => {
                    logger::warn!(
                        "OnChanges reflection probe capture failed for render_space_id={} renderable_index={} unique_id={}: {error}",
                        capture.request.render_space_id,
                        capture.request.renderable_index,
                        capture.request.unique_id
                    );
                    completed_results.push(changed_probe_completion(
                        capture.request.render_space_id,
                        capture.request.unique_id,
                        true,
                    ));
                }
            }
        }
        self.tick_state.active_onchanges_reflection_probe_captures = still_active;
        self.frontend
            .enqueue_rendered_reflection_probes(completed_results);
    }

    fn start_pending_onchanges_reflection_probe_captures(&mut self, gpu: &GpuContext) {
        profiling::scope!("reflection_probe_onchanges::start_pending");
        let mut pending =
            std::mem::take(&mut self.tick_state.pending_onchanges_reflection_probe_requests);
        if pending.is_empty() {
            return;
        }
        let mut completed_results = Vec::new();
        for request in pending.drain(..) {
            let generation = self.tick_state.next_onchanges_reflection_probe_generation;
            self.tick_state.next_onchanges_reflection_probe_generation = self
                .tick_state
                .next_onchanges_reflection_probe_generation
                .saturating_add(1);
            match start_onchanges_reflection_probe_capture(gpu, &self.scene, request, generation) {
                Ok(OnChangesCaptureStart::ImmediateComplete) => {
                    completed_results.push(changed_probe_completion(
                        request.render_space_id,
                        request.unique_id,
                        false,
                    ));
                }
                Ok(OnChangesCaptureStart::Capture(capture)) => {
                    self.tick_state
                        .active_onchanges_reflection_probe_captures
                        .push(*capture);
                }
                Err(error) => {
                    logger::warn!(
                        "OnChanges reflection probe capture could not start for render_space_id={} renderable_index={} unique_id={}: {error}",
                        request.render_space_id,
                        request.renderable_index,
                        request.unique_id
                    );
                    completed_results.push(changed_probe_completion(
                        request.render_space_id,
                        request.unique_id,
                        true,
                    ));
                }
            }
        }
        self.frontend
            .enqueue_rendered_reflection_probes(completed_results);
    }
}

enum OnChangesCaptureStart {
    ImmediateComplete,
    Capture(Box<ActiveOnChangesReflectionProbeCapture>),
}

enum OnChangesCaptureStep {
    Pending,
    Complete,
}

struct OnChangesCaptureStepCtx<'a> {
    gpu: &'a mut GpuContext,
    backend: &'a mut RenderBackend,
    scene: &'a SceneCoordinator,
    base_camera: &'a HostCameraFrame,
    capture: &'a mut ActiveOnChangesReflectionProbeCapture,
}

fn start_onchanges_reflection_probe_capture(
    gpu: &GpuContext,
    scene: &SceneCoordinator,
    request: ReflectionProbeOnChangesRenderRequest,
    generation: u64,
) -> Result<OnChangesCaptureStart, ReflectionProbeBakeError> {
    let space_id = RenderSpaceId(request.render_space_id);
    let space = scene
        .space(space_id)
        .ok_or(ReflectionProbeBakeError::MissingRenderSpace(
            request.render_space_id,
        ))?;
    if !space.is_active() {
        return Err(ReflectionProbeBakeError::InactiveRenderSpace(
            request.render_space_id,
        ));
    }
    let probe_index = usize::try_from(request.renderable_index).map_err(|_err| {
        ReflectionProbeBakeError::InvalidRenderableIndex(request.renderable_index)
    })?;
    let probe = space.reflection_probes().get(probe_index).ok_or(
        ReflectionProbeBakeError::MissingProbe(request.renderable_index),
    )?;
    if probe.state.clear_flags == ReflectionProbeClear::Color {
        return Ok(OnChangesCaptureStart::ImmediateComplete);
    }
    if probe.state.r#type != crate::shared::ReflectionProbeType::OnChanges {
        return Err(ReflectionProbeBakeError::MissingProbe(
            request.renderable_index,
        ));
    }
    let extent = ProbeTaskExtent::from_size(probe.state.resolution)?;
    let targets = ProbeTaskTargets::create(gpu, extent)?;
    Ok(OnChangesCaptureStart::Capture(Box::new(
        ActiveOnChangesReflectionProbeCapture {
            request,
            generation,
            extent,
            targets,
            rendered_faces: 0,
            queued_unique_id: None,
        },
    )))
}

fn step_onchanges_reflection_probe_capture(
    ctx: OnChangesCaptureStepCtx<'_>,
) -> Result<OnChangesCaptureStep, ReflectionProbeBakeError> {
    profiling::scope!("reflection_probe_onchanges::step");
    let state = onchanges_capture_state(ctx.scene, ctx.capture.request)?;
    let faces = onchanges_faces_for_step(state.time_slicing_mode, ctx.capture.rendered_faces);
    if faces.is_empty() {
        return Ok(OnChangesCaptureStep::Complete);
    }
    let plans = plan_onchanges_reflection_probe_faces(
        ctx.scene,
        ctx.base_camera,
        ctx.capture,
        state,
        &faces,
    )?;
    let view_ids = plans.iter().map(|plan| plan.view_id).collect::<Vec<_>>();
    let render_result =
        render_reflection_probe_faces_offscreen(ctx.gpu, ctx.backend, ctx.scene, plans);
    ctx.backend.retire_one_shot_views(&view_ids);
    render_result?;
    for face in faces {
        ctx.capture.rendered_faces |= face.bit();
    }
    if ctx.capture.rendered_faces == ProbeCubeFace::ALL_MASK {
        Ok(OnChangesCaptureStep::Complete)
    } else {
        Ok(OnChangesCaptureStep::Pending)
    }
}

fn onchanges_capture_state(
    scene: &SceneCoordinator,
    request: ReflectionProbeOnChangesRenderRequest,
) -> Result<ReflectionProbeState, ReflectionProbeBakeError> {
    let space_id = RenderSpaceId(request.render_space_id);
    let space = scene
        .space(space_id)
        .ok_or(ReflectionProbeBakeError::MissingRenderSpace(
            request.render_space_id,
        ))?;
    if !space.is_active() {
        return Err(ReflectionProbeBakeError::InactiveRenderSpace(
            request.render_space_id,
        ));
    }
    let probe_index = usize::try_from(request.renderable_index).map_err(|_err| {
        ReflectionProbeBakeError::InvalidRenderableIndex(request.renderable_index)
    })?;
    let probe = space.reflection_probes().get(probe_index).ok_or(
        ReflectionProbeBakeError::MissingProbe(request.renderable_index),
    )?;
    if probe.state.r#type != crate::shared::ReflectionProbeType::OnChanges {
        return Err(ReflectionProbeBakeError::MissingProbe(
            request.renderable_index,
        ));
    }
    Ok(probe.state)
}

fn plan_onchanges_reflection_probe_faces(
    scene: &SceneCoordinator,
    base_camera: &HostCameraFrame,
    capture: &ActiveOnChangesReflectionProbeCapture,
    state: ReflectionProbeState,
    faces: &[ProbeCubeFace],
) -> Result<Vec<FrameViewPlan<'static>>, ReflectionProbeBakeError> {
    let space_id = RenderSpaceId(capture.request.render_space_id);
    let space = scene
        .space(space_id)
        .ok_or(ReflectionProbeBakeError::MissingRenderSpace(space_id.0))?;
    let probe = space
        .reflection_probes()
        .get(capture.request.renderable_index as usize)
        .ok_or(ReflectionProbeBakeError::MissingProbe(
            capture.request.renderable_index,
        ))?;
    let transform_index = usize::try_from(probe.transform_id)
        .map_err(|_err| ReflectionProbeBakeError::InvalidProbeTransform(probe.transform_id))?;
    let probe_world = scene
        .world_matrix_for_render_context(
            space_id,
            transform_index,
            RenderingContext::RenderToAsset,
            base_camera.head_output_transform,
        )
        .ok_or(ReflectionProbeBakeError::MissingProbeTransform(
            probe.transform_id,
        ))?;
    let filter = draw_filter_from_reflection_probe_state(&state);
    let probe_position = probe_world.col(3).truncate();
    Ok(faces
        .iter()
        .copied()
        .map(|face| FrameViewPlan {
            host_camera: host_camera_frame_for_probe_face(
                base_camera,
                state,
                capture.extent.tuple(),
                probe_position,
                face,
            ),
            render_context: RenderingContext::RenderToAsset,
            render_space_filter: Some(space_id),
            draw_filter: Some(filter.clone()),
            view_id: ViewId::reflection_probe_render_task(
                space_id,
                capture.request.unique_id,
                face.view_id_face_index(),
            ),
            viewport_px: capture.extent.tuple(),
            clear: clear_from_reflection_probe_state(state),
            post_processing: reflection_probe_bake_post_processing(),
            target: FrameViewPlanTarget::SecondaryRt(capture.targets.to_offscreen_handles(face)),
        })
        .collect())
}

pub(in crate::runtime) fn onchanges_faces_for_step(
    mode: ReflectionProbeTimeSlicingMode,
    rendered_faces: u8,
) -> Vec<ProbeCubeFace> {
    let remaining = ProbeCubeFace::ALL
        .into_iter()
        .filter(|face| rendered_faces & face.bit() == 0);
    if mode == ReflectionProbeTimeSlicingMode::IndividualFaces {
        remaining.take(1).collect()
    } else {
        remaining.collect()
    }
}

pub(in crate::runtime) fn same_onchanges_probe(
    a: ReflectionProbeOnChangesRenderRequest,
    b: ReflectionProbeOnChangesRenderRequest,
) -> bool {
    a.render_space_id == b.render_space_id && a.renderable_index == b.renderable_index
}

impl ActiveOnChangesReflectionProbeCapture {
    fn into_runtime_capture(self) -> RuntimeReflectionProbeCapture {
        RuntimeReflectionProbeCapture {
            key: RuntimeReflectionProbeCaptureKey {
                space_id: RenderSpaceId(self.request.render_space_id),
                renderable_index: self.request.renderable_index,
            },
            generation: self.generation,
            face_size: self.extent.size,
            mip_levels: self.extent.mip_levels,
            texture: Arc::clone(&self.targets.cube_texture),
            view: self.targets.cube_sample_view(),
            array_view: self.targets.array_sample_view(),
        }
    }
}
