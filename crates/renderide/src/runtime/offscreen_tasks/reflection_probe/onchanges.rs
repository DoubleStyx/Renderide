//! Time-sliced runtime reflection-probe captures.

use std::sync::Arc;

use hashbrown::HashSet;

use crate::backend::RenderBackend;
use crate::camera::{HostCameraFrame, ViewId};
use crate::gpu::GpuContext;
use crate::reflection_probes::specular::{
    RuntimeReflectionProbeCapture, RuntimeReflectionProbeCaptureKey,
};
use crate::render_graph::RenderPathProfile;
use crate::scene::{
    ReflectionProbeOnChangesRenderRequest, RenderSpaceId, SceneCoordinator,
    changed_probe_completion, reflection_probe_solid_color,
};
use crate::shared::{
    ReflectionProbeState, ReflectionProbeTimeSlicingMode, ReflectionProbeType, RenderingContext,
};

use super::{
    FrameViewPlan, FrameViewPlanParams, FrameViewPlanTarget, ProbeCubeFace, ProbeTaskExtent,
    ProbeTaskTargets, ReflectionProbeBakeError, RendererRuntime, clear_from_reflection_probe_state,
    create_probe_task_targets, draw_filter_from_reflection_probe_state,
    host_camera_frame_for_probe_face, render_reflection_probe_faces_offscreen,
};

/// Active OnChanges probe capture, retained across ticks when host time slicing requests it.
pub(crate) struct ActiveOnChangesReflectionProbeCapture {
    /// Host render request that started this capture.
    pub(in crate::runtime) request: ReflectionProbeOnChangesRenderRequest,
    /// Renderer-side capture generation.
    generation: u64,
    /// Cubemap target size and mip count.
    extent: ProbeTaskExtent,
    /// GPU targets that retain partially rendered faces between ticks.
    targets: ProbeTaskTargets,
    /// Face rendering and post-render time-slice progress.
    progress: RuntimeProbeCaptureProgress,
    /// Latest host unique id queued while this probe is still capturing.
    pub(in crate::runtime) queued_unique_id: Option<i32>,
}

/// Active realtime probe capture, retained across ticks while multi-frame time slicing advances.
pub(crate) struct ActiveRealtimeReflectionProbeCapture {
    /// Stable host probe identity.
    key: RuntimeReflectionProbeCaptureKey,
    /// Renderer-side capture generation.
    generation: u64,
    /// Cubemap target size and mip count.
    extent: ProbeTaskExtent,
    /// GPU targets that retain partially rendered faces between ticks.
    targets: ProbeTaskTargets,
    /// Face rendering and post-render time-slice progress.
    progress: RuntimeProbeCaptureProgress,
}

/// Progress state for a runtime cubemap capture.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct RuntimeProbeCaptureProgress {
    /// Bit mask of cubemap faces rendered into the capture target.
    rendered_faces: u8,
    /// Remaining post-render ticks before the capture can be published.
    filter_ticks_remaining: Option<u8>,
}

/// Inputs used to build render plans for a runtime cubemap capture step.
struct RuntimeReflectionProbeFacePlan<'a> {
    /// Scene data used to resolve probe transforms and render spaces.
    scene: &'a SceneCoordinator,
    /// Base camera frame supplying renderer-wide camera state.
    base_camera: &'a HostCameraFrame,
    /// Stable host probe identity.
    key: RuntimeReflectionProbeCaptureKey,
    /// View task id used to make per-face view ids stable.
    view_task_id: i32,
    /// Cubemap render target size.
    extent: ProbeTaskExtent,
    /// Cubemap face render targets.
    targets: &'a ProbeTaskTargets,
    /// Current host reflection probe state.
    state: ReflectionProbeState,
    /// Elapsed renderer runtime in seconds for Unity-style shader time inputs.
    frame_time_seconds: f32,
}

/// Result of advancing one runtime capture by one renderer tick.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RuntimeProbeCaptureStep {
    /// Capture still has more work before it can be published.
    Pending,
    /// Capture is ready to publish.
    Complete,
}

impl RuntimeProbeCaptureProgress {
    /// Cubemap faces that should render this tick.
    fn faces_for_step(self, mode: ReflectionProbeTimeSlicingMode) -> Vec<ProbeCubeFace> {
        if self.filter_ticks_remaining.is_some() || self.rendered_faces == ProbeCubeFace::ALL_MASK {
            return Vec::new();
        }
        capture_faces_for_step(mode, self.rendered_faces)
    }

    /// Marks a set of cubemap faces as rendered.
    fn mark_rendered(&mut self, faces: &[ProbeCubeFace]) {
        for face in faces {
            self.rendered_faces |= face.bit();
        }
    }

    /// Advances post-render filtering delay and reports whether the capture is ready.
    fn advance_after_step(
        &mut self,
        mode: ReflectionProbeTimeSlicingMode,
    ) -> RuntimeProbeCaptureStep {
        if self.rendered_faces != ProbeCubeFace::ALL_MASK {
            return RuntimeProbeCaptureStep::Pending;
        }
        match self.filter_ticks_remaining {
            None => {
                let delay = filter_delay_ticks(mode);
                if delay == 0 {
                    RuntimeProbeCaptureStep::Complete
                } else {
                    self.filter_ticks_remaining = Some(delay);
                    RuntimeProbeCaptureStep::Pending
                }
            }
            Some(0) => RuntimeProbeCaptureStep::Complete,
            Some(ticks) => {
                let remaining = ticks.saturating_sub(1);
                self.filter_ticks_remaining = Some(remaining);
                if remaining == 0 {
                    RuntimeProbeCaptureStep::Complete
                } else {
                    RuntimeProbeCaptureStep::Pending
                }
            }
        }
    }
}

/// Returns the next set of cubemap faces for the requested time-slicing mode.
fn capture_faces_for_step(
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

/// Post-render delay after the last face is rendered.
fn filter_delay_ticks(mode: ReflectionProbeTimeSlicingMode) -> u8 {
    match mode {
        ReflectionProbeTimeSlicingMode::NoTimeSlicing => 0,
        ReflectionProbeTimeSlicingMode::AllFacesAtOnce
        | ReflectionProbeTimeSlicingMode::IndividualFaces => 8,
    }
}

impl RendererRuntime {
    /// Advances host-requested OnChanges reflection-probe captures.
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
        let frame_time_seconds = self.tick_state.frame_time_seconds();
        let active =
            std::mem::take(&mut self.tick_state.active_onchanges_reflection_probe_captures);
        let mut still_active = Vec::with_capacity(active.len());
        let mut completed_results = Vec::new();
        for mut capture in active {
            match step_onchanges_reflection_probe_capture(OnChangesCaptureStepCtx {
                gpu,
                backend: &mut self.backend,
                scene: &self.scene,
                base_camera,
                frame_time_seconds,
                capture: &mut capture,
            }) {
                Ok(RuntimeProbeCaptureStep::Pending) => still_active.push(capture),
                Ok(RuntimeProbeCaptureStep::Complete) => {
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

    /// Starts any queued OnChanges reflection-probe captures.
    fn start_pending_onchanges_reflection_probe_captures(&mut self, gpu: &GpuContext) {
        profiling::scope!("reflection_probe_onchanges::start_pending");
        let pending =
            std::mem::take(&mut self.tick_state.pending_onchanges_reflection_probe_requests);
        if pending.is_empty() {
            return;
        }
        let mut completed_results = Vec::new();
        for request in pending {
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

    /// Advances continuously refreshing realtime reflection-probe captures.
    pub(in crate::runtime) fn drain_realtime_reflection_probe_captures(
        &mut self,
        gpu: &mut GpuContext,
    ) {
        profiling::scope!("reflection_probe_realtime::drain");
        self.start_missing_realtime_reflection_probe_captures(gpu);
        if self
            .tick_state
            .active_realtime_reflection_probe_captures
            .is_empty()
        {
            return;
        }

        let base_camera = &self.host_camera;
        let frame_time_seconds = self.tick_state.frame_time_seconds();
        let active = std::mem::take(&mut self.tick_state.active_realtime_reflection_probe_captures);
        let mut still_active = Vec::with_capacity(active.len());
        for mut capture in active {
            if !realtime_capture_is_still_valid(&self.scene, &capture) {
                continue;
            }
            match step_realtime_reflection_probe_capture(RealtimeCaptureStepCtx {
                gpu,
                backend: &mut self.backend,
                scene: &self.scene,
                base_camera,
                frame_time_seconds,
                capture: &mut capture,
            }) {
                Ok(RuntimeProbeCaptureStep::Pending) => still_active.push(capture),
                Ok(RuntimeProbeCaptureStep::Complete) => self
                    .backend
                    .register_runtime_reflection_probe_capture(capture.into_runtime_capture()),
                Err(error) => {
                    logger::debug!(
                        "Realtime reflection probe capture failed for render_space_id={} renderable_index={} generation={}: {error}",
                        capture.key.space_id.0,
                        capture.key.renderable_index,
                        capture.generation
                    );
                }
            }
        }
        self.tick_state.active_realtime_reflection_probe_captures = still_active;
    }

    /// Starts realtime capture cycles for active probes that do not already have one in flight.
    fn start_missing_realtime_reflection_probe_captures(&mut self, gpu: &GpuContext) {
        profiling::scope!("reflection_probe_realtime::start_missing");
        let active_keys = active_realtime_probe_keys(&self.scene);
        self.tick_state
            .active_realtime_reflection_probe_captures
            .retain(|capture| {
                active_keys.contains(&capture.key)
                    && realtime_capture_is_still_valid(&self.scene, capture)
            });
        for key in active_keys {
            let already_active = self
                .tick_state
                .active_realtime_reflection_probe_captures
                .iter()
                .any(|capture| capture.key == key);
            if already_active {
                continue;
            }
            let generation = self.tick_state.next_realtime_reflection_probe_generation;
            self.tick_state.next_realtime_reflection_probe_generation = self
                .tick_state
                .next_realtime_reflection_probe_generation
                .saturating_add(1);
            match start_realtime_reflection_probe_capture(gpu, &self.scene, key, generation) {
                Ok(capture) => self
                    .tick_state
                    .active_realtime_reflection_probe_captures
                    .push(*capture),
                Err(error) => {
                    logger::debug!(
                        "Realtime reflection probe capture could not start for render_space_id={} renderable_index={}: {error}",
                        key.space_id.0,
                        key.renderable_index
                    );
                }
            }
        }
    }
}

/// Result of attempting to start an OnChanges capture.
enum OnChangesCaptureStart {
    /// No GPU capture is required before notifying the host.
    ImmediateComplete,
    /// GPU cubemap capture is active.
    Capture(Box<ActiveOnChangesReflectionProbeCapture>),
}

/// Inputs for advancing one OnChanges capture.
struct OnChangesCaptureStepCtx<'a> {
    /// GPU context used to render cubemap faces.
    gpu: &'a mut GpuContext,
    /// Backend facade used by offscreen render extraction.
    backend: &'a mut RenderBackend,
    /// Scene snapshot containing probe and transform state.
    scene: &'a SceneCoordinator,
    /// Current host camera used as the base for offscreen face cameras.
    base_camera: &'a HostCameraFrame,
    /// Elapsed renderer runtime in seconds for Unity-style shader time inputs.
    frame_time_seconds: f32,
    /// Capture state advanced by this step.
    capture: &'a mut ActiveOnChangesReflectionProbeCapture,
}

/// Starts an OnChanges capture for one host request.
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
    if reflection_probe_solid_color(probe.state) {
        return Ok(OnChangesCaptureStart::ImmediateComplete);
    }
    if probe.state.r#type != ReflectionProbeType::OnChanges {
        return Err(ReflectionProbeBakeError::MissingProbe(
            request.renderable_index,
        ));
    }
    let extent = ProbeTaskExtent::from_size(probe.state.resolution)?;
    let targets = create_probe_task_targets(gpu, extent)?;
    Ok(OnChangesCaptureStart::Capture(Box::new(
        ActiveOnChangesReflectionProbeCapture {
            request,
            generation,
            extent,
            targets,
            progress: RuntimeProbeCaptureProgress::default(),
            queued_unique_id: None,
        },
    )))
}

/// Returns active realtime probe identities in deterministic render-space order.
fn active_realtime_probe_keys(scene: &SceneCoordinator) -> Vec<RuntimeReflectionProbeCaptureKey> {
    let mut keys = Vec::new();
    let mut seen = HashSet::new();
    for space_id in scene.render_space_ids() {
        let Some(space) = scene.space(space_id) else {
            continue;
        };
        if !space.is_active() {
            continue;
        }
        for probe in space.reflection_probes() {
            if !realtime_probe_state_needs_capture(probe.state) {
                continue;
            }
            let key = RuntimeReflectionProbeCaptureKey {
                space_id,
                renderable_index: probe.renderable_index,
            };
            if seen.insert(key) {
                keys.push(key);
            }
        }
    }
    keys
}

/// Returns whether a probe state should have a realtime cubemap capture in flight.
pub(in crate::runtime) fn realtime_probe_state_needs_capture(state: ReflectionProbeState) -> bool {
    state.r#type == ReflectionProbeType::Realtime && !reflection_probe_solid_color(state)
}

/// Starts a realtime capture generation for one active probe.
fn start_realtime_reflection_probe_capture(
    gpu: &GpuContext,
    scene: &SceneCoordinator,
    key: RuntimeReflectionProbeCaptureKey,
    generation: u64,
) -> Result<Box<ActiveRealtimeReflectionProbeCapture>, ReflectionProbeBakeError> {
    let state = realtime_capture_state(scene, key)?;
    let extent = ProbeTaskExtent::from_size(state.resolution)?;
    let targets = create_probe_task_targets(gpu, extent)?;
    Ok(Box::new(ActiveRealtimeReflectionProbeCapture {
        key,
        generation,
        extent,
        targets,
        progress: RuntimeProbeCaptureProgress::default(),
    }))
}

/// Returns whether an in-flight realtime capture still matches the scene state.
fn realtime_capture_is_still_valid(
    scene: &SceneCoordinator,
    capture: &ActiveRealtimeReflectionProbeCapture,
) -> bool {
    let Ok(state) = realtime_capture_state(scene, capture.key) else {
        return false;
    };
    match ProbeTaskExtent::from_size(state.resolution) {
        Ok(extent) => extent == capture.extent,
        Err(_error) => false,
    }
}

/// Inputs for advancing one realtime capture.
struct RealtimeCaptureStepCtx<'a> {
    /// GPU context used to render cubemap faces.
    gpu: &'a mut GpuContext,
    /// Backend facade used by offscreen render extraction.
    backend: &'a mut RenderBackend,
    /// Scene snapshot containing probe and transform state.
    scene: &'a SceneCoordinator,
    /// Current host camera used as the base for offscreen face cameras.
    base_camera: &'a HostCameraFrame,
    /// Elapsed renderer runtime in seconds for Unity-style shader time inputs.
    frame_time_seconds: f32,
    /// Capture state advanced by this step.
    capture: &'a mut ActiveRealtimeReflectionProbeCapture,
}

/// Advances one realtime capture by one renderer tick.
fn step_realtime_reflection_probe_capture(
    ctx: RealtimeCaptureStepCtx<'_>,
) -> Result<RuntimeProbeCaptureStep, ReflectionProbeBakeError> {
    profiling::scope!("reflection_probe_realtime::step");
    let state = realtime_capture_state(ctx.scene, ctx.capture.key)?;
    let faces = ctx.capture.progress.faces_for_step(state.time_slicing_mode);
    if !faces.is_empty() {
        let plans = plan_realtime_reflection_probe_faces(
            ctx.scene,
            ctx.base_camera,
            ctx.capture,
            state,
            ctx.frame_time_seconds,
            &faces,
        )?;
        let render_result =
            render_reflection_probe_faces_offscreen(ctx.gpu, ctx.backend, ctx.scene, plans);
        render_result?;
        ctx.capture.progress.mark_rendered(&faces);
    }
    Ok(ctx
        .capture
        .progress
        .advance_after_step(state.time_slicing_mode))
}

/// Advances one OnChanges capture by one renderer tick.
fn step_onchanges_reflection_probe_capture(
    ctx: OnChangesCaptureStepCtx<'_>,
) -> Result<RuntimeProbeCaptureStep, ReflectionProbeBakeError> {
    profiling::scope!("reflection_probe_onchanges::step");
    let state = onchanges_capture_state(ctx.scene, ctx.capture.request)?;
    let faces = ctx.capture.progress.faces_for_step(state.time_slicing_mode);
    if !faces.is_empty() {
        let plans = plan_onchanges_reflection_probe_faces(
            ctx.scene,
            ctx.base_camera,
            ctx.capture,
            state,
            ctx.frame_time_seconds,
            &faces,
        )?;
        let render_result =
            render_reflection_probe_faces_offscreen(ctx.gpu, ctx.backend, ctx.scene, plans);
        render_result?;
        ctx.capture.progress.mark_rendered(&faces);
    }
    Ok(ctx
        .capture
        .progress
        .advance_after_step(state.time_slicing_mode))
}

/// Returns the current OnChanges probe state for a capture request.
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
    if probe.state.r#type != ReflectionProbeType::OnChanges {
        return Err(ReflectionProbeBakeError::MissingProbe(
            request.renderable_index,
        ));
    }
    Ok(probe.state)
}

/// Returns the current realtime probe state for a capture key.
fn realtime_capture_state(
    scene: &SceneCoordinator,
    key: RuntimeReflectionProbeCaptureKey,
) -> Result<ReflectionProbeState, ReflectionProbeBakeError> {
    let space = scene
        .space(key.space_id)
        .ok_or(ReflectionProbeBakeError::MissingRenderSpace(key.space_id.0))?;
    if !space.is_active() {
        return Err(ReflectionProbeBakeError::InactiveRenderSpace(
            key.space_id.0,
        ));
    }
    let probe_index = usize::try_from(key.renderable_index)
        .map_err(|_err| ReflectionProbeBakeError::InvalidRenderableIndex(key.renderable_index))?;
    let probe = space
        .reflection_probes()
        .get(probe_index)
        .ok_or(ReflectionProbeBakeError::MissingProbe(key.renderable_index))?;
    if probe.state.r#type != ReflectionProbeType::Realtime {
        return Err(ReflectionProbeBakeError::MissingProbe(key.renderable_index));
    }
    Ok(probe.state)
}

/// Builds render plans for OnChanges cubemap faces.
fn plan_onchanges_reflection_probe_faces(
    scene: &SceneCoordinator,
    base_camera: &HostCameraFrame,
    capture: &ActiveOnChangesReflectionProbeCapture,
    state: ReflectionProbeState,
    frame_time_seconds: f32,
    faces: &[ProbeCubeFace],
) -> Result<Vec<FrameViewPlan<'static>>, ReflectionProbeBakeError> {
    let key = RuntimeReflectionProbeCaptureKey {
        space_id: RenderSpaceId(capture.request.render_space_id),
        renderable_index: capture.request.renderable_index,
    };
    plan_runtime_reflection_probe_faces(
        RuntimeReflectionProbeFacePlan {
            scene,
            base_camera,
            key,
            view_task_id: capture.request.unique_id,
            extent: capture.extent,
            targets: &capture.targets,
            state,
            frame_time_seconds,
        },
        faces,
    )
}

/// Builds render plans for realtime cubemap faces.
fn plan_realtime_reflection_probe_faces(
    scene: &SceneCoordinator,
    base_camera: &HostCameraFrame,
    capture: &ActiveRealtimeReflectionProbeCapture,
    state: ReflectionProbeState,
    frame_time_seconds: f32,
    faces: &[ProbeCubeFace],
) -> Result<Vec<FrameViewPlan<'static>>, ReflectionProbeBakeError> {
    plan_runtime_reflection_probe_faces(
        RuntimeReflectionProbeFacePlan {
            scene,
            base_camera,
            key: capture.key,
            view_task_id: runtime_capture_view_task_id(capture.generation),
            extent: capture.extent,
            targets: &capture.targets,
            state,
            frame_time_seconds,
        },
        faces,
    )
}

/// Builds render plans for dynamic reflection-probe cubemap faces.
fn plan_runtime_reflection_probe_faces(
    plan: RuntimeReflectionProbeFacePlan<'_>,
    faces: &[ProbeCubeFace],
) -> Result<Vec<FrameViewPlan<'static>>, ReflectionProbeBakeError> {
    let RuntimeReflectionProbeFacePlan {
        scene,
        base_camera,
        key,
        view_task_id,
        extent,
        targets,
        state,
        frame_time_seconds,
    } = plan;
    let space_id = key.space_id;
    let space = scene
        .space(space_id)
        .ok_or(ReflectionProbeBakeError::MissingRenderSpace(space_id.0))?;
    let probe = space
        .reflection_probes()
        .get(key.renderable_index as usize)
        .ok_or(ReflectionProbeBakeError::MissingProbe(key.renderable_index))?;
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
        .map(|face| {
            let host_camera = host_camera_frame_for_probe_face(
                base_camera,
                state,
                extent.tuple(),
                probe_position,
                face,
            );
            let mut plan = FrameViewPlan::new(
                &host_camera,
                FrameViewPlanParams {
                    render_context: RenderingContext::RenderToAsset,
                    frame_time_seconds,
                    view_id: ViewId::reflection_probe_render_task(
                        space_id,
                        view_task_id,
                        face.view_id_face_index(),
                    ),
                    viewport_px: extent.tuple(),
                    clear: clear_from_reflection_probe_state(state),
                    profile: RenderPathProfile::reflection_probe(),
                    target: FrameViewPlanTarget::offscreen(targets.to_offscreen_handles(face)),
                },
            );
            plan.render_space_filter = Some(space_id);
            plan.draw_filter = Some(filter.clone());
            plan
        })
        .collect())
}

/// Returns the next faces for OnChanges scheduler tests.
#[cfg(test)]
pub(in crate::runtime) fn onchanges_faces_for_step(
    mode: ReflectionProbeTimeSlicingMode,
    rendered_faces: u8,
) -> Vec<ProbeCubeFace> {
    capture_faces_for_step(mode, rendered_faces)
}

/// Simulates a complete runtime capture and returns its tick count.
#[cfg(test)]
pub(in crate::runtime) fn runtime_capture_ticks_to_complete(
    mode: ReflectionProbeTimeSlicingMode,
) -> usize {
    let mut progress = RuntimeProbeCaptureProgress::default();
    let mut ticks = 0usize;
    loop {
        ticks = ticks.saturating_add(1);
        let faces = progress.faces_for_step(mode);
        progress.mark_rendered(&faces);
        if progress.advance_after_step(mode) == RuntimeProbeCaptureStep::Complete {
            return ticks;
        }
    }
}

/// Returns whether two OnChanges requests target the same probe.
pub(in crate::runtime) fn same_onchanges_probe(
    a: ReflectionProbeOnChangesRenderRequest,
    b: ReflectionProbeOnChangesRenderRequest,
) -> bool {
    a.render_space_id == b.render_space_id && a.renderable_index == b.renderable_index
}

/// Maps an internal capture generation into the existing one-shot probe view id space.
fn runtime_capture_view_task_id(generation: u64) -> i32 {
    (generation & i32::MAX as u64) as i32
}

impl ActiveOnChangesReflectionProbeCapture {
    /// Converts a completed capture into the backend runtime-cubemap source.
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
            view: self
                .targets
                .cube_sample_view("renderide-reflection-probe-onchanges-cube-view"),
            array_view: self
                .targets
                .array_sample_view("renderide-reflection-probe-onchanges-array-view"),
        }
    }
}

impl ActiveRealtimeReflectionProbeCapture {
    /// Converts a completed capture into the backend runtime-cubemap source.
    fn into_runtime_capture(self) -> RuntimeReflectionProbeCapture {
        RuntimeReflectionProbeCapture {
            key: self.key,
            generation: self.generation,
            face_size: self.extent.size,
            mip_levels: self.extent.mip_levels,
            texture: Arc::clone(&self.targets.cube_texture),
            view: self
                .targets
                .cube_sample_view("renderide-reflection-probe-realtime-cube-view"),
            array_view: self
                .targets
                .array_sample_view("renderide-reflection-probe-realtime-array-view"),
        }
    }
}
