//! GPU frustum/previous-Hi-Z culling and indexed-indirect command compaction.
use std::fmt;
use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec4};

use crate::embedded_shaders::embedded_wgsl;
use crate::frame_upload_batch::GraphUploadSink;
use crate::gpu::GpuRetainedResources;
use crate::gpu::indirect_buffer::IndexedIndirectCommand;
use crate::gpu_resource::OnceGpu;

const WORKGROUP_SIZE: u32 = 64;
const INDIRECT_COMMAND_BYTES: u64 = size_of::<IndexedIndirectCommand>() as u64;
const COUNT_BYTES: u64 = size_of::<u32>() as u64;
const PARAMS_BYTES: u64 = size_of::<GpuCullParams>() as u64;

/// Candidate is eligible for previous-frame Hi-Z rejection.
pub(crate) const GPU_CULL_CANDIDATE_HIZ_ELIGIBLE: u32 = 1 << 0;
/// Candidate must be emitted without frustum or Hi-Z rejection.
pub(crate) const GPU_CULL_CANDIDATE_ALWAYS_VISIBLE: u32 = 1 << 1;

const PARAM_FIXED_SLOTS: u32 = 1 << 0;
const PARAM_HIZ_ENABLED: u32 = 1 << 1;

/// One arena-resident indexed draw and its local-space conservative bounds.
///
/// The 64-byte layout is mirrored exactly by `gpu_cull_compact.wgsl`.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct GpuCullCandidate {
    pub command: IndexedIndirectCommand,
    pub run_index: u32,
    pub matrix_index: u32,
    pub flags: u32,
    pub bounds_min: [f32; 4],
    pub bounds_max: [f32; 4],
}

impl GpuCullCandidate {
    pub(crate) fn new(
        command: IndexedIndirectCommand,
        run_index: u32,
        matrix_index: u32,
        bounds_min: [f32; 3],
        bounds_max: [f32; 3],
    ) -> Self {
        Self {
            command,
            run_index,
            matrix_index,
            flags: 0,
            bounds_min: [bounds_min[0], bounds_min[1], bounds_min[2], 0.0],
            bounds_max: [bounds_max[0], bounds_max[1], bounds_max[2], 0.0],
        }
    }

    #[must_use]
    pub(crate) fn with_previous_hiz(mut self, eligible: bool) -> Self {
        if eligible {
            self.flags |= GPU_CULL_CANDIDATE_HIZ_ELIGIBLE;
        } else {
            self.flags &= !GPU_CULL_CANDIDATE_HIZ_ELIGIBLE;
        }
        self
    }

    #[must_use]
    pub(crate) fn with_always_visible(mut self, always_visible: bool) -> Self {
        if always_visible {
            self.flags |= GPU_CULL_CANDIDATE_ALWAYS_VISIBLE;
        } else {
            self.flags &= !GPU_CULL_CANDIDATE_ALWAYS_VISIBLE;
        }
        self
    }
}

/// A consecutive candidate range whose visible commands share render state.
///
/// Compact output for a run starts at `output_start`; its atomic count lives at `count_index`.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub(crate) struct GpuCullRun {
    pub candidate_start: u32,
    pub candidate_count: u32,
    pub output_start: u32,
    pub output_capacity: u32,
    pub count_index: u32,
    _pad: [u32; 3],
}

impl GpuCullRun {
    pub(crate) const fn new(
        candidate_start: u32,
        candidate_count: u32,
        output_start: u32,
        output_capacity: u32,
        count_index: u32,
    ) -> Self {
        Self {
            candidate_start,
            candidate_count,
            output_start,
            output_capacity,
            count_index,
            _pad: [0; 3],
        }
    }
}

/// Per-render-space matrices used by candidates.
///
/// Current frustum planes and previous local-to-clip matrices are stored for up to two eyes.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct GpuCullMatrix {
    pub current_frustum_planes: [[[f32; 4]; 6]; 2],
    pub previous_view_proj: [[[f32; 4]; 4]; 2],
    pub eye_count: u32,
    pub previous_valid_mask: u32,
    _pad: [u32; 2],
}

impl GpuCullMatrix {
    /// Creates a matrix row from one or two local-to-clip transforms.
    ///
    /// Plane extraction uses WebGPU homogeneous clip inequalities (`-w <= x,y <= w`,
    /// `0 <= z <= w`) and therefore remains correct for this renderer's reverse-Z projection.
    pub(crate) fn from_view_projections(
        current: &[Mat4],
        previous: &[Option<Mat4>],
    ) -> Result<Self, GpuCullMatrixError> {
        if !(1..=2).contains(&current.len()) {
            return Err(GpuCullMatrixError::CurrentEyeCount(current.len()));
        }
        if previous.len() > current.len() {
            return Err(GpuCullMatrixError::PreviousEyeCount {
                current: current.len(),
                previous: previous.len(),
            });
        }

        let mut result = Self::zeroed();
        result.eye_count = current.len() as u32;
        for (eye, matrix) in current.iter().copied().enumerate() {
            result.current_frustum_planes[eye] = extract_clip_planes(matrix);
            if let Some(Some(previous)) = previous.get(eye) {
                result.previous_view_proj[eye] = previous.to_cols_array_2d();
                result.previous_valid_mask |= 1 << eye;
            }
        }
        Ok(result)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GpuCullMatrixError {
    CurrentEyeCount(usize),
    PreviousEyeCount { current: usize, previous: usize },
}

impl fmt::Display for GpuCullMatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CurrentEyeCount(count) => {
                write!(
                    f,
                    "GPU cull matrix needs one or two current eyes, got {count}"
                )
            }
            Self::PreviousEyeCount { current, previous } => write!(
                f,
                "GPU cull matrix has {previous} previous eyes but only {current} current eyes"
            ),
        }
    }
}

impl std::error::Error for GpuCullMatrixError {}

/// Output packing selected according to adapter draw-count support.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum GpuCullOutputMode {
    /// Atomically compact each run and consume the count buffer with
    /// `multi_draw_indexed_indirect_count`.
    CompactCount,
    /// Keep one deterministic output slot per candidate and zero `instance_count` when culled.
    #[default]
    FixedSlots,
}

/// Optional previous-frame reverse-Z min Hi-Z pyramid views.
///
/// Views must be `D2`, `R32Float`, and cover the complete mip chain for their eye.
#[derive(Clone, Copy)]
pub(crate) struct GpuCullPreviousHiZ<'a> {
    pub left: &'a wgpu::TextureView,
    pub right: Option<&'a wgpu::TextureView>,
    pub depth_bias: f32,
}

/// CPU inputs for one cull dispatch.
pub(crate) struct GpuCullEncode<'a> {
    pub candidates: &'a [GpuCullCandidate],
    pub runs: &'a [GpuCullRun],
    pub matrices: &'a [GpuCullMatrix],
    /// Caller-owned generation for the immutable candidate/run payload.
    ///
    /// Reusing the same non-`None` value promises that `candidates`, `runs`, and their matrix
    /// indices are byte-for-byte identical to the last successful encode on this compaction
    /// object. The camera matrix contents may still change every frame. This lets retained scene
    /// users avoid validating and uploading the large structural buffers again.
    pub static_input_generation: Option<u64>,
    pub output_mode: GpuCullOutputMode,
    pub previous_hiz: Option<GpuCullPreviousHiZ<'a>>,
}

impl<'a> GpuCullEncode<'a> {
    pub(crate) fn new(
        candidates: &'a [GpuCullCandidate],
        runs: &'a [GpuCullRun],
        matrices: &'a [GpuCullMatrix],
    ) -> Self {
        Self {
            candidates,
            runs,
            matrices,
            static_input_generation: None,
            output_mode: GpuCullOutputMode::FixedSlots,
            previous_hiz: None,
        }
    }
}

/// Metadata for consuming one run after compute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GpuCullRunDraw {
    pub indirect_offset: u64,
    pub count_offset: u64,
    pub max_count: u32,
    pub fixed_count: u32,
    pub output_mode: GpuCullOutputMode,
}

/// Summary of the commands recorded by [`GpuCullCompaction::encode`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GpuCullDispatch {
    pub candidate_count: u32,
    pub run_count: u32,
    pub workgroup_count: u32,
    pub output_mode: GpuCullOutputMode,
    /// Whether candidate/run buffers were validated and uploaded for this dispatch.
    pub static_inputs_uploaded: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum GpuCullError {
    CountOverflow(&'static str),
    CapacityExceeded {
        resource: &'static str,
        needed: u64,
        maximum: u64,
    },
    RunCandidateRange {
        run: usize,
    },
    RunOutputRange {
        run: usize,
    },
    OutputTooSmall {
        run: usize,
        candidates: u32,
        capacity: u32,
    },
    CandidateNotOwned {
        candidate: usize,
    },
    CandidateOwnedTwice {
        candidate: usize,
    },
    CandidateRunMismatch {
        candidate: usize,
        expected: usize,
        actual: u32,
    },
    CandidateMatrixOutOfRange {
        candidate: usize,
        matrix: u32,
        matrix_count: usize,
    },
    InvalidBounds {
        candidate: usize,
    },
    OutputOverlap {
        first_run: usize,
        second_run: usize,
    },
    CountIndexReused {
        first_run: usize,
        second_run: usize,
        count_index: u32,
    },
}

impl fmt::Display for GpuCullError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CountOverflow(name) => write!(f, "{name} count exceeds u32"),
            Self::CapacityExceeded {
                resource,
                needed,
                maximum,
            } => write!(
                f,
                "GPU cull {resource} needs {needed} elements but device permits {maximum}"
            ),
            Self::RunCandidateRange { run } => {
                write!(f, "GPU cull run {run} candidate range is out of bounds")
            }
            Self::RunOutputRange { run } => {
                write!(f, "GPU cull run {run} output range overflows")
            }
            Self::OutputTooSmall {
                run,
                candidates,
                capacity,
            } => write!(
                f,
                "GPU cull run {run} has {candidates} candidates but only {capacity} output slots"
            ),
            Self::CandidateNotOwned { candidate } => {
                write!(f, "GPU cull candidate {candidate} is not covered by a run")
            }
            Self::CandidateOwnedTwice { candidate } => {
                write!(
                    f,
                    "GPU cull candidate {candidate} is covered by multiple runs"
                )
            }
            Self::CandidateRunMismatch {
                candidate,
                expected,
                actual,
            } => write!(
                f,
                "GPU cull candidate {candidate} declares run {actual}, expected {expected}"
            ),
            Self::CandidateMatrixOutOfRange {
                candidate,
                matrix,
                matrix_count,
            } => write!(
                f,
                "GPU cull candidate {candidate} uses matrix {matrix}, but only {matrix_count} exist"
            ),
            Self::InvalidBounds { candidate } => {
                write!(f, "GPU cull candidate {candidate} has invalid bounds")
            }
            Self::OutputOverlap {
                first_run,
                second_run,
            } => write!(
                f,
                "GPU cull runs {first_run} and {second_run} overlap in the output buffer"
            ),
            Self::CountIndexReused {
                first_run,
                second_run,
                count_index,
            } => write!(
                f,
                "GPU cull runs {first_run} and {second_run} both use count index {count_index}"
            ),
        }
    }
}

impl std::error::Error for GpuCullError {}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuCullParams {
    candidate_count: u32,
    run_count: u32,
    matrix_count: u32,
    output_capacity: u32,
    count_capacity: u32,
    flags: u32,
    hiz_eye_mask: u32,
    hiz_depth_bias: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ValidatedLayout {
    output_elements: u32,
    count_elements: u32,
}

struct GpuCullBindGroupCacheEntry {
    hiz_left: wgpu::TextureView,
    hiz_right: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct UploadedStaticInputs {
    generation: u64,
    candidate_count: u32,
    run_count: u32,
    matrix_count: u32,
    layout: ValidatedLayout,
}

impl UploadedStaticInputs {
    fn matches(
        &self,
        generation: u64,
        candidate_count: u32,
        run_count: u32,
        matrix_count: u32,
    ) -> bool {
        self.generation == generation
            && self.candidate_count == candidate_count
            && self.run_count == run_count
            && self.matrix_count == matrix_count
    }
}

/// Persistent buffers and bind groups for the GPU cull/compact compute dispatch.
pub(crate) struct GpuCullCompaction {
    candidates: wgpu::Buffer,
    runs: wgpu::Buffer,
    matrices: wgpu::Buffer,
    output: wgpu::Buffer,
    counts: wgpu::Buffer,
    params: wgpu::Buffer,
    candidate_capacity: u32,
    run_capacity: u32,
    matrix_capacity: u32,
    output_capacity: u32,
    count_capacity: u32,
    fallback_hiz: wgpu::TextureView,
    /// LRU order. The last entry is the bind group selected by the most recent encode.
    ///
    /// Three entries retain both Hi-Z ping-pong halves plus the no-history fallback without
    /// rebuilding bindings when the selected history texture alternates each frame.
    bind_groups: Vec<GpuCullBindGroupCacheEntry>,
    /// Last caller-versioned candidate/run payload resident in `candidates` and `runs`.
    uploaded_static_inputs: Option<UploadedStaticInputs>,
}

impl GpuCullCompaction {
    const MAX_BIND_GROUPS: usize = 3;

    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let candidates = create_storage_buffer::<GpuCullCandidate>(
            device,
            "gpu_cull_candidates",
            1,
            wgpu::BufferUsages::empty(),
        );
        let runs = create_storage_buffer::<GpuCullRun>(
            device,
            "gpu_cull_runs",
            1,
            wgpu::BufferUsages::empty(),
        );
        let matrices = create_storage_buffer::<GpuCullMatrix>(
            device,
            "gpu_cull_matrices",
            1,
            wgpu::BufferUsages::empty(),
        );
        let output = create_storage_buffer::<IndexedIndirectCommand>(
            device,
            "gpu_cull_output",
            1,
            wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_SRC,
        );
        let counts = create_storage_buffer::<u32>(
            device,
            "gpu_cull_counts",
            1,
            wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_SRC,
        );
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_cull_params"),
            size: PARAMS_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let fallback_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gpu_cull_fallback_hiz"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let fallback_hiz = fallback_texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            candidates,
            runs,
            matrices,
            output,
            counts,
            params,
            candidate_capacity: 1,
            run_capacity: 1,
            matrix_capacity: 1,
            output_capacity: 1,
            count_capacity: 1,
            fallback_hiz,
            bind_groups: Vec::with_capacity(Self::MAX_BIND_GROUPS),
            uploaded_static_inputs: None,
        }
    }

    pub(crate) fn output_buffer(&self) -> &wgpu::Buffer {
        &self.output
    }

    pub(crate) fn count_buffer(&self) -> &wgpu::Buffer {
        &self.counts
    }

    pub(crate) fn run_draw(
        &self,
        run: &GpuCullRun,
        output_mode: GpuCullOutputMode,
    ) -> GpuCullRunDraw {
        GpuCullRunDraw {
            indirect_offset: u64::from(run.output_start) * INDIRECT_COMMAND_BYTES,
            count_offset: u64::from(run.count_index) * COUNT_BYTES,
            max_count: run.candidate_count.min(run.output_capacity),
            fixed_count: run.candidate_count.min(run.output_capacity),
            output_mode,
        }
    }

    /// Uploads inputs, clears compact-mode run counters, and records the culling dispatch.
    pub(crate) fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        uploads: GraphUploadSink<'_>,
        request: GpuCullEncode<'_>,
    ) -> Result<GpuCullDispatch, GpuCullError> {
        let candidate_count = u32::try_from(request.candidates.len())
            .map_err(|_| GpuCullError::CountOverflow("candidate"))?;
        let run_count =
            u32::try_from(request.runs.len()).map_err(|_| GpuCullError::CountOverflow("run"))?;
        let matrix_count = u32::try_from(request.matrices.len())
            .map_err(|_| GpuCullError::CountOverflow("matrix"))?;
        let workgroup_count = candidate_count.div_ceil(WORKGROUP_SIZE);
        let max_workgroups = device.limits().max_compute_workgroups_per_dimension;
        if workgroup_count > max_workgroups {
            return Err(GpuCullError::CapacityExceeded {
                resource: "dispatch workgroups",
                needed: u64::from(workgroup_count),
                maximum: u64::from(max_workgroups),
            });
        }
        let cached_static = request.static_input_generation.and_then(|generation| {
            self.uploaded_static_inputs.filter(|cached| {
                cached.matches(generation, candidate_count, run_count, matrix_count)
            })
        });
        let static_inputs_uploaded = cached_static.is_none();
        let layout = match cached_static {
            Some(cached) => cached.layout,
            None => validate_layout(request.candidates, request.runs, request.matrices.len())?,
        };

        self.ensure_capacities(
            device,
            candidate_count,
            run_count,
            matrix_count,
            layout.output_elements,
            layout.count_elements,
        )?;

        if static_inputs_uploaded {
            if !request.candidates.is_empty() {
                uploads.write_buffer(
                    &self.candidates,
                    0,
                    bytemuck::cast_slice(request.candidates),
                );
            }
            if !request.runs.is_empty() {
                uploads.write_buffer(&self.runs, 0, bytemuck::cast_slice(request.runs));
            }
            self.uploaded_static_inputs =
                request
                    .static_input_generation
                    .map(|generation| UploadedStaticInputs {
                        generation,
                        candidate_count,
                        run_count,
                        matrix_count,
                        layout,
                    });
        }
        if !request.matrices.is_empty() {
            uploads.write_buffer(&self.matrices, 0, bytemuck::cast_slice(request.matrices));
        }

        let (hiz_left, hiz_right, hiz_eye_mask, hiz_bias, hiz_enabled) =
            if let Some(hiz) = request.previous_hiz {
                (
                    hiz.left.clone(),
                    hiz.right.unwrap_or(&self.fallback_hiz).clone(),
                    1 | u32::from(hiz.right.is_some()) << 1,
                    if hiz.depth_bias.is_finite() {
                        hiz.depth_bias.max(0.0)
                    } else {
                        0.0
                    },
                    true,
                )
            } else {
                (
                    self.fallback_hiz.clone(),
                    self.fallback_hiz.clone(),
                    0,
                    0.0,
                    false,
                )
            };
        let params = GpuCullParams {
            candidate_count,
            run_count,
            matrix_count,
            output_capacity: self.output_capacity,
            count_capacity: self.count_capacity,
            flags: u32::from(request.output_mode == GpuCullOutputMode::FixedSlots)
                * PARAM_FIXED_SLOTS
                | u32::from(hiz_enabled) * PARAM_HIZ_ENABLED,
            hiz_eye_mask,
            hiz_depth_bias: hiz_bias,
        };
        uploads.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));

        if let Some(clear_bytes) =
            visible_count_clear_bytes(request.output_mode, layout.count_elements)
        {
            encoder.clear_buffer(&self.counts, 0, Some(clear_bytes));
        }

        self.ensure_bind_group(device, &hiz_left, &hiz_right);
        if workgroup_count > 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_cull_compact"),
                timestamp_writes: None,
            });
            pass.set_pipeline(gpu_cull_pipelines().pipeline(device));
            pass.set_bind_group(
                0,
                self.bind_groups.last().map(|entry| &entry.bind_group),
                &[],
            );
            pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        Ok(GpuCullDispatch {
            candidate_count,
            run_count,
            workgroup_count,
            output_mode: request.output_mode,
            static_inputs_uploaded,
        })
    }

    /// Retains every resource referenced by the last encoded command buffer for driver-thread
    /// submission. Call before this object can resize or rebind on the following frame.
    pub(crate) fn retain_submit_resources(&self, retained: &mut GpuRetainedResources) {
        retained.retain_buffers([
            self.candidates.clone(),
            self.runs.clone(),
            self.matrices.clone(),
            self.output.clone(),
            self.counts.clone(),
            self.params.clone(),
        ]);
        if let Some(entry) = self.bind_groups.last() {
            retained.retain_bind_group(entry.bind_group.clone());
        }
    }

    fn ensure_capacities(
        &mut self,
        device: &wgpu::Device,
        candidates: u32,
        runs: u32,
        matrices: u32,
        output: u32,
        counts: u32,
    ) -> Result<(), GpuCullError> {
        let mut changed = false;
        changed |= ensure_buffer_capacity::<GpuCullCandidate>(
            device,
            &mut self.candidates,
            &mut self.candidate_capacity,
            candidates,
            "candidates",
            "gpu_cull_candidates",
            wgpu::BufferUsages::empty(),
        )?;
        changed |= ensure_buffer_capacity::<GpuCullRun>(
            device,
            &mut self.runs,
            &mut self.run_capacity,
            runs,
            "runs",
            "gpu_cull_runs",
            wgpu::BufferUsages::empty(),
        )?;
        changed |= ensure_buffer_capacity::<GpuCullMatrix>(
            device,
            &mut self.matrices,
            &mut self.matrix_capacity,
            matrices,
            "matrices",
            "gpu_cull_matrices",
            wgpu::BufferUsages::empty(),
        )?;
        changed |= ensure_buffer_capacity::<IndexedIndirectCommand>(
            device,
            &mut self.output,
            &mut self.output_capacity,
            output,
            "output commands",
            "gpu_cull_output",
            wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_SRC,
        )?;
        changed |= ensure_buffer_capacity::<u32>(
            device,
            &mut self.counts,
            &mut self.count_capacity,
            counts,
            "run counts",
            "gpu_cull_counts",
            wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_SRC,
        )?;
        if changed {
            // Every group binds the old storage buffers, so a capacity change invalidates all
            // history variants at once.
            self.bind_groups.clear();
        }
        Ok(())
    }

    fn ensure_bind_group(
        &mut self,
        device: &wgpu::Device,
        hiz_left: &wgpu::TextureView,
        hiz_right: &wgpu::TextureView,
    ) {
        if let Some(index) = self
            .bind_groups
            .iter()
            .position(|entry| entry.hiz_left == *hiz_left && entry.hiz_right == *hiz_right)
        {
            if index + 1 != self.bind_groups.len() {
                let entry = self.bind_groups.remove(index);
                self.bind_groups.push(entry);
            }
            return;
        }
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_cull_compact"),
            layout: gpu_cull_pipelines().bind_group_layout(device),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.candidates.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.runs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.matrices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(hiz_left),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(hiz_right),
                },
            ],
        });
        if self.bind_groups.len() == Self::MAX_BIND_GROUPS {
            self.bind_groups.remove(0);
        }
        self.bind_groups.push(GpuCullBindGroupCacheEntry {
            hiz_left: hiz_left.clone(),
            hiz_right: hiz_right.clone(),
            bind_group,
        });
    }
}

fn extract_clip_planes(matrix: Mat4) -> [[f32; 4]; 6] {
    let columns = matrix.to_cols_array_2d();
    let row = |index: usize| {
        Vec4::new(
            columns[0][index],
            columns[1][index],
            columns[2][index],
            columns[3][index],
        )
    };
    let r0 = row(0);
    let r1 = row(1);
    let r2 = row(2);
    let r3 = row(3);
    [
        normalize_plane(r3 + r0),
        normalize_plane(r3 - r0),
        normalize_plane(r3 + r1),
        normalize_plane(r3 - r1),
        normalize_plane(r2),
        normalize_plane(r3 - r2),
    ]
}

fn normalize_plane(plane: Vec4) -> [f32; 4] {
    let length = plane.truncate().length();
    if length.is_finite() && length > f32::EPSILON {
        (plane / length).to_array()
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

fn validate_layout(
    candidates: &[GpuCullCandidate],
    runs: &[GpuCullRun],
    matrix_count: usize,
) -> Result<ValidatedLayout, GpuCullError> {
    let mut owners = vec![None; candidates.len()];
    let mut output_ranges = Vec::with_capacity(runs.len());
    let mut count_indices = Vec::with_capacity(runs.len());
    let mut output_elements = 0u32;
    let mut count_elements = 0u32;

    for (run_index, run) in runs.iter().enumerate() {
        if run.output_capacity < run.candidate_count {
            return Err(GpuCullError::OutputTooSmall {
                run: run_index,
                candidates: run.candidate_count,
                capacity: run.output_capacity,
            });
        }
        let candidate_end = run
            .candidate_start
            .checked_add(run.candidate_count)
            .ok_or(GpuCullError::RunCandidateRange { run: run_index })?;
        if candidate_end as usize > candidates.len() {
            return Err(GpuCullError::RunCandidateRange { run: run_index });
        }
        let output_end = run
            .output_start
            .checked_add(run.output_capacity)
            .ok_or(GpuCullError::RunOutputRange { run: run_index })?;
        output_elements = output_elements.max(output_end);
        count_elements = count_elements.max(run.count_index.saturating_add(1));

        for candidate_index in run.candidate_start..candidate_end {
            let candidate_index = candidate_index as usize;
            if owners[candidate_index].replace(run_index).is_some() {
                return Err(GpuCullError::CandidateOwnedTwice {
                    candidate: candidate_index,
                });
            }
            let candidate = &candidates[candidate_index];
            if candidate.run_index != run_index as u32 {
                return Err(GpuCullError::CandidateRunMismatch {
                    candidate: candidate_index,
                    expected: run_index,
                    actual: candidate.run_index,
                });
            }
        }
        output_ranges.push((run.output_start, output_end, run_index));
        count_indices.push((run.count_index, run_index));
    }

    for (candidate_index, candidate) in candidates.iter().enumerate() {
        if owners[candidate_index].is_none() {
            return Err(GpuCullError::CandidateNotOwned {
                candidate: candidate_index,
            });
        }
        if candidate.matrix_index as usize >= matrix_count {
            return Err(GpuCullError::CandidateMatrixOutOfRange {
                candidate: candidate_index,
                matrix: candidate.matrix_index,
                matrix_count,
            });
        }
        let valid_bounds = (0..3).all(|axis| {
            candidate.bounds_min[axis].is_finite()
                && candidate.bounds_max[axis].is_finite()
                && candidate.bounds_min[axis] <= candidate.bounds_max[axis]
        });
        if !valid_bounds {
            return Err(GpuCullError::InvalidBounds {
                candidate: candidate_index,
            });
        }
    }

    output_ranges.sort_unstable_by_key(|range| range.0);
    for ranges in output_ranges.windows(2) {
        if ranges[0].1 > ranges[1].0 {
            return Err(GpuCullError::OutputOverlap {
                first_run: ranges[0].2,
                second_run: ranges[1].2,
            });
        }
    }
    count_indices.sort_unstable();
    for indices in count_indices.windows(2) {
        if indices[0].0 == indices[1].0 {
            return Err(GpuCullError::CountIndexReused {
                first_run: indices[0].1,
                second_run: indices[1].1,
                count_index: indices[0].0,
            });
        }
    }

    Ok(ValidatedLayout {
        output_elements,
        count_elements,
    })
}

fn grow_capacity(current: u32, needed: u32, maximum: u32) -> Option<u32> {
    if needed > maximum {
        return None;
    }
    let mut capacity = current.max(1).min(maximum);
    while capacity < needed {
        capacity = capacity.saturating_mul(2).min(maximum);
    }
    Some(capacity)
}

fn visible_count_clear_bytes(output_mode: GpuCullOutputMode, count_elements: u32) -> Option<u64> {
    (output_mode == GpuCullOutputMode::CompactCount && count_elements != 0)
        .then_some(u64::from(count_elements) * COUNT_BYTES)
}

fn ensure_buffer_capacity<T>(
    device: &wgpu::Device,
    buffer: &mut wgpu::Buffer,
    capacity: &mut u32,
    needed: u32,
    resource: &'static str,
    label: &'static str,
    extra_usage: wgpu::BufferUsages,
) -> Result<bool, GpuCullError> {
    if needed <= *capacity {
        return Ok(false);
    }
    let stride = size_of::<T>() as u64;
    let limits = device.limits();
    let maximum_bytes = limits
        .max_buffer_size
        .min(limits.max_storage_buffer_binding_size);
    let maximum = (maximum_bytes / stride).min(u64::from(u32::MAX)) as u32;
    let Some(next) = grow_capacity(*capacity, needed, maximum) else {
        return Err(GpuCullError::CapacityExceeded {
            resource,
            needed: u64::from(needed),
            maximum: u64::from(maximum),
        });
    };
    *buffer = create_storage_buffer::<T>(device, label, next, extra_usage);
    *capacity = next;
    Ok(true)
}

fn create_storage_buffer<T>(
    device: &wgpu::Device,
    label: &'static str,
    capacity: u32,
    extra_usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: u64::from(capacity.max(1)) * size_of::<T>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra_usage,
        mapped_at_creation: false,
    });
    crate::profiling::note_resource_churn!(Buffer, "gpu::cull_compact_buffer");
    buffer
}

#[derive(Default)]
struct GpuCullPipelineCache {
    bind_group_layout: OnceGpu<wgpu::BindGroupLayout>,
    pipeline: OnceGpu<wgpu::ComputePipeline>,
}

impl GpuCullPipelineCache {
    fn bind_group_layout(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.bind_group_layout.get_or_create(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_cull_compact"),
                entries: &[
                    buffer_layout(0, wgpu::BufferBindingType::Uniform, PARAMS_BYTES),
                    buffer_layout(
                        1,
                        wgpu::BufferBindingType::Storage { read_only: true },
                        size_of::<GpuCullCandidate>() as u64,
                    ),
                    buffer_layout(
                        2,
                        wgpu::BufferBindingType::Storage { read_only: true },
                        size_of::<GpuCullRun>() as u64,
                    ),
                    buffer_layout(
                        3,
                        wgpu::BufferBindingType::Storage { read_only: true },
                        size_of::<GpuCullMatrix>() as u64,
                    ),
                    buffer_layout(
                        4,
                        wgpu::BufferBindingType::Storage { read_only: false },
                        INDIRECT_COMMAND_BYTES,
                    ),
                    buffer_layout(
                        5,
                        wgpu::BufferBindingType::Storage { read_only: false },
                        COUNT_BYTES,
                    ),
                    texture_layout(6),
                    texture_layout(7),
                ],
            })
        })
    }

    fn pipeline(&self, device: &wgpu::Device) -> &wgpu::ComputePipeline {
        self.pipeline.get_or_create(|| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpu_cull_compact"),
                bind_group_layouts: &[Some(self.bind_group_layout(device))],
                immediate_size: 0,
            });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpu_cull_compact"),
                source: wgpu::ShaderSource::Wgsl(embedded_wgsl!("gpu_cull_compact").into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gpu_cull_compact"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            crate::profiling::note_resource_churn!(ComputePipeline, "gpu::cull_compact_pipeline");
            pipeline
        })
    }
}

fn buffer_layout(
    binding: u32,
    ty: wgpu::BufferBindingType,
    minimum_size: u64,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(minimum_size),
        },
        count: None,
    }
}

fn texture_layout(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn gpu_cull_pipelines() -> &'static GpuCullPipelineCache {
    static CACHE: std::sync::LazyLock<GpuCullPipelineCache> =
        std::sync::LazyLock::new(GpuCullPipelineCache::default);
    &CACHE
}

#[cfg(test)]
mod tests {
    use super::*;

    fn command() -> IndexedIndirectCommand {
        IndexedIndirectCommand {
            index_count: 3,
            instance_count: 1,
            first_index: 0,
            base_vertex: 0,
            first_instance: 0,
        }
    }

    fn candidate(run: u32, matrix: u32) -> GpuCullCandidate {
        GpuCullCandidate::new(command(), run, matrix, [-1.0; 3], [1.0; 3])
    }

    #[test]
    fn host_layouts_match_wgsl_storage_strides() {
        assert_eq!(size_of::<GpuCullCandidate>(), 64);
        assert_eq!(size_of::<GpuCullRun>(), 32);
        assert_eq!(size_of::<GpuCullMatrix>(), 336);
        assert_eq!(size_of::<GpuCullParams>(), 32);
        assert_eq!(std::mem::offset_of!(GpuCullCandidate, command), 0);
        assert_eq!(std::mem::offset_of!(GpuCullCandidate, run_index), 20);
        assert_eq!(std::mem::offset_of!(GpuCullCandidate, matrix_index), 24);
        assert_eq!(std::mem::offset_of!(GpuCullCandidate, bounds_min), 32);
    }

    #[test]
    fn capacity_doubles_and_honours_maximum() {
        assert_eq!(grow_capacity(1, 1, 16), Some(1));
        assert_eq!(grow_capacity(1, 3, 16), Some(4));
        assert_eq!(grow_capacity(4, 9, 12), Some(12));
        assert_eq!(grow_capacity(4, 13, 12), None);
    }

    #[test]
    fn retained_static_input_generation_requires_exact_layout_identity() {
        let uploaded = UploadedStaticInputs {
            generation: 9,
            candidate_count: 120,
            run_count: 7,
            matrix_count: 2,
            layout: ValidatedLayout {
                output_elements: 120,
                count_elements: 7,
            },
        };

        assert!(uploaded.matches(9, 120, 7, 2));
        assert!(!uploaded.matches(10, 120, 7, 2));
        assert!(!uploaded.matches(9, 121, 7, 2));
        assert!(!uploaded.matches(9, 120, 8, 2));
        assert!(!uploaded.matches(9, 120, 7, 1));
    }

    #[test]
    fn candidate_visibility_flags_toggle_independently() {
        let candidate = candidate(0, 0)
            .with_previous_hiz(true)
            .with_always_visible(true);
        assert_eq!(
            candidate.flags,
            GPU_CULL_CANDIDATE_HIZ_ELIGIBLE | GPU_CULL_CANDIDATE_ALWAYS_VISIBLE
        );
        let candidate = candidate
            .with_previous_hiz(false)
            .with_always_visible(false);
        assert_eq!(candidate.flags, 0);
    }

    #[test]
    fn validates_runs_and_computes_sparse_offsets() {
        let candidates = [candidate(0, 0), candidate(0, 0), candidate(1, 0)];
        let runs = [
            GpuCullRun::new(0, 2, 4, 2, 2),
            GpuCullRun::new(2, 1, 9, 1, 6),
        ];
        assert_eq!(
            validate_layout(&candidates, &runs, 1),
            Ok(ValidatedLayout {
                output_elements: 10,
                count_elements: 7,
            })
        );
    }

    #[test]
    fn rejects_overlapping_output_runs() {
        let candidates = [candidate(0, 0), candidate(1, 0)];
        let runs = [
            GpuCullRun::new(0, 1, 2, 2, 0),
            GpuCullRun::new(1, 1, 3, 1, 1),
        ];
        assert!(matches!(
            validate_layout(&candidates, &runs, 1),
            Err(GpuCullError::OutputOverlap { .. })
        ));
    }

    #[test]
    fn run_draw_offsets_are_tightly_packed() {
        let run = GpuCullRun::new(0, 7, 11, 7, 3);
        let draw = GpuCullRunDraw {
            indirect_offset: u64::from(run.output_start) * INDIRECT_COMMAND_BYTES,
            count_offset: u64::from(run.count_index) * COUNT_BYTES,
            max_count: run.candidate_count,
            fixed_count: run.candidate_count,
            output_mode: GpuCullOutputMode::CompactCount,
        };
        assert_eq!(draw.indirect_offset, 220);
        assert_eq!(draw.count_offset, 12);
    }

    #[test]
    fn fixed_slots_dispatch_does_not_clear_visible_counts() {
        assert_eq!(
            visible_count_clear_bytes(GpuCullOutputMode::FixedSlots, 7),
            None
        );
    }

    #[test]
    fn compact_count_dispatch_clears_visible_counts() {
        assert_eq!(
            visible_count_clear_bytes(GpuCullOutputMode::CompactCount, 7),
            Some(7 * COUNT_BYTES)
        );
        assert_eq!(
            visible_count_clear_bytes(GpuCullOutputMode::CompactCount, 0),
            None
        );
    }

    #[test]
    fn extracted_planes_match_homogeneous_webgpu_clip_volume() {
        let projection =
            crate::camera::reverse_z_perspective(16.0 / 9.0, 60f32.to_radians(), 0.1, 100.0);
        let planes = extract_clip_planes(projection);
        let points = [
            Vec4::new(0.0, 0.0, -1.0, 1.0),
            Vec4::new(0.0, 0.0, -0.05, 1.0),
            Vec4::new(0.0, 0.0, -101.0, 1.0),
            Vec4::new(50.0, 0.0, -1.0, 1.0),
        ];
        for point in points {
            let clip = projection * point;
            let homogeneous_inside = clip.x >= -clip.w
                && clip.x <= clip.w
                && clip.y >= -clip.w
                && clip.y <= clip.w
                && clip.z >= 0.0
                && clip.z <= clip.w;
            let plane_inside = planes.iter().all(|plane| {
                Vec4::from_array(*plane).dot(Vec4::new(point.x, point.y, point.z, 1.0)) >= -1e-5
            });
            assert_eq!(plane_inside, homogeneous_inside, "{point:?} -> {clip:?}");
        }
    }

    #[test]
    fn matrix_builder_tracks_stereo_history_per_eye() {
        let previous = Mat4::from_translation(glam::Vec3::Z);
        let matrix = GpuCullMatrix::from_view_projections(
            &[Mat4::IDENTITY, Mat4::IDENTITY],
            &[Some(previous), None],
        )
        .unwrap();
        assert_eq!(matrix.eye_count, 2);
        assert_eq!(matrix.previous_valid_mask, 1);
        assert_eq!(matrix.previous_view_proj[0], previous.to_cols_array_2d());
        assert_eq!(matrix.previous_view_proj[1], [[0.0; 4]; 4]);
    }

    #[test]
    fn hiz_mip_rounds_up_so_four_corner_taps_cover_the_footprint() {
        // A 3.8-pixel footprint aligned this way spans three texels at floor(log2(3.8)) = 1.
        // Sampling only its endpoint columns misses the middle texel. At the ceil mip it spans
        // two columns, so the shader's four corner taps cover the complete 2D footprint.
        let start_px = 1.8_f32;
        let end_px = 5.6_f32;
        let extent_px = end_px - start_px;
        let floor_mip = extent_px.log2().floor() as u32;
        let ceil_mip = extent_px.log2().ceil() as u32;
        let touched_texels = |mip: u32| {
            let scale = (1_u32 << mip) as f32;
            (end_px / scale).floor() as i32 - (start_px / scale).floor() as i32 + 1
        };

        assert_eq!(floor_mip, 1);
        assert_eq!(touched_texels(floor_mip), 3);
        assert_eq!(ceil_mip, 2);
        assert_eq!(touched_texels(ceil_mip), 2);

        let source = include_str!("../../shaders/passes/compute/gpu_cull_compact.wgsl");
        assert!(
            source.contains("u32(ceil(log2(largest_extent)))"),
            "GPU Hi-Z mip selection must round up for conservative four-tap coverage"
        );
    }

    #[test]
    fn compute_shader_parses_and_validates() {
        let source = include_str!("../../shaders/passes/compute/gpu_cull_compact.wgsl");
        let module = naga::front::wgsl::parse_str(source).expect("GPU cull WGSL must parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("GPU cull WGSL must validate");

        let mut layouter = naga::proc::Layouter::default();
        layouter
            .update(module.to_ctx())
            .expect("GPU cull WGSL types must have valid layouts");
        for (name, expected_stride) in [
            ("GpuCullCandidate", size_of::<GpuCullCandidate>() as u32),
            ("GpuCullRun", size_of::<GpuCullRun>() as u32),
            ("GpuCullMatrix", size_of::<GpuCullMatrix>() as u32),
            ("GpuCullParams", size_of::<GpuCullParams>() as u32),
        ] {
            let (handle, _) = module
                .types
                .iter()
                .find(|(_, ty)| ty.name.as_deref() == Some(name))
                .unwrap_or_else(|| panic!("missing shader type {name}"));
            assert_eq!(
                layouter[handle].to_stride(),
                expected_stride,
                "{name} host/shader stride mismatch"
            );
        }
    }

    #[test]
    fn shader_counter_access_is_compact_count_only() {
        let source = include_str!("../../shaders/passes/compute/gpu_cull_compact.wgsl");
        let fixed_start = source
            .find("if (params.flags & PARAM_FIXED_SLOTS) != 0u {")
            .expect("fixed-slot shader branch");
        let compact_start = source[fixed_start..]
            .find("if !visible || run.count_index >= params.count_capacity {")
            .map(|offset| fixed_start + offset)
            .expect("compact-count shader branch");
        let fixed_branch = &source[fixed_start..compact_start];
        let compact_branch = &source[compact_start..];

        assert!(!fixed_branch.contains("visible_counts"));
        assert!(!fixed_branch.contains("run.count_index"));
        assert!(compact_branch.contains("run.count_index >= params.count_capacity"));
        assert!(compact_branch.contains("atomicAdd(&visible_counts[run.count_index], 1u)"));
    }
}
