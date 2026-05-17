//! Shared mip dispatch helpers for IBL compute passes.

use crate::profiling::{GpuProfilerHandle, compute_pass_timestamp_writes};

use super::key::dispatch_groups;
use super::pipeline::ComputePipeline;

/// Runs the boilerplate of one mip-0 compute pass: profiler query, dispatch, query close.
pub(super) fn dispatch_mip0_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &ComputePipeline,
    bind_group: &wgpu::BindGroup,
    face_size: u32,
    pass_label: &'static str,
    profiler: Option<&GpuProfilerHandle>,
    profiler_label: &'static str,
) {
    let pass_query = profiler.map(|p| p.begin_pass_query(profiler_label, encoder));
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(pass_label),
            timestamp_writes: compute_pass_timestamp_writes(pass_query.as_ref()),
        });
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(dispatch_groups(face_size), dispatch_groups(face_size), 6);
    }
    if let (Some(p), Some(q)) = (profiler, pass_query) {
        p.end_query(encoder, q);
    }
}
