//! Per-view command recording fan-out helpers.

use super::{
    CompiledRenderGraph, GraphExecuteError, PerViewRecordInputs, PerViewRecordOutput,
    PerViewWorkItem,
};
use crate::cpu_parallelism::{
    FrameCpuWorkload, FrameParallelPolicy, ParallelAdmission, record_parallel_admission,
};

/// Per-view work items assigned to one recording worker.
const PER_VIEW_RECORD_PARALLEL_CHUNK_VIEWS: usize = 1;
/// Minimum actual draw work before automatic per-view command recording uses Rayon.
const PER_VIEW_RECORD_PARALLEL_MIN_DRAWS: usize = 512;
/// Draw-equivalent work assigned to each per-view graph pass during recording admission.
const PER_VIEW_RECORD_PASS_DRAW_EQUIVALENT: usize = 16;

impl CompiledRenderGraph {
    /// Computes per-view recording work and Rayon admission for a prepared work-item batch.
    pub(super) fn per_view_record_admission_for_work_items(
        &self,
        per_view_work_items: &[PerViewWorkItem],
        n_views: usize,
    ) -> (usize, usize, ParallelAdmission) {
        let total_draw_count = per_view_work_items
            .iter()
            .map(|work_item| work_item.estimated_draw_count)
            .sum::<usize>();
        let estimated_record_work =
            per_view_record_draw_equivalent(n_views, total_draw_count, self.pass_count());
        let admission = per_view_record_admission(
            FrameParallelPolicy::for_current_thread_pool(),
            n_views,
            total_draw_count,
            PER_VIEW_RECORD_PARALLEL_CHUNK_VIEWS,
        );
        (total_draw_count, estimated_record_work, admission)
    }

    /// Drives the per-view recording phase serially for a single view or across Rayon workers for
    /// multi-view batches, returning one [`PerViewRecordOutput`] per input work item in submission
    /// order.
    pub(super) fn record_per_view_outputs(
        &self,
        per_view_work_items: Vec<PerViewWorkItem>,
        inputs: PerViewRecordInputs<'_>,
        n_views: usize,
        estimated_record_work: usize,
        admission: ParallelAdmission,
    ) -> Result<Vec<PerViewRecordOutput>, GraphExecuteError> {
        profiling::scope!("graph::record_per_view_outputs");
        record_parallel_admission(
            "graph_record_per_view",
            estimated_record_work,
            n_views,
            admission,
        );
        if admission.is_parallel() {
            self.record_per_view_outputs_parallel(
                per_view_work_items,
                inputs,
                n_views,
                admission.chunk_size().unwrap_or(1),
            )
        } else {
            self.record_per_view_outputs_serial(per_view_work_items, inputs, n_views)
        }
    }

    fn record_per_view_outputs_parallel(
        &self,
        per_view_work_items: Vec<PerViewWorkItem>,
        inputs: PerViewRecordInputs<'_>,
        n_views: usize,
        parallel_chunk_views: usize,
    ) -> Result<Vec<PerViewRecordOutput>, GraphExecuteError> {
        profiling::scope!("graph::per_view_fan_out");
        if n_views == 2 {
            return self.record_two_view_outputs_parallel(per_view_work_items, inputs);
        }

        let PerViewRecordInputs {
            transient_by_key,
            upload_batch,
            per_view_shared,
            strategy,
            profiler,
        } = inputs;
        let indexed_outputs: Vec<(usize, PerViewRecordOutput)> = {
            profiling::scope!("graph::per_view_fan_out::spawn_workers");
            use rayon::prelude::*;
            per_view_work_items
                .into_par_iter()
                .with_min_len(parallel_chunk_views)
                .map(|work_item| {
                    profiling::scope!("graph::per_view_fan_out::worker");
                    self.record_per_view_work_item_output(
                        work_item,
                        transient_by_key,
                        upload_batch,
                        per_view_shared,
                        strategy,
                        profiler,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        {
            profiling::scope!("graph::per_view_fan_out::collect_outputs");
            collect_ordered_per_view_outputs(indexed_outputs, n_views)
        }
    }

    fn record_two_view_outputs_parallel(
        &self,
        per_view_work_items: Vec<PerViewWorkItem>,
        inputs: PerViewRecordInputs<'_>,
    ) -> Result<Vec<PerViewRecordOutput>, GraphExecuteError> {
        profiling::scope!("graph::per_view_fan_out::two_view_join");
        let PerViewRecordInputs {
            transient_by_key,
            upload_batch,
            per_view_shared,
            strategy,
            profiler,
        } = inputs;
        let mut items = per_view_work_items.into_iter();
        let first = items.next().ok_or(GraphExecuteError::NoViewsInBatch)?;
        let second = items.next().ok_or(GraphExecuteError::NoViewsInBatch)?;
        let extra = items.next();
        debug_assert!(extra.is_none());
        let (first, second) = rayon::join(
            || {
                profiling::scope!("graph::per_view_fan_out::two_view_worker");
                self.record_per_view_work_item_output(
                    first,
                    transient_by_key,
                    upload_batch,
                    per_view_shared,
                    strategy,
                    profiler,
                )
            },
            || {
                profiling::scope!("graph::per_view_fan_out::two_view_worker");
                self.record_per_view_work_item_output(
                    second,
                    transient_by_key,
                    upload_batch,
                    per_view_shared,
                    strategy,
                    profiler,
                )
            },
        );
        let (first_idx, first_output) = first?;
        let (second_idx, second_output) = second?;
        debug_assert_ne!(first_idx, second_idx);
        Ok(if first_idx <= second_idx {
            vec![first_output, second_output]
        } else {
            vec![second_output, first_output]
        })
    }

    fn record_per_view_outputs_serial(
        &self,
        per_view_work_items: Vec<PerViewWorkItem>,
        inputs: PerViewRecordInputs<'_>,
        n_views: usize,
    ) -> Result<Vec<PerViewRecordOutput>, GraphExecuteError> {
        profiling::scope!("graph::per_view_serial");
        let PerViewRecordInputs {
            transient_by_key,
            upload_batch,
            per_view_shared,
            strategy,
            profiler,
        } = inputs;
        let mut outputs = Vec::with_capacity(n_views);
        for work_item in per_view_work_items {
            let (_, output) = self.record_per_view_work_item_output(
                work_item,
                transient_by_key,
                upload_batch,
                per_view_shared,
                strategy,
                profiler,
            )?;
            outputs.push(output);
        }
        Ok(outputs)
    }
}

fn per_view_record_admission(
    policy: FrameParallelPolicy,
    view_count: usize,
    total_draw_count: usize,
    chunk_views: usize,
) -> ParallelAdmission {
    if total_draw_count < per_view_record_parallel_min_draws(policy) {
        return ParallelAdmission::Serial;
    }
    policy.admit_draw_heavy_views(
        FrameCpuWorkload::view_draws(view_count, total_draw_count),
        chunk_views,
    )
}

fn per_view_record_parallel_min_draws(policy: FrameParallelPolicy) -> usize {
    policy
        .draw_heavy_threshold()
        .max(PER_VIEW_RECORD_PARALLEL_MIN_DRAWS)
}

fn per_view_record_draw_equivalent(
    view_count: usize,
    total_draw_count: usize,
    pass_count: usize,
) -> usize {
    let pass_record_work = view_count
        .saturating_mul(pass_count)
        .saturating_mul(PER_VIEW_RECORD_PASS_DRAW_EQUIVALENT);
    total_draw_count.saturating_add(pass_record_work)
}

/// Returns outputs sorted by view index, validating that every expected view was recorded once.
fn collect_ordered_per_view_outputs(
    mut indexed_outputs: Vec<(usize, PerViewRecordOutput)>,
    n_views: usize,
) -> Result<Vec<PerViewRecordOutput>, GraphExecuteError> {
    if indexed_outputs.len() != n_views {
        return Err(GraphExecuteError::NoViewsInBatch);
    }
    {
        profiling::scope!("graph::per_view_fan_out::collect_outputs::sort");
        indexed_outputs.sort_unstable_by_key(|(view_idx, _)| *view_idx);
    }
    {
        profiling::scope!("graph::per_view_fan_out::collect_outputs::validate");
        if indexed_outputs
            .iter()
            .enumerate()
            .any(|(expected, (view_idx, _))| *view_idx != expected)
        {
            return Err(GraphExecuteError::NoViewsInBatch);
        }
    }
    Ok(indexed_outputs
        .into_iter()
        .map(|(_, output)| output)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_view_record_admission_keeps_single_view_serial() {
        let admission = per_view_record_admission(FrameParallelPolicy::new(4), 1, usize::MAX, 1);

        assert_eq!(admission, ParallelAdmission::Serial);
    }

    #[test]
    fn per_view_record_admission_keeps_pass_only_two_view_graphs_serial() {
        let estimated_record_work = per_view_record_draw_equivalent(2, 0, 3);
        let admission = per_view_record_admission(FrameParallelPolicy::new(4), 2, 0, 1);

        assert_eq!(estimated_record_work, 96);
        assert_eq!(admission, ParallelAdmission::Serial);
    }

    #[test]
    fn per_view_record_admission_counts_pass_recording_work_for_diagnostics_only() {
        let estimated_record_work = per_view_record_draw_equivalent(2, 0, 4);
        let admission = per_view_record_admission(FrameParallelPolicy::new(4), 2, 0, 1);

        assert_eq!(estimated_record_work, 128);
        assert_eq!(admission, ParallelAdmission::Serial);
    }

    #[test]
    fn per_view_record_admission_counts_draw_work() {
        let estimated_record_work = per_view_record_draw_equivalent(2, 512, 0);
        let admission = per_view_record_admission(FrameParallelPolicy::new(4), 2, 512, 1);

        assert_eq!(estimated_record_work, 512);
        assert_eq!(admission, ParallelAdmission::Parallel { chunk_size: 1 });
    }

    #[test]
    fn per_view_record_admission_uses_command_recording_draw_floor() {
        assert_eq!(
            per_view_record_parallel_min_draws(FrameParallelPolicy::new(4)),
            512
        );
        assert_eq!(
            per_view_record_parallel_min_draws(FrameParallelPolicy::new(64)),
            512
        );
    }
}
