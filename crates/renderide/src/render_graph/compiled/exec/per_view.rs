//! Per-view command recording fan-out helpers.

use super::{
    CompiledRenderGraph, GraphExecuteError, PerViewRecordInputs, PerViewRecordOutput,
    PerViewWorkItem,
};

/// Per-view work items assigned to one recording worker.
const PER_VIEW_RECORD_PARALLEL_CHUNK_VIEWS: usize = 1;
/// View count at which per-view command recording fans out.
const MIN_VIEWS_FOR_PARALLEL_RECORD: usize = PER_VIEW_RECORD_PARALLEL_CHUNK_VIEWS * 2;

impl CompiledRenderGraph {
    /// Drives the per-view recording phase serially for a single view or across Rayon workers for
    /// multi-view batches, returning one [`PerViewRecordOutput`] per input work item in submission
    /// order.
    pub(super) fn record_per_view_outputs(
        &self,
        per_view_work_items: Vec<PerViewWorkItem>,
        inputs: PerViewRecordInputs<'_>,
        n_views: usize,
    ) -> Result<Vec<PerViewRecordOutput>, GraphExecuteError> {
        profiling::scope!("graph::record_per_view_outputs");
        // One view records serially. Two or more independent views can use worker threads, which
        // lets OpenXR stereo record both eyes in parallel.
        if n_views >= MIN_VIEWS_FOR_PARALLEL_RECORD {
            self.record_per_view_outputs_parallel(
                per_view_work_items,
                inputs,
                n_views,
                PER_VIEW_RECORD_PARALLEL_CHUNK_VIEWS,
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
            profiler,
        } = inputs;
        let mut outputs = Vec::with_capacity(n_views);
        for work_item in per_view_work_items {
            let (_, output) = self.record_per_view_work_item_output(
                work_item,
                transient_by_key,
                upload_batch,
                per_view_shared,
                profiler,
            )?;
            outputs.push(output);
        }
        Ok(outputs)
    }
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
