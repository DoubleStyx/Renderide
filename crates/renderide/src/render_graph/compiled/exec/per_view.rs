//! Per-view command recording fan-out helpers.

use super::{
    CompiledRenderGraph, GraphExecuteError, PerViewRecordInputs, PerViewRecordOutput,
    PerViewWorkItem,
};

impl CompiledRenderGraph {
    /// Drives the per-view recording phase either serially or across a `rayon::scope` fan-out,
    /// returning one [`PerViewRecordOutput`] per input work item in submission order.
    ///
    /// `PerViewParallel` scaffolding is in place across the pass traits, frame-params split,
    /// upload batch, and transient resolve; remaining serialization points around shared backend
    /// systems are gated in [`super::super::record_parallel`].
    pub(super) fn record_per_view_outputs(
        &self,
        per_view_work_items: Vec<PerViewWorkItem>,
        inputs: PerViewRecordInputs<'_>,
        record_parallelism: crate::config::RecordParallelism,
        n_views: usize,
    ) -> Result<Vec<PerViewRecordOutput>, GraphExecuteError> {
        profiling::scope!("graph::record_per_view_outputs");
        // One view records serially. Two or more independent views can use worker threads, which
        // lets OpenXR stereo record both eyes in parallel.
        const MIN_VIEWS_FOR_PARALLEL_RECORD: usize = 2;
        if record_parallelism == crate::config::RecordParallelism::PerViewParallel
            && n_views >= MIN_VIEWS_FOR_PARALLEL_RECORD
        {
            self.record_per_view_outputs_parallel(per_view_work_items, inputs, n_views)
        } else {
            self.record_per_view_outputs_serial(per_view_work_items, inputs, n_views)
        }
    }

    fn record_per_view_outputs_parallel(
        &self,
        per_view_work_items: Vec<PerViewWorkItem>,
        inputs: PerViewRecordInputs<'_>,
        n_views: usize,
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
        use std::sync::OnceLock;
        // Per-slot mailbox: each rayon worker writes its own `view_idx` exactly once via
        // `OnceLock::set`, which is wait-free in the uncontended case (single atomic CAS).
        // Replaces the prior `parking_lot::Mutex<Vec<Option<_>>>` that allocated a Vec
        // and acquired a global lock for every successful write.
        let slots: Vec<OnceLock<PerViewRecordOutput>> = {
            profiling::scope!("graph::per_view_fan_out::setup_mailboxes");
            std::iter::repeat_with(OnceLock::new)
                .take(n_views)
                .collect()
        };
        let first_error: OnceLock<GraphExecuteError> = OnceLock::new();
        {
            profiling::scope!("graph::per_view_fan_out::spawn_workers");
            rayon::scope(|scope| {
                for work_item in per_view_work_items {
                    let slots = &slots;
                    let first_error = &first_error;
                    let shared = per_view_shared;
                    scope.spawn(move |_| {
                        if first_error.get().is_some() {
                            return;
                        }
                        match self.record_per_view_work_item_output(
                            work_item,
                            transient_by_key,
                            upload_batch,
                            shared,
                            profiler,
                        ) {
                            Ok((view_idx, output)) => {
                                // Each rayon worker owns a unique `view_idx`, so this `set`
                                // always succeeds on the happy path; the `Err` arm is dead code
                                // in practice but discarded silently to keep the closure infallible.
                                let _ = slots[view_idx].set(output);
                            }
                            Err(err) => {
                                // Only the first erroring worker wins; later sets discard.
                                let _ = first_error.set(err);
                            }
                        }
                    });
                }
            });
        }
        {
            profiling::scope!("graph::per_view_fan_out::collect_outputs");
            if let Some(err) = first_error.into_inner() {
                return Err(err);
            }
            slots
                .into_iter()
                .map(|slot| slot.into_inner().ok_or(GraphExecuteError::NoViewsInBatch))
                .collect::<Result<Vec<_>, _>>()
        }
    }

    fn record_two_view_outputs_parallel(
        &self,
        per_view_work_items: Vec<PerViewWorkItem>,
        inputs: PerViewRecordInputs<'_>,
    ) -> Result<Vec<PerViewRecordOutput>, GraphExecuteError> {
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
                self.record_per_view_work_item_output(
                    first,
                    transient_by_key,
                    upload_batch,
                    per_view_shared,
                    profiler,
                )
            },
            || {
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
