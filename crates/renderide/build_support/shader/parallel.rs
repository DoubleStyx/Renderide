//! Parallel shader job execution with Cargo jobserver awareness.

use std::io::ErrorKind;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use super::compose::compile_shader_job;
use super::error::BuildError;
use super::model::{CompiledShader, ShaderJob};
use super::modules::ShaderModuleSources;

/// Conservative non-jobserver worker cap used when no stronger Cargo parallelism signal exists.
const FALLBACK_LOCAL_SHADER_WORKERS: usize = 4;

/// Returns the total worker count, including the main thread, for shader composition.
fn configured_shader_worker_limit(job_count: usize) -> usize {
    if job_count == 0 {
        return 0;
    }

    let requested = std::env::var("NUM_JOBS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .or_else(|| {
            std::thread::available_parallelism()
                .ok()
                .map(NonZeroUsize::get)
        })
        .unwrap_or(FALLBACK_LOCAL_SHADER_WORKERS);

    requested
        .clamp(1, FALLBACK_LOCAL_SHADER_WORKERS)
        .min(job_count)
}

/// Connects to Cargo's inherited jobserver when one is available for this build script.
fn inherited_jobserver_client() -> Option<jobserver::Client> {
    // SAFETY: `build.rs` reads the inherited Cargo jobserver immediately during shader compilation,
    // before this code path opens any other file descriptors. That matches `jobserver`'s safety
    // contract for taking ownership of the inherited handles.
    unsafe { jobserver::Client::from_env() }
}

/// Waits until an additional worker thread may consume CPU time under Cargo's jobserver budget.
fn wait_for_worker_token(
    client: &jobserver::Client,
    total_jobs: usize,
    next_job: &AtomicUsize,
    cancelled: &AtomicBool,
) -> Result<Option<jobserver::Acquired>, BuildError> {
    loop {
        if cancelled.load(Ordering::Acquire) || next_job.load(Ordering::Acquire) >= total_jobs {
            return Ok(None);
        }
        match client.try_acquire() {
            Ok(Some(token)) => return Ok(Some(token)),
            Ok(None) => std::thread::yield_now(),
            Err(err) if err.kind() == ErrorKind::Unsupported => return Ok(None),
            Err(err) => {
                return Err(BuildError::Message(format!(
                    "acquire renderide shader build jobserver token: {err}"
                )));
            }
        }
    }
}

/// Returns the next shader job index, or `None` when work is exhausted or cancelled.
fn next_shader_job(
    total_jobs: usize,
    next_job: &AtomicUsize,
    cancelled: &AtomicBool,
) -> Option<usize> {
    if cancelled.load(Ordering::Acquire) {
        return None;
    }
    let job_index = next_job.fetch_add(1, Ordering::AcqRel);
    (job_index < total_jobs).then_some(job_index)
}

/// Locks a mutex while ignoring poisoning so worker panics do not hide the original build error.
fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Stores the first build error and requests that all workers stop after their current job.
fn record_build_error(
    first_error: &Mutex<Option<BuildError>>,
    cancelled: &AtomicBool,
    error: BuildError,
) {
    let mut slot = lock_unpoisoned(first_error);
    if slot.is_none() {
        *slot = Some(error);
    }
    drop(slot);
    cancelled.store(true, Ordering::Release);
}

/// Compiles shader jobs on one worker lane, optionally waiting for a jobserver token first.
fn compile_shader_worker(
    modules: &ShaderModuleSources,
    jobs: &[ShaderJob],
    next_job: &AtomicUsize,
    cancelled: &AtomicBool,
    results: &Mutex<Vec<Option<CompiledShader>>>,
    first_error: &Mutex<Option<BuildError>>,
    inherited_jobserver: Option<&jobserver::Client>,
) {
    let _token = match inherited_jobserver {
        Some(client) => match wait_for_worker_token(client, jobs.len(), next_job, cancelled) {
            Ok(Some(token)) => Some(token),
            Ok(None) => return,
            Err(err) => {
                record_build_error(first_error, cancelled, err);
                return;
            }
        },
        None => None,
    };

    while let Some(job_index) = next_shader_job(jobs.len(), next_job, cancelled) {
        match compile_shader_job(modules, &jobs[job_index]) {
            Ok(compiled) => {
                let mut slots = lock_unpoisoned(results);
                slots[job_index] = Some(compiled);
            }
            Err(err) => {
                record_build_error(first_error, cancelled, err);
                return;
            }
        }
    }
}

/// Compiles all discovered shader jobs while keeping output order deterministic.
pub(super) fn compile_shader_jobs(
    modules: &ShaderModuleSources,
    jobs: &[ShaderJob],
) -> Result<Vec<CompiledShader>, BuildError> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }

    let worker_limit = configured_shader_worker_limit(jobs.len());
    if worker_limit <= 1 {
        let mut compiled = jobs
            .iter()
            .map(|job| compile_shader_job(modules, job))
            .collect::<Result<Vec<_>, _>>()?;
        sort_compiled_shader_results(&mut compiled);
        return Ok(compiled);
    }

    let inherited_jobserver = inherited_jobserver_client();
    let next_job = AtomicUsize::new(0);
    let cancelled = AtomicBool::new(false);
    let results = Mutex::new(
        std::iter::repeat_with(|| None)
            .take(jobs.len())
            .collect::<Vec<Option<CompiledShader>>>(),
    );
    let first_error = Mutex::new(None);

    std::thread::scope(|scope| {
        let next_job_ref = &next_job;
        let cancelled_ref = &cancelled;
        let results_ref = &results;
        let first_error_ref = &first_error;
        for _ in 1..worker_limit {
            let inherited_jobserver = inherited_jobserver.as_ref();
            scope.spawn(move || {
                compile_shader_worker(
                    modules,
                    jobs,
                    next_job_ref,
                    cancelled_ref,
                    results_ref,
                    first_error_ref,
                    inherited_jobserver,
                );
            });
        }

        compile_shader_worker(
            modules,
            jobs,
            next_job_ref,
            cancelled_ref,
            results_ref,
            first_error_ref,
            None,
        );
    });

    let first_error = lock_unpoisoned(&first_error).take();
    if let Some(err) = first_error {
        return Err(err);
    }

    let mut compiled = results
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .into_iter()
        .enumerate()
        .map(|(job_index, result)| {
            result.ok_or_else(|| {
                BuildError::Message(format!(
                    "parallel shader compilation did not produce a result for job {job_index}"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    sort_compiled_shader_results(&mut compiled);
    Ok(compiled)
}

/// Sorts compiled shader results back into the serial discovery order.
pub(super) fn sort_compiled_shader_results(results: &mut [CompiledShader]) {
    results.sort_by_key(|result| result.compile_order);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shader::model::{CompiledShaderTarget, ShaderSourceClass};

    /// Sort preserves source discovery order even when workers finish out of order.
    #[test]
    fn compiled_shader_results_sort_by_compile_order() {
        let mut compiled = vec![
            fake_compiled_shader(2, ShaderSourceClass::Present, "gamma"),
            fake_compiled_shader(0, ShaderSourceClass::Material, "alpha"),
            fake_compiled_shader(1, ShaderSourceClass::Post, "beta"),
        ];

        sort_compiled_shader_results(&mut compiled);

        let stems = compiled
            .iter()
            .map(|compiled| compiled.targets[0].target_stem.as_str())
            .collect::<Vec<_>>();
        assert_eq!(stems, ["alpha", "beta", "gamma"]);
    }

    fn fake_compiled_shader(
        compile_order: usize,
        source_class: ShaderSourceClass,
        target_stem: &str,
    ) -> CompiledShader {
        CompiledShader {
            compile_order,
            source_class,
            pass_directives: Vec::new(),
            targets: vec![CompiledShaderTarget {
                target_stem: target_stem.to_string(),
                wgsl: "wgsl".to_string(),
                pass_directives: Vec::new(),
            }],
        }
    }
}
