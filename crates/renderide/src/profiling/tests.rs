use super::*;

/// Verifies that `rayon_thread_start_handler` produces a valid closure that does not panic
/// when called with arbitrary thread indices.
#[test]
fn rayon_start_handler_does_not_panic_for_any_index() {
    let handler = rayon_thread_start_handler();
    handler(0);
    handler(1);
    handler(usize::MAX);
}

/// Confirms that the public surface of this module compiles and is callable without the
/// `tracy` feature active. All calls must be no-ops; the test itself is the compile check.
#[cfg(not(feature = "tracy"))]
#[test]
fn stubs_are_accessible_without_tracy_feature() {
    register_main_thread();
    emit_frame_mark();
    emit_render_submit_frame_mark();
    plot_fps_cap_active(240);
    plot_window_focused(true);
    plot_surface_acquire_outcome(true, false, false);
    plot_surface_in_flight_count(1);
    plot_surface_previous_present_wait_ms(std::time::Duration::from_millis(2));
    plot_surface_get_current_texture_ms(std::time::Duration::from_millis(3));
    plot_event_loop_wait_ms(11.0);
    plot_event_loop_idle_ms(11.0);
    plot_render_world_maintenance(crate::world_mesh::RenderWorldMaintenanceStats::default());
    plot_world_mesh_prepare(10, 4, 3);
    plot_ipc_poll(&IpcPollProfileSample {
        waited: std::time::Duration::from_millis(1),
        messages: 2,
        bytes: 128,
        decode_duration: std::time::Duration::from_micros(20),
        timed_out: false,
    });
    static CHURN_SITE: ResourceChurnSite =
        ResourceChurnSite::new(ResourceChurnKind::Buffer, "profiling_no_tracy_stub_test");
    CHURN_SITE.note();
    flush_resource_churn_plots();
    let profiler = GpuProfilerHandle;
    assert!(!profiler.has_queries_opened_since_frame_end());
    assert!(!profiler.end_frame_if_queries_opened(1));
    let _ = rayon_thread_start_handler();
}

/// Verifies that `timestamp_query_features_if_supported` has the correct function signature
/// and can be referenced as a function pointer when the `tracy` feature is off.
///
/// The `cfg(not(feature = "tracy"))` branch returns `wgpu::Features::empty()` without ever
/// calling `adapter.features()`, so no real wgpu instance is required.
#[cfg(not(feature = "tracy"))]
#[test]
fn timestamp_features_fn_signature_compiles_without_tracy() {
    let _: fn(&wgpu::Adapter) -> wgpu::Features = timestamp_query_features_if_supported;
}

/// `register_main_thread` and `emit_frame_mark` must be safely callable more than once per
/// process; calling them repeatedly should never panic under any feature configuration.
#[test]
fn thread_registration_and_frame_mark_are_idempotent() {
    register_main_thread();
    register_main_thread();
    emit_frame_mark();
    emit_frame_mark();
}

/// The no-tracy [`PhaseQuery`] placeholder is zero-sized so its presence in per-phase structs
/// cannot regress memory layout when profiling is disabled.
#[cfg(not(feature = "tracy"))]
#[test]
fn phase_query_stub_is_zero_sized() {
    assert_eq!(size_of::<PhaseQuery>(), 0);
}

/// The no-tracy [`GpuProfilerHandle`] stub is also zero-sized; construction is unreachable via
/// [`GpuProfilerHandle::try_new`] (always returns [`None`]), so the placeholder must stay free.
#[cfg(not(feature = "tracy"))]
#[test]
fn gpu_profiler_handle_stub_is_zero_sized() {
    assert_eq!(size_of::<GpuProfilerHandle>(), 0);
}

/// The no-tracy `render_pass_timestamp_writes` helper must always return `None` regardless
/// of what `query` is -- the `PhaseQuery` placeholder carries no data to reserve writes from.
#[cfg(not(feature = "tracy"))]
#[test]
fn render_pass_timestamp_writes_is_none_without_tracy() {
    let q = PhaseQuery;
    assert!(render_pass_timestamp_writes(Some(&q)).is_none());
    assert!(render_pass_timestamp_writes(None).is_none());
}

/// The no-tracy `compute_pass_timestamp_writes` helper must always return `None`.
#[cfg(not(feature = "tracy"))]
#[test]
fn compute_pass_timestamp_writes_is_none_without_tracy() {
    let q = PhaseQuery;
    assert!(compute_pass_timestamp_writes(Some(&q)).is_none());
    assert!(compute_pass_timestamp_writes(None).is_none());
}

/// GPU profiler snapshots default to an empty, zero-query frame for HUD startup.
#[test]
fn gpu_profiler_snapshot_default_is_empty() {
    let snapshot = GpuProfilerSnapshot::default();
    assert!(snapshot.entries.is_empty());
    assert_eq!(snapshot.stats.opened_queries, 0);
    assert_eq!(snapshot.stats.skipped_queries, 0);
}

/// Snapshot construction preserves timing rows and query accounting.
#[test]
fn gpu_profiler_snapshot_preserves_stats() {
    let stats = GpuProfilerFrameStats {
        frame_order: 1,
        opened_queries: 12,
        skipped_queries: 1,
        soft_query_budget: 64,
    };
    let snapshot = GpuProfilerSnapshot {
        entries: vec![GpuPassEntry {
            name: "test_pass".to_owned(),
            ms: 0.5,
            depth: 2,
        }],
        stats,
    };
    assert_eq!(snapshot.entries.len(), 1);
    assert_eq!(snapshot.entries[0].name, "test_pass");
    assert_eq!(snapshot.stats, stats);
}
