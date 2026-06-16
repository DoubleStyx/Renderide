use parking_lot::Mutex;

use super::state::reset_for_test;
use super::{
    CpuRenderPhase, CrashContextSnapshot, DriverStage, GraphErrorKind, InitState, OpenXrCall,
    RenderMode, TargetMode, TickPhase, XrFinalizeKind, clear_openxr_call_if,
    clear_xr_finalize_state, format_snapshot_from, set_cpu_render_phase, set_driver_stage,
    set_init_state, set_last_graph_error, set_last_host_frame_index, set_openxr_call,
    set_prepared_view_count, set_render_mode, set_target_mode, set_tick_phase,
    set_xr_finalize_state, snapshot, write_minimal_snapshot,
};

static CRASH_CONTEXT_TEST_LOCK: Mutex<()> = Mutex::new(());

fn lock_reset_crash_context() -> parking_lot::MutexGuard<'static, ()> {
    let guard = CRASH_CONTEXT_TEST_LOCK.lock();
    reset_for_test();
    guard
}

#[test]
fn snapshot_format_includes_high_value_fields() {
    let s = CrashContextSnapshot {
        uptime_ms: 123,
        tick_sequence: 45,
        tick_phase: TickPhase::RenderViews,
        cpu_render_phase: CpuRenderPhase::CommandRecord,
        render_mode: RenderMode::HmdMultiview,
        init_state: InitState::Finalized,
        target_mode: TargetMode::OpenXr,
        last_host_frame_index: 9001,
        prepared_view_count: 3,
        primary_ipc_drop_streak: 2,
        background_ipc_drop_streak: 1,
        driver_backlog: 4,
        last_graph_error: GraphErrorKind::Pass,
        driver_stage: DriverStage::XrFinalizeStart,
        openxr_call: OpenXrCall::EndFrameProjection,
        xr_finalize_kind: XrFinalizeKind::Projection,
        xr_finalize_image_index: Some(2),
        xr_finalize_frame_seq: 88,
        xr_finalize_command_buffers: 4,
        xr_finalize_extent: Some((2520, 2772)),
        xr_finalize_predicted_display_time_nanos: Some(123_456_789),
    };
    let line = format_snapshot_from(&s);
    assert!(line.contains("phase=render-views"));
    assert!(line.contains("cpu_phase=command-record"));
    assert!(line.contains("mode=hmd-multiview"));
    assert!(line.contains("target=openxr"));
    assert!(line.contains("init=finalized"));
    assert!(line.contains("last_host_frame=9001"));
    assert!(line.contains("last_graph_error=pass"));
    assert!(line.contains("driver_stage=xr-finalize-start"));
    assert!(line.contains("openxr_call=end-frame-projection"));
    assert!(line.contains("xr_finalize=projection"));
    assert!(line.contains("xr_image=2"));
    assert!(line.contains("xr_extent=2520x2772"));
    assert!(line.contains("xr_predicted_time_ns=123456789"));
}

#[test]
fn atomics_snapshot_round_trips_core_fields() {
    let _guard = lock_reset_crash_context();

    set_tick_phase(TickPhase::AssetIntegration);
    set_cpu_render_phase(CpuRenderPhase::Sort);
    set_render_mode(RenderMode::IpcDesktop);
    set_target_mode(TargetMode::Desktop);
    set_init_state(InitState::InitializationComplete);
    set_last_host_frame_index(77);
    set_prepared_view_count(2);
    set_last_graph_error(GraphErrorKind::TransientPool);
    set_driver_stage(DriverStage::XrFinalizeStart);
    set_openxr_call(OpenXrCall::EndFrameProjection);
    set_xr_finalize_state(
        XrFinalizeKind::Projection,
        Some(1),
        13,
        2,
        Some((640, 480)),
        Some(99),
    );
    let s = snapshot();
    assert_eq!(s.tick_phase, TickPhase::AssetIntegration);
    assert_eq!(s.cpu_render_phase, CpuRenderPhase::Sort);
    assert_eq!(s.render_mode, RenderMode::IpcDesktop);
    assert_eq!(s.target_mode, TargetMode::Desktop);
    assert_eq!(s.init_state, InitState::InitializationComplete);
    assert_eq!(s.last_host_frame_index, 77);
    assert_eq!(s.prepared_view_count, 2);
    assert_eq!(s.last_graph_error, GraphErrorKind::TransientPool);
    assert_eq!(s.driver_stage, DriverStage::XrFinalizeStart);
    assert_eq!(s.openxr_call, OpenXrCall::EndFrameProjection);
    assert_eq!(s.xr_finalize_kind, XrFinalizeKind::Projection);
    assert_eq!(s.xr_finalize_image_index, Some(1));
    assert_eq!(s.xr_finalize_frame_seq, 13);
    assert_eq!(s.xr_finalize_command_buffers, 2);
    assert_eq!(s.xr_finalize_extent, Some((640, 480)));
    assert_eq!(s.xr_finalize_predicted_display_time_nanos, Some(99));
    clear_openxr_call_if(OpenXrCall::EndFrameProjection);
    clear_xr_finalize_state();
    let cleared = snapshot();
    assert_eq!(cleared.openxr_call, OpenXrCall::None);
    assert_eq!(cleared.xr_finalize_kind, XrFinalizeKind::None);
}

#[test]
fn clear_openxr_call_if_preserves_newer_call() {
    let _guard = lock_reset_crash_context();

    set_openxr_call(OpenXrCall::EndFrameProjection);
    set_openxr_call(OpenXrCall::WaitPreviousFinalize);
    clear_openxr_call_if(OpenXrCall::EndFrameProjection);
    assert_eq!(snapshot().openxr_call, OpenXrCall::WaitPreviousFinalize);
    clear_openxr_call_if(OpenXrCall::WaitPreviousFinalize);
    assert_eq!(snapshot().openxr_call, OpenXrCall::None);
}

#[test]
fn minimal_snapshot_is_ascii_and_includes_labels() {
    let _guard = lock_reset_crash_context();

    set_tick_phase(TickPhase::Shutdown);
    set_cpu_render_phase(CpuRenderPhase::Cleanup);
    set_render_mode(RenderMode::Headless);
    set_driver_stage(DriverStage::XrFinalizeStart);
    set_openxr_call(OpenXrCall::EndFrameProjection);
    set_xr_finalize_state(
        XrFinalizeKind::Projection,
        Some(3),
        21,
        4,
        Some((2520, 2772)),
        Some(123),
    );
    let mut out = [0u8; 1024];
    let n = write_minimal_snapshot(&mut out);
    let line = std::str::from_utf8(&out[..n]).expect("utf8");
    assert!(line.starts_with("CRASH_CONTEXT"));
    assert!(line.contains("phase=shutdown"));
    assert!(line.contains("cpu_phase=cleanup"));
    assert!(line.contains("mode=headless"));
    assert!(line.contains("driver_stage=xr-finalize-start"));
    assert!(line.contains("openxr_call=end-frame-projection"));
    assert!(line.contains("xr_finalize=projection"));
    assert!(line.contains("xr_image=3"));
    assert!(line.contains("xr_extent=2520x2772"));
    assert!(line.ends_with('\n'));
}
