//! Orchestration tests: IPC dispatch, frame submit, and init-state routing.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use glam::IVec2;
use renderide_shared::ipc::HostDualQueueIpc;

use crate::config::{RendererSettings, RendererSettingsHandle, VsyncMode};
use crate::connection::ConnectionParams;
use crate::ipc::SharedMemoryAccessor;
use crate::shared::buffer::SharedMemoryBufferDescriptor;
use crate::shared::{
    CameraRenderParameters, CameraRenderTask, DesktopConfig, FrameSubmitData, FreeSharedMemoryView,
    Guid, HeadOutputDevice, KeepAlive, MeshRenderablesUpdate, PostProcessingConfig, QualityConfig,
    ReflectionProbeRenderResult, ReflectionProbeRenderTask, RenderSpaceUpdate, RendererCommand,
    RendererEngineReady, RendererInitData, RendererInitFinalizeData, RendererShutdown,
    SetTexture2DFormat, ShaderUpload, SkinWeightMode, TextureFormat,
};

use super::RendererRuntime;
use super::state::tick::QueuedReflectionProbeRenderTask;

static IPC_TEST_QUEUE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn test_settings_handle() -> RendererSettingsHandle {
    Arc::new(std::sync::RwLock::new(RendererSettings::default()))
}

fn test_runtime_standalone() -> RendererRuntime {
    RendererRuntime::new(
        None,
        test_settings_handle(),
        PathBuf::from("/tmp/renderide_orchestration_test_config.toml"),
    )
}

fn test_runtime_ipc_shape() -> RendererRuntime {
    RendererRuntime::new(
        Some(ConnectionParams {
            queue_name: "orchestration_test_queue".into(),
            queue_capacity: crate::connection::DEFAULT_QUEUE_CAPACITY,
        }),
        test_settings_handle(),
        PathBuf::from("/tmp/renderide_orchestration_test_config_ipc.toml"),
    )
}

fn test_runtime_connected_ipc() -> (HostDualQueueIpc, RendererRuntime) {
    let seq = IPC_TEST_QUEUE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let params = ConnectionParams {
        queue_name: format!("orchestration_test_queue_{}_{}", std::process::id(), seq),
        queue_capacity: 4096,
    };
    let host = HostDualQueueIpc::connect(&params).expect("host IPC connects");
    let mut runtime = RendererRuntime::new(
        Some(params),
        test_settings_handle(),
        PathBuf::from("/tmp/renderide_orchestration_test_config_connected_ipc.toml"),
    );
    runtime.connect_ipc().expect("renderer IPC connects");
    runtime.handle_ipc_command(RendererCommand::RendererInitData(test_renderer_init_data()));
    runtime.handle_ipc_command(RendererCommand::RendererInitFinalizeData(
        RendererInitFinalizeData::default(),
    ));
    (host, runtime)
}

fn test_renderer_init_data() -> RendererInitData {
    RendererInitData {
        shared_memory_prefix: Some("test_shm_prefix".into()),
        unique_session_id: Guid::default(),
        main_process_id: 0,
        debug_frame_pacing: false,
        output_device: HeadOutputDevice::default(),
        window_title: None,
        set_window_icon: None,
        splash_screen_override: None,
    }
}

fn apply_running_command(rt: &mut RendererRuntime, cmd: RendererCommand) {
    let effect = crate::frontend::dispatch::commands::handle_running_command(cmd);
    rt.apply_running_command_effect(effect);
}

#[test]
fn dispatch_shutdown_sets_shutdown_requested() {
    let mut rt = test_runtime_standalone();
    apply_running_command(
        &mut rt,
        RendererCommand::RendererShutdown(RendererShutdown::default()),
    );
    assert!(rt.shutdown_requested());
}

#[test]
fn dispatch_frame_submit_updates_lockstep_fields() {
    let mut rt = test_runtime_standalone();
    rt.test_set_shared_memory("test_shm");
    let data = FrameSubmitData {
        frame_index: 101,
        ..Default::default()
    };
    apply_running_command(&mut rt, RendererCommand::FrameSubmitData(data));
    assert_eq!(rt.last_frame_index(), 101);
    assert!(rt.last_frame_data_processed());
    assert!(rt.pending_frame_submit_render());
    assert!(!rt.fatal_error());
}

#[test]
fn render_attempt_clears_pending_frame_submit_render() {
    let mut rt = test_runtime_standalone();
    rt.test_set_shared_memory("test_shm");
    let data = FrameSubmitData {
        frame_index: 101,
        ..Default::default()
    };
    apply_running_command(&mut rt, RendererCommand::FrameSubmitData(data));
    assert!(rt.pending_frame_submit_render());

    rt.note_frame_render_attempted();

    assert!(!rt.pending_frame_submit_render());
}

#[test]
fn submit_completion_work_drained_waits_for_camera_tasks() {
    let mut rt = test_runtime_standalone();
    assert!(rt.submit_completion_work_drained());

    rt.tick_state
        .pending_camera_render_tasks
        .push(CameraRenderTask::default());

    assert!(!rt.submit_completion_work_drained());
}

#[test]
fn regular_begin_frame_waits_for_late_camera_task_after_render_attempt() {
    let (_host, mut rt) = test_runtime_connected_ipc();
    rt.test_set_shared_memory("test_late_camera_task");
    rt.apply_frame_submit_data(FrameSubmitData {
        frame_index: 101,
        render_tasks: vec![CameraRenderTask {
            parameters: Some(CameraRenderParameters {
                resolution: IVec2 { x: 1, y: 1 },
                texture_format: TextureFormat::RGBA32,
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    });
    assert_eq!(rt.pending_camera_render_task_count(), 1);

    rt.note_frame_render_attempted();

    assert!(!rt.pending_frame_submit_render());
    assert!(!rt.submit_completion_work_drained());
    assert!(!rt.should_send_begin_frame());

    rt.tick_state.pending_camera_render_tasks.clear();

    assert!(rt.submit_completion_work_drained());
    assert!(rt.should_send_begin_frame());
}

#[test]
fn regular_begin_frame_waits_for_reflection_probe_task_after_render_attempt() {
    let (_host, mut rt) = test_runtime_connected_ipc();
    rt.apply_frame_submit_data(FrameSubmitData {
        frame_index: 101,
        ..Default::default()
    });
    rt.note_frame_render_attempted();

    rt.tick_state
        .pending_reflection_probe_render_tasks
        .push(QueuedReflectionProbeRenderTask {
            render_space_id: crate::scene::RenderSpaceId(1),
            task: ReflectionProbeRenderTask::default(),
        });

    assert!(!rt.submit_completion_work_drained());
    assert!(!rt.should_send_begin_frame());

    rt.tick_state.pending_reflection_probe_render_tasks.clear();

    assert!(rt.submit_completion_work_drained());
    assert!(rt.should_send_begin_frame());
}

#[test]
fn regular_begin_frame_waits_for_reflection_probe_result_flush_after_render_attempt() {
    let (_host, mut rt) = test_runtime_connected_ipc();
    rt.apply_frame_submit_data(FrameSubmitData {
        frame_index: 101,
        ..Default::default()
    });
    rt.note_frame_render_attempted();

    rt.tick_state
        .pending_reflection_probe_render_results
        .push(ReflectionProbeRenderResult {
            render_task_id: 7,
            success: true,
        });

    assert!(!rt.submit_completion_work_drained());
    assert!(!rt.should_send_begin_frame());

    rt.tick_state
        .pending_reflection_probe_render_results
        .clear();

    assert!(rt.submit_completion_work_drained());
    assert!(rt.should_send_begin_frame());
}

#[test]
fn submit_completion_work_drained_waits_for_reflection_probe_tasks() {
    let mut rt = test_runtime_standalone();
    assert!(rt.submit_completion_work_drained());

    rt.tick_state
        .pending_reflection_probe_render_tasks
        .push(QueuedReflectionProbeRenderTask {
            render_space_id: crate::scene::RenderSpaceId(1),
            task: ReflectionProbeRenderTask::default(),
        });

    assert!(!rt.submit_completion_work_drained());
}

#[test]
fn renderer_engine_ready_activates_host_lockstep_gate() {
    let mut rt = test_runtime_standalone();
    assert!(!rt.host_lockstep_activated());

    apply_running_command(
        &mut rt,
        RendererCommand::RendererEngineReady(RendererEngineReady::default()),
    );

    assert!(rt.host_lockstep_activated());
}

#[test]
fn dispatch_quality_config_updates_skin_weight_mode() {
    let mut rt = test_runtime_standalone();
    let before = rt.unhandled_ipc_command_event_total();
    apply_running_command(
        &mut rt,
        RendererCommand::QualityConfig(QualityConfig {
            skin_weight_mode: SkinWeightMode::TwoBones,
            ..Default::default()
        }),
    );
    assert_eq!(rt.skin_weight_mode(), SkinWeightMode::TwoBones);
    assert_eq!(rt.unhandled_ipc_command_event_total(), before);
}

#[test]
fn dispatch_desktop_config_overrides_effective_caps_without_mutating_renderer_settings() {
    let mut rt = test_runtime_standalone();
    let before = rt.unhandled_ipc_command_event_total();
    {
        let mut settings = rt.settings().write().expect("settings writable");
        settings.rendering.vsync = VsyncMode::On;
        settings.display.focused_fps_cap = 144;
        settings.display.unfocused_fps_cap = 30;
    };

    apply_running_command(
        &mut rt,
        RendererCommand::DesktopConfig(DesktopConfig {
            maximum_background_framerate: Some(30),
            maximum_foreground_framerate: Some(120),
            v_sync: true,
        }),
    );

    let caps = rt.desktop_frame_pacing_caps();
    assert_eq!(caps.foreground_fps_cap, 120);
    assert_eq!(caps.background_fps_cap, 30);
    {
        let settings = rt.settings().read().expect("settings readable");
        assert_eq!(settings.rendering.vsync, VsyncMode::On);
        assert_eq!(settings.display.focused_fps_cap, 144);
        assert_eq!(settings.display.unfocused_fps_cap, 30);
        drop(settings);
    };
    assert_eq!(rt.unhandled_ipc_command_event_total(), before);
}

#[test]
fn dispatch_desktop_config_ignores_negative_and_zero_host_caps_for_effective_caps() {
    let mut rt = test_runtime_standalone();
    {
        let mut settings = rt.settings().write().expect("settings writable");
        settings.rendering.vsync = VsyncMode::On;
        settings.display.focused_fps_cap = 240;
        settings.display.unfocused_fps_cap = 60;
    };

    apply_running_command(
        &mut rt,
        RendererCommand::DesktopConfig(DesktopConfig {
            maximum_background_framerate: Some(0),
            maximum_foreground_framerate: Some(-10),
            v_sync: false,
        }),
    );

    let caps = rt.desktop_frame_pacing_caps();
    assert_eq!(caps.foreground_fps_cap, 240);
    assert_eq!(caps.background_fps_cap, 60);
    {
        let settings = rt.settings().read().expect("settings readable");
        assert_eq!(settings.rendering.vsync, VsyncMode::On);
        assert_eq!(settings.display.focused_fps_cap, 240);
        assert_eq!(settings.display.unfocused_fps_cap, 60);
        drop(settings);
    }
}

#[test]
fn dispatch_keep_alive_is_noop_for_shutdown() {
    let mut rt = test_runtime_standalone();
    apply_running_command(&mut rt, RendererCommand::KeepAlive(KeepAlive::default()));
    assert!(!rt.shutdown_requested());
}

#[test]
fn dispatch_free_shared_memory_view_routes_without_fatal() {
    let mut rt = test_runtime_standalone();
    rt.test_set_shared_memory("pfx");
    apply_running_command(
        &mut rt,
        RendererCommand::FreeSharedMemoryView(FreeSharedMemoryView { buffer_id: 42 }),
    );
    assert!(!rt.fatal_error());
}

#[test]
fn run_asset_integration_at_most_once_per_tick() {
    let mut rt = test_runtime_standalone();
    rt.test_set_shared_memory("asset_integ_test");
    rt.tick_frame_wall_clock_begin(Instant::now());
    rt.run_asset_integration();
    assert!(rt.did_integrate_assets_this_tick());
    rt.run_asset_integration();
    assert!(rt.did_integrate_assets_this_tick());
    rt.tick_frame_wall_clock_begin(Instant::now());
    rt.run_asset_integration();
}

#[test]
fn frame_submit_fatal_on_scene_shared_memory_error() {
    let mut rt = test_runtime_standalone();
    rt.test_set_shared_memory("pfx");
    let data = FrameSubmitData {
        frame_index: 1,
        render_spaces: vec![RenderSpaceUpdate {
            id: 1,
            mesh_renderers_update: Some(MeshRenderablesUpdate {
                removals: SharedMemoryBufferDescriptor {
                    buffer_id: 0,
                    buffer_capacity: 0,
                    offset: 0,
                    length: SharedMemoryAccessor::MAX_ACCESS_COPY_BYTES + 1,
                },
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };
    rt.apply_frame_submit_data(data);
    assert!(rt.fatal_error());
}

#[test]
fn ipc_init_uninitialized_non_init_command_is_fatal() {
    let mut rt = test_runtime_ipc_shape();
    rt.handle_ipc_command(RendererCommand::QualityConfig(QualityConfig::default()));
    assert!(rt.fatal_error());
}

#[test]
fn ipc_init_uninitialized_keep_alive_not_fatal() {
    let mut rt = test_runtime_ipc_shape();
    rt.handle_ipc_command(RendererCommand::KeepAlive(KeepAlive::default()));
    assert!(!rt.fatal_error());
}

#[test]
fn ipc_init_renderer_init_data_moves_to_init_received() {
    let mut rt = test_runtime_ipc_shape();
    rt.handle_ipc_command(RendererCommand::RendererInitData(test_renderer_init_data()));
    assert_eq!(rt.init_state(), crate::frontend::InitState::InitReceived);
    assert!(!rt.fatal_error());
}

#[test]
fn ipc_init_finalize_then_running_dispatch_unhandled() {
    let mut rt = test_runtime_ipc_shape();
    rt.handle_ipc_command(RendererCommand::RendererInitData(test_renderer_init_data()));
    rt.handle_ipc_command(RendererCommand::RendererInitFinalizeData(
        RendererInitFinalizeData::default(),
    ));
    assert_eq!(rt.init_state(), crate::frontend::InitState::Finalized);
    let before = rt.unhandled_ipc_command_event_total();
    rt.handle_ipc_command(RendererCommand::PostProcessingConfig(
        PostProcessingConfig::default(),
    ));
    assert_eq!(rt.unhandled_ipc_command_event_total(), before + 1);
}

#[test]
fn ipc_init_init_received_defers_unrelated_command() {
    let mut rt = test_runtime_ipc_shape();
    rt.handle_ipc_command(RendererCommand::RendererInitData(test_renderer_init_data()));
    assert_eq!(rt.init_state(), crate::frontend::InitState::InitReceived);
    rt.handle_ipc_command(RendererCommand::QualityConfig(QualityConfig::default()));
    assert_eq!(rt.init_state(), crate::frontend::InitState::InitReceived);
    assert_eq!(rt.ipc_state.deferred_pre_finalize_command_count(), 1);
    assert!(!rt.fatal_error());
}

#[test]
fn ipc_init_init_received_dispatches_startup_asset_commands() {
    let mut rt = test_runtime_ipc_shape();
    rt.handle_ipc_command(RendererCommand::RendererInitData(test_renderer_init_data()));

    rt.handle_ipc_command(RendererCommand::SetTexture2DFormat(
        SetTexture2DFormat::default(),
    ));
    rt.handle_ipc_command(RendererCommand::ShaderUpload(ShaderUpload::default()));

    assert_eq!(rt.init_state(), crate::frontend::InitState::InitReceived);
    assert_eq!(rt.ipc_state.deferred_pre_finalize_command_count(), 0);
    assert_eq!(rt.ipc_state.pending_shader_resolutions.len(), 1);
    assert!(!rt.fatal_error());
}

#[test]
fn ipc_init_engine_ready_defers_until_finalize() {
    let mut rt = test_runtime_ipc_shape();
    rt.handle_ipc_command(RendererCommand::RendererInitData(test_renderer_init_data()));

    rt.handle_ipc_command(RendererCommand::RendererEngineReady(
        RendererEngineReady::default(),
    ));

    assert!(!rt.host_lockstep_activated());
    assert_eq!(rt.ipc_state.deferred_pre_finalize_command_count(), 1);

    rt.handle_ipc_command(RendererCommand::RendererInitFinalizeData(
        RendererInitFinalizeData::default(),
    ));

    assert_eq!(rt.init_state(), crate::frontend::InitState::Finalized);
    assert!(rt.host_lockstep_activated());
    assert_eq!(rt.ipc_state.deferred_pre_finalize_command_count(), 0);
}

#[test]
fn post_wait_asset_integration_drains_completed_shader_uploads() {
    let mut rt = test_runtime_ipc_shape();
    rt.handle_ipc_command(RendererCommand::RendererInitData(test_renderer_init_data()));
    rt.handle_ipc_command(RendererCommand::ShaderUpload(ShaderUpload::default()));
    assert_eq!(rt.ipc_state.pending_shader_resolutions.len(), 1);

    for _ in 0..100 {
        rt.run_asset_integration_after_wait_poll();
        if rt.ipc_state.pending_shader_resolutions.is_empty() {
            return;
        }
        std::thread::sleep(Duration::from_millis(1));
    }

    panic!("shader upload resolution did not drain");
}
