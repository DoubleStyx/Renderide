//! Orchestration tests: IPC dispatch, frame submit, and init-state routing.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use crate::config::{RendererSettings, RendererSettingsHandle};
use crate::connection::ConnectionParams;
use crate::ipc::SharedMemoryAccessor;
use crate::shared::buffer::SharedMemoryBufferDescriptor;
use crate::shared::{
    FrameSubmitData, FreeSharedMemoryView, Guid, HeadOutputDevice, KeepAlive,
    MeshRenderablesUpdate, QualityConfig, RenderSpaceUpdate, RendererCommand, RendererInitData,
    RendererInitFinalizeData, RendererShutdown,
};

use super::commands::handle_running_command;
use super::frame_submit::process_frame_submit;
use super::ipc_init_dispatch::dispatch_ipc_command;
use super::RendererRuntime;

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

#[test]
fn dispatch_shutdown_sets_shutdown_requested() {
    let mut rt = test_runtime_standalone();
    handle_running_command(
        &mut rt,
        RendererCommand::RendererShutdown(RendererShutdown::default()),
    );
    assert!(rt.shutdown_requested());
}

#[test]
fn dispatch_frame_submit_updates_lockstep_fields() {
    let mut rt = test_runtime_standalone();
    rt.test_set_shared_memory("test_shm");
    let mut data = FrameSubmitData::default();
    data.frame_index = 101;
    handle_running_command(&mut rt, RendererCommand::FrameSubmitData(data));
    assert_eq!(rt.last_frame_index(), 101);
    assert!(rt.last_frame_data_processed());
    assert!(!rt.fatal_error());
}

#[test]
fn dispatch_quality_config_increments_unhandled_when_no_handler() {
    let mut rt = test_runtime_standalone();
    let before = rt.unhandled_ipc_command_event_total();
    handle_running_command(
        &mut rt,
        RendererCommand::QualityConfig(QualityConfig::default()),
    );
    assert_eq!(rt.unhandled_ipc_command_event_total(), before + 1);
}

#[test]
fn dispatch_keep_alive_is_noop_for_shutdown() {
    let mut rt = test_runtime_standalone();
    handle_running_command(&mut rt, RendererCommand::KeepAlive(KeepAlive::default()));
    assert!(!rt.shutdown_requested());
}

#[test]
fn dispatch_free_shared_memory_view_routes_without_fatal() {
    let mut rt = test_runtime_standalone();
    rt.test_set_shared_memory("pfx");
    handle_running_command(
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
    let mut data = FrameSubmitData::default();
    data.frame_index = 1;
    data.render_spaces.push(RenderSpaceUpdate {
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
    });
    process_frame_submit(&mut rt, data);
    assert!(rt.fatal_error());
}

#[test]
fn ipc_init_uninitialized_non_init_command_is_fatal() {
    let mut rt = test_runtime_ipc_shape();
    dispatch_ipc_command(
        &mut rt,
        RendererCommand::QualityConfig(QualityConfig::default()),
    );
    assert!(rt.fatal_error());
}

#[test]
fn ipc_init_uninitialized_keep_alive_not_fatal() {
    let mut rt = test_runtime_ipc_shape();
    dispatch_ipc_command(&mut rt, RendererCommand::KeepAlive(KeepAlive::default()));
    assert!(!rt.fatal_error());
}

#[test]
fn ipc_init_renderer_init_data_moves_to_init_received() {
    let mut rt = test_runtime_ipc_shape();
    dispatch_ipc_command(
        &mut rt,
        RendererCommand::RendererInitData(test_renderer_init_data()),
    );
    assert_eq!(rt.init_state(), crate::frontend::InitState::InitReceived);
    assert!(!rt.fatal_error());
}

#[test]
fn ipc_init_finalize_then_running_dispatch_unhandled() {
    let mut rt = test_runtime_ipc_shape();
    dispatch_ipc_command(
        &mut rt,
        RendererCommand::RendererInitData(test_renderer_init_data()),
    );
    dispatch_ipc_command(
        &mut rt,
        RendererCommand::RendererInitFinalizeData(RendererInitFinalizeData::default()),
    );
    assert_eq!(rt.init_state(), crate::frontend::InitState::Finalized);
    let before = rt.unhandled_ipc_command_event_total();
    dispatch_ipc_command(
        &mut rt,
        RendererCommand::QualityConfig(QualityConfig::default()),
    );
    assert_eq!(rt.unhandled_ipc_command_event_total(), before + 1);
}

#[test]
fn ipc_init_init_received_defers_unrelated_command() {
    let mut rt = test_runtime_ipc_shape();
    dispatch_ipc_command(
        &mut rt,
        RendererCommand::RendererInitData(test_renderer_init_data()),
    );
    assert_eq!(rt.init_state(), crate::frontend::InitState::InitReceived);
    dispatch_ipc_command(
        &mut rt,
        RendererCommand::QualityConfig(QualityConfig::default()),
    );
    assert_eq!(rt.init_state(), crate::frontend::InitState::InitReceived);
    assert!(!rt.fatal_error());
}
