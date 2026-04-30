//! End-to-end orchestration of the harness lifecycle: open IPC → spawn renderer → handshake →
//! upload sphere mesh → swap to scene `FrameSubmitData` → wait for the renderer to write a fresh
//! PNG → request shutdown.
//!
//! The implementation is split across focused submodules:
//!
//! - [`config`] — public configuration and outcome types.
//! - [`consts`] — centralized timing, asset-id, and tessellation constants.
//! - [`spawn`] — renderer process spawn + RAII guard.
//! - [`scene_state`] — scene-state SHM construction and first-submit pump.
//! - [`png_readback`] — PNG stability state machine + readback driver loop.
//! - [`shutdown`] — graceful shutdown sequence.

use std::time::{Duration, Instant, SystemTime};

use crate::error::HarnessError;
use crate::scene::mesh_payload::pack_sphere_mesh_upload;
use crate::scene::sphere::SphereMesh;

use super::asset_upload::{DEFAULT_ASSET_UPLOAD_TIMEOUT, upload_sphere_mesh};
use super::handshake::{DEFAULT_HANDSHAKE_TIMEOUT, run_handshake};
use super::ipc_setup::{DEFAULT_QUEUE_CAPACITY_BYTES, connect_session};
use super::lockstep::{FrameSubmitScalars, LockstepDriver};

mod config;
mod consts;
pub mod png_readback;
mod scene_state;
mod shutdown;
pub mod spawn;

pub use config::SceneSessionConfig;

use config::SceneSessionOutcome;
use consts::{asset_ids, sphere_tessellation};
use png_readback::{PngStabilityWaitTiming, run_lockstep_until_png_stable};
use scene_state::{build_scene_state, ensure_scene_submitted};
use shutdown::request_shutdown_and_wait;
use spawn::spawn_renderer;

/// Drives the full session end-to-end. The renderer process is killed on `Err` via [`Drop`] of
/// the spawned-renderer guard.
pub(super) fn run_session(cfg: &SceneSessionConfig) -> Result<SceneSessionOutcome, HarnessError> {
    if !cfg.renderer_path.exists() {
        return Err(HarnessError::RendererBinaryMissing(
            cfg.renderer_path.clone(),
        ));
    }

    let mut session = connect_session(DEFAULT_QUEUE_CAPACITY_BYTES)?;
    let prefix = session.shared_memory_prefix.clone();
    let backing_dir = session.tempdir_guard.path().to_path_buf();
    logger::info!(
        "Session: opened authority queues (prefix={prefix}, backing_dir={})",
        backing_dir.display()
    );

    let mut spawned = spawn_renderer(cfg, &session.connection_params.queue_name, &backing_dir)?;

    let mut lockstep = LockstepDriver::new(FrameSubmitScalars::default());
    run_handshake(
        &mut session.queues,
        &mut lockstep,
        &prefix,
        DEFAULT_HANDSHAKE_TIMEOUT,
    )?;

    let mesh = SphereMesh::generate(
        sphere_tessellation::LATITUDE_BANDS,
        sphere_tessellation::LONGITUDE_BANDS,
    );
    let upload = pack_sphere_mesh_upload(&mesh)
        .map_err(|e| HarnessError::QueueOptions(format!("pack sphere upload: {e}")))?;
    let _uploaded = upload_sphere_mesh(
        &mut session.queues,
        &mut lockstep,
        &prefix,
        asset_ids::SPHERE_MESH_BUFFER,
        asset_ids::SPHERE_MESH,
        &upload,
        DEFAULT_ASSET_UPLOAD_TIMEOUT,
    )?;

    let scene = build_scene_state(&prefix, &mut lockstep)?;

    let scene_submit_index =
        ensure_scene_submitted(&mut session.queues, &mut lockstep, cfg.timeout)?;
    let scene_submitted_at = SystemTime::now();
    let scene_submit_instant = Instant::now();
    logger::info!(
        "Session: scene submitted at frame_index={scene_submit_index}, mtime_baseline={scene_submitted_at:?}; waiting for fresh PNG"
    );

    let png_outcome = run_lockstep_until_png_stable(
        &mut session.queues,
        &mut lockstep,
        &cfg.output_path,
        PngStabilityWaitTiming {
            scene_submitted_at,
            scene_submit_instant,
            overall_timeout: cfg.timeout,
            interval: Duration::from_millis(cfg.interval_ms.max(1)),
        },
        #[expect(
            clippy::expect_used,
            reason = "child set immediately above by spawn_renderer"
        )]
        spawned.child.as_mut().expect("child set"),
    )?;
    drop(scene);

    request_shutdown_and_wait(&mut session.queues, &mut spawned)?;

    Ok(png_outcome)
}
