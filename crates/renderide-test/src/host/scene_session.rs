//! End-to-end orchestration of the harness lifecycle: open IPC → spawn renderer → handshake →
//! upload sphere mesh → swap to scene `FrameSubmitData` → wait for the renderer to write a fresh
//! PNG → request shutdown.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant, SystemTime};

use renderide_shared::shared::{RendererCommand, RendererShutdownRequest};
use renderide_shared::wire_writer::render_space::{
    build_sphere_render_space_update, SphereSceneInputs, SphereSceneSharedMemoryLayout,
    SphereSceneSharedMemoryRegions,
};
use renderide_shared::{SharedMemoryWriter, SharedMemoryWriterConfig};

use crate::error::HarnessError;
use crate::scene::mesh_payload::pack_sphere_mesh_upload;
use crate::scene::sphere::SphereMesh;

use super::asset_upload::{upload_sphere_mesh, UploadedMesh, DEFAULT_ASSET_UPLOAD_TIMEOUT};
use super::handshake::{run_handshake, DEFAULT_HANDSHAKE_TIMEOUT};
use super::ipc_setup::{connect_session, IpcSession, DEFAULT_QUEUE_CAPACITY_BYTES};
use super::lockstep::{FrameSubmitScalars, LockstepDriver};

/// Asset / buffer ids used by the harness. These never collide with anything the renderer
/// allocates internally because the renderer treats the shared memory only as host-driven input.
pub(super) const SPHERE_MESH_ASSET_ID: i32 = 2;
pub(super) const SPHERE_MATERIAL_ASSET_ID: i32 = 4;
pub(super) const SPHERE_MESH_BUFFER_ID: i32 = 0;
pub(super) const SCENE_STATE_BUFFER_ID: i32 = 1;
pub(super) const RENDER_SPACE_ID: i32 = 1;

/// Configuration for [`run_session`].
#[derive(Clone, Debug)]
pub(crate) struct SceneSessionConfig {
    /// Path to the `renderide` binary to spawn.
    pub renderer_path: PathBuf,
    /// Output PNG path the renderer writes to (also where the harness reads from).
    pub output_path: PathBuf,
    /// Offscreen render target width.
    pub width: u32,
    /// Offscreen render target height.
    pub height: u32,
    /// Renderer interval between consecutive PNG writes (ms).
    pub interval_ms: u64,
    /// Wall-clock budget for the entire session (handshake + upload + first stable PNG).
    pub timeout: Duration,
    /// When `true`, inherit the renderer's stdout/stderr.
    pub verbose_renderer: bool,
}

/// Result of a successful [`run_session`] call.
#[derive(Clone, Debug)]
pub(super) struct SceneSessionOutcome {
    /// Path to the freshly written PNG produced by the renderer.
    pub png_path: PathBuf,
}

/// Drives the full session end-to-end. The renderer process is killed on `Err` via [`Drop`] of the
/// [`SpawnedRenderer`] guard.
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

    let mesh = SphereMesh::generate(16, 24);
    let upload = pack_sphere_mesh_upload(&mesh)
        .map_err(|e| HarnessError::QueueOptions(format!("pack sphere upload: {e}")))?;
    let _uploaded: UploadedMesh = upload_sphere_mesh(
        &mut session.queues,
        &mut lockstep,
        &prefix,
        SPHERE_MESH_BUFFER_ID,
        SPHERE_MESH_ASSET_ID,
        &upload,
        DEFAULT_ASSET_UPLOAD_TIMEOUT,
    )?;

    let scene = build_scene_state(&prefix, &mut session.queues, &mut lockstep)?;

    // Pump the lockstep until the scene has actually been submitted to the renderer at least
    // once. That way we know the renderer has the sphere in its scene state before we start
    // accepting PNG mtimes.
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
            reason = "child set immediately above by spawn_renderer_child"
        )]
        spawned.child.as_mut().expect("child set"),
    )?;
    drop(scene); // explicitly hold scene SHM writer alive until after readback

    request_shutdown_and_wait(&mut session.queues, &mut spawned)?;

    Ok(png_outcome)
}

/// Holds the scene SHM writer alive so the renderer can keep reading the descriptor over many
/// lockstep ticks.
struct SceneState {
    _writer: SharedMemoryWriter,
    /// Kept around so the harness can re-send if needed; currently unused after the initial set.
    _regions: SphereSceneSharedMemoryRegions,
}

fn build_scene_state(
    prefix: &str,
    _queues: &mut renderide_shared::ipc::HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
) -> Result<SceneState, HarnessError> {
    let inputs = SphereSceneInputs {
        render_space_id: RENDER_SPACE_ID,
        camera_world_pose: SphereSceneInputs::default().camera_world_pose,
        object_pose: SphereSceneInputs::default().object_pose,
        mesh_asset_id: SPHERE_MESH_ASSET_ID,
        material_asset_id: SPHERE_MATERIAL_ASSET_ID,
    };
    let regions = SphereSceneSharedMemoryRegions::build(&inputs);
    let total_bytes = regions.total_bytes();
    let cfg = SharedMemoryWriterConfig {
        prefix: prefix.to_string(),
        destroy_on_drop: true,
    };
    let mut writer =
        SharedMemoryWriter::open(cfg, SCENE_STATE_BUFFER_ID, total_bytes).map_err(|e| {
            HarnessError::QueueOptions(format!("open scene-state SHM (cap={total_bytes}): {e}"))
        })?;

    let layout = SphereSceneSharedMemoryLayout::pack_back_to_back(
        SCENE_STATE_BUFFER_ID,
        total_bytes as i32,
        &regions,
    );
    writer
        .write_at(
            layout.pose_updates_offset as usize,
            &regions.pose_updates_bytes,
        )
        .map_err(|e| HarnessError::QueueOptions(format!("write pose_updates: {e}")))?;
    writer
        .write_at(layout.additions_offset as usize, &regions.additions_bytes)
        .map_err(|e| HarnessError::QueueOptions(format!("write additions: {e}")))?;
    writer
        .write_at(
            layout.mesh_states_offset as usize,
            &regions.mesh_states_bytes,
        )
        .map_err(|e| HarnessError::QueueOptions(format!("write mesh_states: {e}")))?;
    writer
        .write_at(
            layout.packed_material_ids_offset as usize,
            &regions.packed_material_ids_bytes,
        )
        .map_err(|e| HarnessError::QueueOptions(format!("write packed_material_ids: {e}")))?;
    writer.flush();

    let render_space = build_sphere_render_space_update(&inputs, &regions, &layout);
    lockstep.set_render_space(Some(render_space));

    Ok(SceneState {
        _writer: writer,
        _regions: regions,
    })
}

/// Pumps the lockstep loop until at least one `FrameSubmitData` carrying the scene has actually
/// been enqueued. Returns the `frame_index` of that submission so callers can log it.
fn ensure_scene_submitted(
    queues: &mut renderide_shared::ipc::HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
    timeout: Duration,
) -> Result<i32, HarnessError> {
    let deadline = Instant::now() + timeout;
    let frame_index_before = lockstep.current_frame_index();
    while Instant::now() < deadline {
        let tick = lockstep.tick(queues);
        if tick.frame_submits_sent > 0 {
            return Ok(frame_index_before);
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    Err(HarnessError::AssetAckTimeout(
        deadline.elapsed(),
        "renderer never sent FrameStartData after scene was loaded",
    ))
}

/// Scene submission moment and the wall-clock budget for [`run_lockstep_until_png_stable`].
struct PngStabilityWaitTiming {
    /// `SystemTime` when the scene was submitted (used to compare against the PNG `mtime`).
    scene_submitted_at: SystemTime,
    /// Monotonic instant at scene submit (used for the "wait at least N intervals" gate).
    scene_submit_instant: Instant,
    /// Wall-clock budget for the entire wait-until-stable-PNG loop.
    overall_timeout: Duration,
    /// Renderer's configured PNG write interval.
    interval: Duration,
}

fn run_lockstep_until_png_stable(
    queues: &mut renderide_shared::ipc::HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
    output_path: &Path,
    timing: PngStabilityWaitTiming,
    renderer: &mut Child,
) -> Result<SceneSessionOutcome, HarnessError> {
    let PngStabilityWaitTiming {
        scene_submitted_at,
        scene_submit_instant,
        overall_timeout,
        interval,
    } = timing;
    // Renderer needs at least one full interval AFTER scene apply to write a fresh PNG that
    // reflects the new scene state, plus slack for IPC round-trip + PNG encoding. Lavapipe is
    // slow so we wait for at least 2 full intervals past scene submit before accepting a PNG.
    let min_wall_after_submit = (interval * 2).max(Duration::from_millis(1500));
    let deadline =
        Instant::now() + overall_timeout.max(min_wall_after_submit + Duration::from_secs(2));
    let mut last_seen_mtime: Option<SystemTime> = None;
    let mut stable_since: Option<Instant> = None;
    let stable_required = Duration::from_millis(200);
    let mut last_log_at = Instant::now();

    while Instant::now() < deadline {
        let _ = lockstep.tick(queues);

        if let Ok(Some(status)) = renderer.try_wait() {
            return Err(HarnessError::AssetAckTimeout(
                deadline.elapsed(),
                if status.success() {
                    "renderer exited cleanly before producing PNG"
                } else {
                    "renderer exited with failure before producing PNG"
                },
            ));
        }

        // Wait for at least `min_wall_after_submit` before considering ANY PNG as scene-ready,
        // even one whose mtime > scene_submitted_at. This handles slow software rendering where
        // the renderer's tick loop apply-then-render takes longer than a single PNG interval.
        let elapsed_since_submit = scene_submit_instant.elapsed();
        let scene_render_window_open = elapsed_since_submit >= min_wall_after_submit;

        if let Ok(meta) = std::fs::metadata(output_path) {
            if let Ok(mtime) = meta.modified() {
                let advanced_past_scene = mtime > scene_submitted_at;
                if scene_render_window_open && advanced_past_scene && meta.len() > 0 {
                    if last_seen_mtime == Some(mtime) {
                        if let Some(since) = stable_since {
                            if since.elapsed() >= stable_required {
                                return Ok(SceneSessionOutcome {
                                    png_path: output_path.to_path_buf(),
                                });
                            }
                        }
                    } else {
                        last_seen_mtime = Some(mtime);
                        stable_since = Some(Instant::now());
                    }
                }
            }
        }
        if last_log_at.elapsed() >= Duration::from_secs(2) {
            logger::info!(
                "Session: still waiting for fresh PNG (elapsed={:?}, scene_window_open={scene_render_window_open}, last_mtime={:?})",
                scene_submit_instant.elapsed(),
                last_seen_mtime
            );
            last_log_at = Instant::now();
        }
        std::thread::sleep(Duration::from_millis(20));
    }

    Err(HarnessError::PngOutputMissing {
        path: output_path.to_path_buf(),
        wait: overall_timeout,
    })
}

fn request_shutdown_and_wait(
    queues: &mut renderide_shared::ipc::HostDualQueueIpc,
    spawned: &mut SpawnedRenderer,
) -> Result<(), HarnessError> {
    let _ = queues.send_primary(RendererCommand::RendererShutdownRequest(
        RendererShutdownRequest {},
    ));
    logger::info!("Session: sent RendererShutdownRequest, waiting for child to exit");

    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if let Some(child) = spawned.child.as_mut() {
            if let Ok(Some(_status)) = child.try_wait() {
                spawned.child = None;
                return Ok(());
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    logger::warn!("Session: renderer did not exit within 5s, killing");
    if let Some(child) = spawned.child.as_mut() {
        let _ = child.kill();
        let _ = child.wait();
        spawned.child = None;
    }
    Ok(())
}

/// RAII-guarded spawned renderer process. Drop kills the child if still running.
struct SpawnedRenderer {
    child: Option<Child>,
}

impl Drop for SpawnedRenderer {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            logger::warn!("SpawnedRenderer: dropping with live child; killing");
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn spawn_renderer(
    cfg: &SceneSessionConfig,
    queue_name: &str,
    backing_dir: &Path,
) -> Result<SpawnedRenderer, HarnessError> {
    let mut cmd = Command::new(&cfg.renderer_path);
    cmd.arg("--headless");
    cmd.arg("--headless-output").arg(&cfg.output_path);
    cmd.arg("--headless-resolution")
        .arg(format!("{}x{}", cfg.width, cfg.height));
    cmd.arg("--headless-interval-ms")
        .arg(cfg.interval_ms.to_string());
    cmd.arg("-QueueName").arg(queue_name);
    cmd.arg("-QueueCapacity")
        .arg(DEFAULT_QUEUE_CAPACITY_BYTES.to_string());
    cmd.arg("-LogLevel").arg("debug");
    cmd.arg("--ignore-config");
    cmd.env("RENDERIDE_INTERPROCESS_DIR", backing_dir);

    if cfg.verbose_renderer {
        cmd.stdout(Stdio::inherit()).stderr(Stdio::inherit());
    } else {
        cmd.stdout(Stdio::null()).stderr(Stdio::null());
    }

    logger::info!(
        "Spawning renderer: {} --headless --headless-output {} --headless-resolution {}x{} --headless-interval-ms {} -QueueName {} -QueueCapacity {} --ignore-config",
        cfg.renderer_path.display(),
        cfg.output_path.display(),
        cfg.width,
        cfg.height,
        cfg.interval_ms,
        queue_name,
        DEFAULT_QUEUE_CAPACITY_BYTES,
    );

    let child = cmd.spawn().map_err(HarnessError::SpawnRenderer)?;
    Ok(SpawnedRenderer { child: Some(child) })
}

fn _ipc_session_used(_: &IpcSession) {}
