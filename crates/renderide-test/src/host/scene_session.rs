//! End-to-end orchestration of the harness lifecycle: open IPC -> spawn renderer -> handshake ->
//! upload mesh -> swap to scene `FrameSubmitData` -> wait for the renderer to write a fresh
//! PNG -> request shutdown.
//!
//! The implementation is split across focused submodules:
//!
//! - [`config`] -- public configuration and outcome types.
//! - [`consts`] -- centralized timing, asset-id, and tessellation constants.
//! - [`spawn`] -- renderer process spawn + RAII guard.
//! - [`scene_state`] -- scene-state SHM construction and first-submit pump.
//! - [`png_readback`] -- PNG stability state machine + readback driver loop.
//! - [`shutdown`] -- graceful shutdown sequence.

use std::time::{Duration, Instant, SystemTime};

use renderide_shared::shared::RenderBoundingBox;

use crate::error::HarnessError;
use crate::scene::mesh::Mesh;
use crate::scene::mesh_payload::pack_mesh_upload;
use crate::scene::sphere::generate_sphere;
use crate::scene::torus::generate_torus;

use super::asset_upload::{
    DEFAULT_ASSET_UPLOAD_TIMEOUT, MaterialBindRequest, MeshUploadRequest, bind_material_shader,
    upload_shader, upload_sphere_mesh,
};
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
use consts::{asset_ids, sphere_tessellation, torus_geometry};
use png_readback::{PngStabilityWaitTiming, run_lockstep_until_png_stable};
use scene_state::{SceneAssetIds, build_scene_state, ensure_scene_submitted};
use shutdown::request_shutdown_and_wait;
use spawn::spawn_renderer;

/// Selects which procedural geometry the session uploads and renders.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SessionTemplate {
    /// Original UV sphere baseline.
    Sphere,
    /// Procedural torus sized to fit comfortably inside a unit cube.
    Torus,
}

/// Drives the full session end-to-end. The renderer process is killed on `Err` via [`Drop`] of
/// the spawned-renderer guard.
pub(super) fn run_session(
    cfg: &SceneSessionConfig,
    template: SessionTemplate,
) -> Result<SceneSessionOutcome, HarnessError> {
    if !cfg.renderer_path.exists() {
        return Err(HarnessError::RendererBinaryMissing(
            cfg.renderer_path.clone(),
        ));
    }

    let mut session = connect_session(DEFAULT_QUEUE_CAPACITY_BYTES)?;
    let prefix = session.shared_memory_prefix.clone();
    let backing_dir = session.tempdir_guard.path().to_path_buf();
    logger::info!(
        "Session: opened authority queues (prefix={prefix}, backing_dir={}, template={template:?})",
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

    let geometry = build_geometry_for_template(template);
    let upload = pack_mesh_upload(&geometry.mesh, geometry.bounds)
        .map_err(|e| HarnessError::QueueOptions(format!("pack {template:?} mesh upload: {e}")))?;
    let _uploaded = upload_sphere_mesh(
        &mut session.queues,
        &mut lockstep,
        MeshUploadRequest {
            shared_memory_prefix: &prefix,
            backing_dir: &backing_dir,
            buffer_id: asset_ids::MESH_BUFFER,
            asset_id: geometry.assets.mesh,
            mesh: &upload,
            timeout: DEFAULT_ASSET_UPLOAD_TIMEOUT,
        },
    )?;

    let _bound_material = if let Some(binding) = geometry.shader_binding.as_ref() {
        upload_shader(
            &mut session.queues,
            &mut lockstep,
            binding.shader_asset_id,
            binding.shader_name,
            DEFAULT_ASSET_UPLOAD_TIMEOUT,
        )?;
        Some(bind_material_shader(
            &mut session.queues,
            &mut lockstep,
            MaterialBindRequest {
                shared_memory_prefix: &prefix,
                backing_dir: &backing_dir,
                buffer_id: asset_ids::MATERIAL_UPDATE_BUFFER,
                update_batch_id: asset_ids::MATERIAL_UPDATE_BATCH_ID,
                material_asset_id: geometry.assets.material,
                shader_asset_id: binding.shader_asset_id,
                timeout: DEFAULT_ASSET_UPLOAD_TIMEOUT,
            },
        )?)
    } else {
        None
    };

    let scene = build_scene_state(&prefix, &backing_dir, geometry.assets, &mut lockstep)?;

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

struct GeometrySetup {
    mesh: Mesh,
    bounds: RenderBoundingBox,
    assets: SceneAssetIds,
    /// When `Some`, the harness uploads the shader and binds it to the geometry's material
    /// before submitting the scene. When `None` the case stays on the renderer's Null
    /// fallback pipeline.
    shader_binding: Option<ShaderBinding>,
}

struct ShaderBinding {
    /// Renderer-side asset id given to the uploaded shader.
    shader_asset_id: i32,
    /// AssetBundle-style shader name (e.g. `"Unlit.shader"`); the renderer's stem-prefix
    /// resolver strips the optional `.shader` extension and lowercases.
    shader_name: &'static str,
}

fn build_geometry_for_template(template: SessionTemplate) -> GeometrySetup {
    match template {
        SessionTemplate::Sphere => GeometrySetup {
            mesh: generate_sphere(
                sphere_tessellation::LATITUDE_BANDS,
                sphere_tessellation::LONGITUDE_BANDS,
            ),
            bounds: RenderBoundingBox {
                center: glam::Vec3::ZERO,
                extents: glam::Vec3::splat(1.05),
            },
            assets: SceneAssetIds {
                mesh: asset_ids::SPHERE_MESH,
                material: asset_ids::SPHERE_MATERIAL,
            },
            shader_binding: None,
        },
        SessionTemplate::Torus => {
            let outer = torus_geometry::MAJOR_RADIUS + torus_geometry::MINOR_RADIUS;
            GeometrySetup {
                mesh: generate_torus(
                    torus_geometry::MAJOR_SEGMENTS,
                    torus_geometry::MINOR_SEGMENTS,
                    torus_geometry::MAJOR_RADIUS,
                    torus_geometry::MINOR_RADIUS,
                ),
                bounds: RenderBoundingBox {
                    center: glam::Vec3::ZERO,
                    extents: glam::Vec3::new(
                        outer * 1.05,
                        torus_geometry::MINOR_RADIUS * 1.1,
                        outer * 1.05,
                    ),
                },
                assets: SceneAssetIds {
                    mesh: asset_ids::TORUS_MESH,
                    material: asset_ids::TORUS_MATERIAL,
                },
                shader_binding: Some(ShaderBinding {
                    shader_asset_id: asset_ids::TORUS_SHADER,
                    shader_name: "Unlit.shader",
                }),
            }
        }
    }
}
