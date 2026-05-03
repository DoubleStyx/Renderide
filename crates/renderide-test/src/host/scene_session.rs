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

use renderide_shared::shared::{ColorProfile, RenderBoundingBox};

use crate::error::HarnessError;
use crate::scene::mesh::Mesh;
use crate::scene::mesh_payload::pack_mesh_upload;
use crate::scene::sphere::generate_sphere;
use crate::scene::torus::generate_torus;

use super::asset_upload::{
    DEFAULT_ASSET_UPLOAD_TIMEOUT, MaterialBatchRequest, MaterialUpdateOp, MeshUploadRequest,
    PropertyIdLookup, Texture2DUploadRequest, apply_material_batch, pack_texture2d_handle,
    request_property_ids, upload_shader, upload_sphere_mesh, upload_texture2d_rgba8,
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

/// Selects which procedural geometry the session uploads and renders, plus per-template
/// content the harness needs at session time (e.g. the texture bytes for a torus case).
#[derive(Clone, Debug)]
pub enum SessionTemplate {
    /// Original UV sphere baseline (no shader uploaded; renders on Null/checkerboard).
    Sphere,
    /// Procedural torus rendered through the embedded `unlit_default` shader with
    /// `texture_rgba` bound to the material's `_Tex` slot.
    Torus {
        /// Caller-provided RGBA8 pixels (`width * height * 4` bytes, row-major) for the
        /// `_Tex` binding. Typically the same Perlin texture the runner writes as a side
        /// artifact, so the rendered torus and the artifact PNG are pixel-for-pixel
        /// related.
        texture_rgba: Vec<u8>,
        /// Texture dimensions in pixels.
        texture_size: (u32, u32),
    },
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
    let template_label = template_label(&template);
    logger::info!(
        "Session: opened authority queues (prefix={prefix}, backing_dir={}, template={template_label})",
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
    let upload = pack_mesh_upload(&geometry.mesh, geometry.bounds).map_err(|e| {
        HarnessError::QueueOptions(format!("pack {template_label} mesh upload: {e}"))
    })?;
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

    let _bound_material = if let Some(binding) = geometry.material_binding.as_ref() {
        Some(apply_material_binding(
            &mut session.queues,
            &mut lockstep,
            &prefix,
            &backing_dir,
            geometry.assets.material,
            binding,
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
    /// When `Some`, the harness uploads the shader, the texture (if any), and binds them
    /// to the geometry's material before submitting the scene. When `None` the case stays
    /// on the renderer's Null fallback pipeline.
    material_binding: Option<MaterialBinding>,
}

struct MaterialBinding {
    /// Renderer-side asset id given to the uploaded shader.
    shader_asset_id: i32,
    /// AssetBundle-style shader name (e.g. `"Unlit.shader"`); the renderer's stem-prefix
    /// resolver strips the optional `.shader` extension and lowercases.
    shader_name: &'static str,
    /// Optional Texture2D bound to a single material property (e.g. `_Tex` for
    /// `unlit_default`). `None` leaves the texture slot at whatever default the renderer
    /// uses for unbound samplers.
    texture: Option<TextureBinding>,
}

struct TextureBinding {
    /// Renderer-side asset id given to the uploaded Texture2D.
    asset_id: i32,
    /// Material property name the texture binds to (e.g. `"_Tex"`). The renderer interns
    /// this on `MaterialPropertyIdRequest`; the returned id is then used in the
    /// `SetTexture` opcode.
    property_name: &'static str,
    /// Companion `_Tex_ST` (scale + offset) property name. The unlit material's vertex
    /// stage reads this as a `vec4` of `(scale_x, scale_y, offset_x, offset_y)`. The
    /// runner sets it to `(1, 1, 0, 0)` so the texture covers the mesh's `[0,1]^2` UVs
    /// without scaling.
    st_property_name: &'static str,
    /// `(scale_x, scale_y, offset_x, offset_y)` written into the `_Tex_ST` property.
    st_value: [f32; 4],
    /// RGBA8 pixels (`width * height * 4` bytes, row-major).
    rgba_bytes: Vec<u8>,
    /// Texture dimensions in pixels.
    size: (u32, u32),
    /// sRGB vs Linear color profile.
    color_profile: ColorProfile,
}

fn template_label(template: &SessionTemplate) -> &'static str {
    match template {
        SessionTemplate::Sphere => "Sphere",
        SessionTemplate::Torus { .. } => "Torus",
    }
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
            material_binding: None,
        },
        SessionTemplate::Torus {
            texture_rgba,
            texture_size,
        } => {
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
                material_binding: Some(MaterialBinding {
                    shader_asset_id: asset_ids::TORUS_SHADER,
                    shader_name: "Unlit.shader",
                    texture: Some(TextureBinding {
                        asset_id: asset_ids::TORUS_TEXTURE,
                        property_name: "_Tex",
                        st_property_name: "_Tex_ST",
                        st_value: [1.0, 1.0, 0.0, 0.0],
                        rgba_bytes: texture_rgba,
                        size: texture_size,
                        color_profile: ColorProfile::SRGB,
                    }),
                }),
            }
        }
    }
}

fn apply_material_binding(
    queues: &mut renderide_shared::ipc::HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
    prefix: &str,
    backing_dir: &std::path::Path,
    material_asset_id: i32,
    binding: &MaterialBinding,
) -> Result<super::asset_upload::BoundMaterial, HarnessError> {
    upload_shader(
        queues,
        lockstep,
        binding.shader_asset_id,
        binding.shader_name,
        DEFAULT_ASSET_UPLOAD_TIMEOUT,
    )?;

    let mut ops: Vec<MaterialUpdateOp> = Vec::new();
    ops.push(MaterialUpdateOp::SelectTarget { material_asset_id });
    ops.push(MaterialUpdateOp::SetShader {
        shader_asset_id: binding.shader_asset_id,
    });

    if let Some(tex) = binding.texture.as_ref() {
        let _uploaded_texture = upload_texture2d_rgba8(
            queues,
            lockstep,
            Texture2DUploadRequest {
                shared_memory_prefix: prefix,
                backing_dir,
                buffer_id: asset_ids::TEXTURE_DATA_BUFFER,
                asset_id: tex.asset_id,
                width: tex.size.0,
                height: tex.size.1,
                rgba_bytes: &tex.rgba_bytes,
                color_profile: tex.color_profile,
                timeout: DEFAULT_ASSET_UPLOAD_TIMEOUT,
            },
        )?;
        // Note: drops at end of function. The renderer reads pixels at upload time and
        // owns its own GPU copy thereafter, so we don't need to keep the SHM alive past
        // `SetTexture2DResult(DATA_UPLOAD)`.

        let property_ids = request_property_ids(
            queues,
            lockstep,
            PropertyIdLookup {
                request_id: asset_ids::PROPERTY_ID_REQUEST_ID,
                names: &[tex.property_name, tex.st_property_name],
                timeout: DEFAULT_ASSET_UPLOAD_TIMEOUT,
            },
        )?;
        if property_ids.len() != 2 {
            return Err(HarnessError::QueueOptions(format!(
                "expected 2 property ids, got {n}",
                n = property_ids.len()
            )));
        }
        let tex_property_id = property_ids[0];
        let tex_st_property_id = property_ids[1];

        ops.push(MaterialUpdateOp::SetTexture {
            property_id: tex_property_id,
            packed_handle: pack_texture2d_handle(tex.asset_id),
        });
        ops.push(MaterialUpdateOp::SetFloat4 {
            property_id: tex_st_property_id,
            value: tex.st_value,
        });
    }

    ops.push(MaterialUpdateOp::UpdateBatchEnd);

    apply_material_batch(
        queues,
        lockstep,
        MaterialBatchRequest {
            shared_memory_prefix: prefix,
            backing_dir,
            base_buffer_id: asset_ids::MATERIAL_UPDATE_BASE_BUFFER,
            update_batch_id: asset_ids::MATERIAL_UPDATE_BATCH_ID,
            material_update_count: 1,
            ops: &ops,
            timeout: DEFAULT_ASSET_UPLOAD_TIMEOUT,
        },
    )
}
