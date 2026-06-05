//! End-to-end orchestration of the harness lifecycle: open IPC -> spawn renderer -> handshake ->
//! upload assets -> submit scene state -> wait for the renderer to write a fresh PNG -> shutdown.
//!
//! The implementation is split across focused submodules:
//!
//! - `config` -- public configuration and outcome types.
//! - `consts` -- centralized timing, asset-id, and tessellation constants.
//! - [`spawn`] -- renderer process spawn + RAII guard.
//! - `scene_state` -- scene-state SHM construction and first-submit pump.
//! - [`png_readback`] -- PNG stability state machine + readback driver loop.
//! - `shutdown` -- graceful shutdown sequence.

use std::time::{Duration, Instant, SystemTime};

use glam::{Quat, Vec3};
use renderide_shared::shared::{ColorProfile, RenderBoundingBox, RenderTransform};

use crate::error::HarnessError;
use crate::scene::mesh::Mesh;
use crate::scene::mesh_payload::pack_mesh_upload;
use crate::scene::primitives::{centered_bounds, generate_cube, generate_quad};
use crate::scene::sphere::generate_sphere;
use crate::scene::torus::generate_torus;

use super::asset_upload::{
    BoundMaterial, DEFAULT_ASSET_UPLOAD_TIMEOUT, MaterialBatchRequest, MaterialUpdateOp,
    MeshUploadRequest, PropertyIdLookup, Texture2DUploadRequest, UploadedMesh,
    apply_material_batch, pack_texture2d_handle, request_property_ids, upload_mesh, upload_shader,
    upload_texture2d_rgba8,
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
use consts::{asset_ids, shader_variants, sphere_tessellation, torus_geometry};
use png_readback::{PngStabilityWaitTiming, run_lockstep_until_png_stable};
use scene_state::{
    SceneLight, SceneRenderable, SceneSubmission, build_scene_state, default_camera_world_pose,
    directional_light, ensure_scene_submitted, identity_transform, point_light,
};
use shutdown::request_shutdown_and_wait;
use spawn::spawn_renderer;

/// Selects which test scene the session uploads and renders.
#[derive(Clone, Debug)]
pub enum SessionTemplate {
    /// Original UV sphere baseline (no shader uploaded; renders on Null/checkerboard).
    Sphere,
    /// Procedural torus rendered through the embedded `unlit_default` shader with a texture.
    Torus {
        /// Caller-provided RGBA8 pixels for the `_Tex` binding.
        texture_rgba: Vec<u8>,
        /// Texture dimensions in pixels.
        texture_size: (u32, u32),
    },
    /// Multi-object unlit scene with cube, sphere, torus, and quad coverage.
    MultiPrimitiveUnlitGrid {
        /// Checker texture RGBA8 pixels.
        checker_rgba: Vec<u8>,
        /// UV ramp texture RGBA8 pixels.
        uv_ramp_rgba: Vec<u8>,
        /// Shared texture dimensions in pixels.
        texture_size: (u32, u32),
    },
    /// PBS material matrix lit by regular scene lights.
    PbsLitMaterialMatrix,
    /// Alpha-test quads driven by a generated alpha mask.
    AlphaCutoutMaskedQuads {
        /// Alpha mask RGBA8 pixels.
        mask_rgba: Vec<u8>,
        /// Texture dimensions in pixels.
        texture_size: (u32, u32),
    },
    /// Textured static mesh imported from a GLB fixture and rendered through Unlit.
    GltfTexturedStaticMesh {
        /// Mesh imported from the fixture.
        mesh: Mesh,
        /// Fixture texture RGBA8 pixels.
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

    let setup = build_scene_setup(template);
    let _uploaded_meshes = upload_scene_meshes(
        &mut session.queues,
        &mut lockstep,
        &prefix,
        &backing_dir,
        &setup,
    )?;
    let _bound_materials = upload_scene_materials(
        &mut session.queues,
        &mut lockstep,
        &prefix,
        &backing_dir,
        &setup,
    )?;

    let scene = build_scene_state(&prefix, &backing_dir, &setup.scene, &mut lockstep)?;

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

struct SceneSetup {
    meshes: Vec<MeshAsset>,
    materials: Vec<MaterialAsset>,
    scene: SceneSubmission,
}

struct MeshAsset {
    asset_id: i32,
    buffer_id: i32,
    mesh: Mesh,
    bounds: RenderBoundingBox,
}

struct MaterialAsset {
    asset_id: i32,
    update_batch_id: i32,
    property_request_id: i32,
    update_base_buffer_id: i32,
    binding: Option<MaterialBinding>,
}

struct MaterialBinding {
    shader_asset_id: i32,
    shader_name: &'static str,
    shader_variant_bits: Option<u32>,
    textures: Vec<TextureBinding>,
    floats: Vec<FloatBinding>,
    float4s: Vec<Float4Binding>,
}

struct TextureBinding {
    asset_id: i32,
    buffer_id: i32,
    property_name: &'static str,
    st_property_name: &'static str,
    st_value: [f32; 4],
    rgba_bytes: Vec<u8>,
    size: (u32, u32),
    color_profile: ColorProfile,
}

struct FloatBinding {
    property_name: &'static str,
    value: f32,
}

struct Float4Binding {
    property_name: &'static str,
    value: [f32; 4],
}

struct UnlitTexturedMaterialParams {
    material_asset_id: i32,
    shader_asset_id: i32,
    texture_asset_id: i32,
    texture_buffer_id: i32,
    update_batch_id: i32,
    property_request_id: i32,
    update_base_buffer_id: i32,
    variant_bits: u32,
    texture_rgba: Vec<u8>,
    texture_size: (u32, u32),
    color: [f32; 4],
    floats: Vec<FloatBinding>,
}

fn template_label(template: &SessionTemplate) -> &'static str {
    match template {
        SessionTemplate::Sphere => "Sphere",
        SessionTemplate::Torus { .. } => "Torus",
        SessionTemplate::MultiPrimitiveUnlitGrid { .. } => "MultiPrimitiveUnlitGrid",
        SessionTemplate::PbsLitMaterialMatrix => "PbsLitMaterialMatrix",
        SessionTemplate::AlphaCutoutMaskedQuads { .. } => "AlphaCutoutMaskedQuads",
        SessionTemplate::GltfTexturedStaticMesh { .. } => "GltfTexturedStaticMesh",
    }
}

fn build_scene_setup(template: SessionTemplate) -> SceneSetup {
    match template {
        SessionTemplate::Sphere => sphere_setup(),
        SessionTemplate::Torus {
            texture_rgba,
            texture_size,
        } => torus_setup(texture_rgba, texture_size),
        SessionTemplate::MultiPrimitiveUnlitGrid {
            checker_rgba,
            uv_ramp_rgba,
            texture_size,
        } => multi_primitive_unlit_grid_setup(checker_rgba, uv_ramp_rgba, texture_size),
        SessionTemplate::PbsLitMaterialMatrix => pbs_lit_material_matrix_setup(),
        SessionTemplate::AlphaCutoutMaskedQuads {
            mask_rgba,
            texture_size,
        } => alpha_cutout_masked_quads_setup(mask_rgba, texture_size),
        SessionTemplate::GltfTexturedStaticMesh {
            mesh,
            texture_rgba,
            texture_size,
        } => gltf_textured_static_mesh_setup(mesh, texture_rgba, texture_size),
    }
}

fn upload_scene_meshes(
    queues: &mut renderide_shared::ipc::HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
    prefix: &str,
    backing_dir: &std::path::Path,
    setup: &SceneSetup,
) -> Result<Vec<UploadedMesh>, HarnessError> {
    let mut uploaded = Vec::with_capacity(setup.meshes.len());
    for mesh in &setup.meshes {
        let upload = pack_mesh_upload(&mesh.mesh, mesh.bounds)
            .map_err(|e| HarnessError::QueueOptions(format!("pack mesh upload: {e}")))?;
        uploaded.push(upload_mesh(
            queues,
            lockstep,
            MeshUploadRequest {
                shared_memory_prefix: prefix,
                backing_dir,
                buffer_id: mesh.buffer_id,
                asset_id: mesh.asset_id,
                mesh: &upload,
                timeout: DEFAULT_ASSET_UPLOAD_TIMEOUT,
            },
        )?);
    }
    Ok(uploaded)
}

fn upload_scene_materials(
    queues: &mut renderide_shared::ipc::HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
    prefix: &str,
    backing_dir: &std::path::Path,
    setup: &SceneSetup,
) -> Result<Vec<BoundMaterial>, HarnessError> {
    let mut materials = Vec::new();
    for material in &setup.materials {
        if let Some(binding) = material.binding.as_ref() {
            materials.push(apply_material_binding(
                queues,
                lockstep,
                prefix,
                backing_dir,
                material.asset_id,
                material,
                binding,
            )?);
        }
    }
    Ok(materials)
}

fn sphere_setup() -> SceneSetup {
    let mesh_asset_id = asset_ids::SPHERE_MESH;
    let material_asset_id = asset_ids::SPHERE_MATERIAL;
    SceneSetup {
        meshes: vec![MeshAsset {
            asset_id: mesh_asset_id,
            buffer_id: asset_ids::MESH_BUFFER,
            mesh: generate_sphere(
                sphere_tessellation::LATITUDE_BANDS,
                sphere_tessellation::LONGITUDE_BANDS,
            ),
            bounds: RenderBoundingBox {
                center: Vec3::ZERO,
                extents: Vec3::splat(1.05),
            },
        }],
        materials: vec![MaterialAsset {
            asset_id: material_asset_id,
            update_batch_id: asset_ids::MATERIAL_UPDATE_BATCH_ID,
            property_request_id: asset_ids::PROPERTY_ID_REQUEST_ID,
            update_base_buffer_id: asset_ids::MATERIAL_UPDATE_BASE_BUFFER,
            binding: None,
        }],
        scene: SceneSubmission::single_mesh(mesh_asset_id, material_asset_id),
    }
}

fn torus_setup(texture_rgba: Vec<u8>, texture_size: (u32, u32)) -> SceneSetup {
    let outer = torus_geometry::MAJOR_RADIUS + torus_geometry::MINOR_RADIUS;
    let mesh_asset_id = asset_ids::TORUS_MESH;
    let material_asset_id = asset_ids::TORUS_MATERIAL;
    SceneSetup {
        meshes: vec![MeshAsset {
            asset_id: mesh_asset_id,
            buffer_id: asset_ids::MESH_BUFFER,
            mesh: generate_torus(
                torus_geometry::MAJOR_SEGMENTS,
                torus_geometry::MINOR_SEGMENTS,
                torus_geometry::MAJOR_RADIUS,
                torus_geometry::MINOR_RADIUS,
            ),
            bounds: RenderBoundingBox {
                center: Vec3::ZERO,
                extents: Vec3::new(
                    outer * 1.05,
                    torus_geometry::MINOR_RADIUS * 1.1,
                    outer * 1.05,
                ),
            },
        }],
        materials: vec![unlit_textured_material(UnlitTexturedMaterialParams {
            material_asset_id,
            shader_asset_id: asset_ids::TORUS_SHADER,
            texture_asset_id: asset_ids::TORUS_TEXTURE,
            texture_buffer_id: asset_ids::TEXTURE_DATA_BUFFER,
            update_batch_id: asset_ids::MATERIAL_UPDATE_BATCH_ID,
            property_request_id: asset_ids::PROPERTY_ID_REQUEST_ID,
            update_base_buffer_id: asset_ids::MATERIAL_UPDATE_BASE_BUFFER,
            variant_bits: shader_variants::UNLIT_TEXTURE,
            texture_rgba,
            texture_size,
            color: [1.0, 1.0, 1.0, 1.0],
            floats: Vec::new(),
        })],
        scene: SceneSubmission::single_mesh(mesh_asset_id, material_asset_id),
    }
}

fn multi_primitive_unlit_grid_setup(
    checker_rgba: Vec<u8>,
    uv_ramp_rgba: Vec<u8>,
    texture_size: (u32, u32),
) -> SceneSetup {
    let meshes = vec![
        mesh_asset(0, generate_cube(), centered_bounds(Vec3::splat(0.55))),
        mesh_asset(
            1,
            generate_sphere(16, 24),
            centered_bounds(Vec3::splat(1.05)),
        ),
        mesh_asset(
            2,
            generate_torus(40, 18, 0.58, 0.18),
            centered_bounds(Vec3::new(0.82, 0.22, 0.82)),
        ),
        mesh_asset(
            3,
            generate_quad(),
            centered_bounds(Vec3::new(0.55, 0.55, 0.05)),
        ),
    ];
    let materials = vec![
        unlit_textured_material_for_scene(
            0,
            checker_rgba.clone(),
            texture_size,
            [1.0, 0.95, 0.8, 1.0],
            Vec::new(),
        ),
        unlit_textured_material_for_scene(
            1,
            uv_ramp_rgba,
            texture_size,
            [1.0, 1.0, 1.0, 1.0],
            Vec::new(),
        ),
        unlit_textured_material_for_scene(
            2,
            checker_rgba,
            texture_size,
            [0.6, 0.9, 1.0, 1.0],
            Vec::new(),
        ),
        unlit_color_material_for_scene(3, [0.95, 0.32, 0.42, 1.0]),
    ];
    let renderables = [
        (0, Vec3::new(-1.05, 0.55, 0.0), Vec3::splat(0.8)),
        (1, Vec3::new(1.05, 0.55, 0.0), Vec3::splat(0.55)),
        (2, Vec3::new(-1.05, -0.65, 0.0), Vec3::splat(0.85)),
        (3, Vec3::new(1.05, -0.65, 0.0), Vec3::new(1.15, 1.15, 1.0)),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, (asset_index, position, scale))| SceneRenderable {
        transform_id: index as i32,
        pose: transform(position, scale, Quat::IDENTITY),
        mesh_asset_id: scene_mesh_id(asset_index),
        material_asset_id: scene_material_id(asset_index),
        sorting_order: 0,
    })
    .collect();
    SceneSetup {
        meshes,
        materials,
        scene: SceneSubmission {
            camera_world_pose: transform(Vec3::new(0.0, 0.0, -4.2), Vec3::ONE, Quat::IDENTITY),
            renderables,
            lights: Vec::new(),
        },
    }
}

fn pbs_lit_material_matrix_setup() -> SceneSetup {
    let meshes = vec![mesh_asset(
        0,
        generate_sphere(18, 28),
        centered_bounds(Vec3::splat(1.05)),
    )];
    let colors = [
        [0.95, 0.25, 0.18, 1.0],
        [0.28, 0.65, 0.95, 1.0],
        [0.85, 0.75, 0.25, 1.0],
        [0.9, 0.9, 0.95, 1.0],
    ];
    let metallic = [0.0, 0.0, 0.65, 1.0];
    let glossiness = [0.25, 0.75, 0.45, 0.9];
    let materials = (0..4)
        .map(|index| {
            pbs_material_for_scene(index, colors[index], metallic[index], glossiness[index])
        })
        .collect();
    let renderables = [
        Vec3::new(-1.05, 0.55, 0.0),
        Vec3::new(1.05, 0.55, 0.0),
        Vec3::new(-1.05, -0.65, 0.0),
        Vec3::new(1.05, -0.65, 0.0),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, position)| SceneRenderable {
        transform_id: index as i32,
        pose: transform(position, Vec3::splat(0.52), Quat::IDENTITY),
        mesh_asset_id: scene_mesh_id(0),
        material_asset_id: scene_material_id(index as i32),
        sorting_order: 0,
    })
    .collect();
    let lights = vec![
        SceneLight {
            transform_id: 10,
            pose: transform(
                Vec3::new(-1.5, 2.2, -1.5),
                Vec3::ONE,
                Quat::from_rotation_x(-0.65),
            ),
            state: directional_light([1.0, 0.92, 0.82], 1.4),
        },
        SceneLight {
            transform_id: 11,
            pose: transform(Vec3::new(1.8, 0.8, -1.2), Vec3::ONE, Quat::IDENTITY),
            state: point_light([0.35, 0.55, 1.0], 6.0, 4.0),
        },
    ];
    SceneSetup {
        meshes,
        materials,
        scene: SceneSubmission {
            camera_world_pose: transform(Vec3::new(0.0, 0.0, -4.0), Vec3::ONE, Quat::IDENTITY),
            renderables,
            lights,
        },
    }
}

fn alpha_cutout_masked_quads_setup(mask_rgba: Vec<u8>, texture_size: (u32, u32)) -> SceneSetup {
    let meshes = vec![mesh_asset(
        0,
        generate_quad(),
        centered_bounds(Vec3::new(0.55, 0.55, 0.05)),
    )];
    let materials = vec![
        unlit_textured_material_for_scene(
            0,
            mask_rgba,
            texture_size,
            [0.2, 0.95, 0.55, 1.0],
            vec![FloatBinding {
                property_name: "_Cutoff",
                value: 0.5,
            }],
        )
        .with_variant(shader_variants::UNLIT_TEXTURE | shader_variants::UNLIT_ALPHATEST),
    ];
    let renderables = [
        (
            Vec3::new(-0.35, 0.0, 0.0),
            Quat::from_rotation_y(0.45),
            [1.4, 1.4, 1.0],
        ),
        (
            Vec3::new(0.35, 0.0, 0.05),
            Quat::from_rotation_y(-0.45),
            [1.4, 1.4, 1.0],
        ),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, (position, rotation, scale))| SceneRenderable {
        transform_id: index as i32,
        pose: transform(position, Vec3::from_array(scale), rotation),
        mesh_asset_id: scene_mesh_id(0),
        material_asset_id: scene_material_id(0),
        sorting_order: index as i32,
    })
    .collect();
    SceneSetup {
        meshes,
        materials,
        scene: SceneSubmission {
            camera_world_pose: default_camera_world_pose(),
            renderables,
            lights: Vec::new(),
        },
    }
}

fn gltf_textured_static_mesh_setup(
    mesh: Mesh,
    texture_rgba: Vec<u8>,
    texture_size: (u32, u32),
) -> SceneSetup {
    let bounds = bounds_for_mesh(&mesh);
    SceneSetup {
        meshes: vec![MeshAsset {
            asset_id: scene_mesh_id(0),
            buffer_id: scene_mesh_buffer_id(0),
            mesh,
            bounds,
        }],
        materials: vec![unlit_textured_material_for_scene(
            0,
            texture_rgba,
            texture_size,
            [1.0, 1.0, 1.0, 1.0],
            Vec::new(),
        )],
        scene: SceneSubmission {
            camera_world_pose: transform(Vec3::new(0.0, 0.0, -3.4), Vec3::ONE, Quat::IDENTITY),
            renderables: vec![SceneRenderable {
                transform_id: 0,
                pose: identity_transform(),
                mesh_asset_id: scene_mesh_id(0),
                material_asset_id: scene_material_id(0),
                sorting_order: 0,
            }],
            lights: Vec::new(),
        },
    }
}

fn mesh_asset(index: i32, mesh: Mesh, bounds: RenderBoundingBox) -> MeshAsset {
    MeshAsset {
        asset_id: scene_mesh_id(index),
        buffer_id: scene_mesh_buffer_id(index),
        mesh,
        bounds,
    }
}

fn pbs_material_for_scene(
    index: usize,
    color: [f32; 4],
    metallic: f32,
    glossiness: f32,
) -> MaterialAsset {
    MaterialAsset {
        asset_id: scene_material_id(index as i32),
        update_batch_id: scene_update_batch_id(index as i32),
        property_request_id: scene_property_request_id(index as i32),
        update_base_buffer_id: scene_material_update_buffer_id(index as i32),
        binding: Some(MaterialBinding {
            shader_asset_id: scene_shader_id(index as i32),
            shader_name: "PBSMetallic.shader",
            shader_variant_bits: None,
            textures: Vec::new(),
            floats: vec![
                FloatBinding {
                    property_name: "_Metallic",
                    value: metallic,
                },
                FloatBinding {
                    property_name: "_Glossiness",
                    value: glossiness,
                },
            ],
            float4s: vec![Float4Binding {
                property_name: "_Color",
                value: color,
            }],
        }),
    }
}

fn unlit_color_material_for_scene(index: i32, color: [f32; 4]) -> MaterialAsset {
    MaterialAsset {
        asset_id: scene_material_id(index),
        update_batch_id: scene_update_batch_id(index),
        property_request_id: scene_property_request_id(index),
        update_base_buffer_id: scene_material_update_buffer_id(index),
        binding: Some(MaterialBinding {
            shader_asset_id: scene_shader_id(index),
            shader_name: "Unlit.shader",
            shader_variant_bits: Some(shader_variants::UNLIT_COLOR),
            textures: Vec::new(),
            floats: Vec::new(),
            float4s: vec![Float4Binding {
                property_name: "_Color",
                value: color,
            }],
        }),
    }
}

fn unlit_textured_material_for_scene(
    index: i32,
    texture_rgba: Vec<u8>,
    texture_size: (u32, u32),
    color: [f32; 4],
    floats: Vec<FloatBinding>,
) -> MaterialAsset {
    unlit_textured_material(UnlitTexturedMaterialParams {
        material_asset_id: scene_material_id(index),
        shader_asset_id: scene_shader_id(index),
        texture_asset_id: scene_texture_id(index),
        texture_buffer_id: scene_texture_buffer_id(index),
        update_batch_id: scene_update_batch_id(index),
        property_request_id: scene_property_request_id(index),
        update_base_buffer_id: scene_material_update_buffer_id(index),
        variant_bits: shader_variants::UNLIT_TEXTURE | shader_variants::UNLIT_COLOR,
        texture_rgba,
        texture_size,
        color,
        floats,
    })
}

fn unlit_textured_material(params: UnlitTexturedMaterialParams) -> MaterialAsset {
    MaterialAsset {
        asset_id: params.material_asset_id,
        update_batch_id: params.update_batch_id,
        property_request_id: params.property_request_id,
        update_base_buffer_id: params.update_base_buffer_id,
        binding: Some(MaterialBinding {
            shader_asset_id: params.shader_asset_id,
            shader_name: "Unlit.shader",
            shader_variant_bits: Some(params.variant_bits),
            textures: vec![TextureBinding {
                asset_id: params.texture_asset_id,
                buffer_id: params.texture_buffer_id,
                property_name: "_Tex",
                st_property_name: "_Tex_ST",
                st_value: [1.0, 1.0, 0.0, 0.0],
                rgba_bytes: params.texture_rgba,
                size: params.texture_size,
                color_profile: ColorProfile::SRGB,
            }],
            floats: params.floats,
            float4s: vec![Float4Binding {
                property_name: "_Color",
                value: params.color,
            }],
        }),
    }
}

trait MaterialAssetVariantExt {
    fn with_variant(self, variant_bits: u32) -> Self;
}

impl MaterialAssetVariantExt for MaterialAsset {
    fn with_variant(mut self, variant_bits: u32) -> Self {
        if let Some(binding) = self.binding.as_mut() {
            binding.shader_variant_bits = Some(variant_bits);
        }
        self
    }
}

fn apply_material_binding(
    queues: &mut renderide_shared::ipc::HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
    prefix: &str,
    backing_dir: &std::path::Path,
    material_asset_id: i32,
    material: &MaterialAsset,
    binding: &MaterialBinding,
) -> Result<BoundMaterial, HarnessError> {
    upload_shader(
        queues,
        lockstep,
        binding.shader_asset_id,
        binding.shader_name,
        binding.shader_variant_bits,
        DEFAULT_ASSET_UPLOAD_TIMEOUT,
    )?;

    for tex in &binding.textures {
        let _uploaded_texture = upload_texture2d_rgba8(
            queues,
            lockstep,
            Texture2DUploadRequest {
                shared_memory_prefix: prefix,
                backing_dir,
                buffer_id: tex.buffer_id,
                asset_id: tex.asset_id,
                width: tex.size.0,
                height: tex.size.1,
                rgba_bytes: &tex.rgba_bytes,
                color_profile: tex.color_profile,
                timeout: DEFAULT_ASSET_UPLOAD_TIMEOUT,
            },
        )?;
    }

    let property_names = material_property_names(binding);
    let property_ids = if property_names.is_empty() {
        Vec::new()
    } else {
        request_property_ids(
            queues,
            lockstep,
            PropertyIdLookup {
                request_id: material.property_request_id,
                names: &property_names,
                timeout: DEFAULT_ASSET_UPLOAD_TIMEOUT,
            },
        )?
    };
    if property_ids.len() != property_names.len() {
        return Err(HarnessError::QueueOptions(format!(
            "expected {} property ids, got {}",
            property_names.len(),
            property_ids.len()
        )));
    }

    let mut ops: Vec<MaterialUpdateOp> = Vec::new();
    ops.push(MaterialUpdateOp::SelectTarget { material_asset_id });
    ops.push(MaterialUpdateOp::SetShader {
        shader_asset_id: binding.shader_asset_id,
    });

    let mut cursor = 0usize;
    for tex in &binding.textures {
        let tex_property_id = property_ids[cursor];
        let tex_st_property_id = property_ids[cursor + 1];
        cursor += 2;
        ops.push(MaterialUpdateOp::SetTexture {
            property_id: tex_property_id,
            packed_handle: pack_texture2d_handle(tex.asset_id),
        });
        ops.push(MaterialUpdateOp::SetFloat4 {
            property_id: tex_st_property_id,
            value: tex.st_value,
        });
    }
    for scalar in &binding.floats {
        ops.push(MaterialUpdateOp::SetFloat {
            property_id: property_ids[cursor],
            value: scalar.value,
        });
        cursor += 1;
    }
    for vector in &binding.float4s {
        ops.push(MaterialUpdateOp::SetFloat4 {
            property_id: property_ids[cursor],
            value: vector.value,
        });
        cursor += 1;
    }
    ops.push(MaterialUpdateOp::UpdateBatchEnd);

    apply_material_batch(
        queues,
        lockstep,
        MaterialBatchRequest {
            shared_memory_prefix: prefix,
            backing_dir,
            base_buffer_id: material.update_base_buffer_id,
            update_batch_id: material.update_batch_id,
            material_update_count: 1,
            ops: &ops,
            timeout: DEFAULT_ASSET_UPLOAD_TIMEOUT,
        },
    )
}

fn material_property_names(binding: &MaterialBinding) -> Vec<&'static str> {
    let mut names = Vec::with_capacity(
        binding.textures.len() * 2 + binding.floats.len() + binding.float4s.len(),
    );
    for texture in &binding.textures {
        names.push(texture.property_name);
        names.push(texture.st_property_name);
    }
    names.extend(binding.floats.iter().map(|value| value.property_name));
    names.extend(binding.float4s.iter().map(|value| value.property_name));
    names
}

fn transform(position: Vec3, scale: Vec3, rotation: Quat) -> RenderTransform {
    RenderTransform {
        position,
        scale,
        rotation,
    }
}

fn bounds_for_mesh(mesh: &Mesh) -> RenderBoundingBox {
    if mesh.vertices.is_empty() {
        return centered_bounds(Vec3::splat(0.05));
    }

    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for vertex in &mesh.vertices {
        let pos = Vec3::from_array(vertex.position);
        min = min.min(pos);
        max = max.max(pos);
    }
    let center = (min + max) * 0.5;
    let extents = ((max - min) * 0.5).max(Vec3::splat(0.05));
    RenderBoundingBox { center, extents }
}

const fn scene_mesh_id(index: i32) -> i32 {
    asset_ids::SCENE_MESH_BASE + index
}

const fn scene_material_id(index: i32) -> i32 {
    asset_ids::SCENE_MATERIAL_BASE + index
}

const fn scene_shader_id(index: i32) -> i32 {
    asset_ids::SCENE_SHADER_BASE + index
}

const fn scene_texture_id(index: i32) -> i32 {
    asset_ids::SCENE_TEXTURE_BASE + index
}

const fn scene_mesh_buffer_id(index: i32) -> i32 {
    asset_ids::SCENE_MESH_BUFFER_BASE + index
}

const fn scene_texture_buffer_id(index: i32) -> i32 {
    asset_ids::SCENE_TEXTURE_BUFFER_BASE + index
}

const fn scene_material_update_buffer_id(index: i32) -> i32 {
    asset_ids::SCENE_MATERIAL_UPDATE_BUFFER_BASE + index * 4
}

const fn scene_update_batch_id(index: i32) -> i32 {
    asset_ids::SCENE_MATERIAL_UPDATE_BATCH_BASE + index
}

const fn scene_property_request_id(index: i32) -> i32 {
    asset_ids::SCENE_PROPERTY_ID_REQUEST_BASE + index
}

#[cfg(test)]
mod tests {
    use super::{
        SessionTemplate, asset_ids, build_scene_setup, material_property_names, scene_material_id,
        shader_variants,
    };

    #[test]
    fn torus_unlit_perlin_requests_texture_variant() {
        let setup = build_scene_setup(SessionTemplate::Torus {
            texture_rgba: vec![255, 255, 255, 255],
            texture_size: (1, 1),
        });
        let binding = setup.materials[0]
            .binding
            .as_ref()
            .expect("torus case must upload an unlit material");

        assert_eq!(binding.shader_name, "Unlit.shader");
        assert_eq!(
            binding.shader_variant_bits,
            Some(shader_variants::UNLIT_TEXTURE)
        );
    }

    #[test]
    fn material_property_names_follow_payload_order() {
        let setup = build_scene_setup(SessionTemplate::Torus {
            texture_rgba: vec![255, 255, 255, 255],
            texture_size: (1, 1),
        });
        let binding = setup.materials[0].binding.as_ref().expect("binding");
        assert_eq!(
            material_property_names(binding),
            vec!["_Tex", "_Tex_ST", "_Color"]
        );
    }

    #[test]
    fn torus_material_update_buffers_do_not_overlap_texture_buffer() {
        let material_buffers =
            asset_ids::MATERIAL_UPDATE_BASE_BUFFER..=asset_ids::MATERIAL_UPDATE_BASE_BUFFER + 3;

        assert!(!material_buffers.contains(&asset_ids::TEXTURE_DATA_BUFFER));
    }

    #[test]
    fn pbs_matrix_builds_four_materials_against_one_mesh() {
        let setup = build_scene_setup(SessionTemplate::PbsLitMaterialMatrix);
        assert_eq!(setup.meshes.len(), 1);
        assert_eq!(setup.materials.len(), 4);
        assert_eq!(setup.scene.renderables.len(), 4);
        assert_eq!(setup.scene.lights.len(), 2);
        assert_eq!(
            setup.scene.renderables[3].material_asset_id,
            scene_material_id(3)
        );
    }
}
