//! Draw batch collection: filters drawables, builds draw entries, and creates space batches.
//!
//! Used by [`Session::collect_draw_batches`](crate::session::Session::collect_draw_batches) and [`Session::collect_draw_batches_for_task`](crate::session::Session::collect_draw_batches_for_task).
//!
//! # Pipeline resolution DAG (CPU → GPU)
//!
//! End-to-end data flow for mesh draws (aligned with Renderite.Unity asset + material routing):
//!
//! 1. **Inputs**: [`crate::scene::Scene`], [`crate::scene::Drawable`] (mesh id, [`crate::scene::MeshMaterialSlot`] list,
//!    stencil / overrides), [`crate::assets::AssetRegistry`] (meshes, shaders, textures,
//!    [`crate::assets::MaterialPropertyStore`]), and [`crate::config::RenderConfig`].
//! 2. **Per-draw shading tuple**: [`super::filter_and_collect_drawables`] produces [`FilteredDrawable`] with
//!    [`crate::gpu::PipelineVariant`] and [`crate::gpu::ShaderKey`] via [`pipeline::resolve_pipeline_for_material_draw`].
//! 3. **GPU key**: [`crate::gpu::PipelineKey`] `(Option<host_shader_asset_id>, PipelineVariant)` indexes
//!    [`crate::gpu::PipelineRegistry`] / [`crate::gpu::PipelineManager::get_pipeline`], which may lazily create
//!    host-material or native-UI pipelines using [`crate::gpu::PipelineDescriptorCache`].
//! 4. **Bind resources**: Per-material texture/uniform binds for native UI and world-unlit paths use
//!    [`crate::gpu::MaterialGpuResources`] (see [`crate::gpu::native_ui_bind_cache`]). Recording is
//!    centralized in the `material_bind_record` submodule (Strangler Fig seam) after `bind_draw`.
//! 5. **MRT / PBR remapping** at record time: [`crate::render::pass::mesh_draw::pipeline::resolve_pipeline_for_group`]
//!    maps variants for the active pass (including `Material` + MRT rules).

mod batch;
mod filter;
mod native_ui;
mod pipeline;

pub(super) use batch::{build_draw_entries, create_space_batch};
pub(super) use filter::filter_and_collect_drawables;
// Stable `crate::session::collect::*` paths for docs and future callers.
#[cfg(test)]
#[allow(unused_imports)]
pub(crate) use native_ui::mesh_has_ui_canvas_vertices;
#[allow(unused_imports)]
pub(crate) use native_ui::{
    apply_native_ui_pipeline_variant, apply_ui_mesh_pbr_fallback_for_non_native_shader,
    mesh_has_native_ui_vertices,
};
#[allow(unused_imports)]
pub(super) use pipeline::FilteredDrawable;
#[allow(unused_imports)]
pub(super) use pipeline::{
    compute_pipeline_variant_for_drawable, resolve_overlay_stencil_state, resolved_material_slots,
};

#[cfg(test)]
mod tests {
    use super::pipeline::maybe_upgrade_pbr_host_albedo;
    use super::{
        FilteredDrawable, apply_native_ui_pipeline_variant,
        apply_ui_mesh_pbr_fallback_for_non_native_shader, build_draw_entries, create_space_batch,
        mesh_has_ui_canvas_vertices, resolved_material_slots,
    };
    use crate::assets::AssetRegistry;
    use crate::assets::MaterialPropertyValue;
    use crate::assets::mesh::MeshAsset;
    use crate::config::{RenderConfig, ShaderDebugOverride};
    use crate::gpu::{PipelineVariant, ShaderKey};
    use crate::render::batch::DrawEntry;
    use crate::scene::MeshMaterialSlot;
    use crate::scene::{Drawable, Scene};
    use crate::session::native_ui_routing_metrics::NativeUiRoutingFrameMetrics;
    use crate::shared::{
        IndexBufferFormat, RenderBoundingBox, ShadowCastMode, VertexAttributeDescriptor,
        VertexAttributeFormat, VertexAttributeType,
    };
    use crate::stencil::StencilState;
    use glam::Mat4;

    fn make_scene(space_id: i32, is_overlay: bool) -> Scene {
        Scene {
            id: space_id,
            is_overlay,
            ..Default::default()
        }
    }

    #[test]
    fn create_space_batch_returns_none_when_empty() {
        let scene = make_scene(0, false);
        let batch = create_space_batch(0, &scene, vec![], None);
        assert!(batch.is_none());
    }

    #[test]
    fn create_space_batch_returns_some_when_non_empty() {
        let mut scene = make_scene(5, false);
        scene.view_transform = crate::shared::RenderTransform::default();
        let draw = DrawEntry {
            model_matrix: Mat4::IDENTITY,
            node_id: 0,
            mesh_asset_id: 1,
            is_skinned: false,
            material_id: -1,
            sort_key: 0,
            bone_transform_ids: None,
            root_bone_transform_id: None,
            blendshape_weights: None,
            pipeline_variant: PipelineVariant::NormalDebug,
            shader_key: ShaderKey::builtin_only(PipelineVariant::NormalDebug),
            stencil_state: None,
            shadow_cast_mode: crate::shared::ShadowCastMode::on,
            mesh_renderer_property_block_slot0_id: None,
            submesh_index_range: None,
        };
        let batch = create_space_batch(5, &scene, vec![draw], None);
        let batch = batch.expect("should have batch");
        assert_eq!(batch.space_id, 5);
        assert!(!batch.is_overlay);
        assert_eq!(batch.draws.len(), 1);
    }

    #[test]
    fn build_draw_entries_preserves_order() {
        let filtered = vec![FilteredDrawable {
            drawable: Drawable {
                node_id: 0,
                mesh_handle: 1,
                material_handle: Some(10),
                sort_key: 5,
                is_skinned: false,
                ..Default::default()
            },
            world_matrix: Mat4::IDENTITY,
            pipeline_variant: PipelineVariant::NormalDebug,
            shader_key: ShaderKey::builtin_only(PipelineVariant::NormalDebug),
            submesh_index_range: Some((12, 30)),
        }];
        let entries = build_draw_entries(filtered);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].material_id, 10);
        assert_eq!(entries[0].sort_key, 5);
        assert_eq!(entries[0].submesh_index_range, Some((12, 30)));
    }

    #[test]
    fn build_draw_entries_propagates_shadow_cast_mode() {
        let filtered = vec![FilteredDrawable {
            drawable: Drawable {
                node_id: 0,
                mesh_handle: 1,
                shadow_cast_mode: ShadowCastMode::off,
                ..Default::default()
            },
            world_matrix: Mat4::IDENTITY,
            pipeline_variant: PipelineVariant::NormalDebug,
            shader_key: ShaderKey::builtin_only(PipelineVariant::NormalDebug),
            submesh_index_range: None,
        }];
        let entries = build_draw_entries(filtered);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].shadow_cast_mode, ShadowCastMode::off);
    }

    #[test]
    fn mesh_has_ui_canvas_vertices_false_without_mesh() {
        let reg = AssetRegistry::new();
        assert!(!mesh_has_ui_canvas_vertices(&reg, 1));
    }

    #[test]
    fn resolved_material_slots_uses_vec_when_non_empty() {
        let d = Drawable {
            material_slots: vec![MeshMaterialSlot {
                material_asset_id: 1,
                property_block_id: Some(2),
            }],
            material_handle: Some(99),
            ..Default::default()
        };
        let s = resolved_material_slots(&d);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].material_asset_id, 1);
        assert_eq!(s[0].property_block_id, Some(2));
    }

    #[test]
    fn resolved_material_slots_falls_back_to_legacy_handle() {
        let d = Drawable {
            material_handle: Some(5),
            mesh_renderer_property_block_slot0_id: Some(6),
            ..Default::default()
        };
        let s = resolved_material_slots(&d);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].material_asset_id, 5);
        assert_eq!(s[0].property_block_id, Some(6));
    }

    #[test]
    fn apply_native_ui_overlay_respects_overlay_only() {
        let reg = AssetRegistry::new();
        let rc = RenderConfig {
            use_native_ui_wgsl: true,
            ..Default::default()
        };
        let v = apply_native_ui_pipeline_variant(
            false,
            false,
            None,
            &rc,
            Some(42),
            7,
            1,
            PipelineVariant::NormalDebug,
            &reg,
        );
        assert_eq!(v, PipelineVariant::NormalDebug);
    }

    #[test]
    fn apply_native_ui_overlay_disabled_when_config_off() {
        let reg = AssetRegistry::new();
        let rc = RenderConfig {
            use_native_ui_wgsl: false,
            ..Default::default()
        };
        let v = apply_native_ui_pipeline_variant(
            true,
            false,
            None,
            &rc,
            Some(99),
            7,
            1,
            PipelineVariant::NormalDebug,
            &reg,
        );
        assert_eq!(v, PipelineVariant::NormalDebug);
    }

    #[test]
    fn apply_native_ui_overlay_skips_legacy_shader_override() {
        let reg = AssetRegistry::new();
        let rc = RenderConfig {
            use_native_ui_wgsl: true,
            shader_debug_override: ShaderDebugOverride::ForceLegacyGlobalShading,
            ..Default::default()
        };
        let v = apply_native_ui_pipeline_variant(
            true,
            false,
            None,
            &rc,
            Some(42),
            7,
            1,
            PipelineVariant::NormalDebug,
            &reg,
        );
        assert_eq!(v, PipelineVariant::NormalDebug);
    }

    fn mesh_with_uv0(id: i32) -> MeshAsset {
        MeshAsset {
            id,
            vertex_data: Vec::new(),
            index_data: Vec::new(),
            vertex_count: 0,
            index_count: 0,
            index_format: IndexBufferFormat::u_int16,
            submeshes: Vec::new(),
            vertex_attributes: vec![VertexAttributeDescriptor {
                attribute: VertexAttributeType::uv0,
                format: VertexAttributeFormat::float32,
                dimensions: 4,
            }],
            bounds: RenderBoundingBox::default(),
            bind_poses: None,
            bone_counts: None,
            bone_weights: None,
            blendshape_offsets: None,
            num_blendshapes: 0,
        }
    }

    #[test]
    fn maybe_upgrade_pbr_host_albedo_upgrades_when_mesh_uv_and_main_tex() {
        let mut reg = AssetRegistry::new();
        reg.insert_mesh_for_tests(mesh_with_uv0(7));
        let mat_id = 10;
        let main_tex_pid = 88;
        reg.material_property_store.set_material(
            mat_id,
            main_tex_pid,
            MaterialPropertyValue::Texture(1),
        );
        let rc = RenderConfig {
            pbr_bind_host_main_texture: true,
            pbr_host_main_tex_property_id: main_tex_pid,
            ..Default::default()
        };
        let drawable = Drawable {
            mesh_handle: 7,
            ..Default::default()
        };
        let v = maybe_upgrade_pbr_host_albedo(
            PipelineVariant::Pbr,
            &rc,
            &reg.material_property_store,
            &drawable,
            mat_id,
            7,
            &reg,
        );
        assert_eq!(v, PipelineVariant::PbrHostAlbedo);
    }

    #[test]
    fn maybe_upgrade_pbr_host_albedo_keeps_pbr_without_main_tex_texture() {
        let mut reg = AssetRegistry::new();
        reg.insert_mesh_for_tests(mesh_with_uv0(7));
        let mat_id = 10;
        let main_tex_pid = 88;
        let rc = RenderConfig {
            pbr_bind_host_main_texture: true,
            pbr_host_main_tex_property_id: main_tex_pid,
            ..Default::default()
        };
        let drawable = Drawable {
            mesh_handle: 7,
            ..Default::default()
        };
        let v = maybe_upgrade_pbr_host_albedo(
            PipelineVariant::Pbr,
            &rc,
            &reg.material_property_store,
            &drawable,
            mat_id,
            7,
            &reg,
        );
        assert_eq!(v, PipelineVariant::Pbr);
    }

    #[test]
    fn ui_mesh_pbr_fallback_default_keeps_fallback_variant() {
        let mut reg = AssetRegistry::new();
        reg.insert_mesh_for_tests(mesh_with_uv0(9));
        let drawable = Drawable {
            mesh_handle: 9,
            ..Default::default()
        };
        let rc = RenderConfig {
            use_native_ui_wgsl: true,
            use_pbr: true,
            native_ui_uivert_pbr_fallback: false,
            ..Default::default()
        };
        let v = apply_ui_mesh_pbr_fallback_for_non_native_shader(
            &rc,
            &reg,
            &drawable,
            PipelineVariant::NormalDebug,
            true,
            PipelineVariant::NormalDebug,
        );
        assert_eq!(v, PipelineVariant::NormalDebug);
    }

    #[test]
    fn ui_mesh_pbr_fallback_legacy_forces_pbr() {
        let mut reg = AssetRegistry::new();
        reg.insert_mesh_for_tests(mesh_with_uv0(9));
        let drawable = Drawable {
            mesh_handle: 9,
            ..Default::default()
        };
        let rc = RenderConfig {
            use_native_ui_wgsl: true,
            use_pbr: true,
            native_ui_uivert_pbr_fallback: true,
            ..Default::default()
        };
        let v = apply_ui_mesh_pbr_fallback_for_non_native_shader(
            &rc,
            &reg,
            &drawable,
            PipelineVariant::NormalDebug,
            true,
            PipelineVariant::NormalDebug,
        );
        assert_eq!(v, PipelineVariant::Pbr);
    }

    #[test]
    fn native_ui_routing_metrics_count_skip_when_wgsl_off() {
        let reg = AssetRegistry::new();
        let rc = RenderConfig {
            use_native_ui_wgsl: false,
            native_ui_routing_metrics: true,
            ..Default::default()
        };
        let _ = apply_native_ui_pipeline_variant(
            true,
            false,
            None,
            &rc,
            Some(1),
            1,
            1,
            PipelineVariant::NormalDebug,
            &reg,
        );
        let m = NativeUiRoutingFrameMetrics::snapshot_and_reset();
        assert_eq!(m.skip_native_ui_wgsl_off, 1);
    }

    #[test]
    fn apply_native_ui_overlay_stencil_selects_stencil_variant() {
        let mut reg = AssetRegistry::new();
        reg.insert_mesh_for_tests(mesh_with_uv0(5));
        reg.handle_shader_upload(crate::shared::ShaderUpload {
            asset_id: 42,
            file: Some("UI_Unlit".to_string()),
        });
        let rc = RenderConfig {
            use_native_ui_wgsl: true,
            native_ui_overlay_stencil_pipelines: true,
            ..Default::default()
        };
        let st = StencilState::default();
        let v = apply_native_ui_pipeline_variant(
            true,
            false,
            Some(&st),
            &rc,
            Some(42),
            3,
            5,
            PipelineVariant::NormalDebug,
            &reg,
        );
        assert_eq!(v, PipelineVariant::NativeUiUnlitStencil { material_id: 3 });
    }

    #[test]
    fn apply_native_ui_world_space_routes_main_pass_canvas() {
        let mut reg = AssetRegistry::new();
        reg.insert_mesh_for_tests(mesh_with_uv0(5));
        reg.handle_shader_upload(crate::shared::ShaderUpload {
            asset_id: 42,
            file: Some("UI_Unlit".to_string()),
        });
        let rc = RenderConfig {
            use_native_ui_wgsl: true,
            native_ui_world_space: true,
            ..Default::default()
        };
        let v = apply_native_ui_pipeline_variant(
            false,
            false,
            None,
            &rc,
            Some(42),
            3,
            5,
            PipelineVariant::NormalDebug,
            &reg,
        );
        assert_eq!(v, PipelineVariant::NativeUiUnlit { material_id: 3 });
    }

    #[test]
    fn apply_native_ui_unrecognized_host_shader_keeps_variant() {
        let mut reg = AssetRegistry::new();
        reg.insert_mesh_for_tests(mesh_with_uv0(5));
        let rc = RenderConfig {
            use_native_ui_wgsl: true,
            native_ui_world_space: true,
            ..Default::default()
        };
        let v = apply_native_ui_pipeline_variant(
            false,
            false,
            None,
            &rc,
            Some(99),
            3,
            5,
            PipelineVariant::NormalDebug,
            &reg,
        );
        assert_eq!(v, PipelineVariant::NormalDebug);
    }
}
