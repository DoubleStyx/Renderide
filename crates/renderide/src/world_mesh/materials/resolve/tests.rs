use std::sync::Arc;

use super::*;
use crate::materials::host_data::{
    MaterialPropertyStore, MaterialPropertyValue, PropertyIdRegistry,
};
use crate::materials::{MaterialRouter, RasterPipelineKind, UNITY_RENDER_QUEUE_TRANSPARENT};

#[test]
fn generated_billboard_mesh_preserves_compatible_source_pipeline() {
    let mut store = MaterialPropertyStore::new();
    store.set_shader_asset_for_material(7, 99);
    let dict = MaterialDictionary::new(&store);
    let mut router = MaterialRouter::new(RasterPipelineKind::Null);
    router.set_shader_pipeline(
        99,
        RasterPipelineKind::EmbeddedStem(Arc::from("pbsmetallic_default")),
    );
    let ids = MaterialPipelinePropertyIds::new(&PropertyIdRegistry::new());
    let resolved =
        resolve_material_batch(7, None, &dict, &router, &ids, ShaderPermutation::default());
    let mut key = batch_key_from_resolved(
        7,
        None,
        false,
        RasterFrontFace::Clockwise,
        RasterPrimitiveTopology::TriangleList,
        &resolved,
    );
    let mesh_asset_id = crate::particles::billboard_render_buffer_mesh_asset_id(3).unwrap();

    apply_render_buffer_mesh_pipeline_override(
        &mut key,
        mesh_asset_id,
        ShaderPermutation::default(),
    );

    let RasterPipelineKind::EmbeddedStem(stem) = &key.pipeline else {
        panic!("expected embedded source pipeline");
    };
    assert_eq!(stem.as_ref(), "pbsmetallic_default");
    assert!(key.uses_render_buffer_billboard);
    assert!(key.embedded_needs_tangent);
    assert!(!key.embedded_raw_tangent_payload);
    assert!(!key.embedded_raw_normal_payload);
}

#[test]
fn generated_billboard_mesh_preserves_source_blend_for_compatible_pipeline() {
    let registry = PropertyIdRegistry::new();
    let render_queue = registry.intern("_RenderQueue");
    let mut store = MaterialPropertyStore::new();
    store.set_shader_asset_for_material(7, 99);
    store.set_material(
        7,
        render_queue,
        MaterialPropertyValue::Float(UNITY_RENDER_QUEUE_TRANSPARENT as f32),
    );
    let dict = MaterialDictionary::new(&store);
    let mut router = MaterialRouter::new(RasterPipelineKind::Null);
    router.set_shader_pipeline(
        99,
        RasterPipelineKind::EmbeddedStem(Arc::from("pbsvertexcolortransparent_default")),
    );
    let ids = MaterialPipelinePropertyIds::new(&registry);
    let resolved =
        resolve_material_batch(7, None, &dict, &router, &ids, ShaderPermutation::default());
    let mut key = batch_key_from_resolved(
        7,
        None,
        false,
        RasterFrontFace::Clockwise,
        RasterPrimitiveTopology::TriangleList,
        &resolved,
    );
    let mesh_asset_id = crate::particles::billboard_render_buffer_mesh_asset_id(3).unwrap();

    assert_eq!(key.blend_mode, MaterialBlendMode::StemDefault);
    assert_eq!(
        key.transparent_class,
        TransparentMaterialClass::OrderedAlpha
    );

    apply_render_buffer_mesh_pipeline_override(
        &mut key,
        mesh_asset_id,
        ShaderPermutation::default(),
    );

    assert_eq!(key.blend_mode, MaterialBlendMode::StemDefault);
    assert_eq!(
        key.transparent_class,
        TransparentMaterialClass::OrderedAlpha
    );
}
