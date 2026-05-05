//! Shared uniform-packing test fixtures.

use std::sync::Arc;

use hashbrown::HashMap;

use super::super::*;
use crate::gpu_pools::{
    CubemapPool, RenderTexturePool, Texture3dPool, TexturePool, VideoTexturePool,
};
use crate::materials::embedded::layout::{EmbeddedSharedKeywordIds, StemEmbeddedPropertyIds};
use crate::materials::host_data::MaterialPropertyLookupIds;
use crate::materials::host_data::PropertyIdRegistry;
use crate::materials::{ReflectedMaterialUniformBlock, ReflectedUniformScalarKind};

pub(super) fn lookup(material_id: i32) -> MaterialPropertyLookupIds {
    MaterialPropertyLookupIds {
        material_asset_id: material_id,
        mesh_property_block_slot0: None,
    }
}

/// Builds an empty texture-pool set for uniform-packer tests that only need binding metadata.
pub(super) fn empty_texture_pools() -> (
    TexturePool,
    Texture3dPool,
    CubemapPool,
    RenderTexturePool,
    VideoTexturePool,
) {
    (
        TexturePool::default_pool(),
        Texture3dPool::default_pool(),
        CubemapPool::default_pool(),
        RenderTexturePool::new(),
        VideoTexturePool::new(),
    )
}

/// Extracts a packed f32x4 uniform from `bytes`.
pub(super) fn read_f32x4(bytes: &[u8], offset: usize) -> [f32; 4] {
    let mut out = [0.0; 4];
    for (i, value) in out.iter_mut().enumerate() {
        let start = offset + i * 4;
        *value = f32::from_le_bytes(
            bytes[start..start + 4]
                .try_into()
                .expect("uniform f32 component bytes"),
        );
    }
    out
}

/// Extracts a packed f32 uniform from `bytes`.
pub(super) fn read_f32_at(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("uniform f32 bytes"),
    )
}

pub(super) fn reflected_with_f32_fields(
    field_specs: &[(&str, u32)],
) -> (
    ReflectedRasterLayout,
    StemEmbeddedPropertyIds,
    PropertyIdRegistry,
) {
    let registry = PropertyIdRegistry::new();
    let mut fields = HashMap::new();
    let mut total_size = 0u32;
    for (field_name, field_offset) in field_specs {
        fields.insert(
            (*field_name).to_string(),
            ReflectedUniformField {
                offset: *field_offset,
                size: 4,
                kind: ReflectedUniformScalarKind::F32,
            },
        );
        total_size = total_size.max(field_offset.saturating_add(4));
    }
    let reflected = ReflectedRasterLayout {
        layout_fingerprint: 0,
        material_entries: Vec::new(),
        per_draw_entries: Vec::new(),
        material_uniform: Some(ReflectedMaterialUniformBlock {
            binding: 0,
            total_size,
            fields,
        }),
        material_group1_names: HashMap::new(),
        vs_vertex_inputs: Vec::new(),
        vs_max_vertex_location: None,
        uses_scene_depth_snapshot: false,
        uses_scene_color_snapshot: false,
        requires_intersection_pass: false,
    };
    let ids = StemEmbeddedPropertyIds::build(
        Arc::new(EmbeddedSharedKeywordIds::new(&registry)),
        &registry,
        &reflected,
    );
    (reflected, ids, registry)
}

pub(super) fn reflected_with_uniform_fields(
    field_specs: &[(&str, ReflectedUniformScalarKind, u32, u32)],
) -> (
    ReflectedRasterLayout,
    StemEmbeddedPropertyIds,
    PropertyIdRegistry,
) {
    let registry = PropertyIdRegistry::new();
    let mut fields = HashMap::new();
    let mut total_size = 0u32;
    for (field_name, field_kind, field_size, field_offset) in field_specs {
        fields.insert(
            (*field_name).to_string(),
            ReflectedUniformField {
                offset: *field_offset,
                size: *field_size,
                kind: *field_kind,
            },
        );
        total_size = total_size.max(field_offset.saturating_add(*field_size));
    }
    let reflected = ReflectedRasterLayout {
        layout_fingerprint: 0,
        material_entries: Vec::new(),
        per_draw_entries: Vec::new(),
        material_uniform: Some(ReflectedMaterialUniformBlock {
            binding: 0,
            total_size,
            fields,
        }),
        material_group1_names: HashMap::new(),
        vs_vertex_inputs: Vec::new(),
        vs_max_vertex_location: None,
        uses_scene_depth_snapshot: false,
        uses_scene_color_snapshot: false,
        requires_intersection_pass: false,
    };
    let ids = StemEmbeddedPropertyIds::build(
        Arc::new(EmbeddedSharedKeywordIds::new(&registry)),
        &registry,
        &reflected,
    );
    (reflected, ids, registry)
}

/// Packs an asset id as a host render-texture material property.
pub(super) fn packed_render_texture(asset_id: i32) -> i32 {
    use crate::assets::texture::HostTextureAssetKind;

    let type_bits = 3u32;
    let pack_type_shift = 32u32.saturating_sub(type_bits);
    asset_id | ((HostTextureAssetKind::RenderTexture as i32) << pack_type_shift)
}

pub(super) fn packed_texture2d(asset_id: i32) -> i32 {
    use crate::assets::texture::HostTextureAssetKind;

    let type_bits = 3u32;
    let pack_type_shift = 32u32.saturating_sub(type_bits);
    asset_id | ((HostTextureAssetKind::Texture2D as i32) << pack_type_shift)
}
