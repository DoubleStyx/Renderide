//! Texture storage-orientation and sampler metadata uniform tests.

use super::super::*;
use std::sync::Arc;

use hashbrown::HashMap;

use crate::assets::texture::HostTextureAssetKind;
use crate::gpu_pools::{
    CubemapPool, RenderTexturePool, Texture3dPool, TexturePool, VideoTexturePool,
};
use crate::materials::ReflectedMaterialUniformBlock;
use crate::materials::embedded::layout::{EmbeddedSharedKeywordIds, StemEmbeddedPropertyIds};
use crate::materials::embedded::texture_pools::EmbeddedTexturePools;
use crate::materials::host_data::PropertyIdRegistry;

fn lookup(material_id: i32) -> MaterialPropertyLookupIds {
    MaterialPropertyLookupIds {
        material_asset_id: material_id,
        mesh_property_block_slot0: None,
    }
}

fn texture_entry(
    binding: u32,
    view_dimension: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension,
            multisampled: false,
        },
        count: None,
    }
}

fn reflected_with_texture_and_fields(
    texture_name: &str,
    view_dimension: wgpu::TextureViewDimension,
    field_specs: &[(&str, ReflectedUniformScalarKind, u32, u32)],
) -> (
    ReflectedRasterLayout,
    StemEmbeddedPropertyIds,
    PropertyIdRegistry,
) {
    let registry = PropertyIdRegistry::new();
    let mut material_group1_names = HashMap::new();
    material_group1_names.insert(1, texture_name.to_string());
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
        material_entries: vec![texture_entry(1, view_dimension)],
        per_draw_entries: Vec::new(),
        material_uniform: Some(ReflectedMaterialUniformBlock {
            binding: 0,
            total_size,
            fields,
        }),
        material_group1_names,
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

fn reflected_with_texture_and_field(
    texture_name: &str,
    view_dimension: wgpu::TextureViewDimension,
    field_name: &str,
    field_kind: ReflectedUniformScalarKind,
    field_size: u32,
) -> (
    ReflectedRasterLayout,
    StemEmbeddedPropertyIds,
    PropertyIdRegistry,
) {
    reflected_with_texture_and_fields(
        texture_name,
        view_dimension,
        &[(field_name, field_kind, field_size, 0)],
    )
}

fn read_f32x4(bytes: &[u8]) -> [f32; 4] {
    [
        f32::from_le_bytes(bytes[0..4].try_into().unwrap()),
        f32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        f32::from_le_bytes(bytes[8..12].try_into().unwrap()),
        f32::from_le_bytes(bytes[12..16].try_into().unwrap()),
    ]
}

fn read_f32_at(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn pack_texture_id(asset_id: i32, kind: HostTextureAssetKind) -> i32 {
    let type_bits = 3u32;
    let pack_type_shift = 32u32.saturating_sub(type_bits);
    asset_id | ((kind as i32) << pack_type_shift)
}

#[test]
fn storage_metadata_marks_texture2d_and_cubemap_bindings() {
    assert!(binding_storage_v_inverted_from_metadata(
        ResolvedTextureBinding::Texture2D { asset_id: 42 },
        Some(true),
        None
    ));
    assert!(!binding_storage_v_inverted_from_metadata(
        ResolvedTextureBinding::Texture2D { asset_id: 42 },
        Some(false),
        None
    ));
    assert!(binding_storage_v_inverted_from_metadata(
        ResolvedTextureBinding::Cubemap { asset_id: 55 },
        None,
        Some(true)
    ));
    assert!(!binding_storage_v_inverted_from_metadata(
        ResolvedTextureBinding::RenderTexture { asset_id: 9 },
        Some(true),
        Some(true)
    ));
    assert_eq!(storage_v_inverted_flag_value(true), 1.0);
    assert_eq!(storage_v_inverted_flag_value(false), 0.0);
}

#[test]
fn lod_bias_metadata_uses_only_wire_supported_texture_kinds() {
    assert_eq!(
        binding_lod_bias_from_metadata(
            ResolvedTextureBinding::Texture2D { asset_id: 42 },
            Some(-0.75),
            Some(1.25)
        ),
        -0.75
    );
    assert_eq!(
        binding_lod_bias_from_metadata(
            ResolvedTextureBinding::Cubemap { asset_id: 55 },
            Some(-0.75),
            Some(1.25)
        ),
        1.25
    );
    assert_eq!(
        binding_lod_bias_from_metadata(
            ResolvedTextureBinding::Texture3D { asset_id: 77 },
            Some(-0.75),
            Some(1.25)
        ),
        0.0
    );
    assert_eq!(
        binding_lod_bias_from_metadata(
            ResolvedTextureBinding::RenderTexture { asset_id: 9 },
            Some(-0.75),
            Some(1.25)
        ),
        0.0
    );
}

#[test]
fn unresolved_texture2d_does_not_rewrite_st() {
    let texture_pool = TexturePool::default_pool();
    let texture3d_pool = Texture3dPool::default_pool();
    let cubemap_pool = CubemapPool::default_pool();
    let render_texture_pool = RenderTexturePool::new();
    let video_texture_pool = VideoTexturePool::new();
    let pools = EmbeddedTexturePools {
        texture: &texture_pool,
        texture3d: &texture3d_pool,
        cubemap: &cubemap_pool,
        render_texture: &render_texture_pool,
        video_texture: &video_texture_pool,
    };
    let (reflected, ids, registry) = reflected_with_texture_and_field(
        "_MainTex",
        wgpu::TextureViewDimension::D2,
        "_MainTex_ST",
        ReflectedUniformScalarKind::Vec4,
        16,
    );
    let mut store = MaterialPropertyStore::new();
    store.set_material(
        7,
        registry.intern("_MainTex"),
        MaterialPropertyValue::Texture(42),
    );
    store.set_material(
        7,
        registry.intern("_MainTex_ST"),
        MaterialPropertyValue::Float4([2.0, 3.0, 0.25, 0.75]),
    );
    let tex_ctx = UniformPackTextureContext {
        pools: &pools,
        primary_texture_2d: -1,
    };

    let bytes =
        build_embedded_uniform_bytes(&reflected, &ids, &store, lookup(7), &tex_ctx).unwrap();
    assert_eq!(read_f32x4(&bytes), [2.0, 3.0, 0.25, 0.75]);
}

#[test]
fn render_texture_populates_storage_field_as_zero() {
    let texture_pool = TexturePool::default_pool();
    let texture3d_pool = Texture3dPool::default_pool();
    let cubemap_pool = CubemapPool::default_pool();
    let render_texture_pool = RenderTexturePool::new();
    let video_texture_pool = VideoTexturePool::new();
    let pools = EmbeddedTexturePools {
        texture: &texture_pool,
        texture3d: &texture3d_pool,
        cubemap: &cubemap_pool,
        render_texture: &render_texture_pool,
        video_texture: &video_texture_pool,
    };
    let (reflected, ids, registry) = reflected_with_texture_and_fields(
        "_MainTex",
        wgpu::TextureViewDimension::D2,
        &[
            ("_MainTex_ST", ReflectedUniformScalarKind::Vec4, 16, 0),
            (
                "_MainTex_StorageVInverted",
                ReflectedUniformScalarKind::F32,
                4,
                16,
            ),
        ],
    );
    let mut store = MaterialPropertyStore::new();
    store.set_material(
        7,
        registry.intern("_MainTex"),
        MaterialPropertyValue::Texture(pack_texture_id(9, HostTextureAssetKind::RenderTexture)),
    );
    store.set_material(
        7,
        registry.intern("_MainTex_ST"),
        MaterialPropertyValue::Float4([2.0, 3.0, 0.25, 0.75]),
    );
    let tex_ctx = UniformPackTextureContext {
        pools: &pools,
        primary_texture_2d: -1,
    };

    let bytes =
        build_embedded_uniform_bytes(&reflected, &ids, &store, lookup(7), &tex_ctx).unwrap();
    assert_eq!(read_f32x4(&bytes), [2.0, 3.0, 0.25, 0.75]);
    assert_eq!(read_f32_at(&bytes, 16), 0.0);
}

#[test]
fn unflagged_texture2d_populates_storage_field_as_zero() {
    let texture_pool = TexturePool::default_pool();
    let texture3d_pool = Texture3dPool::default_pool();
    let cubemap_pool = CubemapPool::default_pool();
    let render_texture_pool = RenderTexturePool::new();
    let video_texture_pool = VideoTexturePool::new();
    let pools = EmbeddedTexturePools {
        texture: &texture_pool,
        texture3d: &texture3d_pool,
        cubemap: &cubemap_pool,
        render_texture: &render_texture_pool,
        video_texture: &video_texture_pool,
    };
    let (reflected, ids, registry) = reflected_with_texture_and_field(
        "_MainTex",
        wgpu::TextureViewDimension::D2,
        "_MainTex_StorageVInverted",
        ReflectedUniformScalarKind::F32,
        4,
    );
    let mut store = MaterialPropertyStore::new();
    store.set_material(
        7,
        registry.intern("_MainTex"),
        MaterialPropertyValue::Texture(42),
    );
    let tex_ctx = UniformPackTextureContext {
        pools: &pools,
        primary_texture_2d: -1,
    };

    let bytes =
        build_embedded_uniform_bytes(&reflected, &ids, &store, lookup(7), &tex_ctx).unwrap();
    assert_eq!(read_f32_at(&bytes, 0), 0.0);
}

#[test]
fn font_atlas_storage_field_resolves_font_atlas_binding() {
    let texture_pool = TexturePool::default_pool();
    let texture3d_pool = Texture3dPool::default_pool();
    let cubemap_pool = CubemapPool::default_pool();
    let render_texture_pool = RenderTexturePool::new();
    let video_texture_pool = VideoTexturePool::new();
    let pools = EmbeddedTexturePools {
        texture: &texture_pool,
        texture3d: &texture3d_pool,
        cubemap: &cubemap_pool,
        render_texture: &render_texture_pool,
        video_texture: &video_texture_pool,
    };
    let (reflected, ids, registry) = reflected_with_texture_and_field(
        "_FontAtlas",
        wgpu::TextureViewDimension::D2,
        "_FontAtlas_StorageVInverted",
        ReflectedUniformScalarKind::F32,
        4,
    );
    let mut store = MaterialPropertyStore::new();
    store.set_material(
        8,
        registry.intern("_FontAtlas"),
        MaterialPropertyValue::Texture(42),
    );
    let tex_ctx = UniformPackTextureContext {
        pools: &pools,
        primary_texture_2d: -1,
    };

    let bytes =
        build_embedded_uniform_bytes(&reflected, &ids, &store, lookup(8), &tex_ctx).unwrap();
    assert_eq!(read_f32_at(&bytes, 0), 0.0);
}

#[test]
fn font_atlas_lod_bias_field_resolves_font_atlas_binding() {
    let texture_pool = TexturePool::default_pool();
    let texture3d_pool = Texture3dPool::default_pool();
    let cubemap_pool = CubemapPool::default_pool();
    let render_texture_pool = RenderTexturePool::new();
    let video_texture_pool = VideoTexturePool::new();
    let pools = EmbeddedTexturePools {
        texture: &texture_pool,
        texture3d: &texture3d_pool,
        cubemap: &cubemap_pool,
        render_texture: &render_texture_pool,
        video_texture: &video_texture_pool,
    };
    let (reflected, ids, registry) = reflected_with_texture_and_field(
        "_FontAtlas",
        wgpu::TextureViewDimension::D2,
        "_FontAtlas_LodBias",
        ReflectedUniformScalarKind::F32,
        4,
    );
    let mut store = MaterialPropertyStore::new();
    store.set_material(
        8,
        registry.intern("_FontAtlas"),
        MaterialPropertyValue::Texture(42),
    );
    store.set_material(
        8,
        registry.intern("_FontAtlas_LodBias"),
        MaterialPropertyValue::Float(7.0),
    );
    let tex_ctx = UniformPackTextureContext {
        pools: &pools,
        primary_texture_2d: -1,
    };

    let bytes =
        build_embedded_uniform_bytes(&reflected, &ids, &store, lookup(8), &tex_ctx).unwrap();
    assert_eq!(read_f32_at(&bytes, 0), 0.0);
}

#[test]
fn nonresident_font_atlas_keeps_text_mode_msdf_fallback() {
    let texture_pool = TexturePool::default_pool();
    let texture3d_pool = Texture3dPool::default_pool();
    let cubemap_pool = CubemapPool::default_pool();
    let render_texture_pool = RenderTexturePool::new();
    let video_texture_pool = VideoTexturePool::new();
    let pools = EmbeddedTexturePools {
        texture: &texture_pool,
        texture3d: &texture3d_pool,
        cubemap: &cubemap_pool,
        render_texture: &render_texture_pool,
        video_texture: &video_texture_pool,
    };
    let (reflected, ids, registry) = reflected_with_texture_and_field(
        "_FontAtlas",
        wgpu::TextureViewDimension::D2,
        "_TextMode",
        ReflectedUniformScalarKind::F32,
        4,
    );
    let mut store = MaterialPropertyStore::new();
    store.set_material(
        8,
        registry.intern("_FontAtlas"),
        MaterialPropertyValue::Texture(42),
    );
    let tex_ctx = UniformPackTextureContext {
        pools: &pools,
        primary_texture_2d: -1,
    };

    let bytes =
        build_embedded_uniform_bytes(&reflected, &ids, &store, lookup(8), &tex_ctx).unwrap();
    assert_eq!(read_f32_at(&bytes, 0), 0.0);
}
