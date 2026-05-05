//! Color-space, scalar-default, and explicit uniform packing tests.

use super::super::*;
use super::common::*;

use crate::materials::ReflectedUniformScalarKind;
use crate::materials::embedded::texture_pools::EmbeddedTexturePools;
use crate::materials::host_data::{MaterialPropertyStore, MaterialPropertyValue};

#[test]
fn srgb_material_color_vec4_uniforms_linearize_rgb_only() {
    let (reflected, ids, registry) = reflected_with_uniform_fields(&[
        ("_Color", ReflectedUniformScalarKind::Vec4, 16, 0),
        ("_Blend", ReflectedUniformScalarKind::Vec4, 16, 16),
        ("_EdgeEmission", ReflectedUniformScalarKind::Vec4, 16, 32),
    ]);
    let mut store = MaterialPropertyStore::new();
    let input = [0.5, 0.25, -0.5, 0.75];
    store.set_material(
        27,
        registry.intern("_Color"),
        MaterialPropertyValue::Float4(input),
    );
    store.set_material(
        27,
        registry.intern("_Blend"),
        MaterialPropertyValue::Float4(input),
    );
    store.set_material(
        27,
        registry.intern("_EdgeEmission"),
        MaterialPropertyValue::Float4(input),
    );
    let (texture, texture3d, cubemap, render_texture, video_texture) = empty_texture_pools();
    let pools = EmbeddedTexturePools {
        texture: &texture,
        texture3d: &texture3d,
        cubemap: &cubemap,
        render_texture: &render_texture,
        video_texture: &video_texture,
    };
    let tex_ctx = UniformPackTextureContext {
        pools: &pools,
        primary_texture_2d: -1,
    };
    let value_spaces =
        MaterialUniformValueSpaces::for_stem("pbsvoronoicrystal_default", &reflected);
    let expected = srgb_vec4_rgb_to_linear(input);

    let bytes = build_embedded_uniform_bytes_with_value_spaces(
        &reflected,
        &ids,
        &value_spaces,
        &store,
        lookup(27),
        &tex_ctx,
    )
    .expect("uniform bytes");

    assert_eq!(read_f32x4(&bytes, 0), expected);
    assert_eq!(read_f32x4(&bytes, 16), expected);
    assert_eq!(read_f32x4(&bytes, 32), expected);
}

#[test]
fn color_named_texture_transform_vec4_uniforms_remain_raw() {
    let (reflected, ids, registry) = reflected_with_uniform_fields(&[
        ("_MainTex_ST", ReflectedUniformScalarKind::Vec4, 16, 0),
        ("_ColorMap_ST", ReflectedUniformScalarKind::Vec4, 16, 16),
        ("_ColorMask_ST", ReflectedUniformScalarKind::Vec4, 16, 32),
        ("_TintTex_ST", ReflectedUniformScalarKind::Vec4, 16, 48),
    ]);
    let mut store = MaterialPropertyStore::new();
    let input = [2.0, 3.0, 0.25, 0.75];
    for field in [
        "_MainTex_ST",
        "_ColorMap_ST",
        "_ColorMask_ST",
        "_TintTex_ST",
    ] {
        store.set_material(
            28,
            registry.intern(field),
            MaterialPropertyValue::Float4(input),
        );
    }
    let (texture, texture3d, cubemap, render_texture, video_texture) = empty_texture_pools();
    let pools = EmbeddedTexturePools {
        texture: &texture,
        texture3d: &texture3d,
        cubemap: &cubemap,
        render_texture: &render_texture,
        video_texture: &video_texture,
    };
    let tex_ctx = UniformPackTextureContext {
        pools: &pools,
        primary_texture_2d: -1,
    };
    let value_spaces = MaterialUniformValueSpaces::for_stem("pbscolorsplat_default", &reflected);

    let bytes = build_embedded_uniform_bytes_with_value_spaces(
        &reflected,
        &ids,
        &value_spaces,
        &store,
        lookup(28),
        &tex_ctx,
    )
    .expect("uniform bytes");

    assert_eq!(read_f32x4(&bytes, 0), input);
    assert_eq!(read_f32x4(&bytes, 16), input);
    assert_eq!(read_f32x4(&bytes, 32), input);
    assert_eq!(read_f32x4(&bytes, 48), input);
}

#[test]
fn srgb_material_color_arrays_linearize_only_when_metadata_marks_them() {
    let (reflected, ids, registry) = reflected_with_uniform_fields(&[(
        "_TintColors",
        ReflectedUniformScalarKind::Unsupported,
        32,
        0,
    )]);
    let mut store = MaterialPropertyStore::new();
    let input = vec![[0.5, 0.25, -0.5, 0.75], [0.04045, 1.25, 0.0, 0.5]];
    store.set_material(
        29,
        registry.intern("_TintColors"),
        MaterialPropertyValue::Float4Array(input.clone()),
    );
    let (texture, texture3d, cubemap, render_texture, video_texture) = empty_texture_pools();
    let pools = EmbeddedTexturePools {
        texture: &texture,
        texture3d: &texture3d,
        cubemap: &cubemap,
        render_texture: &render_texture,
        video_texture: &video_texture,
    };
    let tex_ctx = UniformPackTextureContext {
        pools: &pools,
        primary_texture_2d: -1,
    };
    let value_spaces = MaterialUniformValueSpaces::for_stem("pbsdistancelerp_default", &reflected);

    let bytes = build_embedded_uniform_bytes_with_value_spaces(
        &reflected,
        &ids,
        &value_spaces,
        &store,
        lookup(29),
        &tex_ctx,
    )
    .expect("uniform bytes");

    assert_eq!(read_f32x4(&bytes, 0), srgb_vec4_rgb_to_linear(input[0]));
    assert_eq!(read_f32x4(&bytes, 16), srgb_vec4_rgb_to_linear(input[1]));
}

#[test]
fn linear_material_color_arrays_remain_raw() {
    let (reflected, ids, registry) = reflected_with_uniform_fields(&[
        ("_Color0", ReflectedUniformScalarKind::Unsupported, 16, 0),
        ("_Color1", ReflectedUniformScalarKind::Unsupported, 16, 16),
    ]);
    let mut store = MaterialPropertyStore::new();
    let color0 = [0.25, 0.5, 0.75, 1.0];
    let color1 = [0.75, 0.5, 0.25, 1.0];
    store.set_material(
        30,
        registry.intern("_Color0"),
        MaterialPropertyValue::Float4Array(vec![color0]),
    );
    store.set_material(
        30,
        registry.intern("_Color1"),
        MaterialPropertyValue::Float4Array(vec![color1]),
    );
    let (texture, texture3d, cubemap, render_texture, video_texture) = empty_texture_pools();
    let pools = EmbeddedTexturePools {
        texture: &texture,
        texture3d: &texture3d,
        cubemap: &cubemap,
        render_texture: &render_texture,
        video_texture: &video_texture,
    };
    let tex_ctx = UniformPackTextureContext {
        pools: &pools,
        primary_texture_2d: -1,
    };
    let value_spaces = MaterialUniformValueSpaces::for_stem("gradientskybox_default", &reflected);

    let bytes = build_embedded_uniform_bytes_with_value_spaces(
        &reflected,
        &ids,
        &value_spaces,
        &store,
        lookup(30),
        &tex_ctx,
    )
    .expect("uniform bytes");

    assert_eq!(read_f32x4(&bytes, 0), color0);
    assert_eq!(read_f32x4(&bytes, 16), color1);
}

#[test]
fn standard_scalar_defaults_pack_without_host_values() {
    let (reflected, ids, _) = reflected_with_f32_fields(&[
        ("_GlossMapScale", 0),
        ("_OcclusionStrength", 4),
        ("_UVSec", 8),
    ]);
    let (textures, texture3d, cubemaps, render_textures, videos) = empty_texture_pools();
    let pools = EmbeddedTexturePools {
        texture: &textures,
        texture3d: &texture3d,
        cubemap: &cubemaps,
        render_texture: &render_textures,
        video_texture: &videos,
    };
    let bytes = build_embedded_uniform_bytes(
        &reflected,
        &ids,
        &MaterialPropertyStore::new(),
        lookup(1),
        &UniformPackTextureContext {
            pools: &pools,
            primary_texture_2d: -1,
        },
    )
    .expect("uniform bytes");

    assert_eq!(read_f32_at(&bytes, 0), 1.0);
    assert_eq!(read_f32_at(&bytes, 4), 1.0);
    assert_eq!(read_f32_at(&bytes, 8), 0.0);
}
