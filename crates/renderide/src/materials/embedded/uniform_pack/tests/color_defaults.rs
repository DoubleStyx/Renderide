//! Color-space, scalar-default, and explicit uniform packing tests.

use super::super::*;
use super::common::*;

use crate::embedded_shaders::EmbeddedMaterialDefaultValue;
use crate::materials::ReflectedUniformScalarKind;
use crate::materials::embedded::texture_pools::EmbeddedTexturePools;
use crate::materials::host_data::{MaterialPropertyStore, MaterialPropertyValue};

#[test]
fn srgb_material_color_vec4_uniforms_linearize_rgb_only() {
    let (reflected, ids, registry) = reflected_with_uniform_fields(&[
        ("_Color", ReflectedUniformScalarKind::Vec4, 16, 0),
        ("_Blend", ReflectedUniformScalarKind::Vec4, 16, 16),
        ("_EdgeEmission", ReflectedUniformScalarKind::Vec4, 16, 32),
        ("_Tint", ReflectedUniformScalarKind::Vec4, 16, 48),
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
    store.set_material(
        27,
        registry.intern("_Tint"),
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
        None,
    )
    .expect("uniform bytes");

    assert_eq!(read_f32x4(&bytes, 0), expected);
    assert_eq!(read_f32x4(&bytes, 16), expected);
    assert_eq!(read_f32x4(&bytes, 32), expected);
    assert_eq!(read_f32x4(&bytes, 48), expected);
}

#[test]
fn srgb_material_color_vec4_defaults_pack_raw() {
    let (reflected, ids, _) =
        reflected_with_uniform_fields(&[("_Color", ReflectedUniformScalarKind::Vec4, 16, 0)]);
    let mut defaults = MaterialUniformDefaults::default();
    let default_color = [0.5, 0.25, 0.75, 1.0];
    defaults.insert(
        "_Color".to_string(),
        EmbeddedMaterialDefaultValue::vec4(default_color),
    );
    let (texture, texture3d, cubemap, render_texture, video_texture) = empty_texture_pools();
    let pools = EmbeddedTexturePools {
        texture: &texture,
        texture3d: &texture3d,
        cubemap: &cubemap,
        render_texture: &render_texture,
        video_texture: &video_texture,
    };
    let value_spaces = MaterialUniformValueSpaces::for_stem("pbsmetallic_default", &reflected);
    let metadata = MaterialUniformPackMetadata {
        value_spaces: &value_spaces,
        material_defaults: &defaults,
    };

    let bytes = build_embedded_uniform_bytes_with_material_defaults(
        &reflected,
        &ids,
        &metadata,
        &MaterialPropertyStore::new(),
        lookup(32),
        &UniformPackTextureContext {
            pools: &pools,
            primary_texture_2d: -1,
        },
        None,
    )
    .expect("uniform bytes");

    assert_eq!(read_f32x4(&bytes, 0), default_color);
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
        None,
    )
    .expect("uniform bytes");

    assert_eq!(read_f32x4(&bytes, 0), input);
    assert_eq!(read_f32x4(&bytes, 16), input);
    assert_eq!(read_f32x4(&bytes, 32), input);
    assert_eq!(read_f32x4(&bytes, 48), input);
}

#[test]
fn unwritten_texture_transform_vec4_uniforms_pack_unity_identity() {
    let (reflected, ids, _) = reflected_with_uniform_fields(&[
        ("_MainTex_ST", ReflectedUniformScalarKind::Vec4, 16, 0),
        ("_NormalMap_ST", ReflectedUniformScalarKind::Vec4, 16, 16),
        ("_FarTex0_ST", ReflectedUniformScalarKind::Vec4, 16, 32),
        ("_NearTex1_ST", ReflectedUniformScalarKind::Vec4, 16, 48),
        ("_Tint", ReflectedUniformScalarKind::Vec4, 16, 64),
    ]);
    let (texture, texture3d, cubemap, render_texture, video_texture) = empty_texture_pools();
    let pools = EmbeddedTexturePools {
        texture: &texture,
        texture3d: &texture3d,
        cubemap: &cubemap,
        render_texture: &render_texture,
        video_texture: &video_texture,
    };

    let bytes = build_embedded_uniform_bytes(
        &reflected,
        &ids,
        &MaterialPropertyStore::new(),
        lookup(31),
        &UniformPackTextureContext {
            pools: &pools,
            primary_texture_2d: -1,
        },
    )
    .expect("uniform bytes");

    let unity_identity = [1.0, 1.0, 0.0, 0.0];
    assert_eq!(read_f32x4(&bytes, 0), unity_identity);
    assert_eq!(read_f32x4(&bytes, 16), unity_identity);
    assert_eq!(read_f32x4(&bytes, 32), unity_identity);
    assert_eq!(read_f32x4(&bytes, 48), unity_identity);
    assert_eq!(read_f32x4(&bytes, 64), [0.0; 4]);
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
        None,
    )
    .expect("uniform bytes");

    assert_eq!(read_f32x4(&bytes, 0), srgb_vec4_rgb_to_linear(input[0]));
    assert_eq!(read_f32x4(&bytes, 16), srgb_vec4_rgb_to_linear(input[1]));
}

#[test]
fn volume_highlight_color_arrays_linearize_host_values() {
    let (reflected, ids, registry) = reflected_with_uniform_fields(&[(
        "_HighlightColor",
        ReflectedUniformScalarKind::Unsupported,
        32,
        0,
    )]);
    let mut store = MaterialPropertyStore::new();
    let input = vec![[0.5, 0.25, -0.5, 0.75], [0.04045, 1.25, 0.0, 0.5]];
    store.set_material(
        33,
        registry.intern("_HighlightColor"),
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
    let value_spaces = MaterialUniformValueSpaces::for_stem("volumeunlit_default", &reflected);

    let bytes = build_embedded_uniform_bytes_with_value_spaces(
        &reflected,
        &ids,
        &value_spaces,
        &store,
        lookup(33),
        &tex_ctx,
        None,
    )
    .expect("uniform bytes");

    assert_eq!(read_f32x4(&bytes, 0), srgb_vec4_rgb_to_linear(input[0]));
    assert_eq!(read_f32x4(&bytes, 16), srgb_vec4_rgb_to_linear(input[1]));
}

#[test]
fn gradient_skybox_color_arrays_stay_raw_for_material_uniform_path() {
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

    for stem in ["gradientskybox_default", "skybox_gradientskybox_default"] {
        let value_spaces = MaterialUniformValueSpaces::for_stem(stem, &reflected);
        let bytes = build_embedded_uniform_bytes_with_value_spaces(
            &reflected,
            &ids,
            &value_spaces,
            &store,
            lookup(30),
            &tex_ctx,
            None,
        )
        .expect("uniform bytes");

        assert_eq!(read_f32x4(&bytes, 0), color0);
        assert_eq!(read_f32x4(&bytes, 16), color1);
    }
}

/// Unwritten host f32 properties pack as zero. The host's `MaterialProviderBase` bootstrap
/// writes every `Sync<X>` on the first batch, so this fallthrough is only observable in the
/// pre-first-batch window that is never rendered.
#[test]
fn unwritten_scalar_fields_pack_as_zero() {
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

    assert_eq!(read_f32_at(&bytes, 0), 0.0);
    assert_eq!(read_f32_at(&bytes, 4), 0.0);
    assert_eq!(read_f32_at(&bytes, 8), 0.0);
}

#[test]
fn material_scalar_defaults_pack_when_host_property_is_missing() {
    let (reflected, ids, _) = reflected_with_f32_fields(&[
        ("_GlossMapScale", 0),
        ("_OcclusionStrength", 4),
        ("_UVSec", 8),
    ]);
    let mut defaults = MaterialUniformDefaults::default();
    defaults.insert(
        "_GlossMapScale".to_string(),
        EmbeddedMaterialDefaultValue::float(1.0),
    );
    defaults.insert(
        "_OcclusionStrength".to_string(),
        EmbeddedMaterialDefaultValue::float(1.0),
    );
    let (textures, texture3d, cubemaps, render_textures, videos) = empty_texture_pools();
    let pools = EmbeddedTexturePools {
        texture: &textures,
        texture3d: &texture3d,
        cubemap: &cubemaps,
        render_texture: &render_textures,
        video_texture: &videos,
    };
    let value_spaces = MaterialUniformValueSpaces::default();
    let metadata = MaterialUniformPackMetadata {
        value_spaces: &value_spaces,
        material_defaults: &defaults,
    };

    let bytes = build_embedded_uniform_bytes_with_material_defaults(
        &reflected,
        &ids,
        &metadata,
        &MaterialPropertyStore::new(),
        lookup(2),
        &UniformPackTextureContext {
            pools: &pools,
            primary_texture_2d: -1,
        },
        None,
    )
    .expect("uniform bytes");

    assert_eq!(read_f32_at(&bytes, 0), 1.0);
    assert_eq!(read_f32_at(&bytes, 4), 1.0);
    assert_eq!(read_f32_at(&bytes, 8), 0.0);
}

#[test]
fn explicit_host_scalar_overrides_material_default() {
    let (reflected, ids, registry) = reflected_with_f32_fields(&[("_GlossMapScale", 0)]);
    let mut defaults = MaterialUniformDefaults::default();
    defaults.insert(
        "_GlossMapScale".to_string(),
        EmbeddedMaterialDefaultValue::float(1.0),
    );
    let mut store = MaterialPropertyStore::new();
    store.set_material(
        3,
        registry.intern("_GlossMapScale"),
        MaterialPropertyValue::Float(0.25),
    );
    let (textures, texture3d, cubemaps, render_textures, videos) = empty_texture_pools();
    let pools = EmbeddedTexturePools {
        texture: &textures,
        texture3d: &texture3d,
        cubemap: &cubemaps,
        render_texture: &render_textures,
        video_texture: &videos,
    };
    let value_spaces = MaterialUniformValueSpaces::default();
    let metadata = MaterialUniformPackMetadata {
        value_spaces: &value_spaces,
        material_defaults: &defaults,
    };

    let bytes = build_embedded_uniform_bytes_with_material_defaults(
        &reflected,
        &ids,
        &metadata,
        &store,
        lookup(3),
        &UniformPackTextureContext {
            pools: &pools,
            primary_texture_2d: -1,
        },
        None,
    )
    .expect("uniform bytes");

    assert_eq!(read_f32_at(&bytes, 0), 0.25);
}

#[test]
fn material_vec4_defaults_pack_before_texture_transform_identity() {
    let (reflected, ids, _) =
        reflected_with_uniform_fields(&[("_MainTex_ST", ReflectedUniformScalarKind::Vec4, 16, 0)]);
    let mut defaults = MaterialUniformDefaults::default();
    let expected = [2.0, 2.0, 0.5, 0.5];
    defaults.insert(
        "_MainTex_ST".to_string(),
        EmbeddedMaterialDefaultValue::vec4(expected),
    );
    let (textures, texture3d, cubemaps, render_textures, videos) = empty_texture_pools();
    let pools = EmbeddedTexturePools {
        texture: &textures,
        texture3d: &texture3d,
        cubemap: &cubemaps,
        render_texture: &render_textures,
        video_texture: &videos,
    };
    let value_spaces = MaterialUniformValueSpaces::default();
    let metadata = MaterialUniformPackMetadata {
        value_spaces: &value_spaces,
        material_defaults: &defaults,
    };

    let bytes = build_embedded_uniform_bytes_with_material_defaults(
        &reflected,
        &ids,
        &metadata,
        &MaterialPropertyStore::new(),
        lookup(4),
        &UniformPackTextureContext {
            pools: &pools,
            primary_texture_2d: -1,
        },
        None,
    )
    .expect("uniform bytes");

    assert_eq!(read_f32x4(&bytes, 0), expected);
}
