//! Uniform-packing regression tests.

mod text_uniform_packing_tests {
    use std::sync::Arc;

    use hashbrown::HashMap;

    use super::super::tables::inferred_keyword_float_f32;
    use super::super::*;
    use crate::gpu_pools::{
        CubemapPool, RenderTexturePool, Texture3dPool, TexturePool, VideoTexturePool,
    };
    use crate::materials::embedded::layout::{EmbeddedSharedKeywordIds, StemEmbeddedPropertyIds};
    use crate::materials::host_data::PropertyIdRegistry;
    use crate::materials::host_data::{MaterialPropertyLookupIds, MaterialPropertyStore};
    use crate::materials::{ReflectedMaterialUniformBlock, ReflectedUniformScalarKind};
    use crate::shared::ColorProfile;

    fn lookup(material_id: i32) -> MaterialPropertyLookupIds {
        MaterialPropertyLookupIds {
            material_asset_id: material_id,
            mesh_property_block_slot0: None,
        }
    }

    /// Builds an empty texture-pool set for uniform-packer tests that only need binding metadata.
    fn empty_texture_pools() -> (
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
    fn read_f32x4(bytes: &[u8], offset: usize) -> [f32; 4] {
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
    fn read_f32_at(bytes: &[u8], offset: usize) -> f32 {
        f32::from_le_bytes(
            bytes[offset..offset + 4]
                .try_into()
                .expect("uniform f32 bytes"),
        )
    }

    fn reflected_with_f32_fields(
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

    fn reflected_with_uniform_fields(
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
    fn packed_render_texture(asset_id: i32) -> i32 {
        use crate::assets::texture::HostTextureAssetKind;

        let type_bits = 3u32;
        let pack_type_shift = 32u32.saturating_sub(type_bits);
        asset_id | ((HostTextureAssetKind::RenderTexture as i32) << pack_type_shift)
    }

    fn packed_texture2d(asset_id: i32) -> i32 {
        use crate::assets::texture::HostTextureAssetKind;

        let type_bits = 3u32;
        let pack_type_shift = 32u32.saturating_sub(type_bits);
        asset_id | ((HostTextureAssetKind::Texture2D as i32) << pack_type_shift)
    }

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
        let value_spaces =
            MaterialUniformValueSpaces::for_stem("pbscolorsplat_default", &reflected);

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
        let value_spaces =
            MaterialUniformValueSpaces::for_stem("pbsdistancelerp_default", &reflected);

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
        let value_spaces =
            MaterialUniformValueSpaces::for_stem("gradientskybox_default", &reflected);

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

    #[test]
    fn unlit_texture_presence_infers_observable_keywords() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        store.set_material(
            50,
            reg.intern("_Tex"),
            MaterialPropertyValue::Texture(packed_render_texture(1)),
        );
        store.set_material(
            50,
            reg.intern("_MaskTex"),
            MaterialPropertyValue::Texture(packed_render_texture(2)),
        );
        store.set_material(
            50,
            reg.intern("_OffsetTex"),
            MaterialPropertyValue::Texture(packed_render_texture(3)),
        );

        assert_eq!(
            inferred_keyword_float_f32("_TEXTURE", &store, lookup(50), &ids),
            Some(1.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("_MASK_TEXTURE_MUL", &store, lookup(50), &ids),
            Some(1.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("_MASK_TEXTURE_CLIP", &store, lookup(50), &ids),
            Some(0.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("_OFFSET_TEXTURE", &store, lookup(50), &ids),
            Some(1.0)
        );
    }

    #[test]
    fn right_eye_keyword_infers_from_right_eye_st_presence() {
        let (_reflected, ids, registry) = reflected_with_f32_fields(&[("_RightEye_ST", 0)]);
        let mut store = MaterialPropertyStore::new();
        store.set_material(
            51,
            registry.intern("_RightEye_ST"),
            MaterialPropertyValue::Float4([0.5, 1.0, 0.5, 0.0]),
        );

        assert_eq!(
            inferred_keyword_float_f32("_RIGHT_EYE_ST", &store, lookup(51), &ids),
            Some(1.0)
        );
    }

    #[test]
    fn cutout_blend_mode_infers_alpha_clip_from_canonical_blend_mode() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let pid = reg.intern("_BlendMode");
        store.set_material(12, pid, MaterialPropertyValue::Float(1.0));

        for field_name in ["_ALPHATEST_ON", "_ALPHATEST", "_ALPHACLIP"] {
            assert_eq!(
                inferred_keyword_float_f32(field_name, &store, lookup(12), &ids),
                Some(1.0),
                "{field_name} should enable for cutout _BlendMode"
            );
        }
        assert_eq!(
            inferred_keyword_float_f32("_ALPHABLEND_ON", &store, lookup(12), &ids),
            Some(0.0)
        );
    }

    /// `MaterialRenderType::TransparentCutout` (1) on the wire enables the alpha-test keyword
    /// family even when the host never sends `_Mode` / `_BlendMode` (the FrooxEngine path).
    #[test]
    fn transparent_cutout_render_type_infers_alpha_test_family() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let render_type_pid = reg.intern("_RenderType");
        store.set_material(7, render_type_pid, MaterialPropertyValue::Float(1.0));

        for field_name in ["_ALPHATEST_ON", "_ALPHATEST", "_ALPHACLIP"] {
            assert_eq!(
                inferred_keyword_float_f32(field_name, &store, lookup(7), &ids),
                Some(1.0),
                "{field_name} should enable for TransparentCutout render type"
            );
        }
        assert_eq!(
            inferred_keyword_float_f32("_ALPHABLEND_ON", &store, lookup(7), &ids),
            Some(0.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("_ALPHAPREMULTIPLY_ON", &store, lookup(7), &ids),
            Some(0.0)
        );
    }

    /// `MaterialRenderType::Opaque` (0) -- neither alpha-test nor alpha-blend keyword fires.
    /// This is the case that previously bit Unlit: default `_Cutoff = 0.98` lit up the
    /// `_Cutoff in (0, 1)` heuristic even though the host had selected Opaque.
    #[test]
    fn opaque_render_type_disables_all_alpha_keywords() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let render_type_pid = reg.intern("_RenderType");
        store.set_material(8, render_type_pid, MaterialPropertyValue::Float(0.0));

        for field_name in [
            "_ALPHATEST_ON",
            "_ALPHATEST",
            "_ALPHACLIP",
            "_ALPHABLEND_ON",
            "_ALPHAPREMULTIPLY_ON",
        ] {
            assert_eq!(
                inferred_keyword_float_f32(field_name, &store, lookup(8), &ids),
                Some(0.0),
                "{field_name} should be disabled for Opaque render type"
            );
        }
    }

    /// `MaterialRenderType::Transparent` (2) with FrooxEngine `BlendMode.Alpha` factors
    /// (`_SrcBlend = SrcAlpha (5)`, `_DstBlend = OneMinusSrcAlpha (10)`) maps to
    /// `_ALPHABLEND_ON`, not `_ALPHAPREMULTIPLY_ON`.
    #[test]
    fn transparent_render_type_with_alpha_factors_infers_alpha_blend() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let render_type_pid = reg.intern("_RenderType");
        let src_blend_pid = reg.intern("_SrcBlend");
        let dst_blend_pid = reg.intern("_DstBlend");
        store.set_material(9, render_type_pid, MaterialPropertyValue::Float(2.0));
        store.set_material(9, src_blend_pid, MaterialPropertyValue::Float(5.0));
        store.set_material(9, dst_blend_pid, MaterialPropertyValue::Float(10.0));

        assert_eq!(
            inferred_keyword_float_f32("_ALPHABLEND_ON", &store, lookup(9), &ids),
            Some(1.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("_ALPHAPREMULTIPLY_ON", &store, lookup(9), &ids),
            Some(0.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("_ALPHATEST_ON", &store, lookup(9), &ids),
            Some(0.0)
        );
    }

    /// `MaterialRenderType::Transparent` (2) with FrooxEngine `BlendMode.Transparent`
    /// (premultiplied) factors `_SrcBlend = One (1)`, `_DstBlend = OneMinusSrcAlpha (10)`
    /// maps to `_ALPHAPREMULTIPLY_ON`, not `_ALPHABLEND_ON`.
    #[test]
    fn transparent_render_type_with_premultiplied_factors_infers_premultiply() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let render_type_pid = reg.intern("_RenderType");
        let src_blend_pid = reg.intern("_SrcBlend");
        let dst_blend_pid = reg.intern("_DstBlend");
        store.set_material(11, render_type_pid, MaterialPropertyValue::Float(2.0));
        store.set_material(11, src_blend_pid, MaterialPropertyValue::Float(1.0));
        store.set_material(11, dst_blend_pid, MaterialPropertyValue::Float(10.0));

        assert_eq!(
            inferred_keyword_float_f32("_ALPHAPREMULTIPLY_ON", &store, lookup(11), &ids),
            Some(1.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("_ALPHABLEND_ON", &store, lookup(11), &ids),
            Some(0.0)
        );
    }

    /// `BlendMode.Additive` writes Transparent render type with `_SrcBlend = One` and
    /// `_DstBlend = One`; Unlit uses that signal to enable `_MUL_RGB_BY_ALPHA`.
    #[test]
    fn transparent_render_type_with_additive_factors_infers_mul_rgb_by_alpha() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let render_type_pid = reg.intern("_RenderType");
        let src_blend_pid = reg.intern("_SrcBlend");
        let dst_blend_pid = reg.intern("_DstBlend");
        store.set_material(13, render_type_pid, MaterialPropertyValue::Float(2.0));
        store.set_material(13, src_blend_pid, MaterialPropertyValue::Float(1.0));
        store.set_material(13, dst_blend_pid, MaterialPropertyValue::Float(1.0));

        assert_eq!(
            inferred_keyword_float_f32("_MUL_RGB_BY_ALPHA", &store, lookup(13), &ids),
            Some(1.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("_ALPHAPREMULTIPLY_ON", &store, lookup(13), &ids),
            Some(0.0)
        );
    }

    /// FrooxEngine drives the LUT `LERP` keyword from `_Lerp > 0` rather than sending a
    /// standalone keyword material property.
    #[test]
    fn lut_lerp_keyword_infers_from_lerp_uniform() {
        let (_reflected, ids, reg) = reflected_with_f32_fields(&[("_Lerp", 0), ("LERP", 4)]);
        let mut store = MaterialPropertyStore::new();
        let lerp_pid = reg.intern("_Lerp");

        assert_eq!(
            inferred_keyword_float_f32("LERP", &store, lookup(24), &ids),
            Some(0.0)
        );

        store.set_material(24, lerp_pid, MaterialPropertyValue::Float(0.0));
        assert_eq!(
            inferred_keyword_float_f32("LERP", &store, lookup(24), &ids),
            Some(0.0)
        );

        store.set_material(24, lerp_pid, MaterialPropertyValue::Float(0.25));
        assert_eq!(
            inferred_keyword_float_f32("LERP", &store, lookup(24), &ids),
            Some(1.0)
        );
    }

    /// Base LUT materials default to the Unity/FrooxEngine `SRGB` variant unless the reflected
    /// keyword field is explicitly supplied as false.
    #[test]
    fn lut_srgb_keyword_defaults_on() {
        let (_reflected, ids, reg) = reflected_with_f32_fields(&[("SRGB", 0)]);
        let mut store = MaterialPropertyStore::new();

        assert_eq!(
            inferred_keyword_float_f32("SRGB", &store, lookup(25), &ids),
            Some(1.0)
        );

        let srgb_pid = reg.intern("SRGB");
        store.set_material(25, srgb_pid, MaterialPropertyValue::Float(0.0));
        assert_eq!(
            inferred_keyword_float_f32("SRGB", &store, lookup(25), &ids),
            Some(0.0)
        );
    }

    /// Additive blend factors alone are not enough; the material must also be in a transparent
    /// render type or queue range.
    #[test]
    fn opaque_render_type_with_additive_factors_does_not_infer_mul_rgb_by_alpha() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let render_type_pid = reg.intern("_RenderType");
        let src_blend_pid = reg.intern("_SrcBlend");
        let dst_blend_pid = reg.intern("_DstBlend");
        store.set_material(14, render_type_pid, MaterialPropertyValue::Float(0.0));
        store.set_material(14, src_blend_pid, MaterialPropertyValue::Float(1.0));
        store.set_material(14, dst_blend_pid, MaterialPropertyValue::Float(1.0));

        assert_eq!(
            inferred_keyword_float_f32("_MUL_RGB_BY_ALPHA", &store, lookup(14), &ids),
            Some(0.0)
        );
    }

    /// Render queue inference covers materials that signal transparency through queue state rather
    /// than `MaterialRenderType`.
    #[test]
    fn render_queue_transparent_with_additive_factors_infers_mul_rgb_by_alpha() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let render_queue_pid = reg.intern("_RenderQueue");
        let src_blend_pid = reg.intern("_SrcBlend");
        let dst_blend_pid = reg.intern("_DstBlend");
        store.set_material(15, render_queue_pid, MaterialPropertyValue::Float(3000.0));
        store.set_material(15, src_blend_pid, MaterialPropertyValue::Float(1.0));
        store.set_material(15, dst_blend_pid, MaterialPropertyValue::Float(1.0));

        assert_eq!(
            inferred_keyword_float_f32("_MUL_RGB_BY_ALPHA", &store, lookup(15), &ids),
            Some(1.0)
        );
    }

    /// PBS materials (`PBS_DualSidedMaterial.cs` and friends) bypass `SetBlendMode` and
    /// only signal `AlphaHandling.AlphaClip` by writing render queue 2450 plus the
    /// `_ALPHACLIP` shader keyword (which is not on the wire). Queue 2450 alone must
    /// enable the alpha-test family.
    #[test]
    fn render_queue_alpha_test_range_enables_alpha_test_family() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let render_queue_pid = reg.intern("_RenderQueue");
        store.set_material(20, render_queue_pid, MaterialPropertyValue::Float(2450.0));

        for field_name in ["_ALPHATEST_ON", "_ALPHATEST", "_ALPHACLIP"] {
            assert_eq!(
                inferred_keyword_float_f32(field_name, &store, lookup(20), &ids),
                Some(1.0),
                "{field_name} should enable for queue 2450 (AlphaTest range)"
            );
        }
        assert_eq!(
            inferred_keyword_float_f32("_ALPHABLEND_ON", &store, lookup(20), &ids),
            Some(0.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("_ALPHAPREMULTIPLY_ON", &store, lookup(20), &ids),
            Some(0.0)
        );
    }

    /// Queue 2000 (Geometry / Opaque) must leave every alpha keyword off -- this is the
    /// PBS `AlphaHandling.Opaque` default.
    #[test]
    fn render_queue_opaque_range_disables_all_alpha_keywords() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let render_queue_pid = reg.intern("_RenderQueue");
        store.set_material(21, render_queue_pid, MaterialPropertyValue::Float(2000.0));

        for field_name in [
            "_ALPHATEST_ON",
            "_ALPHATEST",
            "_ALPHACLIP",
            "_ALPHABLEND_ON",
            "_ALPHAPREMULTIPLY_ON",
        ] {
            assert_eq!(
                inferred_keyword_float_f32(field_name, &store, lookup(21), &ids),
                Some(0.0),
                "{field_name} should be disabled for queue 2000 (Opaque range)"
            );
        }
    }

    /// Queue 3000 (Transparent) without premultiplied blend factors enables `_ALPHABLEND_ON`.
    #[test]
    fn render_queue_transparent_range_enables_alpha_blend() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let render_queue_pid = reg.intern("_RenderQueue");
        store.set_material(22, render_queue_pid, MaterialPropertyValue::Float(3000.0));

        assert_eq!(
            inferred_keyword_float_f32("_ALPHABLEND_ON", &store, lookup(22), &ids),
            Some(1.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("_ALPHAPREMULTIPLY_ON", &store, lookup(22), &ids),
            Some(0.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("_ALPHATEST_ON", &store, lookup(22), &ids),
            Some(0.0)
        );
    }

    /// Queue 3000 (Transparent) with premultiplied factors `_SrcBlend = 1`,
    /// `_DstBlend = 10` is `BlendMode.Transparent` -- enables `_ALPHAPREMULTIPLY_ON`.
    #[test]
    fn render_queue_transparent_with_premultiplied_factors_infers_premultiply() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let render_queue_pid = reg.intern("_RenderQueue");
        let src_blend_pid = reg.intern("_SrcBlend");
        let dst_blend_pid = reg.intern("_DstBlend");
        store.set_material(23, render_queue_pid, MaterialPropertyValue::Float(3000.0));
        store.set_material(23, src_blend_pid, MaterialPropertyValue::Float(1.0));
        store.set_material(23, dst_blend_pid, MaterialPropertyValue::Float(10.0));

        assert_eq!(
            inferred_keyword_float_f32("_ALPHAPREMULTIPLY_ON", &store, lookup(23), &ids),
            Some(1.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("_ALPHABLEND_ON", &store, lookup(23), &ids),
            Some(0.0)
        );
    }

    /// Render-texture bindings must not rewrite Unity `_ST` values behind the shader's back.
    #[test]
    fn render_texture_binding_leaves_st_uniform_unchanged() {
        let mut fields = HashMap::new();
        fields.insert(
            "_MainTex_ST".to_string(),
            ReflectedUniformField {
                offset: 0,
                size: 16,
                kind: ReflectedUniformScalarKind::Vec4,
            },
        );
        let mut material_group1_names = HashMap::new();
        material_group1_names.insert(1, "_MainTex".to_string());
        let reflected = ReflectedRasterLayout {
            layout_fingerprint: 0,
            material_entries: Vec::new(),
            per_draw_entries: Vec::new(),
            material_uniform: Some(ReflectedMaterialUniformBlock {
                binding: 0,
                total_size: 16,
                fields,
            }),
            material_group1_names,
            vs_vertex_inputs: Vec::new(),
            vs_max_vertex_location: None,
            uses_scene_depth_snapshot: false,
            uses_scene_color_snapshot: false,
            requires_intersection_pass: false,
        };

        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let mut ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let main_tex_st = reg.intern("_MainTex_ST");
        let main_tex = reg.intern("_MainTex");
        ids.uniform_field_ids
            .insert("_MainTex_ST".to_string(), main_tex_st);
        ids.texture_binding_property_ids
            .insert(1, Arc::from(vec![main_tex].into_boxed_slice()));
        store.set_material(
            24,
            main_tex,
            MaterialPropertyValue::Texture(packed_render_texture(9)),
        );
        store.set_material(
            24,
            main_tex_st,
            MaterialPropertyValue::Float4([2.0, 3.0, 0.25, 0.75]),
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

        let bytes = build_embedded_uniform_bytes(&reflected, &ids, &store, lookup(24), &tex_ctx)
            .expect("uniform bytes");

        assert_eq!(read_f32x4(&bytes, 0), [2.0, 3.0, 0.25, 0.75]);
    }

    #[test]
    fn explicit_ui_text_control_fields_pack_canonical_values() {
        let (reflected, ids, registry) =
            reflected_with_f32_fields(&[("_TextMode", 0), ("_RectClip", 4), ("_OVERLAY", 8)]);
        let mut store = MaterialPropertyStore::new();
        store.set_material(
            25,
            registry.intern("_TextMode"),
            MaterialPropertyValue::Float(2.0),
        );
        store.set_material(
            25,
            registry.intern("_RectClip"),
            MaterialPropertyValue::Float(1.0),
        );
        store.set_material(
            25,
            registry.intern("_OVERLAY"),
            MaterialPropertyValue::Float(1.0),
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

        let bytes = build_embedded_uniform_bytes(&reflected, &ids, &store, lookup(25), &tex_ctx)
            .expect("uniform bytes");

        assert_eq!(read_f32_at(&bytes, 0), 2.0);
        assert_eq!(read_f32_at(&bytes, 4), 1.0);
        assert_eq!(read_f32_at(&bytes, 8), 1.0);
    }

    #[test]
    fn font_atlas_profile_metadata_infers_text_mode() {
        let binding = ResolvedTextureBinding::Texture2D { asset_id: 42 };

        assert_eq!(
            binding_text_mode_from_metadata(binding, Some(ColorProfile::Linear)),
            Some(0.0)
        );
        assert_eq!(
            binding_text_mode_from_metadata(binding, Some(ColorProfile::SRGB)),
            Some(1.0)
        );
        assert_eq!(
            binding_text_mode_from_metadata(binding, Some(ColorProfile::SRGBAlpha)),
            Some(1.0)
        );
        assert_eq!(binding_text_mode_from_metadata(binding, None), None);
        assert_eq!(
            binding_text_mode_from_metadata(
                ResolvedTextureBinding::RenderTexture { asset_id: 42 },
                Some(ColorProfile::SRGB)
            ),
            None
        );
    }

    #[test]
    fn explicit_ui_text_control_fields_ignore_keyword_aliases() {
        let (reflected, ids, registry) =
            reflected_with_f32_fields(&[("_TextMode", 0), ("_RectClip", 4), ("_OVERLAY", 8)]);
        let mut store = MaterialPropertyStore::new();
        for property_name in [
            "TextMode", "textmode", "RectClip", "rectclip", "OVERLAY", "overlay",
        ] {
            store.set_material(
                26,
                registry.intern(property_name),
                MaterialPropertyValue::Float(1.0),
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

        let bytes = build_embedded_uniform_bytes(&reflected, &ids, &store, lookup(26), &tex_ctx)
            .expect("uniform bytes");

        assert_eq!(read_f32_at(&bytes, 0), 0.0);
        assert_eq!(read_f32_at(&bytes, 4), 0.0);
        assert_eq!(read_f32_at(&bytes, 8), 0.0);
    }

    #[test]
    fn inferred_pbs_keyword_enables_from_texture_presence() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let pid = reg.intern("_SpecularMap");
        store.set_material(4, pid, MaterialPropertyValue::Texture(123));
        assert_eq!(
            inferred_keyword_float_f32("_SPECULARMAP", &store, lookup(4), &ids),
            Some(1.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("_SPECGLOSSMAP", &store, lookup(4), &ids),
            Some(0.0)
        );
        let spec_gloss_pid = reg.intern("_SpecGlossMap");
        store.set_material(4, spec_gloss_pid, MaterialPropertyValue::Texture(456));
        assert_eq!(
            inferred_keyword_float_f32("_SPECGLOSSMAP", &store, lookup(4), &ids),
            Some(1.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("_ALBEDOTEX", &store, lookup(4), &ids),
            Some(0.0)
        );
    }

    #[test]
    fn fresnel_texture_keyword_infers_from_far_or_near_textures() {
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);

        for (i, property_name) in [
            "_FarTex",
            "_NearTex",
            "_FarTex0",
            "_NearTex0",
            "_FarTex1",
            "_NearTex1",
        ]
        .iter()
        .enumerate()
        {
            let material_id = 50 + i as i32;
            let mut store = MaterialPropertyStore::new();
            store.set_material(
                material_id,
                reg.intern(property_name),
                MaterialPropertyValue::Texture(packed_texture2d(100 + i as i32)),
            );
            assert_eq!(
                inferred_keyword_float_f32("_TEXTURE", &store, lookup(material_id), &ids),
                Some(1.0),
                "{property_name} should enable _TEXTURE"
            );
        }
    }

    #[test]
    fn inferred_pbs_splat_keywords_enable_from_texture_presence() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        for property_name in [
            "_PackedHeightMap",
            "_PackedNormalMap23",
            "_PackedEmissionMap",
            "_MetallicGloss23",
            "_SpecularMap3",
        ] {
            store.set_material(
                54,
                reg.intern(property_name),
                MaterialPropertyValue::Texture(packed_texture2d(123)),
            );
        }

        for field_name in [
            "_HEIGHTMAP",
            "_PACKED_NORMALMAP",
            "_PACKED_EMISSIONTEX",
            "_METALLICMAP",
            "_SPECULARMAP",
        ] {
            assert_eq!(
                inferred_keyword_float_f32(field_name, &store, lookup(54), &ids),
                Some(1.0),
                "{field_name} should infer from its selected texture family"
            );
        }
    }

    #[test]
    fn gradient_keyword_infers_from_gradient_texture() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        store.set_material(
            60,
            reg.intern("_Gradient"),
            MaterialPropertyValue::Texture(packed_texture2d(14)),
        );

        assert_eq!(
            inferred_keyword_float_f32("GRADIENT", &store, lookup(60), &ids),
            Some(1.0)
        );
    }

    #[test]
    fn normalmap_keyword_infers_from_normal_map_zero() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        store.set_material(
            61,
            reg.intern("_NormalMap0"),
            MaterialPropertyValue::Texture(packed_texture2d(15)),
        );

        assert_eq!(
            inferred_keyword_float_f32("_NORMALMAP", &store, lookup(61), &ids),
            Some(1.0)
        );
    }

    #[test]
    fn clip_keyword_stays_off_from_clip_range_properties_only() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        store.set_material(
            62,
            reg.intern("_ClipMin"),
            MaterialPropertyValue::Float(0.0),
        );
        store.set_material(
            62,
            reg.intern("_ClipMax"),
            MaterialPropertyValue::Float(10.0),
        );

        assert_eq!(
            inferred_keyword_float_f32("CLIP", &store, lookup(62), &ids),
            Some(0.0)
        );
    }

    #[test]
    fn inferred_pbs_displace_keywords_follow_host_keyword_predicates() {
        let (_reflected, ids, reg) =
            reflected_with_f32_fields(&[("_VertexOffsetBias", 0), ("_UVOffsetBias", 4)]);
        let mut store = MaterialPropertyStore::new();
        store.set_material(
            55,
            reg.intern("_VertexOffsetMap"),
            MaterialPropertyValue::Texture(packed_texture2d(123)),
        );
        store.set_material(
            55,
            reg.intern("_UVOffsetBias"),
            MaterialPropertyValue::Float(0.25),
        );
        store.set_material(
            55,
            reg.intern("_PositionOffsetMap"),
            MaterialPropertyValue::Texture(packed_texture2d(456)),
        );

        assert_eq!(
            inferred_keyword_float_f32("VERTEX_OFFSET", &store, lookup(55), &ids),
            Some(1.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("UV_OFFSET", &store, lookup(55), &ids),
            Some(1.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("OBJECT_POS_OFFSET", &store, lookup(55), &ids),
            Some(1.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("VERTEX_POS_OFFSET", &store, lookup(55), &ids),
            Some(0.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("VERTEX_OFFSET", &store, lookup(56), &ids),
            Some(0.0)
        );
    }

    #[test]
    fn vec4_defaults_match_documented_unity_conventions() {
        // Spot-check a few entries in the generic vec4 default table that DO need a non-zero
        // value because the relevant WGSL shaders rely on them prior to host writes.
        assert_eq!(
            default_vec4_for_field("_EmissionColor"),
            [0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            default_vec4_for_field("_SpecularColor"),
            [1.0, 1.0, 1.0, 0.5]
        );
        assert_eq!(default_vec4_for_field("_Rect"), [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(default_vec4_for_field("_Point"), [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(default_vec4_for_field("_OverlayTint"), [1.0, 1.0, 1.0, 0.5]);
        assert_eq!(
            default_vec4_for_field("_BackgroundColor"),
            [0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(default_vec4_for_field("_Range"), [0.001, 0.001, 0.0, 0.0]);
        assert_eq!(
            default_vec4_for_field("_BehindFarColor"),
            [0.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(default_vec4_for_field("_Tint0_"), [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn only_main_texture_bindings_fallback_to_primary_texture() {
        use crate::materials::embedded::texture_resolve::should_fallback_to_primary_texture;
        assert!(should_fallback_to_primary_texture("_MainTex"));
        assert!(!should_fallback_to_primary_texture("_MainTex1"));
        assert!(!should_fallback_to_primary_texture("_SpecularMap"));
    }

    /// `_ALBEDOTEX` keyword inference must treat a packed [`HostTextureAssetKind::RenderTexture`] like a
    /// bound texture (parity with 2D-only `texture_property_asset_id_by_pid`).
    #[test]
    fn albedo_keyword_infers_from_render_texture_packed_id() {
        use crate::assets::texture::{HostTextureAssetKind, unpack_host_texture_packed};

        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let main_tex = reg.intern("_MainTex");
        let type_bits = 3u32;
        let pack_type_shift = 32u32.saturating_sub(type_bits);
        let asset_id = 7i32;
        let packed = asset_id | ((HostTextureAssetKind::RenderTexture as i32) << pack_type_shift);
        assert_eq!(
            unpack_host_texture_packed(packed),
            Some((asset_id, HostTextureAssetKind::RenderTexture))
        );
        store.set_material(6, main_tex, MaterialPropertyValue::Texture(packed));
        assert_eq!(
            inferred_keyword_float_f32("_ALBEDOTEX", &store, lookup(6), &ids),
            Some(1.0)
        );
    }

    /// Builds a `StemEmbeddedPropertyIds` that mirrors the projection360 family -- every
    /// uniform-field probe used by the keyword inference is registered. Texture-binding
    /// pids live on `EmbeddedSharedKeywordIds` so they don't need per-stem registration.
    fn projection360_ids(reg: &PropertyIdRegistry) -> StemEmbeddedPropertyIds {
        let mut ids = StemEmbeddedPropertyIds::minimal_for_tests(reg);
        for field_name in [
            "_FOV",
            "_PerspectiveFOV",
            "_TextureLerp",
            "_CubeLOD",
            "_MaxIntensity",
            "_Tint0",
        ] {
            ids.uniform_field_ids
                .insert(field_name.to_string(), reg.intern(field_name));
        }
        ids
    }

    /// Inserts the default full-sphere `_FOV` value `Projection360Material` writes after
    /// `OnAwake()` (`FieldOfView = (360 deg, 180 deg)` converted to radians).
    fn set_full_sphere_fov(store: &mut MaterialPropertyStore, reg: &PropertyIdRegistry, mat: i32) {
        store.set_material(
            mat,
            reg.intern("_FOV"),
            MaterialPropertyValue::Float4([std::f32::consts::TAU, std::f32::consts::PI, 0.0, 0.0]),
        );
    }

    /// Packs a host texture id with an explicit kind tag, matching the shared
    /// `IdPacker<TextureAssetType>` layout `unpack_host_texture_packed` decodes.
    fn packed_texture(asset_id: i32, kind: crate::assets::texture::HostTextureAssetKind) -> i32 {
        let type_bits = 3u32;
        let pack_type_shift = 32u32.saturating_sub(type_bits);
        asset_id | ((kind as i32) << pack_type_shift)
    }

    /// Default `Projection360` materials send `_FOV = (TAU, PI, 0, 0)` -- the OUTSIDE-mode
    /// inference must leave every keyword field at zero so the fragment shader's existing
    /// fallthrough behaves like Unity's default `OUTSIDE_CLIP` (and the choice is moot
    /// since every direction is in-FOV anyway).
    #[test]
    fn projection360_full_sphere_fov_keeps_outside_keywords_zero() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = projection360_ids(&reg);
        store.set_material(
            30,
            reg.intern("_FOV"),
            MaterialPropertyValue::Float4([std::f32::consts::TAU, std::f32::consts::PI, 0.0, 0.0]),
        );
        for field_name in ["OUTSIDE_CLIP", "OUTSIDE_COLOR", "OUTSIDE_CLAMP"] {
            assert_eq!(
                inferred_keyword_float_f32(field_name, &store, lookup(30), &ids),
                Some(0.0),
                "{field_name} should be 0 for full-sphere FOV"
            );
        }
    }

    /// Narrow FOV is what the video player writes -- the renderer must enable
    /// `OUTSIDE_CLAMP` so the partial-FOV pixels render edge-clamped instead of being
    /// discarded by the default-clip fallthrough.
    #[test]
    fn projection360_narrow_fov_enables_outside_clamp() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = projection360_ids(&reg);
        let one_deg = std::f32::consts::PI / 180.0;
        store.set_material(
            31,
            reg.intern("_FOV"),
            MaterialPropertyValue::Float4([one_deg, one_deg, 0.0, 0.0]),
        );
        assert_eq!(
            inferred_keyword_float_f32("OUTSIDE_CLAMP", &store, lookup(31), &ids),
            Some(1.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("OUTSIDE_CLIP", &store, lookup(31), &ids),
            Some(0.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("OUTSIDE_COLOR", &store, lookup(31), &ids),
            Some(0.0)
        );
    }

    /// Hemispherical FOVs (`180 deg x 180 deg`) are partial in X -- must enable `OUTSIDE_CLAMP`.
    #[test]
    fn projection360_hemispherical_fov_enables_outside_clamp() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = projection360_ids(&reg);
        store.set_material(
            32,
            reg.intern("_FOV"),
            MaterialPropertyValue::Float4([std::f32::consts::PI, std::f32::consts::PI, 0.0, 0.0]),
        );
        assert_eq!(
            inferred_keyword_float_f32("OUTSIDE_CLAMP", &store, lookup(32), &ids),
            Some(1.0)
        );
    }

    /// Full-azimuth, half-elevation FOV (`360 deg x 90 deg`) is partial in Y -- must enable
    /// `OUTSIDE_CLAMP`.
    #[test]
    fn projection360_full_azimuth_half_elevation_enables_outside_clamp() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = projection360_ids(&reg);
        store.set_material(
            33,
            reg.intern("_FOV"),
            MaterialPropertyValue::Float4([
                std::f32::consts::TAU,
                std::f32::consts::FRAC_PI_2,
                0.0,
                0.0,
            ]),
        );
        assert_eq!(
            inferred_keyword_float_f32("OUTSIDE_CLAMP", &store, lookup(33), &ids),
            Some(1.0)
        );
    }

    /// Float-precision residuals from the host's degrees->radians conversion
    /// (`* PI / 180`) must still classify whole-sphere defaults as full-sphere.
    #[test]
    fn projection360_full_sphere_tolerates_float_residual() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = projection360_ids(&reg);
        let fov_x = 360.0_f32 * std::f32::consts::PI / 180.0;
        let fov_y = 180.0_f32 * std::f32::consts::PI / 180.0;
        store.set_material(
            34,
            reg.intern("_FOV"),
            MaterialPropertyValue::Float4([fov_x, fov_y, 0.0, 0.0]),
        );
        assert_eq!(
            inferred_keyword_float_f32("OUTSIDE_CLAMP", &store, lookup(34), &ids),
            Some(0.0),
            "host-computed (TAU, PI) within tolerance must remain full-sphere"
        );
    }

    /// `Mode.Perspective` is the video player's mode -- `_PerspectiveFOV` is sent only when
    /// `Projection.Value == Mode.Perspective`, so its presence drives `_PERSPECTIVE = 1`.
    /// Without this inference the shader silently downgrades to `_VIEW`, producing a
    /// vertical-line-stretched-vertically render because object-space view-direction's
    /// `atan2(view_dir.x, view_dir.z)` is near-constant across a flat quad.
    #[test]
    fn projection360_perspective_fov_present_enables_perspective_keyword() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = projection360_ids(&reg);
        set_full_sphere_fov(&mut store, &reg, 40);
        store.set_material(
            40,
            reg.intern("_PerspectiveFOV"),
            MaterialPropertyValue::Float4([
                std::f32::consts::FRAC_PI_4,
                std::f32::consts::FRAC_PI_4,
                0.0,
                0.0,
            ]),
        );
        assert_eq!(
            inferred_keyword_float_f32("_PERSPECTIVE", &store, lookup(40), &ids),
            Some(1.0)
        );
    }

    /// `Mode.View` (default) never sends `_PerspectiveFOV` -- the keyword stays off and the
    /// shader falls through to the object-space view direction.
    #[test]
    fn projection360_no_perspective_fov_keeps_perspective_keyword_off() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = projection360_ids(&reg);
        set_full_sphere_fov(&mut store, &reg, 41);
        assert_eq!(
            inferred_keyword_float_f32("_PERSPECTIVE", &store, lookup(41), &ids),
            Some(0.0)
        );
    }

    /// Cubemap textures bound on `_MainCube` or `_SecondCube` enable the `CUBEMAP` keyword
    /// (no `_CubeLOD` written -> fixed-mip cubemap path).
    #[test]
    fn projection360_main_cube_present_enables_cubemap_keyword() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = projection360_ids(&reg);
        set_full_sphere_fov(&mut store, &reg, 42);
        store.set_material(
            42,
            reg.intern("_MainCube"),
            MaterialPropertyValue::Texture(packed_texture(
                3,
                crate::assets::texture::HostTextureAssetKind::Cubemap,
            )),
        );
        assert_eq!(
            inferred_keyword_float_f32("CUBEMAP", &store, lookup(42), &ids),
            Some(1.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("CUBEMAP_LOD", &store, lookup(42), &ids),
            Some(0.0)
        );
    }

    /// `_CubeLOD` written alongside a cubemap routes to `CUBEMAP_LOD` (mirrors host's
    /// `CubemapLOD.Value.HasValue` predicate).
    #[test]
    fn projection360_cubemap_with_cube_lod_enables_cubemap_lod_keyword() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = projection360_ids(&reg);
        set_full_sphere_fov(&mut store, &reg, 43);
        store.set_material(
            43,
            reg.intern("_MainCube"),
            MaterialPropertyValue::Texture(packed_texture(
                3,
                crate::assets::texture::HostTextureAssetKind::Cubemap,
            )),
        );
        store.set_material(
            43,
            reg.intern("_CubeLOD"),
            MaterialPropertyValue::Float(2.0),
        );
        assert_eq!(
            inferred_keyword_float_f32("CUBEMAP_LOD", &store, lookup(43), &ids),
            Some(1.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("CUBEMAP", &store, lookup(43), &ids),
            Some(0.0)
        );
    }

    /// Secondary 2D texture bound on `_SecondTex` enables `SECOND_TEXTURE`.
    #[test]
    fn projection360_secondary_texture_enables_second_texture_keyword() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = projection360_ids(&reg);
        set_full_sphere_fov(&mut store, &reg, 44);
        store.set_material(
            44,
            reg.intern("_SecondTex"),
            MaterialPropertyValue::Texture(packed_texture(
                7,
                crate::assets::texture::HostTextureAssetKind::Texture2D,
            )),
        );
        assert_eq!(
            inferred_keyword_float_f32("SECOND_TEXTURE", &store, lookup(44), &ids),
            Some(1.0)
        );
    }

    /// `TextureLerp != 0` on its own enables `SECOND_TEXTURE` (mirrors the host's
    /// `state2 = ... || TextureLerp.Value != 0f` predicate). Even without a secondary
    /// asset, the host turns on the keyword so the shader's lerp branch runs.
    #[test]
    fn projection360_nonzero_texture_lerp_enables_second_texture_keyword() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = projection360_ids(&reg);
        set_full_sphere_fov(&mut store, &reg, 45);
        store.set_material(
            45,
            reg.intern("_TextureLerp"),
            MaterialPropertyValue::Float(0.5),
        );
        assert_eq!(
            inferred_keyword_float_f32("SECOND_TEXTURE", &store, lookup(45), &ids),
            Some(1.0)
        );
    }

    /// `_OffsetTex` texture binding enables the `_OFFSET` direction-perturbation path.
    #[test]
    fn projection360_offset_texture_enables_offset_keyword() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = projection360_ids(&reg);
        set_full_sphere_fov(&mut store, &reg, 46);
        store.set_material(
            46,
            reg.intern("_OffsetTex"),
            MaterialPropertyValue::Texture(packed_texture(
                11,
                crate::assets::texture::HostTextureAssetKind::Texture2D,
            )),
        );
        assert_eq!(
            inferred_keyword_float_f32("_OFFSET", &store, lookup(46), &ids),
            Some(1.0)
        );
    }

    /// `_MaxIntensity` is sent only when `MaxIntensity.HasValue || HDR texture` -- its
    /// presence drives `_CLAMP_INTENSITY`.
    #[test]
    fn projection360_max_intensity_present_enables_clamp_intensity_keyword() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = projection360_ids(&reg);
        set_full_sphere_fov(&mut store, &reg, 47);
        store.set_material(
            47,
            reg.intern("_MaxIntensity"),
            MaterialPropertyValue::Float(8.0),
        );
        assert_eq!(
            inferred_keyword_float_f32("_CLAMP_INTENSITY", &store, lookup(47), &ids),
            Some(1.0)
        );
    }

    /// `TintTexture` set with `TintTextureMode == Direct` sends `_TintTex` but not
    /// `_Tint0`/`_Tint1` -- `TINT_TEX_DIRECT` enables, `TINT_TEX_LERP` stays off.
    #[test]
    fn projection360_tint_texture_without_tint0_routes_to_direct() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = projection360_ids(&reg);
        set_full_sphere_fov(&mut store, &reg, 48);
        store.set_material(
            48,
            reg.intern("_TintTex"),
            MaterialPropertyValue::Texture(packed_texture(
                13,
                crate::assets::texture::HostTextureAssetKind::Texture2D,
            )),
        );
        assert_eq!(
            inferred_keyword_float_f32("TINT_TEX_DIRECT", &store, lookup(48), &ids),
            Some(1.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("TINT_TEX_LERP", &store, lookup(48), &ids),
            Some(0.0)
        );
    }

    /// `TintTextureMode == Lerp` sends `_Tint0`/`_Tint1` alongside `_TintTex` --
    /// `TINT_TEX_LERP` enables, `TINT_TEX_DIRECT` stays off.
    #[test]
    fn projection360_tint_texture_with_tint0_routes_to_lerp() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = projection360_ids(&reg);
        set_full_sphere_fov(&mut store, &reg, 49);
        store.set_material(
            49,
            reg.intern("_TintTex"),
            MaterialPropertyValue::Texture(packed_texture(
                13,
                crate::assets::texture::HostTextureAssetKind::Texture2D,
            )),
        );
        store.set_material(
            49,
            reg.intern("_Tint0"),
            MaterialPropertyValue::Float4([1.0, 0.0, 0.0, 1.0]),
        );
        assert_eq!(
            inferred_keyword_float_f32("TINT_TEX_LERP", &store, lookup(49), &ids),
            Some(1.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("TINT_TEX_DIRECT", &store, lookup(49), &ids),
            Some(0.0)
        );
    }

    /// Stems without an `_FOV` uniform field (i.e., not `Projection360`) must not
    /// participate in the inference -- the call falls through to the generic
    /// keyword-like-field default of `0`.
    #[test]
    fn outside_mode_inference_does_not_fire_for_non_projection360_stems() {
        let store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        for field_name in [
            "OUTSIDE_CLIP",
            "OUTSIDE_COLOR",
            "OUTSIDE_CLAMP",
            "_PERSPECTIVE",
            "CUBEMAP",
            "CUBEMAP_LOD",
            "SECOND_TEXTURE",
            "_OFFSET",
            "_CLAMP_INTENSITY",
            "TINT_TEX_DIRECT",
            "TINT_TEX_LERP",
        ] {
            assert_eq!(
                inferred_keyword_float_f32(field_name, &store, lookup(35), &ids),
                Some(0.0),
                "{field_name} should default to 0 when stem has no _FOV"
            );
        }
    }
}

mod storage_orientation_uniform_tests {
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
}
