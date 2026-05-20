//! Tests for Unity shader asset resolution.

use super::*;

#[test]
fn container_asset_path_strips_shader_suffix() {
    assert_eq!(
        shader_asset_name_from_container_asset_path("assets/foo/my_shader.shader").as_deref(),
        Some("my_shader")
    );
    assert_eq!(
        shader_asset_name_from_container_asset_path("archive:/CAB-deadbeef").as_deref(),
        None
    );
}

#[test]
fn container_asset_path_handles_backslashes_whitespace_and_plain_stems() {
    assert_eq!(
        shader_asset_name_from_container_asset_path("Assets\\Shaders\\UI Text Unlit.shader")
            .as_deref(),
        Some("ui text unlit")
    );
    assert_eq!(
        shader_asset_name_from_container_asset_path("  assets/foo/ToonLit.shader  ").as_deref(),
        Some("toonlit")
    );
    assert_eq!(
        shader_asset_name_from_container_asset_path("assets/foo/AlreadyStem").as_deref(),
        Some("alreadystem")
    );
    assert_eq!(
        shader_asset_name_from_container_asset_path("").as_deref(),
        None
    );
    assert_eq!(
        shader_asset_name_from_container_asset_path("assets/foo/   ").as_deref(),
        None
    );
}

#[test]
fn prefix_formatters_are_stable_for_empty_short_and_truncated_inputs() {
    assert_eq!(format_hex_prefix(&[], 8), "");
    assert_eq!(format_hex_prefix(&[0, 1, 0xab, 0xff], 8), "00 01 ab ff");
    assert_eq!(format_hex_prefix(&[0, 1, 2, 3], 2), "00 01");
    assert_eq!(short_hex_prefix("00 01 02 03", 2), "00 01");
    assert_eq!(short_hex_prefix("00 01", 8), "00 01");
}

#[test]
fn ascii_prefix_hint_only_returns_printable_prefixes() {
    assert_eq!(ascii_prefix_hint(b"", 8), "");
    assert_eq!(ascii_prefix_hint(b"UnityFS\nBundle", 32), "UnityFS\nBundle");
    assert_eq!(ascii_prefix_hint(&[0xff, b'A', b'B'], 32), "");
    assert_eq!(ascii_prefix_hint(b"abcdef", 3), "abc");
}

#[test]
fn truncate_display_preserves_short_errors_and_truncates_long_errors() {
    assert_eq!(truncate_display("short", 16), "short");
    let truncated = truncate_display("abcdefghijklmnopqrstuvwxyz", 8);
    assert_eq!(truncated, "abcdefg...");
}

#[test]
fn internal_shader_name_strips_variant_suffix() {
    assert_eq!(
        parse_internal_shader_name("Unlit_00002202"),
        Some(InternalShaderName {
            full_name: "Unlit_00002202".to_string(),
            shader_asset_name: "Unlit".to_string(),
            shader_variant_bits: Some(0x2202),
        })
    );
    assert_eq!(
        parse_internal_shader_name("Custom/With_Underscore_00000080"),
        Some(InternalShaderName {
            full_name: "Custom/With_Underscore_00000080".to_string(),
            shader_asset_name: "With_Underscore".to_string(),
            shader_variant_bits: Some(0x80),
        })
    );
    assert_eq!(
        parse_internal_shader_name("Unlit_nothex123"),
        Some(InternalShaderName {
            full_name: "Unlit_nothex123".to_string(),
            shader_asset_name: "Unlit_nothex123".to_string(),
            shader_variant_bits: None,
        })
    );
}

#[test]
fn shader_lab_name_parser_uses_top_level_shader_before_fallback() {
    let source = r#"
            Shader "PBSLerp" {
                Properties {}
                FallBack "Transparent/Cutout/VertexLit"
            }
        "#;

    assert_eq!(
        parse_shader_lab_internal_name(source),
        Some(InternalShaderName {
            full_name: "PBSLerp".to_string(),
            shader_asset_name: "PBSLerp".to_string(),
            shader_variant_bits: None,
        })
    );
}

#[test]
fn shader_lab_name_parser_preserves_variant_suffix() {
    let source = r#"
            Shader "PBSLerpSpecular_000000B1" {
                FallBack "Transparent/Cutout/VertexLit"
            }
        "#;

    assert_eq!(
        parse_shader_lab_internal_name(source),
        Some(InternalShaderName {
            full_name: "PBSLerpSpecular_000000B1".to_string(),
            shader_asset_name: "PBSLerpSpecular".to_string(),
            shader_variant_bits: Some(0xB1),
        })
    );
}

#[test]
fn shader_lab_name_parser_ignores_comments_strings_and_fallback_only_text() {
    assert_eq!(
        parse_shader_lab_internal_name(r#"FallBack "Transparent/Cutout/VertexLit""#),
        None
    );

    let source = r#"
            // Shader "CommentedOut"
            CustomEditor "ShaderGUI"
            /* Shader "AlsoCommentedOut" */
            Shader "PBSLerpMetallic_000000B1" {}
        "#;

    assert_eq!(
        parse_shader_lab_internal_name(source),
        Some(InternalShaderName {
            full_name: "PBSLerpMetallic_000000B1".to_string(),
            shader_asset_name: "PBSLerpMetallic".to_string(),
            shader_variant_bits: Some(0xB1),
        })
    );
}

#[test]
fn shader_lab_value_parser_prefers_declaration_over_parsed_form_name() {
    let parsed_form = UnityValue::Object(
        [
            (
                "m_Name".to_string(),
                UnityValue::String("Legacy Shaders/Transparent/Cutout/VertexLit".to_string()),
            ),
            (
                "m_SerializedShader".to_string(),
                UnityValue::String(
                    r#"Shader "PBSLerpSpecular_000000B1" {
                            FallBack "Transparent/Cutout/VertexLit"
                        }"#
                    .to_string(),
                ),
            ),
        ]
        .into_iter()
        .collect(),
    );

    assert_eq!(
        shader_lab_internal_name_from_unity_value(&parsed_form),
        Some(InternalShaderName {
            full_name: "PBSLerpSpecular_000000B1".to_string(),
            shader_asset_name: "PBSLerpSpecular".to_string(),
            shader_variant_bits: Some(0xB1),
        })
    );
    assert_eq!(
        parsed_form_internal_shader_name(&parsed_form),
        Some(InternalShaderName {
            full_name: "Legacy Shaders/Transparent/Cutout/VertexLit".to_string(),
            shader_asset_name: "VertexLit".to_string(),
            shader_variant_bits: None,
        })
    );
}

#[test]
fn resolution_uses_lowercase_container_filename_route_and_shader_lab_variant_bits() {
    let shader_asset_name =
        shader_asset_name_from_container_asset_path("Assets/Shaders/PBSLerpSpecular.shader");
    assert_eq!(
        shader_resolution_from_candidates(&[shader_candidate(
            1,
            shader_asset_name.as_deref(),
            Some(internal_shader_name("PBSLerpSpecular_000000B1", Some(0xB1))),
        )]),
        Some(ResolvedUnityShaderAsset {
            shader_asset_name: "pbslerpspecular".to_string(),
            shader_variant_bits: Some(0xB1),
        })
    );
}

#[test]
fn resolution_skips_fallback_shader_for_matching_variant_candidate() {
    assert_eq!(
        shader_resolution_from_candidates(&[
            shader_candidate(
                3_464_988_009_001_945_076,
                Some("pbslerpspecular"),
                Some(internal_shader_name(
                    "Legacy Shaders/Transparent/Cutout/VertexLit",
                    None,
                )),
            ),
            shader_candidate(
                4_060_164_223_764_131_682,
                None,
                Some(internal_shader_name("PBSLerpSpecular_000000B1", Some(0xB1))),
            ),
        ]),
        Some(ResolvedUnityShaderAsset {
            shader_asset_name: "pbslerpspecular".to_string(),
            shader_variant_bits: Some(0xB1),
        })
    );
}

#[test]
fn resolution_does_not_use_internal_name_as_route() {
    assert_eq!(
        shader_resolution_from_candidates(&[shader_candidate(
            1,
            None,
            Some(internal_shader_name("PBSLerpSpecular_000000B1", Some(0xB1))),
        )]),
        None
    );
}

#[test]
fn resolution_keeps_container_route_when_internal_variant_name_differs() {
    assert_eq!(
        shader_resolution_from_candidates(&[shader_candidate(
            1,
            Some("ui_unlit"),
            Some(internal_shader_name("UI/Unlit_00000014", Some(0x14))),
        )]),
        Some(ResolvedUnityShaderAsset {
            shader_asset_name: "ui_unlit".to_string(),
            shader_variant_bits: Some(0x14),
        })
    );
}

#[test]
fn resolution_skips_fallback_variant_names() {
    assert_eq!(
        shader_resolution_from_candidates(&[shader_candidate(
            1,
            Some("pbslerpspecular"),
            Some(internal_shader_name(
                "Legacy Shaders/Transparent/Cutout/VertexLit_00000001",
                Some(1),
            )),
        )]),
        Some(ResolvedUnityShaderAsset {
            shader_asset_name: "pbslerpspecular".to_string(),
            shader_variant_bits: None,
        })
    );
}

#[test]
fn resolution_ignores_internal_names_without_variant_suffixes() {
    assert_eq!(
        shader_resolution_from_candidates(&[shader_candidate(
            1,
            Some("pbslerpspecular"),
            Some(internal_shader_name("PBSLerpSpecular", None)),
        )]),
        Some(ResolvedUnityShaderAsset {
            shader_asset_name: "pbslerpspecular".to_string(),
            shader_variant_bits: None,
        })
    );
}

#[test]
fn parsed_form_name_field_is_internal_shader_name() {
    let parsed_form = UnityValue::Object(
        std::iter::once((
            "m_Name".to_string(),
            UnityValue::String("Unlit_00000200".to_string()),
        ))
        .collect(),
    );

    assert_eq!(
        parsed_form_internal_shader_name(&parsed_form),
        Some(InternalShaderName {
            full_name: "Unlit_00000200".to_string(),
            shader_asset_name: "Unlit".to_string(),
            shader_variant_bits: Some(0x200),
        })
    );
}

#[test]
fn parsed_form_plain_name_is_stem_without_variant_bits() {
    let parsed_form = UnityValue::Object(
        std::iter::once((
            "m_Name".to_string(),
            UnityValue::String("Unlit".to_string()),
        ))
        .collect(),
    );

    assert_eq!(
        parsed_form_internal_shader_name(&parsed_form),
        Some(InternalShaderName {
            full_name: "Unlit".to_string(),
            shader_asset_name: "Unlit".to_string(),
            shader_variant_bits: None,
        })
    );
}

#[test]
fn parsed_form_name_missing_returns_none() {
    let parsed_form = UnityValue::Object(
        std::iter::once((
            "m_Script".to_string(),
            UnityValue::String("Unlit_00000200".to_string()),
        ))
        .collect(),
    );

    assert_eq!(parsed_form_internal_shader_name(&parsed_form), None);
}

#[test]
fn file_binary_probe_records_prefixes_without_parsing() {
    let probe = FileBinaryProbe::new(b"UnityFS\0binary");
    assert_eq!(probe.bytes_len, 14);
    assert!(probe.prefix_hex.starts_with("55 6e 69 74 79 46 53 00"));
    assert_eq!(probe.prefix_ascii, "");
    assert!(!probe.bundle_parse_ok);
    assert_eq!(probe.bundle_assets, 0);
    assert_eq!(probe.bundle_err, None);
}

fn shader_candidate(
    path_id: i64,
    container_name: Option<&str>,
    internal_name: Option<InternalShaderName>,
) -> ShaderObjectCandidate {
    let internal_source = internal_name
        .as_ref()
        .map(|_| InternalShaderNameSource::ParsedFormName);
    ShaderObjectCandidate {
        path_id,
        class_id: SHADER,
        container_name: container_name.map(str::to_string),
        internal_name,
        internal_source,
    }
}

fn internal_shader_name(full_name: &str, shader_variant_bits: Option<u32>) -> InternalShaderName {
    let shader_asset_name = full_name
        .rsplit_once('_')
        .map_or(full_name, |(stem, _)| stem)
        .rsplit('/')
        .next()
        .unwrap_or(full_name)
        .to_string();
    InternalShaderName {
        full_name: full_name.to_string(),
        shader_asset_name,
        shader_variant_bits,
    }
}

#[test]
fn path_hint_rejects_missing_paths_and_empty_directories() {
    let temp = tempfile::tempdir().expect("tempdir");
    assert_eq!(
        try_resolve_shader_asset_name_from_path(&temp.path().join("missing")),
        None
    );
    assert_eq!(try_resolve_shader_asset_name_from_path(temp.path()), None);
}
