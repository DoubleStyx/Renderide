//! Composes `shaders/source/modules/*.wgsl` with [`naga_oil`] (`#import`), validates with naga, and
//! writes flat `shaders/target/*.wgsl` plus `OUT_DIR/embedded_shaders.rs`.
//!
//! Optional per-material metadata: `shaders/source/materials/<stem>.meta.json`:
//! `{ "unity_names": ["MyGame/CustomUnlit"] }` (normalized keys are applied at resolve time).
//!
//! `debug_world_normals.wgsl` is special-cased: two targets (`debug_world_normals_default`,
//! `debug_world_normals_multiview`) are emitted from one source using naga-oil `shader_defs`
//! (`MULTIVIEW` true/false).

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use naga::back::wgsl::WriterFlags;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga_oil::compose::{
    ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderDefValue, ShaderLanguage,
    ShaderType,
};

fn validate_and_write_wgsl(module: &naga::Module, label: &str, out_path: &std::path::Path) {
    let has_vs = module
        .entry_points
        .iter()
        .any(|e| e.stage == naga::ShaderStage::Vertex && e.name == "vs_main");
    let has_fs = module
        .entry_points
        .iter()
        .any(|e| e.stage == naga::ShaderStage::Fragment && e.name == "fs_main");
    if !has_vs || !has_fs {
        panic!(
            "{label}: expected entry points vs_main and fs_main (vertex={has_vs} fragment={has_fs})",
        );
    }

    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    let info = validator
        .validate(module)
        .unwrap_or_else(|e| panic!("validate {label}: {e}"));
    let wgsl = naga::back::wgsl::write_string(module, &info, WriterFlags::EXPLICIT_TYPES)
        .unwrap_or_else(|e| panic!("wgsl out {label}: {e}"));
    fs::write(out_path, &wgsl).unwrap_or_else(|e| panic!("write {}: {e}", out_path.display()));
}

fn add_globals_module(composer: &mut Composer, globals_source: &str) {
    composer
        .add_composable_module(ComposableModuleDescriptor {
            source: globals_source,
            file_path: "shaders/source/modules/globals.wgsl",
            language: ShaderLanguage::Wgsl,
            ..Default::default()
        })
        .unwrap_or_else(|e| panic!("add globals module: {e}"));
}

fn compose_material(
    globals_source: &str,
    material_source: &str,
    material_file_path: &str,
    shader_defs: HashMap<String, ShaderDefValue>,
) -> naga::Module {
    let mut composer = Composer::default().with_capabilities(Capabilities::all());
    add_globals_module(&mut composer, globals_source);
    composer
        .make_naga_module(NagaModuleDescriptor {
            source: material_source,
            file_path: material_file_path,
            shader_type: ShaderType::Wgsl,
            shader_defs,
            ..Default::default()
        })
        .unwrap_or_else(|e| panic!("compose {material_file_path}: {e}"))
}

fn main() {
    let manifest_dir =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let globals_path = manifest_dir.join("shaders/source/modules/globals.wgsl");
    let materials_dir = manifest_dir.join("shaders/source/materials");
    let target_dir = manifest_dir.join("shaders/target");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));

    println!("cargo:rerun-if-changed=shaders/source");
    println!("cargo:rerun-if-changed=build.rs");

    fs::create_dir_all(materials_dir.parent().expect("materials parent")).expect("mkdir");
    fs::create_dir_all(&target_dir).expect("mkdir target");

    let globals_source = fs::read_to_string(&globals_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", globals_path.display()));

    let mut manifest_entries: Vec<serde_json::Value> = Vec::new();
    let mut embedded_arms = String::new();
    let mut output_stems: Vec<String> = Vec::new();

    let mut material_paths: Vec<PathBuf> = fs::read_dir(&materials_dir)
        .expect("read materials")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "wgsl").unwrap_or(false))
        .collect();
    material_paths.sort();

    for path in &material_paths {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("material stem");

        if stem == "debug_world_normals" {
            let material_source =
                fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));

            let meta_path = materials_dir.join("debug_world_normals.meta.json");
            let unity_names: Vec<String> = if meta_path.is_file() {
                let raw = fs::read_to_string(&meta_path)
                    .unwrap_or_else(|e| panic!("read {}: {e}", meta_path.display()));
                let v: serde_json::Value = serde_json::from_str(&raw)
                    .unwrap_or_else(|e| panic!("parse {}: {e}", meta_path.display()));
                v.get("unity_names")
                    .and_then(|u| u.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|x| x.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default()
            } else {
                Vec::new()
            };

            let variants = [
                ("debug_world_normals_default", false),
                ("debug_world_normals_multiview", true),
            ];
            for (target_stem, multiview) in variants {
                let mut defs = HashMap::new();
                defs.insert("MULTIVIEW".to_string(), ShaderDefValue::Bool(multiview));
                let module = compose_material(
                    &globals_source,
                    &material_source,
                    path.to_str().expect("utf8 path"),
                    defs,
                );
                let label = format!("{target_stem} (MULTIVIEW={multiview})");
                let out_path = target_dir.join(format!("{target_stem}.wgsl"));
                validate_and_write_wgsl(&module, &label, &out_path);

                manifest_entries.push(serde_json::json!({
                    "stem": target_stem,
                    "file": format!("shaders/target/{target_stem}.wgsl"),
                    "unity_names": unity_names.clone(),
                }));

                embedded_arms.push_str(&format!(
                    "        \"{target_stem}\" => Some(include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/shaders/target/{target_stem}.wgsl\"))),\n"
                ));
                output_stems.push(target_stem.to_string());
            }
            continue;
        }

        if stem == "world_unlit" {
            let material_source =
                fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));

            let meta_path = materials_dir.join("world_unlit.meta.json");
            let unity_names: Vec<String> = if meta_path.is_file() {
                let raw = fs::read_to_string(&meta_path)
                    .unwrap_or_else(|e| panic!("read {}: {e}", meta_path.display()));
                let v: serde_json::Value = serde_json::from_str(&raw)
                    .unwrap_or_else(|e| panic!("parse {}: {e}", meta_path.display()));
                v.get("unity_names")
                    .and_then(|u| u.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|x| x.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default()
            } else {
                Vec::new()
            };

            let variants = [
                ("world_unlit_default", false),
                ("world_unlit_multiview", true),
            ];
            for (target_stem, multiview) in variants {
                let mut defs = HashMap::new();
                defs.insert("MULTIVIEW".to_string(), ShaderDefValue::Bool(multiview));
                let module = compose_material(
                    &globals_source,
                    &material_source,
                    path.to_str().expect("utf8 path"),
                    defs,
                );
                let label = format!("{target_stem} (MULTIVIEW={multiview})");
                let out_path = target_dir.join(format!("{target_stem}.wgsl"));
                validate_and_write_wgsl(&module, &label, &out_path);

                let entry_names = if multiview {
                    Vec::new()
                } else {
                    unity_names.clone()
                };
                manifest_entries.push(serde_json::json!({
                    "stem": target_stem,
                    "file": format!("shaders/target/{target_stem}.wgsl"),
                    "unity_names": entry_names,
                }));

                embedded_arms.push_str(&format!(
                    "        \"{target_stem}\" => Some(include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/shaders/target/{target_stem}.wgsl\"))),\n"
                ));
                output_stems.push(target_stem.to_string());
            }
            continue;
        }

        let material_source =
            fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));

        let meta_path = materials_dir.join(format!("{stem}.meta.json"));
        let unity_names: Vec<String> = if meta_path.is_file() {
            let raw = fs::read_to_string(&meta_path)
                .unwrap_or_else(|e| panic!("read {}: {e}", meta_path.display()));
            let v: serde_json::Value = serde_json::from_str(&raw)
                .unwrap_or_else(|e| panic!("parse {}: {e}", meta_path.display()));
            v.get("unity_names")
                .and_then(|u| u.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|x| x.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        let module = compose_material(
            &globals_source,
            &material_source,
            path.to_str().expect("utf8 path"),
            HashMap::new(),
        );
        let out_path = target_dir.join(format!("{stem}.wgsl"));
        validate_and_write_wgsl(&module, stem, &out_path);

        manifest_entries.push(serde_json::json!({
            "stem": stem,
            "file": format!("shaders/target/{stem}.wgsl"),
            "unity_names": unity_names,
        }));

        embedded_arms.push_str(&format!(
            "        \"{stem}\" => Some(include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/shaders/target/{stem}.wgsl\"))),\n"
        ));
        output_stems.push(stem.to_string());
    }

    let manifest = serde_json::json!({
        "materials": manifest_entries,
        "globals_module": "shaders/source/modules/globals.wgsl",
    });
    let manifest_path = target_dir.join("manifest.json");
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).expect("manifest json"),
    )
    .expect("write manifest");

    let stems_list = output_stems
        .iter()
        .map(|s| format!("    \"{s}\","))
        .collect::<Vec<_>>()
        .join("\n");

    let embedded_rs = format!(
        r#"// Generated by `build.rs` — do not edit.

/// Flattened WGSL for `stem` (see `shaders/target/{{stem}}.wgsl`).
pub fn embedded_target_wgsl(stem: &str) -> Option<&'static str> {{
    match stem {{
{embedded_arms}        _ => None,
    }}
}}

/// Stems under `shaders/target/*.wgsl` present at build time.
pub const COMPILED_MATERIAL_STEMS: &[&str] = &[
{stems}
];

/// JSON manifest written next to composed shaders.
pub const SHADER_MANIFEST_JSON: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/target/manifest.json"));
"#,
        embedded_arms = embedded_arms,
        stems = stems_list
    );

    let gen_path = out_dir.join("embedded_shaders.rs");
    fs::write(&gen_path, embedded_rs).expect("write embedded_shaders.rs");
}
