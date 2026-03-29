//! Composes Renderide WGSL under `RENDERIDESHADERS/` with [naga_oil] at build time and writes flat
//! WGSL into `OUT_DIR` for `include_str!` in pipeline modules.
//!
//! [naga_oil]: https://docs.rs/naga_oil/

use std::fs;
use std::path::Path;

use naga::back::wgsl::WriterFlags;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga_oil::compose::{ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderType};

fn main() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let shader_dir = manifest_dir.join("RENDERIDESHADERS");
    let out_dir_var = std::env::var("OUT_DIR").expect("OUT_DIR");
    let out_dir = Path::new(&out_dir_var);

    println!("cargo:rerun-if-changed=RENDERIDESHADERS/common/uniform_ring.wgsl");
    println!("cargo:rerun-if-changed=RENDERIDESHADERS/common/color_util.wgsl");
    println!("cargo:rerun-if-changed=RENDERIDESHADERS/common/ui_common.wgsl");
    println!("cargo:rerun-if-changed=RENDERIDESHADERS/world/unlit.wgsl");
    println!("cargo:rerun-if-changed=RENDERIDESHADERS/world/overlay_unlit.wgsl");
    println!("cargo:rerun-if-changed=RENDERIDESHADERS/world/volume_unlit.wgsl");
    println!("cargo:rerun-if-changed=RENDERIDESHADERS/world/billboard_unlit.wgsl");
    println!("cargo:rerun-if-changed=RENDERIDESHADERS/ui/ui_unlit.wgsl");
    println!("cargo:rerun-if-changed=RENDERIDESHADERS/ui/ui_circle_segment.wgsl");
    println!("cargo:rerun-if-changed=RENDERIDESHADERS/ui/ui_text_unlit.wgsl");
    println!("cargo:rerun-if-changed=RENDERIDESHADERS/ui/text_unlit.wgsl");
    println!("cargo:rerun-if-changed=RENDERIDESHADERS/pbr/pbs_metallic.wgsl");
    println!("cargo:rerun-if-changed=RENDERIDESHADERS/pbr/pbs_metallic_host_albedo.wgsl");
    println!("cargo:rerun-if-changed=build.rs");

    for (out_name, rel_path) in [
        ("world_unlit", "world/unlit.wgsl"),
        ("overlay_unlit", "world/overlay_unlit.wgsl"),
        ("volume_unlit", "world/volume_unlit.wgsl"),
        ("billboard_unlit", "world/billboard_unlit.wgsl"),
        ("ui_unlit", "ui/ui_unlit.wgsl"),
        ("ui_circle_segment", "ui/ui_circle_segment.wgsl"),
        ("ui_text_unlit", "ui/ui_text_unlit.wgsl"),
        ("text_unlit", "ui/text_unlit.wgsl"),
        ("pbs_metallic", "pbr/pbs_metallic.wgsl"),
        (
            "pbs_metallic_host_albedo",
            "pbr/pbs_metallic_host_albedo.wgsl",
        ),
    ] {
        let composed = compose_entry(&shader_dir, rel_path).unwrap_or_else(|e| {
            panic!("naga_oil compose failed for {rel_path}: {e:?}");
        });
        let path = out_dir.join(format!("{out_name}.wgsl"));
        fs::write(&path, composed).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    }
}

fn compose_entry(shader_dir: &Path, entry_rel_path: &str) -> Result<String, String> {
    let mut composer = Composer::default();

    for rel in [
        "common/uniform_ring.wgsl",
        "common/color_util.wgsl",
        "common/ui_common.wgsl",
    ] {
        let path = shader_dir.join(rel);
        let source = fs::read_to_string(&path).map_err(|e| e.to_string())?;
        let file_path = path.to_string_lossy().into_owned();
        composer
            .add_composable_module(ComposableModuleDescriptor {
                source: &source,
                file_path: &file_path,
                ..Default::default()
            })
            .map_err(|e| format!("{e:?}"))?;
    }

    let entry_path = shader_dir.join(entry_rel_path);
    let entry_source = fs::read_to_string(&entry_path).map_err(|e| e.to_string())?;
    let module = composer
        .make_naga_module(NagaModuleDescriptor {
            source: &entry_source,
            file_path: &entry_path.to_string_lossy(),
            shader_type: ShaderType::Wgsl,
            ..Default::default()
        })
        .map_err(|e| format!("{e:?}"))?;

    let caps = Capabilities::all();
    let info = Validator::new(ValidationFlags::all(), caps)
        .validate(&module)
        .map_err(|e| format!("{e:?}"))?;

    naga::back::wgsl::write_string(&module, &info, WriterFlags::EXPLICIT_TYPES)
        .map_err(|e| format!("{e:?}"))
}
