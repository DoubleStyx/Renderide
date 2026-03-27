//! Composes modular WGSL under `wgsl_modules/` with [naga_oil] at build time and writes flat WGSL
//! into `OUT_DIR` for `include_str!` in [`crate::gpu::pipeline::shaders`].
//!
//! [naga_oil]: https://docs.rs/naga_oil/

use std::fs;
use std::path::Path;

use naga::back::wgsl::WriterFlags;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga_oil::compose::{ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderType};

fn main() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let wgsl_dir = manifest_dir.join("wgsl_modules");
    let out_dir_var = std::env::var("OUT_DIR").expect("OUT_DIR");
    let out_dir = Path::new(&out_dir_var);

    println!("cargo:rerun-if-changed=wgsl_modules/uniform_ring.wgsl");
    println!("cargo:rerun-if-changed=wgsl_modules/color_util.wgsl");
    println!("cargo:rerun-if-changed=wgsl_modules/normal_debug.wgsl");
    println!("cargo:rerun-if-changed=wgsl_modules/uv_debug.wgsl");
    println!("cargo:rerun-if-changed=wgsl_modules/host_unlit.wgsl");
    println!("cargo:rerun-if-changed=wgsl_modules/world_unlit.wgsl");
    println!("cargo:rerun-if-changed=wgsl_modules/ui_common.wgsl");
    println!("cargo:rerun-if-changed=wgsl_modules/ui_unlit.wgsl");
    println!("cargo:rerun-if-changed=wgsl_modules/ui_text_unlit.wgsl");
    println!("cargo:rerun-if-changed=build.rs");

    for name in [
        "normal_debug",
        "uv_debug",
        "host_unlit",
        "world_unlit",
        "ui_unlit",
        "ui_text_unlit",
    ] {
        let composed = compose_entry(&wgsl_dir, name).unwrap_or_else(|e| {
            panic!("naga_oil compose failed for {name}: {e:?}");
        });
        let path = out_dir.join(format!("{name}.wgsl"));
        fs::write(&path, composed).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    }
}

fn compose_entry(wgsl_dir: &Path, entry_name: &str) -> Result<String, String> {
    let mut composer = Composer::default();

    // Only library modules: entry shaders (`normal_debug`, `uv_debug`) define `vs_main` / `fs_main`
    // and must not be registered as composable modules (would collide on entry point names).
    let library_modules = ["uniform_ring", "color_util", "ui_common"];
    for stem in library_modules {
        let path = wgsl_dir.join(format!("{stem}.wgsl"));
        if !path.exists() {
            continue;
        }
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

    let entry_path = wgsl_dir.join(format!("{entry_name}.wgsl"));
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
