//! Build-script entry point for shader composition and embedded WGSL registry generation.
//!
//! Source layout:
//! - `shaders/modules/*.wgsl` provides naga-oil composable modules.
//! - `shaders/materials/*.wgsl` provides host-routed material shaders.
//! - `shaders/passes/{post,backend,compute,present}/*.wgsl` provides non-material pass shaders.
//! - `shaders/target/*.wgsl` is generated inspection output and is not consumed by Rust code.

#[path = "shader/compose.rs"]
mod compose;
#[path = "shader/directives.rs"]
mod directives;
#[path = "shader/emit.rs"]
mod emit;
#[path = "shader/error.rs"]
mod error;
#[path = "shader/mirror_once.rs"]
mod mirror_once;
#[path = "shader/model.rs"]
mod model;
#[path = "shader/modules.rs"]
mod modules;
#[path = "shader/parallel.rs"]
mod parallel;
#[path = "shader/source.rs"]
mod source;
#[path = "shader/validation.rs"]
mod validation;

use std::fs;
use std::path::Path;

pub use error::BuildError;

use emit::{ComposedShaders, clean_target_dir, emit_compiled_shader, render_embedded_shaders_rs};
use modules::discover_shader_modules;
use parallel::compile_shader_jobs;
use source::discover_shader_jobs;

/// Reads a required build-script environment variable.
pub fn env_var(name: &'static str) -> Result<String, BuildError> {
    error::env_var(name)
}

/// Composes all shader sources into flattened WGSL targets and generated Rust metadata.
fn compose_all_shaders(
    manifest_dir: &Path,
    shader_root: &Path,
    target_dir: &Path,
) -> Result<ComposedShaders, BuildError> {
    let shader_modules = discover_shader_modules(manifest_dir)?;
    let jobs = discover_shader_jobs(shader_root)?;
    let compiled = compile_shader_jobs(&shader_modules, &jobs)?;
    let mut out = ComposedShaders::new();
    clean_target_dir(target_dir)?;
    for compiled_shader in &compiled {
        emit_compiled_shader(compiled_shader, target_dir, &mut out)?;
    }
    Ok(out)
}

/// Composes WGSL, writes `shaders/target/*.wgsl`, and writes `OUT_DIR/embedded_shaders.rs`.
pub fn compile(manifest_dir: &Path, out_dir: &Path) -> Result<(), BuildError> {
    let shader_root = manifest_dir.join("shaders");
    let target_dir = shader_root.join("target");

    println!("cargo:rerun-if-changed=shaders/modules");
    println!("cargo:rerun-if-changed=shaders/materials");
    println!("cargo:rerun-if-changed=shaders/passes");

    fs::create_dir_all(&shader_root)?;
    fs::create_dir_all(&target_dir)?;

    let composed = compose_all_shaders(manifest_dir, &shader_root, &target_dir)?;
    let embedded_rs = render_embedded_shaders_rs(&composed);

    let gen_path = out_dir.join("embedded_shaders.rs");
    fs::write(&gen_path, embedded_rs)?;
    Ok(())
}
