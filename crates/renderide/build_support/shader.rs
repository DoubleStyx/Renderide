//! Build-script entry point for shader composition and runtime shader package generation.
//!
//! Source layout:
//! - `shaders/modules/*.wgsl` provides naga-oil composable modules.
//! - `shaders/materials/*.wgsl` provides host-routed material shaders.
//! - `shaders/passes/{post,backend,compute,present}/*.wgsl` provides non-material pass shaders.
//! - `target/<profile>/shaders/*.wgsl` is generated runtime package output.

#[path = "shader/compose.rs"]
mod compose;
#[path = "shader/directives.rs"]
mod directives;
#[path = "shader/emit.rs"]
mod emit;
#[path = "shader/error.rs"]
mod error;
#[path = "shader/manifest.rs"]
mod manifest;
#[path = "shader/mirror_once.rs"]
mod mirror_once;
#[path = "shader/model.rs"]
mod model;
#[path = "shader/modules.rs"]
mod modules;
#[path = "shader/parallel.rs"]
mod parallel;
#[path = "../src/shader_package/schema.rs"]
mod shader_package_schema;
#[path = "shader/source.rs"]
mod source;
#[path = "shader/validation.rs"]
mod validation;

use std::fs;
use std::path::{Path, PathBuf};

pub use error::BuildError;

use emit::{ComposedShaders, clean_target_dir, emit_compiled_shader, render_embedded_shaders_rs};
use manifest::write_shader_package_manifest;
use modules::discover_shader_modules;
use parallel::compile_shader_jobs;
use source::discover_shader_jobs;

const SHADER_INPUT_STAMP_FILE: &str = ".shader-inputs-fnv";

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
    write_shader_package_manifest(&compiled, target_dir)?;
    Ok(out)
}

fn artifact_shader_package_dir(out_dir: &Path) -> Result<PathBuf, BuildError> {
    out_dir
        .ancestors()
        .nth(3)
        .map(|artifact_dir| artifact_dir.join("shaders"))
        .ok_or_else(|| {
            BuildError::Message(format!(
                "cannot derive shader artifact directory from OUT_DIR {}",
                out_dir.display()
            ))
        })
}

fn collect_shader_input_files(root: &Path) -> Result<Vec<PathBuf>, BuildError> {
    let mut files = Vec::new();
    for relative in ["modules", "materials", "passes"] {
        collect_shader_input_files_recursive(&root.join(relative), &mut files)?;
    }
    files.sort();
    Ok(files)
}

fn collect_shader_input_files_recursive(
    dir: &Path,
    out: &mut Vec<PathBuf>,
) -> Result<(), BuildError> {
    if !dir.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(dir)
        .map_err(|e| BuildError::Message(format!("read {}: {e}", dir.display())))?
    {
        let entry =
            entry.map_err(|e| BuildError::Message(format!("read {} entry: {e}", dir.display())))?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|e| BuildError::Message(format!("stat {}: {e}", path.display())))?;
        if file_type.is_dir() {
            collect_shader_input_files_recursive(&path, out)?;
        } else if !file_type.is_dir() && path.extension().is_some_and(|ext| ext == "wgsl") {
            out.push(path);
        }
    }
    Ok(())
}

fn shader_input_fingerprint(shader_root: &Path) -> Result<String, BuildError> {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    fn update(hash: &mut u64, bytes: &[u8]) {
        for byte in bytes {
            *hash ^= u64::from(*byte);
            *hash = hash.wrapping_mul(FNV_PRIME);
        }
    }

    let mut hash = FNV_OFFSET;
    for path in collect_shader_input_files(shader_root)? {
        let relative = path.strip_prefix(shader_root).map_err(|e| {
            BuildError::Message(format!(
                "shader input {} is not under {}: {e}",
                path.display(),
                shader_root.display()
            ))
        })?;
        let relative = relative.to_string_lossy().replace('\\', "/");
        update(&mut hash, relative.as_bytes());
        update(&mut hash, b"\0");
        let contents = fs::read(&path)?;
        update(&mut hash, &contents);
        update(&mut hash, b"\0");
    }
    Ok(format!("{hash:016x}"))
}

fn package_is_complete(target_dir: &Path) -> Result<Option<ComposedShaders>, BuildError> {
    let manifest_path = target_dir.join(shader_package_schema::SHADER_PACKAGE_MANIFEST_FILE);
    if !manifest_path.is_file() {
        return Ok(None);
    }
    let manifest_text = fs::read_to_string(&manifest_path)?;
    let manifest: shader_package_schema::ShaderPackageManifest = toml::from_str(&manifest_text)
        .map_err(|e| {
            BuildError::Message(format!(
                "parse shader package manifest {}: {e}",
                manifest_path.display()
            ))
        })?;
    if manifest.version != shader_package_schema::SHADER_PACKAGE_MANIFEST_VERSION {
        return Ok(None);
    }
    let mut composed = ComposedShaders::new();
    for target in manifest.targets {
        let target_path = target_dir.join(&target.file);
        let Ok(wgsl) = fs::read_to_string(&target_path) else {
            return Ok(None);
        };
        if shader_package_schema::stable_source_hash(&wgsl) != target.wgsl_hash {
            return Ok(None);
        }
        composed.record_target_stem(target.class.into(), target.stem);
    }
    Ok(Some(composed))
}

fn cached_composed_shaders(
    shader_root: &Path,
    target_dir: &Path,
    embedded_rs_path: &Path,
) -> Result<Option<ComposedShaders>, BuildError> {
    let fingerprint = shader_input_fingerprint(shader_root)?;
    let stamp_path = target_dir.join(SHADER_INPUT_STAMP_FILE);
    if !embedded_rs_path.is_file() {
        return Ok(None);
    }
    if fs::read_to_string(&stamp_path)
        .ok()
        .is_none_or(|stamp| stamp.trim() != fingerprint)
    {
        return Ok(None);
    }
    package_is_complete(target_dir)
}

fn write_shader_input_stamp(shader_root: &Path, target_dir: &Path) -> Result<(), BuildError> {
    let fingerprint = shader_input_fingerprint(shader_root)?;
    fs::write(
        target_dir.join(SHADER_INPUT_STAMP_FILE),
        format!("{fingerprint}\n"),
    )?;
    Ok(())
}

impl From<shader_package_schema::ShaderTargetClass> for model::ShaderSourceClass {
    fn from(value: shader_package_schema::ShaderTargetClass) -> Self {
        match value {
            shader_package_schema::ShaderTargetClass::Material => Self::Material,
            shader_package_schema::ShaderTargetClass::Post => Self::Post,
            shader_package_schema::ShaderTargetClass::Backend => Self::Backend,
            shader_package_schema::ShaderTargetClass::Compute => Self::Compute,
            shader_package_schema::ShaderTargetClass::Present => Self::Present,
        }
    }
}

/// Composes WGSL, writes `target/<profile>/shaders/*.wgsl`, and writes `OUT_DIR/embedded_shaders.rs`.
pub fn compile(manifest_dir: &Path, out_dir: &Path) -> Result<(), BuildError> {
    let shader_root = manifest_dir.join("shaders");
    let target_dir = artifact_shader_package_dir(out_dir)?;
    println!(
        "cargo:rustc-env=RENDERIDE_SHADER_PACKAGE_DIR_DEFAULT={}",
        target_dir.display()
    );

    println!("cargo:rerun-if-changed=shaders/modules");
    println!("cargo:rerun-if-changed=shaders/materials");
    println!("cargo:rerun-if-changed=shaders/passes");

    fs::create_dir_all(&shader_root)?;
    fs::create_dir_all(&target_dir)?;

    let gen_path = out_dir.join("embedded_shaders.rs");
    let composed = match cached_composed_shaders(&shader_root, &target_dir, &gen_path)? {
        Some(cached) => cached,
        None => {
            let composed = compose_all_shaders(manifest_dir, &shader_root, &target_dir)?;
            write_shader_input_stamp(&shader_root, &target_dir)?;
            composed
        }
    };
    let embedded_rs = render_embedded_shaders_rs(&composed);
    fs::write(&gen_path, embedded_rs)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn shader_package_dir_from_out_dir_handles_debug_dev_fast_and_cross_target() {
        let cases = [
            (
                "/repo/target/debug/build/renderide-123/out",
                "/repo/target/debug/shaders",
            ),
            (
                "/repo/target/dev-fast/build/renderide-123/out",
                "/repo/target/dev-fast/shaders",
            ),
            (
                "/repo/target/x86_64-pc-windows-msvc/release/build/renderide-123/out",
                "/repo/target/x86_64-pc-windows-msvc/release/shaders",
            ),
        ];
        for (out_dir, expected) in cases {
            assert_eq!(
                artifact_shader_package_dir(Path::new(out_dir)).expect("artifact shader dir"),
                PathBuf::from(expected),
            );
        }
    }

    #[test]
    fn shader_input_fingerprint_changes_with_contents() -> Result<(), BuildError> {
        let temp = tempfile::tempdir()?;
        let material_dir = temp.path().join("materials");
        fs::create_dir_all(&material_dir)?;
        let path = material_dir.join("unlit.wgsl");
        fs::write(&path, "first")?;
        let first = shader_input_fingerprint(temp.path())?;

        fs::write(&path, "second")?;
        let second = shader_input_fingerprint(temp.path())?;

        assert_ne!(first, second);
        Ok(())
    }
}
