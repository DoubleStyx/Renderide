//! Source audit for GPU timestamp instrumentation coverage.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Files whose copy commands are covered by a caller-level GPU profiler scope.
const PARENT_SCOPED_COPY_FILES: &[&str] = &[
    "src/app/headless/readback.rs",
    "src/backend/frame_gpu/scene_snapshot.rs",
    "src/gpu/profiling/frame_bracket.rs",
    "src/occlusion/gpu/encode/staging_copy.rs",
    "src/render_graph/frame_upload_batch/batch.rs",
    "src/runtime/offscreen_tasks/reflection_probe/readback.rs",
];

/// GPU copy/clear commands that must be timestamped or explicitly parent-scoped.
const COPY_OR_CLEAR_COMMANDS: &[&str] = &[
    ".copy_buffer_to_buffer(",
    ".copy_texture_to_buffer(",
    ".copy_texture_to_texture(",
    ".clear_buffer(",
];

/// GPU pass constructors that must use descriptor timestamp writes.
const PASS_CONSTRUCTORS: &[&str] = &[".begin_render_pass(", ".begin_compute_pass("];

/// Returns the renderide crate directory.
fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Recursively returns all Rust source files below `src`.
fn rust_sources() -> io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    collect_rust_sources(&manifest_dir().join("src"), &mut out)?;
    out.sort();
    Ok(out)
}

/// Recursively appends Rust source files under `dir`.
fn collect_rust_sources(dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_rust_sources(&path, out)?;
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
    Ok(())
}

/// Returns a normalized crate-relative file label.
fn file_label(path: &Path) -> String {
    path.strip_prefix(manifest_dir())
        .unwrap_or(path)
        .display()
        .to_string()
        .replace('\\', "/")
}

/// Returns a local source window around `line_idx`.
fn source_window(lines: &[&str], line_idx: usize, before: usize, after: usize) -> String {
    let start = line_idx.saturating_sub(before);
    let end = line_idx
        .saturating_add(after)
        .saturating_add(1)
        .min(lines.len());
    lines[start..end].join("\n")
}

/// Returns whether a source window includes timestamp writes for a GPU pass descriptor.
fn pass_window_has_timestamp_writes(window: &str) -> bool {
    window.contains("timestamp_writes")
}

/// Returns whether a source window includes a GPU-profiler encoder scope.
fn copy_window_has_gpu_scope(window: &str) -> bool {
    window.contains("begin_query(") || window.contains("GpuEncoderScope::begin(")
}

/// Returns whether this file's copy commands are covered by a broader query scope.
fn copy_file_parent_scoped(label: &str) -> bool {
    PARENT_SCOPED_COPY_FILES.contains(&label)
}

#[test]
fn gpu_passes_have_timestamp_writes() -> io::Result<()> {
    let mut offenders = Vec::new();
    for path in rust_sources()? {
        let label = file_label(&path);
        let src = fs::read_to_string(&path)?;
        let lines: Vec<&str> = src.lines().collect();
        for (idx, line) in lines.iter().enumerate() {
            if PASS_CONSTRUCTORS.iter().any(|needle| line.contains(needle)) {
                let window = source_window(&lines, idx, 8, 16);
                if !pass_window_has_timestamp_writes(&window) {
                    offenders.push(format!("{}:{}", label, idx + 1));
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "GPU render/compute passes must thread wgpu-profiler timestamp_writes or be explicitly redesigned:\n  - {}",
        offenders.join("\n  - ")
    );
    Ok(())
}

#[test]
fn gpu_copies_and_clears_have_profiler_scopes() -> io::Result<()> {
    let mut offenders = Vec::new();
    for path in rust_sources()? {
        let label = file_label(&path);
        let src = fs::read_to_string(&path)?;
        let lines: Vec<&str> = src.lines().collect();
        for (idx, line) in lines.iter().enumerate() {
            if COPY_OR_CLEAR_COMMANDS
                .iter()
                .any(|needle| line.contains(needle))
            {
                let window = source_window(&lines, idx, 48, 8);
                if !copy_window_has_gpu_scope(&window) && !copy_file_parent_scoped(&label) {
                    offenders.push(format!("{}:{}", label, idx + 1));
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "GPU copies and clears must use `GpuEncoderScope`/`begin_query` or be listed in PARENT_SCOPED_COPY_FILES with a real reason:\n  - {}",
        offenders.join("\n  - ")
    );
    Ok(())
}
