//! Import-graph guards for the renderer's intended module layering.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Edge {
    from: &'static str,
    to: &'static str,
}

const ROOT_MODULES: &[&str] = &[
    "app",
    "assets",
    "backend",
    "camera",
    "color_space",
    "concurrency",
    "config",
    "diagnostics",
    "frontend",
    "graph_inputs",
    "gpu",
    "gpu_pools",
    "gpu_resource",
    "ipc",
    "materials",
    "mesh_deform",
    "occlusion",
    "passes",
    "process_io",
    "profiling",
    "reflection_probes",
    "render_graph",
    "run_error",
    "runtime",
    "scene",
    "shared",
    "skybox",
    "world_mesh",
    "xr",
];

const FORBIDDEN_EDGES: &[Edge] = &[
    Edge {
        from: "assets",
        to: "backend",
    },
    Edge {
        from: "frontend",
        to: "assets",
    },
    Edge {
        from: "frontend",
        to: "graph_inputs",
    },
    Edge {
        from: "frontend",
        to: "gpu",
    },
    Edge {
        from: "graph_inputs",
        to: "backend",
    },
    Edge {
        from: "graph_inputs",
        to: "passes",
    },
    Edge {
        from: "frontend",
        to: "xr",
    },
    Edge {
        from: "gpu",
        to: "xr",
    },
    Edge {
        from: "materials",
        to: "backend",
    },
    Edge {
        from: "passes",
        to: "backend",
    },
    Edge {
        from: "render_graph",
        to: "backend",
    },
    Edge {
        from: "render_graph",
        to: "passes",
    },
    Edge {
        from: "scene",
        to: "backend",
    },
    Edge {
        from: "world_mesh",
        to: "render_graph",
    },
];

const REFACTORED_MODULE_FILES: &[&str] = &[
    "assets/mesh/gpu_mesh/upload.rs",
    "assets/mesh/gpu_mesh/upload/derived_streams.rs",
    "assets/mesh/gpu_mesh/upload/generated.rs",
    "backend/facade/graph_access.rs",
    "backend/facade/graph_access/warmup.rs",
    "backend/asset_transfers/reliable_ack.rs",
    "config/persist/load/previous_layout.rs",
    "passes/world_mesh_forward/prepare.rs",
    "passes/world_mesh_forward/prepare/cache.rs",
    "render_graph/compiled/exec/recording/per_view/batch_plan.rs",
    "render_graph/compiled/exec/recording/per_view/frame_params.rs",
    "render_graph/compiled/exec/recording/per_view/offscreen_copy.rs",
    "render_graph/compiled/exec.rs",
    "render_graph/compiled/exec/command_recording.rs",
    "render_graph/compiled/exec/prepare.rs",
    "render_graph/compiled/exec/recording_path.rs",
    "render_graph/compiled/exec/swapchain.rs",
    "render_graph/compiled/frame_view.rs",
    "render_graph/compiled/frame_view/profile.rs",
    "render_graph/schedule.rs",
    "render_graph/schedule/hud.rs",
    "render_graph/schedule/tests.rs",
    "runtime/frame/extract.rs",
    "runtime/frame/extract/cull.rs",
    "runtime/frame/extract/queue.rs",
    "runtime/frame/extract/sort.rs",
    "runtime/frame/extract/visible_deform.rs",
    "world_mesh/draw_prep/prepared_renderables.rs",
    "world_mesh/draw_prep/prepared_renderables/lod.rs",
    "world_mesh/draw_prep/prepared_renderables/tests.rs",
    "world_mesh/instances.rs",
    "world_mesh/instances/builder.rs",
    "world_mesh/instances/prepass.rs",
];

#[test]
fn layer_boundary_edges_do_not_regress() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let root_modules = ROOT_MODULES.iter().copied().collect::<BTreeSet<_>>();
    let edges = collect_crate_edges(&src, &root_modules);

    let violations = FORBIDDEN_EDGES
        .iter()
        .filter_map(|forbidden| {
            let files = edges.get(&(forbidden.from, forbidden.to))?;
            Some(format!(
                "{} -> {} in {}",
                forbidden.from,
                forbidden.to,
                files.iter().cloned().collect::<Vec<_>>().join(", ")
            ))
        })
        .collect::<Vec<_>>();

    assert!(
        violations.is_empty(),
        "forbidden renderide layer import(s):\n{}",
        violations.join("\n")
    );
}

#[test]
fn refactored_renderer_modules_stay_split() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let missing = REFACTORED_MODULE_FILES
        .iter()
        .filter(|relative| !src.join(relative).exists())
        .copied()
        .collect::<Vec<_>>();

    assert!(
        missing.is_empty(),
        "refactored renderer module file(s) are missing:\n{}",
        missing.join("\n")
    );
}

#[test]
fn renderer_source_modules_stay_under_line_limit() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let oversized = rust_files(&src)
        .into_iter()
        .filter_map(|path| {
            let source = fs::read_to_string(&path).ok()?;
            let line_count = source.lines().count();
            (line_count > 1_000)
                .then(|| format!("{}: {line_count} lines", relative_path(&src, &path)))
        })
        .collect::<Vec<_>>();

    assert!(
        oversized.is_empty(),
        "renderer source module(s) exceeded the 1,000-line limit:\n{}",
        oversized.join("\n")
    );
}

#[test]
fn main_graph_assembly_lives_in_backend() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    assert!(
        !src.join("render_graph/main_graph.rs").exists(),
        "concrete main graph assembly belongs under backend/graph, not render_graph core"
    );
    assert!(
        src.join("backend/graph/main_graph.rs").exists(),
        "backend-owned main graph assembly module is missing"
    );
}

#[test]
fn pass_context_does_not_import_or_expose_graph_pass_frame() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let pass_context = src.join("render_graph/context/pass.rs");
    let source = fs::read_to_string(&pass_context).expect("read pass context");
    let stripped = strip_comments(&source);

    assert!(
        !contains_identifier(&stripped, "GraphPassFrame"),
        "render graph pass contexts must not import or expose GraphPassFrame"
    );
}

fn collect_crate_edges(
    src: &Path,
    root_modules: &BTreeSet<&'static str>,
) -> BTreeMap<(&'static str, &'static str), BTreeSet<String>> {
    let mut edges = BTreeMap::new();
    for file in rust_files(src) {
        let Ok(source) = fs::read_to_string(&file) else {
            continue;
        };
        let Some(from) = root_module_for(src, &file, root_modules) else {
            continue;
        };
        for target in crate_path_targets(&source, root_modules) {
            if target == from {
                continue;
            }
            edges
                .entry((from, target))
                .or_insert_with(BTreeSet::new)
                .insert(relative_path(src, &file));
        }
    }
    edges
}

fn rust_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(path) = stack.pop() {
        let Ok(entries) = fs::read_dir(&path) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                out.push(path);
            }
        }
    }
    out
}

fn root_module_for(
    src: &Path,
    file: &Path,
    root_modules: &BTreeSet<&'static str>,
) -> Option<&'static str> {
    let relative = file.strip_prefix(src).ok()?;
    let first = relative.components().next()?.as_os_str().to_str()?;
    let module_name = first.strip_suffix(".rs").unwrap_or(first);
    root_modules.get(module_name).copied()
}

fn crate_path_targets(
    source: &str,
    root_modules: &BTreeSet<&'static str>,
) -> BTreeSet<&'static str> {
    let stripped = strip_comments(source);
    let mut targets = BTreeSet::new();
    for segment in stripped.split("crate::").skip(1) {
        let module = segment
            .chars()
            .take_while(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
            .collect::<String>();
        if let Some(module) = root_modules.get(module.as_str()) {
            targets.insert(*module);
        }
    }
    targets
}

fn strip_comments(source: &str) -> String {
    let mut stripped = String::with_capacity(source.len());
    let mut chars = source.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '/' && chars.peek() == Some(&'/') {
            for next in chars.by_ref() {
                if next == '\n' {
                    stripped.push('\n');
                    break;
                }
            }
        } else if ch == '/' && chars.peek() == Some(&'*') {
            chars.next();
            let mut previous = '\0';
            for next in chars.by_ref() {
                if previous == '*' && next == '/' {
                    break;
                }
                previous = next;
            }
        } else {
            stripped.push(ch);
        }
    }
    stripped
}

fn contains_identifier(source: &str, ident: &str) -> bool {
    source
        .split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
        .any(|token| token == ident)
}

fn relative_path(src: &Path, file: &Path) -> String {
    file.strip_prefix(src)
        .unwrap_or(file)
        .to_string_lossy()
        .replace('\\', "/")
}
