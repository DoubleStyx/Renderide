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
    "build_info",
    "blackboard_contract",
    "bounds",
    "camera",
    "color_space",
    "concurrency",
    "config",
    "cpu_parallelism",
    "crash_context",
    "cull_contract",
    "diagnostics",
    "embedded_shaders",
    "frame_contract",
    "frame_upload_batch",
    "frontend",
    "graph_inputs",
    "gpu",
    "gpu_jobs",
    "gpu_pools",
    "gpu_resource",
    "hi_z_cpu",
    "hi_z_temporal",
    "history_texture",
    "hud_contract",
    "ipc",
    "log_throttle",
    "materials",
    "mesh_deform",
    "occlusion",
    "particles",
    "passes",
    "process_io",
    "profiling",
    "reflection_probes",
    "render_contract",
    "render_graph",
    "render_phase",
    "run_error",
    "runtime",
    "scene",
    "shared",
    "skybox",
    "upload_arena",
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
fn root_module_list_matches_lib_declarations() -> Result<(), String> {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let expected = ROOT_MODULES.iter().copied().collect::<BTreeSet<_>>();
    let declared = declared_root_modules(&src)?;

    let missing = declared
        .iter()
        .filter(|module| !expected.contains(module.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let stale = expected
        .iter()
        .filter(|module| !declared.contains(**module))
        .copied()
        .collect::<Vec<_>>();

    if !missing.is_empty() || !stale.is_empty() {
        return Err(format!(
            "architecture root-module list is out of sync with src/lib.rs\nmissing from ROOT_MODULES: {}\nstale in ROOT_MODULES: {}",
            missing.join(", "),
            stale.join(", ")
        ));
    }

    Ok(())
}

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
fn root_module_dependency_graph_is_acyclic() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let root_modules = ROOT_MODULES.iter().copied().collect::<BTreeSet<_>>();
    let edges = collect_production_crate_edges(&src, &root_modules);
    let cycles = root_module_cycles(&edges);

    assert!(
        cycles.is_empty(),
        "renderer root module dependency cycle(s):\n{}",
        format_cycle_reports(&cycles, &edges)
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
    collect_crate_edges_with_mode(src, root_modules, EdgeScanMode::AllCode)
}

fn collect_production_crate_edges(
    src: &Path,
    root_modules: &BTreeSet<&'static str>,
) -> BTreeMap<(&'static str, &'static str), BTreeSet<String>> {
    collect_crate_edges_with_mode(src, root_modules, EdgeScanMode::ProductionCode)
}

#[derive(Clone, Copy)]
enum EdgeScanMode {
    AllCode,
    ProductionCode,
}

fn collect_crate_edges_with_mode(
    src: &Path,
    root_modules: &BTreeSet<&'static str>,
    mode: EdgeScanMode,
) -> BTreeMap<(&'static str, &'static str), BTreeSet<String>> {
    let mut edges = BTreeMap::new();
    for file in rust_files(src) {
        if matches!(mode, EdgeScanMode::ProductionCode) && is_test_source_file(src, &file) {
            continue;
        }
        let Ok(source) = fs::read_to_string(&file) else {
            continue;
        };
        let Some(from) = root_module_for(src, &file, root_modules) else {
            continue;
        };
        let stripped = strip_comments(&source);
        let searchable = match mode {
            EdgeScanMode::AllCode => stripped,
            EdgeScanMode::ProductionCode => strip_cfg_test_modules(&stripped),
        };
        for target in crate_path_targets_in_stripped_source(&searchable, root_modules) {
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

fn declared_root_modules(src: &Path) -> Result<BTreeSet<String>, String> {
    let lib = src.join("lib.rs");
    let source =
        fs::read_to_string(&lib).map_err(|err| format!("read {}: {err}", lib.display()))?;
    Ok(strip_comments(&source)
        .lines()
        .filter_map(root_module_declared_on_line)
        .collect())
}

fn root_module_declared_on_line(line: &str) -> Option<String> {
    let line = line.trim_start();
    let rest = line
        .strip_prefix("mod ")
        .or_else(|| line.strip_prefix("pub mod "))
        .or_else(|| line.strip_prefix("pub(crate) mod "))?;
    let module = rest
        .chars()
        .take_while(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
        .collect::<String>();
    (!module.is_empty()).then_some(module)
}

fn is_test_source_file(src: &Path, file: &Path) -> bool {
    let relative = file.strip_prefix(src).unwrap_or(file);
    relative.components().any(|component| {
        let name = component.as_os_str().to_string_lossy();
        name == "tests" || name == "tests.rs" || name.ends_with("_tests.rs")
    })
}

fn strip_cfg_test_modules(source: &str) -> String {
    const CFG_TEST: &str = "#[cfg(test)]";

    let mut stripped = String::with_capacity(source.len());
    let mut cursor = 0;
    while let Some(relative_attr_start) = source[cursor..].find(CFG_TEST) {
        let attr_start = cursor + relative_attr_start;
        stripped.push_str(&source[cursor..attr_start]);

        let mut item_start = attr_start + CFG_TEST.len();
        item_start = skip_ascii_whitespace(source, item_start);
        let Some(after_mod) = source[item_start..].strip_prefix("mod ") else {
            stripped.push_str(&source[attr_start..item_start]);
            cursor = item_start;
            continue;
        };

        let ident_start = item_start + "mod ".len();
        let ident_end = skip_identifier(source, ident_start);
        if ident_end == ident_start {
            stripped.push_str(&source[attr_start..item_start + after_mod.len()]);
            cursor = item_start + after_mod.len();
            continue;
        }

        let item_body_start = skip_ascii_whitespace(source, ident_end);
        match source.as_bytes().get(item_body_start).copied() {
            Some(b';') => {
                cursor = item_body_start + 1;
            }
            Some(b'{') => {
                let Some(item_end) = matching_brace_end(source, item_body_start) else {
                    stripped.push_str(&source[attr_start..]);
                    cursor = source.len();
                    continue;
                };
                cursor = item_end + 1;
            }
            _ => {
                stripped.push_str(&source[attr_start..item_body_start]);
                cursor = item_body_start;
            }
        }
    }
    stripped.push_str(&source[cursor..]);
    stripped
}

fn skip_ascii_whitespace(source: &str, mut index: usize) -> usize {
    while source
        .as_bytes()
        .get(index)
        .is_some_and(u8::is_ascii_whitespace)
    {
        index += 1;
    }
    index
}

fn skip_identifier(source: &str, mut index: usize) -> usize {
    while source
        .as_bytes()
        .get(index)
        .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
    {
        index += 1;
    }
    index
}

fn matching_brace_end(source: &str, open_index: usize) -> Option<usize> {
    let bytes = source.as_bytes();
    let mut depth = 0usize;
    let mut index = open_index;
    while index < bytes.len() {
        if let Some(raw_end) = raw_string_end(source, index) {
            index = raw_end;
            continue;
        }

        match bytes[index] {
            b'{' => {
                depth += 1;
                index += 1;
            }
            b'}' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(index);
                }
                index += 1;
            }
            b'"' => {
                index = string_literal_end(source, index)?;
            }
            _ => {
                index += 1;
            }
        }
    }
    None
}

fn raw_string_end(source: &str, start: usize) -> Option<usize> {
    let bytes = source.as_bytes();
    if bytes.get(start).copied()? != b'r' {
        return None;
    }

    let mut hashes = 0usize;
    let mut quote = start + 1;
    while bytes.get(quote).copied() == Some(b'#') {
        hashes += 1;
        quote += 1;
    }
    if bytes.get(quote).copied() != Some(b'"') {
        return None;
    }

    let mut index = quote + 1;
    while index < bytes.len() {
        if bytes[index] == b'"'
            && (0..hashes).all(|offset| bytes.get(index + 1 + offset).copied() == Some(b'#'))
        {
            return Some(index + 1 + hashes);
        }
        index += 1;
    }
    None
}

fn string_literal_end(source: &str, quote_index: usize) -> Option<usize> {
    let bytes = source.as_bytes();
    let mut index = quote_index + 1;
    while index < bytes.len() {
        match bytes[index] {
            b'\\' => {
                index += 2;
            }
            b'"' => {
                return Some(index + 1);
            }
            _ => {
                index += 1;
            }
        }
    }
    None
}

fn root_module_cycles(
    edges: &BTreeMap<(&'static str, &'static str), BTreeSet<String>>,
) -> Vec<Vec<&'static str>> {
    let mut adjacency = BTreeMap::<&'static str, Vec<&'static str>>::new();
    for &(from, to) in edges.keys() {
        adjacency.entry(from).or_default().push(to);
        adjacency.entry(to).or_default();
    }

    let mut state = SccState::new(&adjacency);
    for node in adjacency.keys().copied() {
        if !state.indices.contains_key(node) {
            state.strong_connect(node);
        }
    }
    state.components.sort();
    state.components
}

struct SccState<'a> {
    adjacency: &'a BTreeMap<&'static str, Vec<&'static str>>,
    next_index: usize,
    stack: Vec<&'static str>,
    on_stack: BTreeSet<&'static str>,
    indices: BTreeMap<&'static str, usize>,
    lowlinks: BTreeMap<&'static str, usize>,
    components: Vec<Vec<&'static str>>,
}

impl<'a> SccState<'a> {
    fn new(adjacency: &'a BTreeMap<&'static str, Vec<&'static str>>) -> Self {
        Self {
            adjacency,
            next_index: 0,
            stack: Vec::new(),
            on_stack: BTreeSet::new(),
            indices: BTreeMap::new(),
            lowlinks: BTreeMap::new(),
            components: Vec::new(),
        }
    }

    fn strong_connect(&mut self, node: &'static str) {
        let index = self.next_index;
        self.next_index += 1;
        self.indices.insert(node, index);
        self.lowlinks.insert(node, index);
        self.stack.push(node);
        self.on_stack.insert(node);

        let neighbors = self.adjacency.get(node).cloned().unwrap_or_default();
        for target in neighbors {
            if !self.indices.contains_key(target) {
                self.strong_connect(target);
                let target_lowlink = self.lowlinks[&target];
                if let Some(node_lowlink) = self.lowlinks.get_mut(node) {
                    *node_lowlink = (*node_lowlink).min(target_lowlink);
                }
            } else if self.on_stack.contains(target) {
                let target_index = self.indices[&target];
                if let Some(node_lowlink) = self.lowlinks.get_mut(node) {
                    *node_lowlink = (*node_lowlink).min(target_index);
                }
            }
        }

        if self.lowlinks[&node] == self.indices[&node] {
            let mut component = Vec::new();
            while let Some(member) = self.stack.pop() {
                self.on_stack.remove(member);
                component.push(member);
                if member == node {
                    break;
                }
            }
            if component.len() > 1 {
                component.sort_unstable();
                self.components.push(component);
            }
        }
    }
}

fn format_cycle_reports(
    cycles: &[Vec<&'static str>],
    edges: &BTreeMap<(&'static str, &'static str), BTreeSet<String>>,
) -> String {
    cycles
        .iter()
        .map(|cycle| {
            let members = cycle.iter().copied().collect::<BTreeSet<_>>();
            let cycle_edges = edges
                .iter()
                .filter(|((from, to), _)| members.contains(from) && members.contains(to))
                .map(|((from, to), files)| {
                    format!(
                        "  {from} -> {to} in {}",
                        files.iter().cloned().collect::<Vec<_>>().join(", ")
                    )
                })
                .collect::<Vec<_>>();
            format!("{}\n{}", cycle.join(" -> "), cycle_edges.join("\n"))
        })
        .collect::<Vec<_>>()
        .join("\n\n")
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

fn crate_path_targets_in_stripped_source(
    source: &str,
    root_modules: &BTreeSet<&'static str>,
) -> BTreeSet<&'static str> {
    let mut targets = BTreeSet::new();
    for segment in source.split("crate::").skip(1) {
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
