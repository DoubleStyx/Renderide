//! Naga-oil composable module discovery and registration.

use std::fs;
use std::path::{Path, PathBuf};

use hashbrown::HashMap;
use naga_oil::compose::{ComposableModuleDescriptor, Composer, ShaderLanguage};

use super::error::BuildError;

/// Source text for a composable WGSL module, keyed by naga-oil file path.
pub(super) type ShaderModuleSources = Vec<(String, String)>;

/// Loads every `*.wgsl` under `shaders/modules/` relative to `manifest_dir`.
///
/// Modules are returned in dependency order: each module's `#import` targets appear before it.
pub(super) fn discover_shader_modules(
    manifest_dir: &Path,
) -> Result<ShaderModuleSources, BuildError> {
    let modules_dir = manifest_dir.join("shaders/modules");
    let mut paths = Vec::new();
    collect_wgsl_paths(&modules_dir, &mut paths)?;
    paths.sort();

    let mut modules = Vec::with_capacity(paths.len());
    for path in paths {
        let source = fs::read_to_string(&path)
            .map_err(|e| BuildError::Message(format!("read {}: {e}", path.display())))?;
        let rel = path.strip_prefix(manifest_dir).map_err(|e| {
            BuildError::Message(format!(
                "module path {} is not under manifest {}: {e}",
                path.display(),
                manifest_dir.display()
            ))
        })?;
        let file_path = rel.to_string_lossy().replace('\\', "/");
        modules.push((file_path, source));
    }

    if modules.is_empty() {
        return Err(BuildError::Message(format!(
            "no *.wgsl modules under {} (naga-oil imports will fail)",
            modules_dir.display()
        )));
    }

    topo_sort_shader_modules(&modules)
}

/// Recursively collects WGSL module files under `dir`.
fn collect_wgsl_paths(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), BuildError> {
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
            collect_wgsl_paths(&path, out)?;
        } else if file_type.is_file() && path.extension().is_some_and(|x| x == "wgsl") {
            out.push(path);
        }
    }
    Ok(())
}

/// Topologically sorts shader modules so each `#import` target is registered before its importer.
fn topo_sort_shader_modules(
    modules: &[(String, String)],
) -> Result<ShaderModuleSources, BuildError> {
    let mut path_to_idx: HashMap<String, usize> = HashMap::default();
    let mut imports_per_module: Vec<Vec<String>> = Vec::with_capacity(modules.len());
    for (i, (file_path, source)) in modules.iter().enumerate() {
        let define = parse_define_import_path(source).ok_or_else(|| {
            BuildError::Message(format!(
                "module {file_path} has no `#define_import_path` directive",
            ))
        })?;
        if let Some(prev) = path_to_idx.insert(define.clone(), i) {
            return Err(BuildError::Message(format!(
                "duplicate `#define_import_path {define}` in {file_path} and {}",
                modules[prev].0,
            )));
        }
        imports_per_module.push(parse_import_paths(source));
    }

    let mut in_degree = vec![0usize; modules.len()];
    let mut children_of: Vec<Vec<usize>> = vec![Vec::new(); modules.len()];
    for (i, imports) in imports_per_module.iter().enumerate() {
        for import_path in imports {
            if let Some(&j) = path_to_idx.get(import_path) {
                if i == j {
                    continue;
                }
                children_of[j].push(i);
                in_degree[i] += 1;
            }
        }
    }

    let mut ready: Vec<usize> = (0..modules.len()).filter(|&i| in_degree[i] == 0).collect();
    let mut sorted = Vec::with_capacity(modules.len());
    while let Some(idx) = ready.first().copied() {
        ready.remove(0);
        sorted.push(idx);
        for &child in &children_of[idx] {
            in_degree[child] -= 1;
            if in_degree[child] == 0 {
                let pos = ready
                    .binary_search_by(|&j| modules[j].0.cmp(&modules[child].0))
                    .unwrap_or_else(|e| e);
                ready.insert(pos, child);
            }
        }
    }
    if sorted.len() != modules.len() {
        let unresolved: Vec<&str> = (0..modules.len())
            .filter(|i| !sorted.contains(i))
            .map(|i| modules[i].0.as_str())
            .collect();
        return Err(BuildError::Message(format!(
            "shader-module import graph has a cycle; unresolved: {unresolved:?}",
        )));
    }

    Ok(sorted.into_iter().map(|i| modules[i].clone()).collect())
}

/// Parses the first `#define_import_path <path>` directive from WGSL source.
fn parse_define_import_path(source: &str) -> Option<String> {
    for line in source.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("#define_import_path") {
            return Some(rest.trim().to_string());
        }
    }
    None
}

/// Parses every `#import <path>` from WGSL source.
fn parse_import_paths(source: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in source.lines() {
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix("#import") else {
            continue;
        };
        let rest = rest.trim();
        let path = rest
            .split_whitespace()
            .next()
            .map(|p| p.trim_end_matches('{').to_string());
        if let Some(p) = path {
            let p = p.split("::{").next().unwrap_or(&p).to_string();
            if !p.is_empty() {
                out.push(p);
            }
        }
    }
    out
}

/// Registers all composable modules on a naga-oil composer.
pub(super) fn register_composable_modules(
    composer: &mut Composer,
    modules: &[(String, String)],
) -> Result<(), BuildError> {
    for (file_path, source) in modules {
        composer
            .add_composable_module(ComposableModuleDescriptor {
                source: source.as_str(),
                file_path: file_path.as_str(),
                language: ShaderLanguage::Wgsl,
                ..Default::default()
            })
            .map_err(|e| BuildError::Message(format!("add composable module {file_path}: {e}")))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Import topo sort places dependencies before importers.
    #[test]
    fn topo_sort_orders_dependencies_first() -> Result<(), BuildError> {
        let modules = vec![
            (
                "shaders/modules/b.wgsl".to_string(),
                "#define_import_path renderide::b\n#import renderide::a as a\n".to_string(),
            ),
            (
                "shaders/modules/a.wgsl".to_string(),
                "#define_import_path renderide::a\n".to_string(),
            ),
        ];

        let sorted = topo_sort_shader_modules(&modules)?;
        let paths = sorted
            .iter()
            .map(|(path, _)| path.as_str())
            .collect::<Vec<_>>();
        assert_eq!(paths, ["shaders/modules/a.wgsl", "shaders/modules/b.wgsl"]);
        Ok(())
    }

    /// Import topo sort still handles modules stored in nested filesystem paths.
    #[test]
    fn topo_sort_accepts_nested_module_paths() -> Result<(), BuildError> {
        let modules = vec![
            (
                "shaders/modules/pbs/lighting.wgsl".to_string(),
                "#define_import_path renderide::pbs::lighting\n#import renderide::pbs::surface as surface\n"
                    .to_string(),
            ),
            (
                "shaders/modules/pbs/surface.wgsl".to_string(),
                "#define_import_path renderide::pbs::surface\n".to_string(),
            ),
        ];

        let sorted = topo_sort_shader_modules(&modules)?;
        let paths = sorted
            .iter()
            .map(|(path, _)| path.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            paths,
            [
                "shaders/modules/pbs/surface.wgsl",
                "shaders/modules/pbs/lighting.wgsl"
            ]
        );
        Ok(())
    }
}
