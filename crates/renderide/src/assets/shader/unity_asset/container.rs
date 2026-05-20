//! AssetBundle `m_Container` shader route extraction.

use std::path::Path;

use unity_asset::AssetBundle;
use unity_asset::class_ids::SHADER;
use unity_asset::environment::{BinarySource, Environment};

use super::log_container_resolution;

/// Shader stem from [`Environment::bundle_container_entries`] by matching Shader `path_id` to
/// `AssetBundle.m_Container`.
pub(super) fn shader_container_names_from_bundle(
    bundle_path: &Path,
    bundle: &AssetBundle,
) -> Vec<(i64, String)> {
    let mut env = Environment::new();
    let _ = env.load_file(bundle_path);
    let source = BinarySource::path(bundle_path);
    if env.bundles().get(&source).is_none() {
        logger::debug!(
            "shader_unity_asset: Environment has no bundle for {:?} (m_Container unavailable)",
            bundle_path.display()
        );
        return Vec::new();
    }
    let Ok(entries) = env.bundle_container_entries(bundle_path) else {
        return Vec::new();
    };
    if entries.is_empty() {
        logger::debug!(
            "shader_unity_asset: no m_Container entries for {:?}",
            bundle_path.display()
        );
        return Vec::new();
    }

    let shader_path_ids: Vec<i64> = bundle
        .assets
        .iter()
        .flat_map(|sf| {
            sf.object_handles()
                .filter(|h| h.class_id() == SHADER)
                .map(|h| h.path_id())
        })
        .collect();

    let mut names = Vec::new();
    for pid in shader_path_ids {
        if let Some(entry) = entries.iter().find(|e| e.path_id == pid)
            && let Some(name) = shader_asset_name_from_container_asset_path(&entry.asset_path)
        {
            log_container_resolution(pid, &name, &entry.asset_path);
            names.push((pid, name));
        }
    }
    names
}

/// Returns the resolved container route for `path_id`.
pub(super) fn container_name_for_path_id(
    container_names: &[(i64, String)],
    path_id: i64,
) -> Option<String> {
    container_names
        .iter()
        .find(|(container_path_id, _)| *container_path_id == path_id)
        .map(|(_, name)| name.clone())
}

/// Derives a lowercase shader asset name from a Unity `m_Container` asset path
/// (e.g. `.../UI_Unlit.shader` -> `ui_unlit`).
pub(super) fn shader_asset_name_from_container_asset_path(asset_path: &str) -> Option<String> {
    let p = asset_path.replace('\\', "/");
    let seg = p.rsplit('/').next()?.trim();
    if seg.is_empty() {
        return None;
    }
    let base = seg
        .strip_suffix(".shader")
        .unwrap_or(seg)
        .rsplit('/')
        .next()
        .unwrap_or(seg)
        .trim();
    if base.is_empty() {
        return None;
    }
    let shader_asset_name = base.to_ascii_lowercase();
    if shader_asset_name.starts_with("cab-") {
        return None;
    }
    Some(shader_asset_name)
}
