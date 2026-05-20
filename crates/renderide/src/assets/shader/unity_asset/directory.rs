//! Directory scanning fallback for loose Unity shader asset layouts.

use std::path::Path;

use super::probe::FileBinaryProbe;
use super::{MAX_DIR_FILES, ResolvedUnityShaderAsset, try_from_file_inner};

/// Scans a directory hint for a Unity shader AssetBundle.
pub(super) fn try_from_directory(dir: &Path) -> Option<ResolvedUnityShaderAsset> {
    let read_dir = match std::fs::read_dir(dir) {
        Ok(d) => d,
        Err(e) => {
            logger::warn!(
                "shader_unity_asset: cannot read directory {:?}: {}",
                dir.display(),
                e
            );
            return None;
        }
    };

    let mut paths: Vec<std::path::PathBuf> = read_dir
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();

    let files_total = paths.len();
    if files_total == 0 {
        logger::warn!(
            "shader_unity_asset: directory {:?} contains no regular files (only subdirs or empty); cannot probe Unity binaries here",
            dir.display()
        );
        return None;
    }
    paths.sort_unstable();
    paths.sort_by_key(|p| {
        let ext = p
            .extension()
            .and_then(|s| s.to_str())
            .map(str::to_ascii_lowercase)
            .unwrap_or_default();
        match ext.as_str() {
            "asset" | "unity" | "shader" => 0,
            _ => 1,
        }
    });

    let mut examined = 0usize;
    let mut bundle_parse_hits = 0usize;
    let mut first_probe: Option<FileBinaryProbe> = None;

    for (idx, p) in paths.into_iter().enumerate() {
        if idx >= MAX_DIR_FILES {
            break;
        }
        examined += 1;
        logger::debug!(
            "shader_unity_asset: directory {:?} examining [{}/{}] {:?}",
            dir.display(),
            examined,
            files_total.min(MAX_DIR_FILES),
            p.display()
        );
        let (name, probe) = try_from_file_inner(&p, false);
        if let Some(name) = name {
            return Some(name);
        }
        if let Some(probe) = probe {
            if probe.bundle_parse_ok {
                bundle_parse_hits += 1;
            }
            if first_probe.is_none() {
                first_probe = Some(probe);
            }
        }
    }

    logger::warn!(
        "shader_unity_asset: directory {:?} -- no shader name (files_total={} examined={} cap={} bundle_hits={})",
        dir.display(),
        files_total,
        examined,
        MAX_DIR_FILES,
        bundle_parse_hits
    );
    if let Some(ref fp) = first_probe {
        logger::debug!("shader_unity_asset: first failed file probe sample");
        fp.log_debug_detail();
    }

    None
}
