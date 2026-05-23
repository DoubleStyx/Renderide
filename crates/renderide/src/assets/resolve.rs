use std::path::PathBuf;

const ASSETS_FOLDER_NAME: &str = "assets";

/// Enumerates directories that might contain assets used by Renderide.
pub fn assets_search_candidates() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let push_unique = |v: &mut Vec<PathBuf>, p: PathBuf| {
        if !v.iter().any(|x| x == &p) {
            v.push(p);
        }
    };

    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        push_unique(&mut out, dir.join(ASSETS_FOLDER_NAME));
        if let Some(parent) = dir.parent() {
            push_unique(&mut out, parent.join(ASSETS_FOLDER_NAME));
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        if let Some(root) = crate::config::find_renderide_workspace_root(&cwd) {
            push_unique(
                &mut out,
                root.join("crates/renderide").join(ASSETS_FOLDER_NAME),
            );
        }
        push_unique(&mut out, cwd.join(ASSETS_FOLDER_NAME));
    }

    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
        && let Some(root) = crate::config::find_renderide_workspace_root(dir)
    {
        push_unique(
            &mut out,
            root.join("crates/renderide").join(ASSETS_FOLDER_NAME),
        );
    }

    out
}
