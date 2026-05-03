//! Disk helpers for Dear ImGui HUD persistence.

use std::io;
use std::path::{Path, PathBuf};

/// Sidecar file containing Dear ImGui's raw window-layout `.ini` payload.
pub const IMGUI_INI_FILE_NAME: &str = "renderide-imgui.ini";

/// Places the ImGui sidecar next to the renderer config file.
pub fn imgui_ini_path_from_config_save_path(config_save_path: &Path) -> PathBuf {
    let parent = config_save_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    parent.join(IMGUI_INI_FILE_NAME)
}

/// Writes UTF-8 text atomically using a hidden sibling temp file and rename.
pub fn write_text_atomic(path: &Path, contents: &str) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(IMGUI_INI_FILE_NAME);
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let tmp = parent.join(format!(".{file_name}.tmp"));
    std::fs::write(&tmp, contents.as_bytes())?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{IMGUI_INI_FILE_NAME, imgui_ini_path_from_config_save_path, write_text_atomic};

    #[test]
    fn imgui_ini_path_sits_next_to_config() {
        let p = imgui_ini_path_from_config_save_path(std::path::Path::new(
            "/tmp/renderide/config.toml",
        ));
        assert_eq!(
            p,
            std::path::Path::new("/tmp/renderide").join(IMGUI_INI_FILE_NAME)
        );
    }

    #[test]
    fn imgui_ini_path_handles_bare_config_filename() {
        let p = imgui_ini_path_from_config_save_path(std::path::Path::new("config.toml"));
        assert_eq!(p, std::path::Path::new(".").join(IMGUI_INI_FILE_NAME));
    }

    #[test]
    fn atomic_write_roundtrips_text() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(IMGUI_INI_FILE_NAME);
        write_text_atomic(&path, "[Window][Renderer]\n").expect("write");
        let text = std::fs::read_to_string(path).expect("read");
        assert_eq!(text, "[Window][Renderer]\n");
    }
}
