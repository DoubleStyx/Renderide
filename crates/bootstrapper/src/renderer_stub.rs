//! Ensures the Host-visible renderer path (`Renderite.Renderer`) points at the real `renderide` binary on Unix.

#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::fs;

use crate::config::ResoBootConfig;

/// Returns `true` when `lhs` and `rhs` refer to the same inode (e.g. a hard link to the renderer binary).
#[cfg(target_os = "macos")]
fn same_filesystem_inode(lhs: &std::path::Path, rhs: &std::path::Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    match (fs::metadata(lhs), fs::metadata(rhs)) {
        (Ok(ma), Ok(mb)) => ma.dev() == mb.dev() && ma.ino() == mb.ino(),
        _ => false,
    }
}

/// On Linux, creates a symlink; on macOS, a hard link so argv0 / process naming match the real binary.
///
/// No-op on Windows and when `renderide` does not exist beside the bootstrapper.
pub fn ensure_link(config: &ResoBootConfig) {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        let symlink = &config.renderite_executable;
        let target = config.renderite_directory.join("renderide");
        let needs_renderer_stub = target.exists() && {
            #[cfg(target_os = "linux")]
            {
                !symlink.exists() || fs::read_link(symlink).is_err()
            }
            #[cfg(target_os = "macos")]
            {
                !symlink.exists() || !same_filesystem_inode(symlink, &target)
            }
        };
        if needs_renderer_stub {
            let _ = fs::remove_file(symlink);
            #[cfg(target_os = "linux")]
            if let Err(e) = std::os::unix::fs::symlink(&target, symlink) {
                logger::warn!("Failed to create Renderite.Renderer symlink: {}", e);
            }
            #[cfg(target_os = "macos")]
            if let Err(e) = fs::hard_link(&target, symlink) {
                logger::warn!("Failed to create Renderite.Renderer link: {}", e);
            }
        }
    }
    #[cfg(windows)]
    {
        let _ = config;
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use std::fs;

    use super::ensure_link;
    use crate::config::ResoBootConfig;

    fn cfg_with_dirs(tmp: &std::path::Path, renderer_name: &std::path::Path) -> ResoBootConfig {
        ResoBootConfig {
            current_directory: tmp.to_path_buf(),
            runtime_config: tmp.join("Renderite.Host.runtimeconfig.json"),
            renderite_directory: tmp.to_path_buf(),
            renderite_executable: renderer_name.to_path_buf(),
            shared_memory_prefix: "t".into(),
            is_wine: false,
            renderide_log_level: None,
        }
    }

    #[test]
    fn ensure_link_creates_symlink_when_missing() {
        let tmp = std::env::temp_dir().join(format!("bootstrapper_stub_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        let target = tmp.join("renderide");
        fs::write(&target, b"").unwrap();
        let link = tmp.join("Renderite.Renderer");
        let c = cfg_with_dirs(&tmp, &link);
        ensure_link(&c);
        assert!(link.exists());
        let dest = fs::read_link(&link).expect("symlink");
        assert_eq!(dest, target);
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn ensure_link_refreshes_broken_symlink() {
        let tmp = std::env::temp_dir().join(format!("bootstrapper_stub2_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        let target = tmp.join("renderide");
        fs::write(&target, b"").unwrap();
        let link = tmp.join("Renderite.Renderer");
        std::os::unix::fs::symlink(tmp.join("wrong"), &link).unwrap();
        let c = cfg_with_dirs(&tmp, &link);
        ensure_link(&c);
        assert_eq!(fs::read_link(&link).unwrap(), target);
        let _ = fs::remove_dir_all(&tmp);
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests_macos {
    use std::fs;

    use super::ensure_link;
    use crate::config::ResoBootConfig;

    #[test]
    fn ensure_link_hardlinks_when_missing() {
        let tmp =
            std::env::temp_dir().join(format!("bootstrapper_stub_mac_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        let target = tmp.join("renderide");
        fs::write(&target, b"bin").unwrap();
        let link = tmp.join("Renderite.Renderer");
        let c = ResoBootConfig {
            current_directory: tmp.clone(),
            runtime_config: tmp.join("Renderite.Host.runtimeconfig.json"),
            renderite_directory: tmp.clone(),
            renderite_executable: link.clone(),
            shared_memory_prefix: "t".into(),
            is_wine: false,
            renderide_log_level: None,
        };
        ensure_link(&c);
        assert!(link.exists());
        let _ = fs::remove_dir_all(&tmp);
    }
}
