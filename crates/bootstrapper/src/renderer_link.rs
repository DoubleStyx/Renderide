//! Ensures the Host-visible renderer path (`Renderite.Renderer`) points at the real renderer binary on Unix.

#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::fs;
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::path::Path;

use crate::config::ResoBootConfig;

/// Returns `true` when `link` is a symlink that resolves to `target`.
#[cfg(target_os = "linux")]
fn symlink_points_to_target(link: &Path, target: &Path) -> bool {
    let Ok(destination) = fs::read_link(link) else {
        return false;
    };
    let destination = if destination.is_absolute() {
        destination
    } else if let Some(parent) = link.parent() {
        parent.join(destination)
    } else {
        destination
    };
    match (fs::canonicalize(destination), fs::canonicalize(target)) {
        (Ok(destination), Ok(target)) => destination == target,
        _ => false,
    }
}

/// Returns `true` when `lhs` and `rhs` refer to the same inode (e.g. a hard link to the renderer binary).
#[cfg(target_os = "macos")]
fn same_filesystem_inode(lhs: &Path, rhs: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    match (fs::metadata(lhs), fs::metadata(rhs)) {
        (Ok(ma), Ok(mb)) => ma.dev() == mb.dev() && ma.ino() == mb.ino(),
        _ => false,
    }
}

/// Removes a stale Host-visible renderer path.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn remove_stale_link(link: &Path) -> bool {
    match fs::remove_file(link) {
        Ok(()) => true,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => true,
        Err(e) => {
            logger::warn!(
                "Failed to remove stale Renderite.Renderer at {}: {}",
                link.display(),
                e
            );
            false
        }
    }
}

/// Copies the renderer binary as a macOS fallback when a hard link cannot be created.
#[cfg(target_os = "macos")]
fn copy_renderer_stub(target: &Path, link: &Path) -> std::io::Result<()> {
    fs::copy(target, link)?;
    let permissions = fs::metadata(target)?.permissions();
    fs::set_permissions(link, permissions)?;
    Ok(())
}

/// On Linux, creates a symlink; on macOS, a hard link so argv0 / process naming match the real binary.
///
/// No-op on Windows and when `renderide-renderer` does not exist beside the launcher.
pub fn ensure_link(config: &ResoBootConfig) {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        let link = &config.renderite_executable;
        let target = config.renderite_directory.join("renderide-renderer");
        if !target.exists() {
            logger::debug!(
                "Renderite.Renderer bridge not created because {} is missing",
                target.display()
            );
            return;
        }

        #[cfg(target_os = "linux")]
        {
            if symlink_points_to_target(link, &target) {
                logger::debug!("Renderite.Renderer symlink already points to renderide-renderer");
                return;
            }
            if !remove_stale_link(link) {
                return;
            }
            if let Err(e) = std::os::unix::fs::symlink(&target, link) {
                logger::warn!("Failed to create Renderite.Renderer symlink: {}", e);
            } else {
                logger::info!("Created Renderite.Renderer symlink for renderide-renderer");
            }
        }

        #[cfg(target_os = "macos")]
        {
            if same_filesystem_inode(link, &target) {
                logger::debug!("Renderite.Renderer hard link already points to renderide-renderer");
                return;
            }
            if !remove_stale_link(link) {
                return;
            }
            match fs::hard_link(&target, link) {
                Ok(()) => {
                    logger::info!("Created Renderite.Renderer hard link for renderide-renderer");
                }
                Err(hard_link_error) => match copy_renderer_stub(&target, link) {
                    Ok(()) => {
                        logger::warn!(
                            "Could not hard-link Renderite.Renderer: {}; copied renderide-renderer instead",
                            hard_link_error
                        );
                    }
                    Err(copy_error) => {
                        logger::warn!(
                            "Failed to create Renderite.Renderer hard link ({}) or copy fallback ({}).",
                            hard_link_error,
                            copy_error
                        );
                    }
                },
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
            resonite_dir: None,
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
        let target = tmp.join("renderide-renderer");
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
        let target = tmp.join("renderide-renderer");
        fs::write(&target, b"").unwrap();
        let link = tmp.join("Renderite.Renderer");
        std::os::unix::fs::symlink(tmp.join("wrong"), &link).unwrap();
        let c = cfg_with_dirs(&tmp, &link);
        ensure_link(&c);
        assert_eq!(fs::read_link(&link).unwrap(), target);
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn ensure_link_refreshes_stale_launcher_symlink() {
        let tmp = std::env::temp_dir().join(format!("bootstrapper_stub4_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        let stale_launcher = tmp.join("renderide");
        fs::write(&stale_launcher, b"launcher").unwrap();
        let target = tmp.join("renderide-renderer");
        fs::write(&target, b"renderer").unwrap();
        let link = tmp.join("Renderite.Renderer");
        std::os::unix::fs::symlink(&stale_launcher, &link).unwrap();
        let c = cfg_with_dirs(&tmp, &link);
        ensure_link(&c);
        assert_eq!(fs::read_link(&link).unwrap(), target);
        let _ = fs::remove_dir_all(&tmp);
    }

    /// When the renderer binary is not yet present (early-startup race or non-bundled deployment),
    /// `ensure_link` must not create a dangling symlink -- the linker stub is only useful when the
    /// real `renderide-renderer` binary already exists in the renderer directory.
    #[test]
    fn ensure_link_is_noop_when_target_missing() {
        let tmp = std::env::temp_dir().join(format!("bootstrapper_stub3_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        let link = tmp.join("Renderite.Renderer");
        let c = cfg_with_dirs(&tmp, &link);
        ensure_link(&c);
        assert!(
            !link.exists() && fs::symlink_metadata(&link).is_err(),
            "no symlink should be created when target is missing"
        );
        let _ = fs::remove_dir_all(&tmp);
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests_macos {
    use std::fs;

    use super::ensure_link;
    use crate::config::ResoBootConfig;

    fn cfg_with_dirs(tmp: &std::path::Path, link: &std::path::Path) -> ResoBootConfig {
        ResoBootConfig {
            current_directory: tmp.to_path_buf(),
            runtime_config: tmp.join("Renderite.Host.runtimeconfig.json"),
            renderite_directory: tmp.to_path_buf(),
            renderite_executable: link.to_path_buf(),
            resonite_dir: None,
            shared_memory_prefix: "t".into(),
            is_wine: false,
            renderide_log_level: None,
        }
    }

    #[test]
    fn ensure_link_hardlinks_when_missing() {
        let tmp =
            std::env::temp_dir().join(format!("bootstrapper_stub_mac_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        let target = tmp.join("renderide-renderer");
        fs::write(&target, b"bin").unwrap();
        let link = tmp.join("Renderite.Renderer");
        let c = cfg_with_dirs(&tmp, &link);
        ensure_link(&c);
        assert!(link.exists());
        let _ = fs::remove_dir_all(&tmp);
    }

    /// When the renderer binary is not yet present (early-startup race or non-bundled deployment),
    /// `ensure_link` must not create a dangling hard link -- the linker stub is only useful when the
    /// real `renderide-renderer` binary already exists in the renderer directory.
    #[test]
    fn ensure_link_is_noop_when_target_missing() {
        let tmp =
            std::env::temp_dir().join(format!("bootstrapper_stub_mac2_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        let link = tmp.join("Renderite.Renderer");
        let c = cfg_with_dirs(&tmp, &link);
        ensure_link(&c);
        assert!(
            !link.exists() && fs::symlink_metadata(&link).is_err(),
            "no link should be created when target is missing"
        );
        let _ = fs::remove_dir_all(&tmp);
    }
}
