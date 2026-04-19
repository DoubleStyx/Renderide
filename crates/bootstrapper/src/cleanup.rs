//! Best-effort removal of shared-memory queue backing files after a Wine session.

#[cfg(target_os = "linux")]
use std::fs;
#[cfg(target_os = "linux")]
use std::path::Path;

#[cfg(target_os = "linux")]
use interprocess::LINUX_SHM_MEMORY_DIR;

/// Deletes files whose names contain `shared_memory_prefix` under Wine-relevant trees.
///
/// ResoBoot scans `/dev/shm` recursively; queue files also live under
/// [`interprocess::LINUX_SHM_MEMORY_DIR`]. Both are walked so orphaned `.qu` files and stray matches are removed.
///
/// Linux only; other platforms compile to a no-op.
#[cfg(target_os = "linux")]
pub fn remove_wine_queue_backing_files(shared_memory_prefix: &str) {
    let shm = Path::new("/dev/shm");
    let mmf = Path::new(LINUX_SHM_MEMORY_DIR);

    for base in [shm, mmf] {
        if base.exists() {
            let _ = remove_files_recursive_matching(base, shared_memory_prefix);
        }
    }
}

/// Non-Linux builds do not use Wine queue paths under `/dev/shm`.
#[cfg(not(target_os = "linux"))]
pub fn remove_wine_queue_backing_files(_shared_memory_prefix: &str) {}

/// Recursively deletes regular files under `dir` whose names contain `needle`.
#[cfg(target_os = "linux")]
fn remove_files_recursive_matching(dir: &Path, needle: &str) -> std::io::Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let Ok(ty) = entry.file_type() else {
            continue;
        };
        if ty.is_dir() {
            let _ = remove_files_recursive_matching(&path, needle);
        } else if ty.is_file()
            && path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.contains(needle))
        {
            let _ = fs::remove_file(&path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[cfg(target_os = "linux")]
    use super::*;
    #[cfg(target_os = "linux")]
    use std::io::Write;

    #[cfg(target_os = "linux")]
    #[test]
    fn remove_matching_only_prefix_files() {
        let tmp = std::env::temp_dir().join(format!("bootstrapper_cleanup_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("nested")).unwrap();
        let needle = "abc123PREFIX";
        let mut f = std::fs::File::create(tmp.join(format!("{needle}.qu"))).unwrap();
        writeln!(f, "x").unwrap();
        std::fs::File::create(tmp.join("other.qu")).unwrap();
        remove_files_recursive_matching(&tmp, needle).unwrap();
        assert!(!tmp.join(format!("{needle}.qu")).exists());
        assert!(tmp.join("other.qu").exists());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn remove_nested_multiple_matches() {
        let tmp = std::env::temp_dir().join(format!(
            "bootstrapper_cleanup_nested_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        let needle = "needleX";
        std::fs::create_dir_all(tmp.join("a/b")).unwrap();
        std::fs::write(tmp.join(format!("x_{needle}_1.qu")), b"1").unwrap();
        std::fs::write(tmp.join("a").join(format!("{needle}.qu")), b"2").unwrap();
        std::fs::write(tmp.join("a/b/keep.qu"), b"k").unwrap();
        remove_files_recursive_matching(&tmp, needle).unwrap();
        assert!(!tmp.join(format!("x_{needle}_1.qu")).exists());
        assert!(!tmp.join("a").join(format!("{needle}.qu")).exists());
        assert!(tmp.join("a/b/keep.qu").exists());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn remove_missing_root_ok() {
        let p = std::env::temp_dir().join(format!(
            "bootstrapper_cleanup_missing_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&p);
        remove_files_recursive_matching(&p, "x").unwrap();
    }
}
