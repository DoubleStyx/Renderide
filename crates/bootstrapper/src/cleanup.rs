//! Best-effort removal of shared-memory queue backing files after a Wine session.

use std::fs;
use std::path::Path;

use interprocess::LINUX_SHM_MEMORY_DIR;

/// Deletes files whose names contain `shared_memory_prefix` under Wine-relevant trees.
///
/// ResoBoot scans `/dev/shm` recursively; queue files also live under
/// [`interprocess::LINUX_SHM_MEMORY_DIR`]. Both are walked so orphaned `.qu` files and stray matches are removed.
pub fn remove_wine_queue_backing_files(shared_memory_prefix: &str) {
    if !cfg!(target_os = "linux") {
        return;
    }

    let shm = Path::new("/dev/shm");
    let mmf = Path::new(LINUX_SHM_MEMORY_DIR);

    for base in [shm, mmf] {
        if base.exists() {
            let _ = remove_files_recursive_matching(base, shared_memory_prefix);
        }
    }
}

fn remove_files_recursive_matching(dir: &Path, needle: &str) -> std::io::Result<()> {
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
    use super::*;
    use std::io::Write;

    #[test]
    fn remove_matching_only_prefix_files() {
        let tmp = std::env::temp_dir().join(format!("bootstrapper_cleanup_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join("nested")).unwrap();
        let needle = "abc123PREFIX";
        let mut f = fs::File::create(tmp.join(format!("{needle}.qu"))).unwrap();
        writeln!(f, "x").unwrap();
        fs::File::create(tmp.join("other.qu")).unwrap();
        remove_files_recursive_matching(&tmp, needle).unwrap();
        assert!(!tmp.join(format!("{needle}.qu")).exists());
        assert!(tmp.join("other.qu").exists());
        let _ = fs::remove_dir_all(&tmp);
    }
}
