//! Helpers for locating the Khronos OpenXR **loader** shared library at run time (`openxr_loader.dll`
//! on Windows, `libopenxr_loader.so` on Linux, `libopenxr_loader.dylib` on macOS).
//!
//! On Windows, the crate `build.rs` also copies a vendored `openxr_loader.dll` into the build output
//! directory so it sits next to built executables; [`openxr_loader_candidate_paths`] still checks the
//! executable directory first so shipped layouts work without relying on `PATH`.
//!
//! # Override path
//!
//! Set [`RENDERIDE_OPENXR_LOADER`] to a filesystem path that is either:
//! - the loader library file itself (e.g. `C:\Path\to\openxr_loader.dll`), or
//! - a directory containing that file (the per-OS basename from [`openxr_loader_library_filename`]
//!   is appended).
//!
//! This is checked after the executable's directory and before optional standard install locations
//! (Windows only) and the platform default search used by [`openxr::Entry::load`].
//!
//! # Khronos arch mapping
//!
//! Khronos arch mapping matches `build.rs` and `openxr_windows_arch.rs`; it is compiled only under
//! `#[cfg(test)]` so non-test library builds stay free of `dead_code` warnings.

use std::path::PathBuf;

/// Environment variable: path to the OpenXR loader library file, or a directory that contains it.
pub const RENDERIDE_OPENXR_LOADER: &str = "RENDERIDE_OPENXR_LOADER";

/// Basename of the Khronos OpenXR loader for the current target OS (matches openxr-rs [`openxr::Entry::load`]).
pub fn openxr_loader_library_filename() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "openxr_loader.dll"
    }
    #[cfg(target_os = "macos")]
    {
        "libopenxr_loader.dylib"
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        "libopenxr_loader.so"
    }
}

fn push_unique(out: &mut Vec<PathBuf>, path: PathBuf) {
    if !out.iter().any(|p| p == &path) {
        out.push(path);
    }
}

fn path_from_renderide_openxr_loader_env(name: &str) -> Option<PathBuf> {
    let raw = std::env::var_os(RENDERIDE_OPENXR_LOADER)?;
    let p = PathBuf::from(raw);
    Some(if p.is_dir() { p.join(name) } else { p })
}

#[cfg(target_os = "windows")]
fn windows_openxr_sdk_bin_candidates(name: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for key in ["ProgramFiles", "ProgramFiles(x86)"] {
        if let Ok(root) = std::env::var(key) {
            let base = PathBuf::from(root).join("OpenXR");
            for rel in ["bin", "bin/win64"] {
                push_unique(&mut out, base.join(rel).join(name));
            }
        }
    }
    out
}

/// Ordered candidate paths for [`openxr::Entry::load_from`]: executable directory, env override,
/// optional OS-specific defaults (Windows: common SDK install locations under Program Files).
pub fn openxr_loader_candidate_paths() -> Vec<PathBuf> {
    let name = openxr_loader_library_filename();
    let mut out = Vec::new();

    if let Ok(exe) = std::env::current_exe()
        && let Some(parent) = exe.parent()
    {
        push_unique(&mut out, parent.join(name));
    }

    if let Some(p) = path_from_renderide_openxr_loader_env(name) {
        push_unique(&mut out, p);
    }

    #[cfg(target_os = "windows")]
    {
        for p in windows_openxr_sdk_bin_candidates(name) {
            push_unique(&mut out, p);
        }
    }

    out
}

#[cfg(test)]
mod khronos_mapping_tests {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/openxr_windows_arch.rs"
    ));

    #[test]
    fn khronos_maps_x86_64_to_x64() {
        assert_eq!(khronos_windows_subdir_for_arch("x86_64"), Some("x64"));
    }

    #[test]
    fn khronos_maps_i686_to_win32_uwp() {
        assert_eq!(khronos_windows_subdir_for_arch("i686"), Some("Win32_uwp"));
    }

    #[test]
    fn khronos_maps_aarch64_to_arm64_uwp() {
        assert_eq!(
            khronos_windows_subdir_for_arch("aarch64"),
            Some("ARM64_uwp")
        );
    }

    #[test]
    fn khronos_unknown_arch_none() {
        assert_eq!(khronos_windows_subdir_for_arch("riscv64gc"), None);
    }
}

#[cfg(test)]
mod path_tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn exe_dir_joined_loader_name_roundtrips() {
        let dir = Path::new("/opt/renderide/bin");
        let p = dir.join(openxr_loader_library_filename());
        assert_eq!(
            p.file_name().unwrap().to_str().unwrap(),
            openxr_loader_library_filename()
        );
    }

    #[test]
    fn library_filename_matches_openxr_crate() {
        #[cfg(target_os = "windows")]
        assert_eq!(openxr_loader_library_filename(), "openxr_loader.dll");
        #[cfg(target_os = "macos")]
        assert_eq!(openxr_loader_library_filename(), "libopenxr_loader.dylib");
        #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
        assert_eq!(openxr_loader_library_filename(), "libopenxr_loader.so");
    }

    #[test]
    fn candidate_paths_include_exe_parent_joined_name() {
        let paths = openxr_loader_candidate_paths();
        let name = openxr_loader_library_filename();
        let exe_parent = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(Path::to_path_buf));
        if let Some(parent) = exe_parent {
            assert!(
                paths.contains(&parent.join(name)),
                "expected {:?} in candidates {:?}",
                parent.join(name),
                paths
            );
        }
    }

    /// [`push_unique`] must not append a duplicate path; insertion order is otherwise preserved.
    #[test]
    fn push_unique_dedupes_existing_entries() {
        let mut out: Vec<PathBuf> = Vec::new();
        let a = PathBuf::from("/tmp/a");
        let b = PathBuf::from("/tmp/b");
        push_unique(&mut out, a.clone());
        push_unique(&mut out, b.clone());
        push_unique(&mut out, a.clone());
        push_unique(&mut out, b.clone());
        assert_eq!(out, vec![a, b]);
    }

    /// All env-mutating cases are exercised in a single test so they serialize under cargo's
    /// default parallel test harness; mutating [`RENDERIDE_OPENXR_LOADER`] from concurrent tests
    /// would race otherwise.
    #[test]
    fn env_override_absent_directory_and_verbatim() {
        let _guard = EnvGuard::capture();
        let name = openxr_loader_library_filename();

        // SAFETY: env mutation in test; this is the only test in this module that mutates env.
        unsafe {
            std::env::remove_var(RENDERIDE_OPENXR_LOADER);
        }
        assert!(
            path_from_renderide_openxr_loader_env(name).is_none(),
            "unset env should return None"
        );

        let tmp_dir = std::env::temp_dir();
        // SAFETY: env mutation in test; this is the only test in this module that mutates env.
        unsafe {
            std::env::set_var(RENDERIDE_OPENXR_LOADER, &tmp_dir);
        }
        let resolved = path_from_renderide_openxr_loader_env(name).expect("directory override");
        assert_eq!(resolved, tmp_dir.join(name));

        let made_up = tmp_dir.join("renderide-openxr-test-this-path-should-not-exist.dll");
        // SAFETY: env mutation in test; this is the only test in this module that mutates env.
        unsafe {
            std::env::set_var(RENDERIDE_OPENXR_LOADER, &made_up);
        }
        let resolved = path_from_renderide_openxr_loader_env(name).expect("file override");
        assert_eq!(resolved, made_up);
    }

    /// RAII guard that restores [`RENDERIDE_OPENXR_LOADER`] to its original state on drop so env
    /// mutations across tests do not leak process-wide.
    struct EnvGuard(Option<std::ffi::OsString>);

    impl EnvGuard {
        fn capture() -> Self {
            Self(std::env::var_os(RENDERIDE_OPENXR_LOADER))
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            // SAFETY: env mutation in test; this Drop runs at end of the single env-mutating test.
            unsafe {
                match &self.0 {
                    Some(v) => std::env::set_var(RENDERIDE_OPENXR_LOADER, v),
                    None => std::env::remove_var(RENDERIDE_OPENXR_LOADER),
                }
            }
        }
    }
}
