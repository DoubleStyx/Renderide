// Maps Rust CARGO_CFG_TARGET_ARCH to Khronos `openxr_loader_windows-*` per-arch directory names.
// Included from build.rs and `src/xr/openxr_loader_paths.rs`.

/// Returns the Khronos Windows package subfolder containing `openxr_loader.dll` for `arch`,
/// or `None` if this project does not ship a matching vendored loader.
pub(crate) fn khronos_windows_subdir_for_arch(arch: &str) -> Option<&'static str> {
    match arch {
        "x86_64" => Some("x64"),
        // Khronos packages ship `Win32_uwp` (not plain `Win32`) in current SDK layouts.
        "i686" | "i586" => Some("Win32_uwp"),
        // Khronos Windows SDK may only ship `ARM64_uwp` (no plain `ARM64`); same loader entry points.
        "aarch64" => Some("ARM64_uwp"),
        _ => None,
    }
}
