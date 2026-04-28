//! Detection of Wine on Linux (e.g. `wine_get_version` from `ntdll`).

/// Returns `true` when the process is running under Wine on Linux.
///
/// Native Windows, macOS, and Linux builds never report Wine except when
/// `ntdll.dll` exports `wine_get_version` as injected by Wine.
pub(crate) fn is_wine() -> bool {
    wine_get_version().is_some()
}

/// Wine version string from `ntdll`, or `None` when not running under Wine.
pub(crate) fn wine_get_version() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        wine_get_version_linux()
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Maximum UTF-16 code units scanned looking for the NUL terminator on Wine's version string.
///
/// Wine returns a short version string in practice (e.g. `"10.0"`); a missing terminator from
/// a corrupted shim must not spin this thread forever during bootstrap.
#[cfg(target_os = "linux")]
const WINE_VERSION_MAX_UTF16_UNITS: usize = 256;

/// Loads `wine_get_version` from `ntdll.dll` via [`libloading::Library`] and returns the UTF-16 version string.
#[cfg(target_os = "linux")]
fn wine_get_version_linux() -> Option<String> {
    use libloading::{Library, Symbol};

    // SAFETY: `Library::new` loads a host module; under native Linux this fails fast; under Wine,
    // `ntdll.dll` is a Wine-provided shim that exposes `wine_get_version`.
    let lib = unsafe { Library::new("ntdll.dll") }.ok()?;
    // SAFETY: We only call a symbol name Wine documents; wrong signatures would be UB.
    let func: Symbol<unsafe extern "C" fn() -> *const u16> =
        unsafe { lib.get(b"wine_get_version\0").ok()? };
    // SAFETY: `wine_get_version` returns a pointer to a static UTF-16 string or null when absent.
    let ptr = unsafe { func() };
    if ptr.is_null() {
        return None;
    }
    let mut len = None;
    for i in 0..WINE_VERSION_MAX_UTF16_UNITS {
        // SAFETY: scanning at most `WINE_VERSION_MAX_UTF16_UNITS` code units before bailing;
        // Wine NUL-terminates the version string well within this bound for any honest shim.
        if unsafe { *ptr.add(i) } == 0 {
            len = Some(i);
            break;
        }
    }
    let len = len?;
    // SAFETY: `ptr` is valid for `len` UTF-16 code units for the lifetime of the returned string;
    // `len` is bounded by `WINE_VERSION_MAX_UTF16_UNITS`.
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
    String::from_utf16(slice).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(target_os = "linux"))]
    #[test]
    fn wine_never_detected_off_linux() {
        assert!(wine_get_version().is_none());
        assert!(!is_wine());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn is_wine_matches_presence_of_version_string() {
        assert_eq!(is_wine(), wine_get_version().is_some());
    }
}
