//! Runtime environment detection (Wine under Linux).

/// Returns `true` when the process is running under Wine on Linux.
///
/// Native Windows, macOS, and Linux builds never report Wine except when
/// `ntdll.dll` exports `wine_get_version` as injected by Wine.
pub fn is_wine() -> bool {
    wine_get_version().is_some()
}

/// Wine version string from `ntdll`, or `None` when not running under Wine.
pub fn wine_get_version() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        wine_get_version_linux()
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (); // Non-Linux targets are never the Wine-on-Linux stack.
        None
    }
}

#[cfg(target_os = "linux")]
fn wine_get_version_linux() -> Option<String> {
    use std::ffi::CString;

    type WineGetVersion = unsafe extern "C" fn() -> *const u16;

    let name = CString::new("ntdll.dll").ok()?;
    let handle = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW) };
    if handle.is_null() {
        return None;
    }
    let sym = CString::new("wine_get_version").ok()?;
    let func = unsafe {
        std::mem::transmute::<*mut libc::c_void, Option<WineGetVersion>>(libc::dlsym(
            handle,
            sym.as_ptr(),
        ))
    };
    let Some(func) = func else {
        unsafe { libc::dlclose(handle) };
        return None;
    };
    let ptr = unsafe { func() };
    if ptr.is_null() {
        unsafe { libc::dlclose(handle) };
        return None;
    }
    let mut len = 0usize;
    while unsafe { *ptr.add(len) } != 0 {
        len += 1;
    }
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
    let result = String::from_utf16(slice).ok();
    unsafe { libc::dlclose(handle) };
    result
}
