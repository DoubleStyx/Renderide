//! Wine detection via ntdll.wine_get_version.
//! Uses dlopen to load ntdll at runtime (only available when running under Wine).

use std::ffi::CString;

type WineGetVersion = unsafe extern "C" fn() -> *const u16;

/// Returns true if running under Wine.
pub fn is_wine() -> bool {
    wine_get_version().is_some()
}

/// Get Wine version string, or None if not running under Wine.
pub fn wine_get_version() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
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
        let func = func?;
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

    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}
