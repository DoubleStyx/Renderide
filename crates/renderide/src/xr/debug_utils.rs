//! Optional [`XR_EXT_debug_utils`](https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#XR_EXT_debug_utils) messenger.
//!
//! Many OpenXR runtimes (including Monado and stacks built on it, such as WiVRn) also print their
//! own diagnostic lines directly to **stderr** from native code. Those messages are not delivered
//! through this extension; controlling them is runtime-specific (for example the Khronos loader's
//! `XR_LOADER_DEBUG`, or a given runtime's own environment variables).

use std::ffi::{CStr, c_char};
use std::os::raw::c_void;
use std::ptr;

use openxr as xr;
use xr::sys::Handle as _;

use logger::LogLevel;

/// Owns an [`xr::sys::DebugUtilsMessengerEXT`] and destroys it on drop.
pub(crate) struct OpenxrDebugUtilsMessenger {
    instance: xr::Instance,
    messenger: xr::sys::DebugUtilsMessengerEXT,
}

impl OpenxrDebugUtilsMessenger {
    /// Registers a debug messenger that forwards messages to the Renderide file logger.
    ///
    /// Returns [`None`] if the extension is not loaded, or if creation fails (failure is logged once).
    pub(crate) fn try_create(instance: &xr::Instance) -> Option<Self> {
        let fp = instance.exts().ext_debug_utils.as_ref()?;
        let message_severities = xr::sys::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
            | xr::sys::DebugUtilsMessageSeverityFlagsEXT::INFO
            | xr::sys::DebugUtilsMessageSeverityFlagsEXT::WARNING
            | xr::sys::DebugUtilsMessageSeverityFlagsEXT::ERROR;
        let message_types = xr::sys::DebugUtilsMessageTypeFlagsEXT::GENERAL
            | xr::sys::DebugUtilsMessageTypeFlagsEXT::VALIDATION
            | xr::sys::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
            | xr::sys::DebugUtilsMessageTypeFlagsEXT::CONFORMANCE;
        let create_info = xr::sys::DebugUtilsMessengerCreateInfoEXT {
            ty: xr::sys::DebugUtilsMessengerCreateInfoEXT::TYPE,
            next: ptr::null(),
            message_severities,
            message_types,
            user_callback: Some(debug_utils_callback),
            user_data: ptr::null_mut(),
        };
        let mut messenger = xr::sys::DebugUtilsMessengerEXT::NULL;
        // SAFETY: `fp` is the runtime-resolved function pointer for an enabled extension;
        // `instance` is a valid live `xr::Instance`; `create_info` is fully initialised and its
        // `next` pointer is null; `&mut messenger` is a valid out-pointer.
        let r = unsafe {
            (fp.create_debug_utils_messenger)(
                instance.as_raw(),
                core::ptr::addr_of!(create_info),
                core::ptr::addr_of_mut!(messenger),
            )
        };
        if r != xr::sys::Result::SUCCESS {
            logger::warn!(
                "OpenXR: xrCreateDebugUtilsMessengerEXT failed ({r:?}); runtime debug messages will not be mirrored to Renderide logs"
            );
            return None;
        }
        Some(Self {
            instance: instance.clone(),
            messenger,
        })
    }
}

impl Drop for OpenxrDebugUtilsMessenger {
    fn drop(&mut self) {
        if self.messenger.into_raw() == 0 {
            return;
        }
        if let Some(fp) = self.instance.exts().ext_debug_utils.as_ref() {
            // SAFETY: `self.messenger` was returned by a successful `xrCreateDebugUtilsMessengerEXT`
            // call and has not been destroyed (non-NULL guarded above); `fp` is still valid since
            // `self.instance` is still alive.
            let r = unsafe { (fp.destroy_debug_utils_messenger)(self.messenger) };
            if r != xr::sys::Result::SUCCESS {
                logger::warn!("OpenXR: xrDestroyDebugUtilsMessengerEXT failed ({r:?})");
            }
        }
        self.messenger = xr::sys::DebugUtilsMessengerEXT::NULL;
    }
}

/// OpenXR-invoked callback that forwards messenger events to the file logger.
///
/// # Safety
///
/// Must only be installed as an `XR_EXT_debug_utils` callback; the OpenXR runtime guarantees
/// `data` is valid (or null) for the call duration.
unsafe extern "system" fn debug_utils_callback(
    severity: xr::sys::DebugUtilsMessageSeverityFlagsEXT,
    _types: xr::sys::DebugUtilsMessageTypeFlagsEXT,
    data: *const xr::sys::DebugUtilsMessengerCallbackDataEXT,
    _user: *mut c_void,
) -> xr::sys::Bool32 {
    if data.is_null() {
        return xr::sys::FALSE;
    }
    // SAFETY: the OpenXR spec requires runtimes to pass a valid, fully-initialised pointer for the
    // duration of the callback; we checked non-null above. We only borrow immutably and never
    // retain the reference past return.
    let data = unsafe { &*data };
    // SAFETY: `data.message` and `data.function_name` are either null or NUL-terminated C strings
    // owned by the runtime for the callback's duration; `c_str_to_str` handles both.
    let msg = unsafe { c_str_to_str(data.message) };
    // SAFETY: see above.
    let fn_name = unsafe { c_str_to_str(data.function_name) };
    let prefix = if fn_name.is_empty() {
        "OpenXR".to_string()
    } else {
        format!("OpenXR ({fn_name})")
    };
    let full = format!("{prefix}: {msg}");

    let level = if severity.intersects(xr::sys::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        LogLevel::Error
    } else if severity.intersects(xr::sys::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        LogLevel::Warn
    } else if severity.intersects(xr::sys::DebugUtilsMessageSeverityFlagsEXT::INFO) {
        LogLevel::Info
    } else {
        LogLevel::Debug
    };

    logger::log(level, format_args!("{full}"));
    xr::sys::FALSE
}

/// Copies a C string into an owned `String`, or returns empty if `p` is null.
///
/// # Safety
///
/// If `p` is non-null, it must point to a NUL-terminated C string that remains valid for the
/// duration of the call.
unsafe fn c_str_to_str(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: non-null was checked; caller guarantees NUL termination per the function contract.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}
