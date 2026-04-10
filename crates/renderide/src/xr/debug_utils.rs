//! Optional [`XR_EXT_debug_utils`](https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#XR_EXT_debug_utils) messenger.
//!
//! Many OpenXR runtimes (including Monado and stacks built on it, such as WiVRn) also print their
//! own diagnostic lines directly to **stderr** from native code. Those messages are not delivered
//! through this extension; controlling them is runtime-specific (for example the Khronos loader’s
//! `XR_LOADER_DEBUG`, or a given runtime’s own environment variables).

use std::ffi::{c_char, CStr};
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
        let r = unsafe {
            (fp.create_debug_utils_messenger)(instance.as_raw(), &create_info, &mut messenger)
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
            let r = unsafe { (fp.destroy_debug_utils_messenger)(self.messenger) };
            if r != xr::sys::Result::SUCCESS {
                logger::warn!("OpenXR: xrDestroyDebugUtilsMessengerEXT failed ({r:?})");
            }
        }
        self.messenger = xr::sys::DebugUtilsMessengerEXT::NULL;
    }
}

unsafe extern "system" fn debug_utils_callback(
    severity: xr::sys::DebugUtilsMessageSeverityFlagsEXT,
    _types: xr::sys::DebugUtilsMessageTypeFlagsEXT,
    data: *const xr::sys::DebugUtilsMessengerCallbackDataEXT,
    _user: *mut c_void,
) -> xr::sys::Bool32 {
    if data.is_null() {
        return xr::sys::FALSE;
    }
    let data = unsafe { &*data };
    let msg = unsafe { c_str_to_str(data.message) };
    let fn_name = unsafe { c_str_to_str(data.function_name) };
    let prefix = if fn_name.is_empty() {
        String::from("OpenXR")
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

unsafe fn c_str_to_str(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    CStr::from_ptr(p).to_string_lossy().into_owned()
}
