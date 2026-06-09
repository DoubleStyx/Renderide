//! Windows security attributes for named IPC objects.

#[cfg(windows)]
use std::io;

#[cfg(windows)]
use std::ptr::null_mut;

#[cfg(windows)]
use windows_sys::Win32::Foundation::LocalFree;
#[cfg(windows)]
use windows_sys::Win32::Security::Authorization::{
    ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1,
};
#[cfg(windows)]
use windows_sys::Win32::Security::{PSECURITY_DESCRIPTOR, SECURITY_ATTRIBUTES};

#[cfg(windows)]
const CURRENT_OWNER_SDDL: &str = "D:P(A;;GA;;;OW)(A;;GA;;;SY)";

/// Owned security descriptor and attributes for current-owner-only named IPC objects.
#[cfg(windows)]
pub(crate) struct CurrentOwnerSecurityAttributes {
    descriptor: PSECURITY_DESCRIPTOR,
    attrs: SECURITY_ATTRIBUTES,
}

#[cfg(windows)]
impl CurrentOwnerSecurityAttributes {
    /// Builds a DACL that grants generic-all to the object owner and LocalSystem.
    pub(crate) fn new() -> io::Result<Self> {
        let mut descriptor: PSECURITY_DESCRIPTOR = null_mut();
        let sddl: Vec<u16> = CURRENT_OWNER_SDDL
            .encode_utf16()
            .chain(std::iter::once(0))
            .collect();
        let ok = unsafe {
            ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl.as_ptr(),
                SDDL_REVISION_1,
                &mut descriptor,
                null_mut(),
            )
        };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            descriptor,
            attrs: SECURITY_ATTRIBUTES {
                nLength: size_of::<SECURITY_ATTRIBUTES>() as u32,
                lpSecurityDescriptor: descriptor.cast(),
                bInheritHandle: 0,
            },
        })
    }

    /// Pointer passed to Win32 object creation calls.
    pub(crate) const fn as_ptr(&self) -> *const SECURITY_ATTRIBUTES {
        &self.attrs
    }
}

#[cfg(windows)]
impl Drop for CurrentOwnerSecurityAttributes {
    fn drop(&mut self) {
        if !self.descriptor.is_null() {
            unsafe {
                LocalFree(self.descriptor.cast());
            }
        }
    }
}
