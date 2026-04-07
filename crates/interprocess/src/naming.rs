//! Windows mapping and semaphore name formatting.

/// Windows file-mapping object name (`CT_IP_{queue}`).
pub(crate) fn windows_mapping_name(memory_view_name: &str) -> String {
    format!("CT_IP_{memory_view_name}")
}

/// Win32 named semaphore (`Global\CT.IP.{queue}`).
pub(crate) fn windows_semaphore_wide_name(memory_view_name: &str) -> String {
    format!("Global\\CT.IP.{memory_view_name}")
}
