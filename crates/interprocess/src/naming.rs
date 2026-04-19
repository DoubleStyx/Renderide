//! Windows mapping and semaphore name formatting (compiled only on `cfg(windows)` targets).

/// Windows file-mapping object name (`CT_IP_{queue}`).
pub(crate) fn windows_mapping_name(memory_view_name: &str) -> String {
    format!("CT_IP_{memory_view_name}")
}

/// Win32 named semaphore (`Global\CT.IP.{queue}`).
pub(crate) fn windows_semaphore_wide_name(memory_view_name: &str) -> String {
    format!("Global\\CT.IP.{memory_view_name}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_mapping_name_prefix() {
        assert_eq!(windows_mapping_name("abc"), "CT_IP_abc");
    }

    #[test]
    fn windows_semaphore_wide_name_format() {
        assert_eq!(windows_semaphore_wide_name("q1"), "Global\\CT.IP.q1");
    }
}
