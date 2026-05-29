//! Compile-time release metadata and platform token resolution.

use super::{NIGHTLY_PREFIX, RELEASE_CHANNEL, ReleaseBuildMetadata};

/// Returns embedded release metadata when this launcher is an official CI release.
pub(super) fn current() -> Option<ReleaseBuildMetadata> {
    let platform = current_platform()?;
    let channel = option_env!("RENDERIDE_RELEASE_CHANNEL")?.trim();
    let tag = option_env!("RENDERIDE_RELEASE_TAG")?.trim();
    let commit = option_env!("RENDERIDE_RELEASE_COMMIT")?.trim();
    let embedded_platform = option_env!("RENDERIDE_RELEASE_PLATFORM")?.trim();
    if channel != RELEASE_CHANNEL
        || !tag.starts_with(NIGHTLY_PREFIX)
        || !is_full_sha(commit)
        || embedded_platform != platform
    {
        return None;
    }
    Some(ReleaseBuildMetadata {
        channel: channel.to_owned(),
        tag: tag.to_owned(),
        commit: commit.to_owned(),
        platform: embedded_platform.to_owned(),
    })
}

/// Returns the compile-time platform token used by release asset names.
pub(super) fn current_platform() -> Option<&'static str> {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        Some("linux-x86_64")
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        Some("linux-aarch64")
    }
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        Some("windows-x86_64")
    }
    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    {
        Some("windows-aarch64")
    }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        Some("macos-x86_64")
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        Some("macos-aarch64")
    }
    #[cfg(not(any(
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "aarch64"),
        all(target_os = "windows", target_arch = "x86_64"),
        all(target_os = "windows", target_arch = "aarch64"),
        all(target_os = "macos", target_arch = "x86_64"),
        all(target_os = "macos", target_arch = "aarch64")
    )))]
    {
        None
    }
}

/// Returns whether a string is a complete hexadecimal Git commit SHA.
pub(super) fn is_full_sha(value: &str) -> bool {
    value.len() == 40 && value.chars().all(|c| c.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::is_full_sha;

    #[test]
    fn full_sha_validation_requires_forty_hex_chars() {
        assert!(is_full_sha("0123456789abcdef0123456789ABCDEF01234567"));
        assert!(!is_full_sha("0123456789abcdef"));
        assert!(!is_full_sha("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"));
    }
}
