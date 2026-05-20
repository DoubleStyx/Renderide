//! Binary probe diagnostics for Unity shader asset resolution.

use std::fmt::Display;
use std::path::Path;

/// Hex prefix length for short probe lines.
const PROBE_HEX_SHORT: usize = 8;

/// Per-file binary probe state for structured failure logs.
pub(super) struct FileBinaryProbe {
    /// Total bytes read from the probed file.
    pub(super) bytes_len: usize,
    /// Hex prefix sampled from the probed file.
    pub(super) prefix_hex: String,
    /// Printable ASCII prefix sampled from the probed file.
    pub(super) prefix_ascii: String,
    /// Whether AssetBundle parsing succeeded.
    pub(super) bundle_parse_ok: bool,
    /// Number of serialized files in the parsed bundle.
    pub(super) bundle_assets: usize,
    /// Truncated AssetBundle parse error.
    pub(super) bundle_err: Option<String>,
}

impl FileBinaryProbe {
    /// Builds a probe summary from file bytes.
    pub(super) fn new(bytes: &[u8]) -> Self {
        Self {
            bytes_len: bytes.len(),
            prefix_hex: format_hex_prefix(bytes, 24),
            prefix_ascii: ascii_prefix_hint(bytes, 40),
            bundle_parse_ok: false,
            bundle_assets: 0,
            bundle_err: None,
        }
    }

    /// Logs one short warning line; full fields are emitted by [`Self::log_debug_detail`].
    pub(super) fn warn_short(&self, path: &Path, reason: &str) {
        logger::warn!(
            "shader_unity_asset: {:?} -- {} | bytes={} hex8={} | bundle_ok={} | err {:?}",
            path.display(),
            reason,
            self.bytes_len,
            short_hex_prefix(&self.prefix_hex, PROBE_HEX_SHORT),
            self.bundle_parse_ok,
            self.bundle_err.as_deref().unwrap_or("")
        );
    }

    /// Logs the complete probe summary at debug level.
    pub(super) fn log_debug_detail(&self) {
        logger::debug!(
            "shader_unity_asset: probe detail bytes={} prefix_hex={} prefix_ascii={:?} bundle_ok={} bundle_assets={} bundle_err={:?}",
            self.bytes_len,
            self.prefix_hex,
            self.prefix_ascii,
            self.bundle_parse_ok,
            self.bundle_assets,
            self.bundle_err
        );
    }
}

/// Returns the first `max_bytes` bytes from a space-separated hex string.
pub(super) fn short_hex_prefix(space_separated_hex: &str, max_bytes: usize) -> String {
    space_separated_hex
        .split_whitespace()
        .take(max_bytes)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Formats up to `max` bytes as a space-separated lowercase hex prefix.
pub(super) fn format_hex_prefix(bytes: &[u8], max: usize) -> String {
    bytes
        .iter()
        .take(max)
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Returns a printable ASCII hint when the sampled prefix is fully printable.
pub(super) fn ascii_prefix_hint(bytes: &[u8], max: usize) -> String {
    let take = bytes.iter().copied().take(max).collect::<Vec<u8>>();
    if take.is_empty() {
        return String::new();
    }
    if take
        .iter()
        .all(|b| b.is_ascii_graphic() || matches!(b, b' ' | b'\t' | b'\n' | b'\r'))
    {
        String::from_utf8_lossy(&take).chars().take(40).collect()
    } else {
        String::new()
    }
}

/// Formats and truncates an error for bounded single-line logging.
pub(super) fn truncate_display(err: impl Display, max: usize) -> String {
    let s = err.to_string();
    if s.len() <= max {
        return s;
    }
    format!("{}...", &s[..max.saturating_sub(1)])
}
