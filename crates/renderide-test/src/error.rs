//! Error types for the renderide-test harness.

use std::path::PathBuf;

use thiserror::Error;

/// Top-level harness error.
#[derive(Debug, Error)]
pub(crate) enum HarnessError {
    /// The renderer binary could not be located on disk.
    #[error("renderer binary not found at {0}; build with `cargo build -p renderide` (use `--profile dev-fast` or `--release` as needed)")]
    RendererBinaryMissing(PathBuf),
    /// Spawning the renderer process failed.
    #[error("spawn renderer process: {0}")]
    SpawnRenderer(#[source] std::io::Error),
    /// Building [`interprocess::QueueOptions`] failed (capacity invalid, etc.).
    #[error("queue options invalid: {0}")]
    QueueOptions(String),
    /// The handshake never completed within the configured timeout.
    #[error("handshake timed out after {0:?}")]
    HandshakeTimeout(std::time::Duration),
    /// An asset upload acknowledgement never arrived.
    #[error("asset ack timed out after {0:?} ({1})")]
    AssetAckTimeout(std::time::Duration, &'static str),
    /// PNG output never appeared / never refreshed within the configured wait.
    #[error("expected fresh PNG output at {path} within {wait:?}")]
    PngOutputMissing {
        /// Output PNG path the renderer was instructed to write.
        path: PathBuf,
        /// Maximum wall-clock wait before giving up.
        wait: std::time::Duration,
    },
    /// Reading or decoding the PNG output failed.
    #[error("read png {path}: {source}")]
    PngRead {
        /// Path that failed to load.
        path: PathBuf,
        /// Underlying image crate error.
        #[source]
        source: image::ImageError,
    },
    /// Writing the diff or actual PNG to disk failed.
    #[error("write png {path}: {source}")]
    PngWrite {
        /// Output path.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: image::ImageError,
    },
    /// Rendered image has no per-channel variation (clear-only or nearly flat); geometry did not draw.
    #[error(
        "rendered image at {path} is a flat single color {color:?}; the renderer produced no draws"
    )]
    FlatImage {
        /// Path to the offending PNG.
        path: PathBuf,
        /// Sample RGBA (typically the first pixel).
        color: [u8; 4],
    },
    /// Golden image is missing on disk; run `generate` first.
    #[error("golden image not found at {0}; run `renderide-test generate` first")]
    GoldenMissing(PathBuf),
    /// Perceptual diff failed against the configured threshold.
    #[error("perceptual diff failed: SSIM={score:.4} below threshold {threshold:.4}; diff written to {diff_path}")]
    GoldenMismatch {
        /// Computed SSIM score.
        score: f64,
        /// Required minimum SSIM score.
        threshold: f64,
        /// Path of the saved diff visualization.
        diff_path: PathBuf,
    },
    /// Generic IO failure (file copy, rename, etc.).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// `image-compare` failed to compute a similarity score.
    #[error("image-compare: {0}")]
    ImageCompare(String),
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::HarnessError;

    #[test]
    fn display_renderer_binary_missing() {
        let p = PathBuf::from("/no/renderide");
        let e = HarnessError::RendererBinaryMissing(p.clone());
        let s = e.to_string();
        assert!(s.contains("renderer binary not found"));
        assert!(s.contains(p.to_string_lossy().as_ref()));
    }

    #[test]
    fn display_golden_missing() {
        let p = PathBuf::from("missing.png");
        let e = HarnessError::GoldenMissing(p);
        assert!(e.to_string().contains("golden image not found"));
        assert!(e.to_string().contains("generate"));
    }

    #[test]
    fn display_golden_mismatch_includes_scores_and_paths() {
        let e = HarnessError::GoldenMismatch {
            score: 0.5,
            threshold: 0.95,
            diff_path: PathBuf::from("target/diff.png"),
        };
        let s = e.to_string();
        assert!(s.contains("SSIM=0.5000"));
        assert!(s.contains("0.9500"));
        assert!(s.contains("diff.png"));
    }
}
