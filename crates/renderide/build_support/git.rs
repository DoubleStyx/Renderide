//! Build-time commit metadata for embedding into the renderer identifier.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Number of leading hex characters of the commit hash embedded into the
/// renderer identifier sent to the host.
pub(crate) const COMMIT_HASH_LEN: usize = 8;
const FULL_COMMIT_HASH_LEN: usize = 40;

/// Source of the commit hash embedded into the renderer binary.
pub(crate) enum CommitSource {
    /// Commit supplied by the release workflow.
    ReleaseEnv,
    /// Commit resolved from the source checkout.
    Git,
}

impl CommitSource {
    /// Stable label written into renderer startup logs.
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Self::ReleaseEnv => "release-env",
            Self::Git => "git",
        }
    }
}

/// Commit metadata selected for the current build.
pub(crate) struct BuildCommit {
    /// First [`COMMIT_HASH_LEN`] hexadecimal characters of the full SHA.
    pub(crate) short: String,
    /// Metadata source used to resolve [`Self::short`].
    pub(crate) source: CommitSource,
}

/// Returns release-provided commit metadata first, falling back to git.
pub(crate) fn build_commit(manifest_dir: &Path) -> Option<BuildCommit> {
    if let Some(short) = release_commit_short() {
        return Some(BuildCommit {
            short,
            source: CommitSource::ReleaseEnv,
        });
    }

    current_commit_short(manifest_dir).map(|short| BuildCommit {
        short,
        source: CommitSource::Git,
    })
}

fn release_commit_short() -> Option<String> {
    let value = std::env::var("RENDERIDE_RELEASE_COMMIT").ok()?;
    full_commit_to_short(value.trim())
}

/// Returns the first [`COMMIT_HASH_LEN`] characters of `HEAD`'s commit
/// hash, or [`None`] if git is unavailable or any step fails.
///
/// Never errors; callers treat [`None`] as "omit the suffix" so a missing
/// or broken git environment falls back to the bare `Renderide <version>`
/// identifier.
pub(crate) fn current_commit_short(manifest_dir: &Path) -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(manifest_dir)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let hash = String::from_utf8(output.stdout).ok()?;
    full_commit_to_short(hash.trim())
}

/// Returns the renderer-visible short SHA for a valid full Git commit hash.
pub(crate) fn full_commit_to_short(hash: &str) -> Option<String> {
    if hash.len() != FULL_COMMIT_HASH_LEN || !hash.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    Some(hash[..COMMIT_HASH_LEN].to_ascii_lowercase())
}

/// Emits `cargo:rerun-if-changed=...` directives for the resolved git
/// directory's `HEAD`, the current branch ref (if `HEAD` is symbolic), and
/// `packed-refs`.
///
/// Resolves the real git dir via `git rev-parse --git-dir` so a `.git`
/// file (the worktree pointer used when this crate is checked out under
/// `worktrees/<name>/Renderide/`) is followed correctly. Silently no-ops
/// on any failure.
pub(crate) fn emit_rerun_if_changed(manifest_dir: &Path) {
    let _ = (|| -> Option<()> {
        let output = Command::new("git")
            .args(["rev-parse", "--git-dir"])
            .current_dir(manifest_dir)
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let raw = String::from_utf8(output.stdout).ok()?;
        let git_dir = PathBuf::from(raw.trim());
        let git_dir = if git_dir.is_absolute() {
            git_dir
        } else {
            manifest_dir.join(git_dir)
        };

        let head = git_dir.join("HEAD");
        println!("cargo:rerun-if-changed={}", head.display());
        println!(
            "cargo:rerun-if-changed={}",
            git_dir.join("packed-refs").display()
        );

        let head_contents = std::fs::read_to_string(&head).ok()?;
        let ref_path = head_contents.strip_prefix("ref: ")?.trim();
        println!(
            "cargo:rerun-if-changed={}",
            git_dir.join(ref_path).display()
        );
        Some(())
    })();
}
