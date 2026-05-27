//! Per-user updater state for skipped release prompts.

use std::fs;
use std::io;
use std::path::PathBuf;

use super::{STATE_FILE, UpdateError};

/// Reads the persisted release tag that the user chose to skip.
pub(super) fn skipped_release_tag() -> Option<String> {
    let path = state_path()?;
    match fs::read_to_string(&path) {
        Ok(contents) => skipped_release_tag_from_contents(&contents),
        Err(e) if e.kind() == io::ErrorKind::NotFound => None,
        Err(e) => {
            logger::warn!("Could not read updater state {}: {e}", path.display());
            None
        }
    }
}

/// Persists that a release tag should not be prompted again.
pub(super) fn persist_skipped_release(tag: &str) -> Result<(), UpdateError> {
    let Some(path) = state_path() else {
        logger::warn!("Could not resolve per-user updater state path.");
        return Ok(());
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| UpdateError::Io {
            context: format!("create updater state directory {}", parent.display()),
            source,
        })?;
    }
    fs::write(&path, format!("skip_release_tag={tag}\n")).map_err(|source| UpdateError::Io {
        context: format!("write updater state {}", path.display()),
        source,
    })
}

/// Resolves the per-user updater state file path.
fn state_path() -> Option<PathBuf> {
    directories::BaseDirs::new().map(|dirs| {
        dirs.config_dir()
            .join("Renderide")
            .join("updater")
            .join(STATE_FILE)
    })
}

/// Parses the skipped release tag from state file contents.
pub(super) fn skipped_release_tag_from_contents(contents: &str) -> Option<String> {
    contents.lines().find_map(|line| {
        line.strip_prefix("skip_release_tag=")
            .filter(|tag| !tag.trim().is_empty())
            .map(|tag| tag.trim().to_owned())
    })
}

#[cfg(test)]
mod tests {
    use super::skipped_release_tag_from_contents;

    #[test]
    fn skipped_release_state_parses_tag() {
        assert_eq!(
            skipped_release_tag_from_contents("skip_release_tag=nightly-1\n"),
            Some("nightly-1".to_owned())
        );
        assert_eq!(skipped_release_tag_from_contents("other=value\n"), None);
    }
}
