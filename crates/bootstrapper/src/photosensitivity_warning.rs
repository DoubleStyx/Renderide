//! Startup photosensitivity warning and per-user suppression state.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use toml_edit::{DocumentMut, Item, Table, value};

/// Title used by the startup photosensitivity warning dialog.
pub const PHOTOSENSITIVITY_WARNING_TITLE: &str = "PHOTOSENSITIVITY WARNING";
/// Body text used by the startup photosensitivity warning dialog.
pub const PHOTOSENSITIVITY_WARNING_MESSAGE: &str = concat!(
    "Renderide is experimental and may have visual bugs that are more severe or unexpected than ",
    "Resonite's default Unity renderer, including flicker, flashing frames, incorrect brightness ",
    "or contrast, broken post-processing, and rapidly changing patterns.\n\n",
    "These renderer artifacts, as well as user-created content, can trigger seizures or other ",
    "symptoms in people with photosensitive epilepsy or related sensitivities.\n\n",
    "Stop using Renderide immediately and move away from the display if you feel dizzy, ",
    "disoriented, nauseated, experience eye discomfort, or notice involuntary movement or vision ",
    "changes."
);

const APPLICATION_DIR: &str = "Renderide";
const STATE_FILE: &str = "warnings.toml";
const PHOTOSENSITIVITY_SECTION: &str = "photosensitivity";
const NEVER_SHOW_AGAIN_KEY: &str = "never_show_again";

/// User choice returned by the startup photosensitivity warning dialog.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhotosensitivityWarningChoice {
    /// Close the dialog for this launch only.
    Close,
    /// Close the dialog and suppress it for future launches by this OS user.
    NeverShowAgain,
}

/// Shows the startup photosensitivity warning when it has not been suppressed.
///
/// The `prompt` callback is supplied by the bootstrapper binary so the library stays free of
/// native dialog dependencies.
pub fn run_startup_photosensitivity_warning<P>(prompt: P)
where
    P: FnOnce() -> PhotosensitivityWarningChoice,
{
    if !should_prompt_warning() || warning_never_show_again() {
        return;
    }

    match prompt() {
        PhotosensitivityWarningChoice::Close => {}
        PhotosensitivityWarningChoice::NeverShowAgain => {
            if let Err(e) = persist_never_show_again() {
                logger::warn!("Could not persist photosensitivity warning preference: {e}");
            }
        }
    }
}

fn should_prompt_warning() -> bool {
    if std::env::var("CI").is_ok() {
        logger::info!("Skipping photosensitivity warning: CI is set.");
        return false;
    }
    if !graphical_session_available() {
        logger::warn!("Skipping photosensitivity warning: no graphical session is available.");
        return false;
    }
    true
}

fn graphical_session_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        let has_x11 = std::env::var_os("DISPLAY").is_some_and(|v| !v.is_empty());
        let has_wayland = std::env::var_os("WAYLAND_DISPLAY").is_some_and(|v| !v.is_empty());
        has_x11 || has_wayland
    }
    #[cfg(not(target_os = "linux"))]
    {
        true
    }
}

fn warning_never_show_again() -> bool {
    let Some(path) = state_path() else {
        logger::warn!("Could not resolve per-user warning state path.");
        return false;
    };
    warning_never_show_again_at(&path)
}

fn warning_never_show_again_at(path: &Path) -> bool {
    match fs::read_to_string(path) {
        Ok(contents) => match never_show_again_from_contents(&contents) {
            Ok(value) => value,
            Err(e) => {
                logger::warn!("Could not parse warning state {}: {e}", path.display());
                false
            }
        },
        Err(e) if e.kind() == io::ErrorKind::NotFound => false,
        Err(e) => {
            logger::warn!("Could not read warning state {}: {e}", path.display());
            false
        }
    }
}

fn persist_never_show_again() -> io::Result<()> {
    let Some(path) = state_path() else {
        logger::warn!("Could not resolve per-user warning state path.");
        return Ok(());
    };
    persist_never_show_again_at(&path)
}

fn persist_never_show_again_at(path: &Path) -> io::Result<()> {
    let mut document = read_state_document(path)?;
    let Some(photosensitivity) =
        get_or_create_table(document.as_table_mut(), PHOTOSENSITIVITY_SECTION)
    else {
        document[PHOTOSENSITIVITY_SECTION] = Item::Table(Table::new());
        let Some(photosensitivity) = document[PHOTOSENSITIVITY_SECTION].as_table_mut() else {
            return atomic_write_toml(path, &document.to_string());
        };
        photosensitivity.insert(NEVER_SHOW_AGAIN_KEY, value(true));
        return atomic_write_toml(path, &document.to_string());
    };
    photosensitivity.insert(NEVER_SHOW_AGAIN_KEY, value(true));
    atomic_write_toml(path, &document.to_string())
}

fn read_state_document(path: &Path) -> io::Result<DocumentMut> {
    match fs::read_to_string(path) {
        Ok(contents) => match contents.parse::<DocumentMut>() {
            Ok(document) => Ok(document),
            Err(e) => {
                logger::warn!(
                    "Could not parse warning state {}; replacing it: {e}",
                    path.display()
                );
                Ok(DocumentMut::new())
            }
        },
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(DocumentMut::new()),
        Err(e) => Err(e),
    }
}

fn never_show_again_from_contents(contents: &str) -> Result<bool, toml_edit::TomlError> {
    let document = contents.parse::<DocumentMut>()?;
    Ok(document
        .get(PHOTOSENSITIVITY_SECTION)
        .and_then(Item::as_table)
        .and_then(|table| table.get(NEVER_SHOW_AGAIN_KEY))
        .and_then(Item::as_bool)
        .unwrap_or(false))
}

fn get_or_create_table<'a>(table: &'a mut Table, key: &str) -> Option<&'a mut Table> {
    table
        .entry(key)
        .or_insert_with(|| Item::Table(Table::new()))
        .as_table_mut()
}

fn state_path() -> Option<PathBuf> {
    directories::BaseDirs::new()
        .map(|dirs| dirs.config_dir().join(APPLICATION_DIR).join(STATE_FILE))
}

fn atomic_write_toml(path: &Path, contents: &str) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(STATE_FILE);
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let tmp = parent.join(format!(".{file_name}.tmp"));
    fs::write(&tmp, contents.as_bytes())?;
    fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        NEVER_SHOW_AGAIN_KEY, PHOTOSENSITIVITY_SECTION, never_show_again_from_contents,
        persist_never_show_again_at, warning_never_show_again_at,
    };

    #[test]
    fn missing_state_does_not_suppress_warning() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("warnings.toml");

        assert!(!warning_never_show_again_at(&path));
    }

    #[test]
    fn missing_key_does_not_suppress_warning() {
        assert!(!never_show_again_from_contents("[photosensitivity]\n").expect("parse"));
    }

    #[test]
    fn true_key_suppresses_warning() {
        assert!(
            never_show_again_from_contents("[photosensitivity]\nnever_show_again = true\n")
                .expect("parse")
        );
    }

    #[test]
    fn invalid_toml_does_not_suppress_warning() {
        assert!(never_show_again_from_contents("[photosensitivity").is_err());
    }

    #[test]
    fn persist_never_show_again_writes_state() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("warnings.toml");

        persist_never_show_again_at(&path).expect("persist");

        assert!(warning_never_show_again_at(&path));
        let text = std::fs::read_to_string(path).expect("read");
        assert!(text.contains(&format!("[{PHOTOSENSITIVITY_SECTION}]")));
        assert!(text.contains(&format!("{NEVER_SHOW_AGAIN_KEY} = true")));
    }

    #[test]
    fn persist_never_show_again_preserves_unrelated_keys() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("warnings.toml");
        std::fs::write(
            &path,
            r#"
[updater]
future_key = "keep"

[photosensitivity]
future_warning_key = "keep"
"#,
        )
        .expect("write fixture");

        persist_never_show_again_at(&path).expect("persist");

        let text = std::fs::read_to_string(path).expect("read");
        assert!(text.contains("[updater]"), "got:\n{text}");
        assert!(text.contains("future_key = \"keep\""), "got:\n{text}");
        assert!(
            text.contains("future_warning_key = \"keep\""),
            "got:\n{text}"
        );
        assert!(text.contains("never_show_again = true"), "got:\n{text}");
    }
}
