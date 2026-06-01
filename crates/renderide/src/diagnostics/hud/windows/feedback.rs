//! Compact feedback panel with links to issue reporting and discussion channels.

use std::io;
use std::process::Command;

use imgui::WindowFlags;

use super::super::layout::{self, Viewport, WindowSlot};
use super::super::state::HudUiState;
use super::super::view::HudWindow;

const CONTENT_WIDTH: f32 = 268.0;
const GITHUB_ISSUES_URL: &str = "https://github.com/DoubleStyx/Renderide/issues";
const TELEGRAM_URL: &str = "https://t.me/DoubleStyx";
const DISCORD_RENDERING_DISCUSSION_URL: &str =
    "https://discord.com/channels/1040316820650991766/1156348246973751487";

/// Tiny always-visible feedback panel anchored in the upper-right corner.
pub struct FeedbackWindow;

impl HudWindow for FeedbackWindow {
    type Data<'a> = ();
    type State = HudUiState;

    fn title(&self) -> &str {
        "Feedback / Bug Report"
    }

    fn anchor(&self, viewport: Viewport) -> WindowSlot {
        layout::feedback_panel_slot(viewport)
    }

    fn flags(&self) -> WindowFlags {
        WindowFlags::ALWAYS_AUTO_RESIZE | WindowFlags::NO_FOCUS_ON_APPEARING | WindowFlags::NO_NAV
    }

    fn bg_alpha(&self) -> f32 {
        0.82
    }

    fn body(&self, ui: &imgui::Ui, _data: Self::Data<'_>, _state: &mut Self::State) {
        ui.dummy([CONTENT_WIDTH, 0.0]);
        link_button(ui, "Issues", GITHUB_ISSUES_URL);
        ui.same_line();
        link_button(ui, "Telegram", TELEGRAM_URL);
        ui.same_line();
        link_button(ui, "Discord", DISCORD_RENDERING_DISCUSSION_URL);
    }
}

fn link_button(ui: &imgui::Ui, label: &str, url: &'static str) {
    if ui.small_button(label)
        && let Err(e) = open_feedback_url(url)
    {
        logger::warn!("Failed to open feedback link: {e}");
    }
    if ui.is_item_hovered() {
        ui.tooltip_text(url);
    }
}

#[derive(Debug, thiserror::Error)]
enum OpenFeedbackUrlError {
    #[error("failed to spawn {program} for {url}: {source}")]
    Spawn {
        program: &'static str,
        url: &'static str,
        #[source]
        source: io::Error,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct UrlOpenCommand {
    program: &'static str,
    leading_args: &'static [&'static str],
    url: &'static str,
}

fn url_open_command(url: &'static str) -> UrlOpenCommand {
    if cfg!(target_os = "windows") {
        UrlOpenCommand {
            program: "cmd",
            leading_args: &["/C", "start", ""],
            url,
        }
    } else if cfg!(target_os = "macos") {
        UrlOpenCommand {
            program: "open",
            leading_args: &[],
            url,
        }
    } else {
        UrlOpenCommand {
            program: "xdg-open",
            leading_args: &[],
            url,
        }
    }
}

fn open_feedback_url(url: &'static str) -> Result<(), OpenFeedbackUrlError> {
    let spec = url_open_command(url);
    let mut command = Command::new(spec.program);
    command.args(spec.leading_args);
    command.arg(spec.url);
    command
        .spawn()
        .map(|_| ())
        .map_err(|source| OpenFeedbackUrlError::Spawn {
            program: spec.program,
            url: spec.url,
            source,
        })
}

#[cfg(test)]
mod tests {
    use super::{
        DISCORD_RENDERING_DISCUSSION_URL, GITHUB_ISSUES_URL, TELEGRAM_URL, url_open_command,
    };

    #[test]
    fn feedback_links_match_readme_targets() {
        assert_eq!(
            GITHUB_ISSUES_URL,
            "https://github.com/DoubleStyx/Renderide/issues"
        );
        assert_eq!(TELEGRAM_URL, "https://t.me/DoubleStyx");
        assert_eq!(
            DISCORD_RENDERING_DISCUSSION_URL,
            "https://discord.com/channels/1040316820650991766/1156348246973751487"
        );
    }

    #[test]
    fn url_open_command_matches_platform() {
        let spec = url_open_command(GITHUB_ISSUES_URL);
        let expected_program = if cfg!(target_os = "windows") {
            "cmd"
        } else if cfg!(target_os = "macos") {
            "open"
        } else {
            "xdg-open"
        };
        let expected_args: &[&str] = if cfg!(target_os = "windows") {
            &["/C", "start", ""]
        } else {
            &[]
        };

        assert_eq!(spec.program, expected_program);
        assert_eq!(spec.leading_args, expected_args);
        assert_eq!(spec.url, GITHUB_ISSUES_URL);
    }
}
