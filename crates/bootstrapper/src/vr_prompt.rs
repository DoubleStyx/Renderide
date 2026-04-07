//! Host argv augmentation so FrooxEngine receives `-Screen` or `-Device <HeadOutputDevice>`
//! before process startup, matching FrooxEngine `LaunchOptions` handling of `-screen` / `-device`.
//!
//! The renderer learns the effective device from IPC `RendererInitData` after connect.

use std::env;

/// When set, the bootstrapper does not show the desktop vs VR dialog (automation / headless).
pub const ENV_SKIP_VR_DIALOG: &str = "RENDERIDE_SKIP_VR_DIALOG";

/// Strips a leading `-` (if present) and lowercases, matching FrooxEngine’s normalized argv tokens.
fn normalized_flag_token(arg: &str) -> String {
    let s = arg.trim();
    if let Some(rest) = s.strip_prefix('-') {
        rest.to_ascii_lowercase()
    } else {
        s.to_ascii_lowercase()
    }
}

/// Returns `true` when `args` already specify FrooxEngine output via `-Screen` or `-Device …`.
///
/// Any `-Device` token counts as explicit (even if the following value is invalid for the host).
pub fn host_args_have_explicit_output_device(args: &[String]) -> bool {
    for a in args {
        let n = normalized_flag_token(a);
        if n == "screen" || n == "device" {
            return true;
        }
    }
    false
}

/// Whether the optional Yes/No dialog should run before spawning the Host.
pub fn should_prompt_vr_dialog(host_args: &[String]) -> bool {
    if host_args_have_explicit_output_device(host_args) {
        return false;
    }
    if env::var("CI").is_ok() {
        return false;
    }
    if env::var(ENV_SKIP_VR_DIALOG).is_ok() {
        return false;
    }
    true
}

/// Desktop vs VR choice: **Yes** → `-Device SteamVR`, **No** → `-Screen`.
///
/// Returns [`None`] when the dialog is dismissed without a choice.
pub fn prompt_desktop_or_vr() -> Option<bool> {
    let res = rfd::MessageDialog::new()
        .set_title("Renderide")
        .set_description(
            "Launch the Host using a VR headset (SteamVR-style OpenXR) instead of desktop?\n\n\
             Yes = VR (-Device SteamVR)\n\
             No = Desktop (-Screen)",
        )
        .set_buttons(rfd::MessageButtons::YesNo)
        .show();
    match res {
        rfd::MessageDialogResult::Yes => Some(true),
        rfd::MessageDialogResult::No => Some(false),
        _ => None,
    }
}

/// Prepends `-Device SteamVR` or `-Screen` to the Host argv list.
pub fn apply_host_vr_choice(mut host_args: Vec<String>, vr: bool) -> Vec<String> {
    if vr {
        host_args.insert(0, "SteamVR".into());
        host_args.insert(0, "-Device".into());
    } else {
        host_args.insert(0, "-Screen".into());
    }
    host_args
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_screen_flags() {
        assert!(host_args_have_explicit_output_device(&["-Screen".into()]));
        assert!(host_args_have_explicit_output_device(&["-screen".into()]));
    }

    #[test]
    fn detects_device_flags() {
        assert!(host_args_have_explicit_output_device(&[
            "-Device".into(),
            "SteamVR".into()
        ]));
    }

    #[test]
    fn no_false_positives() {
        assert!(!host_args_have_explicit_output_device(&[
            "-Invisible".into(),
            "-Data".into()
        ]));
    }

    #[test]
    fn apply_vr_prepends_device_steamvr() {
        let out = apply_host_vr_choice(vec!["-Invisible".into()], true);
        assert_eq!(out, vec!["-Device", "SteamVR", "-Invisible"]);
    }

    #[test]
    fn apply_desktop_prepends_screen() {
        let out = apply_host_vr_choice(vec![], false);
        assert_eq!(out, vec!["-Screen"]);
    }
}
