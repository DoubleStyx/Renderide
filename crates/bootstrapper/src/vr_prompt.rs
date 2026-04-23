//! Host argv augmentation so FrooxEngine receives `-Screen` or `-Device <HeadOutputDevice>`
//! before process startup, matching FrooxEngine `LaunchOptions` handling of `-screen` / `-device`.
//!
//! The renderer learns the effective device from IPC `RendererInitData` after connect.

use std::env;

/// When set, the bootstrapper does not show the desktop vs VR dialog (automation / headless).
pub const ENV_SKIP_VR_DIALOG: &str = "RENDERIDE_SKIP_VR_DIALOG";

/// Strips a leading `-` (if present) and lowercases, matching FrooxEngine's normalized argv tokens.
///
/// Used so `-Screen`, `-screen`, and `Screen` are treated consistently when scanning for output flags.
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

/// Labels used for the custom dialog buttons; also returned by `rfd` as the
/// `MessageDialogResult::Custom(label)` payload, so they double as match keys.
const VR_BUTTON_LABEL: &str = "VR";
const DESKTOP_BUTTON_LABEL: &str = "Desktop";
const CANCEL_BUTTON_LABEL: &str = "Cancel";

/// Desktop vs VR choice: **VR** → `-Device SteamVR`, **Desktop** → `-Screen`.
///
/// Returns [`None`] when the user clicks **Cancel** or otherwise dismisses
/// the dialog; callers treat this as a request to abort the launch.
pub fn prompt_desktop_or_vr() -> Option<bool> {
    let res = rfd::MessageDialog::new()
        .set_title("Renderide")
        .set_description("Launch Resonite in VR or desktop mode?")
        .set_buttons(rfd::MessageButtons::YesNoCancelCustom(
            VR_BUTTON_LABEL.into(),
            DESKTOP_BUTTON_LABEL.into(),
            CANCEL_BUTTON_LABEL.into(),
        ))
        .show();
    match res {
        // Native backends that honor custom labels return them verbatim.
        rfd::MessageDialogResult::Custom(label) if label == VR_BUTTON_LABEL => Some(true),
        rfd::MessageDialogResult::Custom(label) if label == DESKTOP_BUTTON_LABEL => Some(false),
        _ => None,
    }
}

/// Prepends `-Device SteamVR` or `-Screen` to the Host argv list.
pub fn apply_host_vr_choice(host_args: Vec<String>, vr: bool) -> Vec<String> {
    if vr {
        let mut out = Vec::with_capacity(host_args.len().saturating_add(2));
        out.push("-Device".into());
        out.push("SteamVR".into());
        out.extend(host_args);
        out
    } else {
        let mut out = Vec::with_capacity(host_args.len().saturating_add(1));
        out.push("-Screen".into());
        out.extend(host_args);
        out
    }
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

    #[test]
    fn normalized_flag_token_trims_and_strips_leading_dash() {
        assert_eq!(super::normalized_flag_token("  -Screen  "), "screen");
        assert_eq!(super::normalized_flag_token("Device"), "device");
        // Only one leading `-` is stripped; `--` prefixes remain normalized for the remainder.
        assert_eq!(super::normalized_flag_token("--Foo"), "-foo");
    }

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn should_prompt_false_when_ci_set() {
        let _g = ENV_LOCK.lock().expect("env lock");
        std::env::set_var("CI", "1");
        assert!(!should_prompt_vr_dialog(&[]));
        std::env::remove_var("CI");
    }

    #[test]
    fn should_prompt_false_when_skip_env_set() {
        let _g = ENV_LOCK.lock().expect("env lock");
        std::env::set_var(ENV_SKIP_VR_DIALOG, "1");
        assert!(!should_prompt_vr_dialog(&[]));
        std::env::remove_var(ENV_SKIP_VR_DIALOG);
    }

    #[test]
    fn should_prompt_false_when_device_explicit() {
        let _g = ENV_LOCK.lock().expect("env lock");
        std::env::remove_var("CI");
        std::env::remove_var(ENV_SKIP_VR_DIALOG);
        assert!(!should_prompt_vr_dialog(&["-Device".into(), "x".into()]));
    }

    #[test]
    fn should_prompt_true_when_unset_and_no_explicit_device() {
        let _g = ENV_LOCK.lock().expect("env lock");
        std::env::remove_var("CI");
        std::env::remove_var(ENV_SKIP_VR_DIALOG);
        assert!(should_prompt_vr_dialog(&[]));
    }
}
