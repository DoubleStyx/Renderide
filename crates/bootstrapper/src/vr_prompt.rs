//! Host argv augmentation so `FrooxEngine` receives `-Screen` or `-Device <HeadOutputDevice>`
//! before process startup, matching `FrooxEngine` `LaunchOptions` handling of `-screen` / `-device`.
//!
//! The renderer learns the effective device from IPC `RendererInitData` after connect.
//!
//! # Hang protection on Linux
//!
//! `rfd::MessageDialog::show()` uses a GTK3 or XDG portal backend on Linux and can block
//! indefinitely when the backend cannot surface a window (headless shell, missing GTK runtime,
//! broken portal). To keep the bootstrapper diagnosable, [`prompt_desktop_or_vr`] is only
//! reached after the logger has been initialized; it emits one log line before and after the
//! blocking `show()` call, and a watchdog thread aborts the process with an actionable error
//! (pointing at [`ENV_SKIP_VR_DIALOG`]) if the dialog does not return within
//! [`DIALOG_WATCHDOG_TIMEOUT`]. When neither `DISPLAY` nor `WAYLAND_DISPLAY` is set on Linux,
//! [`should_prompt_vr_dialog`] returns `false` so the dialog is skipped entirely.

use std::env;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// When set, the bootstrapper does not show the desktop vs VR dialog (automation / headless).
pub(crate) const ENV_SKIP_VR_DIALOG: &str = "RENDERIDE_SKIP_VR_DIALOG";

/// Maximum time [`prompt_desktop_or_vr`] waits for `rfd::MessageDialog::show()` to return
/// before the watchdog thread aborts the process with an actionable log line.
const DIALOG_WATCHDOG_TIMEOUT: Duration = Duration::from_secs(60);

/// Strips a leading `-` (if present) and lowercases, matching `FrooxEngine`'s normalized argv tokens.
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

/// Returns `true` when `args` already specify `FrooxEngine` output via `-Screen` or `-Device …`.
///
/// Any `-Device` token counts as explicit (even if the following value is invalid for the host).
pub(crate) fn host_args_have_explicit_output_device(args: &[String]) -> bool {
    for a in args {
        let n = normalized_flag_token(a);
        if n == "screen" || n == "device" {
            return true;
        }
    }
    false
}

/// On Linux, returns `true` when at least one of `DISPLAY` (X11) or `WAYLAND_DISPLAY` (Wayland)
/// is set in the environment. On other platforms returns `true` unconditionally.
///
/// Used by [`should_prompt_vr_dialog`] to skip the `rfd` popup on headless / TTY launches where
/// GTK cannot open a window and `show()` would block forever without writing any log.
fn linux_graphical_session_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        let has_x11 = env::var_os("DISPLAY").is_some_and(|v| !v.is_empty());
        let has_wayland = env::var_os("WAYLAND_DISPLAY").is_some_and(|v| !v.is_empty());
        has_x11 || has_wayland
    }
    #[cfg(not(target_os = "linux"))]
    {
        true
    }
}

/// Whether the optional Yes/No dialog should run before spawning the Host.
///
/// Returns `false` when explicit output flags are already present, `CI` is set,
/// [`ENV_SKIP_VR_DIALOG`] is set, or (on Linux) neither `DISPLAY` nor `WAYLAND_DISPLAY`
/// is set — the latter case is logged at `warn` level so headless launches are not silent.
pub(crate) fn should_prompt_vr_dialog(host_args: &[String]) -> bool {
    if host_args_have_explicit_output_device(host_args) {
        return false;
    }
    if env::var("CI").is_ok() {
        return false;
    }
    if env::var(ENV_SKIP_VR_DIALOG).is_ok() {
        return false;
    }
    if !linux_graphical_session_available() {
        logger::warn!(
            "Skipping desktop/VR dialog: neither DISPLAY nor WAYLAND_DISPLAY is set. \
             Pass -Screen or -Device SteamVR, or set {ENV_SKIP_VR_DIALOG}=1 to silence this warning.",
        );
        return false;
    }
    true
}

/// Labels used for the custom dialog buttons; also returned by `rfd` as the
/// `MessageDialogResult::Custom(label)` payload, so they double as match keys.
const VR_BUTTON_LABEL: &str = "VR";
/// Desktop-mode button label. Also the `MessageDialogResult::Custom` payload returned by `rfd`.
const DESKTOP_BUTTON_LABEL: &str = "Desktop";
/// Cancel button label. Also the `MessageDialogResult::Custom` payload returned by `rfd`.
const CANCEL_BUTTON_LABEL: &str = "Cancel";

/// Desktop vs VR choice: **VR** → `-Device SteamVR`, **Desktop** → `-Screen`.
///
/// Returns [`None`] when the user clicks **Cancel** or otherwise dismisses
/// the dialog; callers treat this as a request to abort the launch.
///
/// Requires the global logger to be initialized before invocation so that the
/// before/after log lines and the watchdog abort message reach disk. Installs a
/// short-lived watchdog thread that aborts the process via [`std::process::exit`]
/// with a pointer to [`ENV_SKIP_VR_DIALOG`] if `rfd::MessageDialog::show()` has
/// not returned after [`DIALOG_WATCHDOG_TIMEOUT`].
pub(crate) fn prompt_desktop_or_vr() -> Option<bool> {
    let completed = Arc::new(AtomicBool::new(false));
    spawn_dialog_watchdog(Arc::clone(&completed));

    logger::info!("Showing desktop/VR selection dialog via rfd backend.");
    let res = rfd::MessageDialog::new()
        .set_title("Renderide")
        .set_description("Launch Resonite in VR or desktop mode?")
        .set_buttons(rfd::MessageButtons::YesNoCancelCustom(
            VR_BUTTON_LABEL.into(),
            DESKTOP_BUTTON_LABEL.into(),
            CANCEL_BUTTON_LABEL.into(),
        ))
        .show();
    completed.store(true, Ordering::SeqCst);

    match res {
        // Native backends that honor custom labels return them verbatim.
        rfd::MessageDialogResult::Custom(label) if label == VR_BUTTON_LABEL => {
            logger::info!("Desktop/VR dialog returned: VR.");
            Some(true)
        }
        rfd::MessageDialogResult::Custom(label) if label == DESKTOP_BUTTON_LABEL => {
            logger::info!("Desktop/VR dialog returned: Desktop.");
            Some(false)
        }
        other => {
            logger::info!("Desktop/VR dialog cancelled or dismissed: {other:?}.");
            None
        }
    }
}

/// Spawns a detached watchdog thread that logs an error and exits the process
/// if `completed` is still `false` after [`DIALOG_WATCHDOG_TIMEOUT`].
///
/// The dialog thread flips `completed` to `true` once `rfd`'s `show()` returns;
/// the watchdog checks the flag after sleeping and quietly exits its own
/// `thread::spawn` closure if the dialog finished in time.
fn spawn_dialog_watchdog(completed: Arc<AtomicBool>) {
    let spawn_result = thread::Builder::new()
        .name("rfd-dialog-watchdog".into())
        .spawn(move || {
            thread::sleep(DIALOG_WATCHDOG_TIMEOUT);
            if completed.load(Ordering::SeqCst) {
                return;
            }
            logger::error!(
                "Desktop/VR dialog did not return within {secs}s. \
                 The rfd GTK/XDG portal backend appears to be hung. \
                 Set {ENV_SKIP_VR_DIALOG}=1 (or pass -Screen / -Device SteamVR) to bypass the dialog.",
                secs = DIALOG_WATCHDOG_TIMEOUT.as_secs(),
            );
            logger::flush();
            std::process::exit(1);
        });
    if let Err(e) = spawn_result {
        // If the OS cannot spawn a watchdog thread, the dialog can still hang silently — log and
        // continue rather than aborting; the dialog's own behavior is unchanged.
        logger::warn!(
            "Could not spawn rfd dialog watchdog thread: {e}. Dialog timeout is disabled."
        );
    }
}

/// Prepends `-Device SteamVR` or `-Screen` to the Host argv list.
pub(crate) fn apply_host_vr_choice(host_args: Vec<String>, vr: bool) -> Vec<String> {
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
        assert_eq!(normalized_flag_token("  -Screen  "), "screen");
        assert_eq!(normalized_flag_token("Device"), "device");
        // Only one leading `-` is stripped; `--` prefixes remain normalized for the remainder.
        assert_eq!(normalized_flag_token("--Foo"), "-foo");
    }

    /// Serializes env-var-mutating tests in this module so parallel runs do not race on
    /// `DISPLAY` / `CI` / [`ENV_SKIP_VR_DIALOG`] state.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Snapshot of the env vars this module's tests mutate, plus a restore guard.
    ///
    /// The `Drop` impl restores the captured values on test exit (including panics),
    /// so individual tests never have to remember to clean up behind themselves.
    struct EnvSnapshot {
        /// Captured `CI` value at snapshot time.
        ci: Option<std::ffi::OsString>,
        /// Captured [`ENV_SKIP_VR_DIALOG`] value at snapshot time.
        skip: Option<std::ffi::OsString>,
        /// Captured `DISPLAY` value at snapshot time.
        display: Option<std::ffi::OsString>,
        /// Captured `WAYLAND_DISPLAY` value at snapshot time.
        wayland: Option<std::ffi::OsString>,
    }

    impl EnvSnapshot {
        /// Captures the current values of the display / bypass env vars for later restore.
        fn capture() -> Self {
            Self {
                ci: env::var_os("CI"),
                skip: env::var_os(ENV_SKIP_VR_DIALOG),
                display: env::var_os("DISPLAY"),
                wayland: env::var_os("WAYLAND_DISPLAY"),
            }
        }
    }

    impl Drop for EnvSnapshot {
        fn drop(&mut self) {
            restore("CI", self.ci.take());
            restore(ENV_SKIP_VR_DIALOG, self.skip.take());
            restore("DISPLAY", self.display.take());
            restore("WAYLAND_DISPLAY", self.wayland.take());
        }
    }

    /// Restores a single env var to `value`, or removes it when `value` is [`None`].
    fn restore(key: &str, value: Option<std::ffi::OsString>) {
        if let Some(v) = value {
            env::set_var(key, v);
        } else {
            env::remove_var(key);
        }
    }

    #[test]
    fn should_prompt_false_when_ci_set() {
        let _g = ENV_LOCK.lock().expect("env lock");
        let _snap = EnvSnapshot::capture();
        env::set_var("CI", "1");
        assert!(!should_prompt_vr_dialog(&[]));
    }

    #[test]
    fn should_prompt_false_when_skip_env_set() {
        let _g = ENV_LOCK.lock().expect("env lock");
        let _snap = EnvSnapshot::capture();
        env::set_var(ENV_SKIP_VR_DIALOG, "1");
        assert!(!should_prompt_vr_dialog(&[]));
    }

    #[test]
    fn should_prompt_false_when_device_explicit() {
        let _g = ENV_LOCK.lock().expect("env lock");
        let _snap = EnvSnapshot::capture();
        env::remove_var("CI");
        env::remove_var(ENV_SKIP_VR_DIALOG);
        assert!(!should_prompt_vr_dialog(&["-Device".into(), "x".into()]));
    }

    #[test]
    fn should_prompt_true_when_unset_and_display_present() {
        let _g = ENV_LOCK.lock().expect("env lock");
        let _snap = EnvSnapshot::capture();
        env::remove_var("CI");
        env::remove_var(ENV_SKIP_VR_DIALOG);
        env::set_var("DISPLAY", ":0");
        env::remove_var("WAYLAND_DISPLAY");
        assert!(should_prompt_vr_dialog(&[]));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn should_prompt_false_on_linux_when_no_display() {
        let _g = ENV_LOCK.lock().expect("env lock");
        let _snap = EnvSnapshot::capture();
        env::remove_var("CI");
        env::remove_var(ENV_SKIP_VR_DIALOG);
        env::remove_var("DISPLAY");
        env::remove_var("WAYLAND_DISPLAY");
        assert!(!should_prompt_vr_dialog(&[]));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn should_prompt_true_on_linux_with_wayland_only() {
        let _g = ENV_LOCK.lock().expect("env lock");
        let _snap = EnvSnapshot::capture();
        env::remove_var("CI");
        env::remove_var(ENV_SKIP_VR_DIALOG);
        env::remove_var("DISPLAY");
        env::set_var("WAYLAND_DISPLAY", "wayland-0");
        assert!(should_prompt_vr_dialog(&[]));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn should_prompt_false_on_linux_when_display_is_empty_string() {
        let _g = ENV_LOCK.lock().expect("env lock");
        let _snap = EnvSnapshot::capture();
        env::remove_var("CI");
        env::remove_var(ENV_SKIP_VR_DIALOG);
        env::set_var("DISPLAY", "");
        env::set_var("WAYLAND_DISPLAY", "");
        assert!(!should_prompt_vr_dialog(&[]));
    }
}
