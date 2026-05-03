//! Command-line parsing and the optional desktop vs VR dialog wiring that precedes [`crate::run`].
//!
//! [`resolve_vr_choice`] takes the dialog as a callback so this library never depends on `rfd`;
//! the production callback lives in the bin-only `dialog` module
//! (`crates/bootstrapper/src/dialog.rs`). The dialog is resolved **after** the global logger has
//! been initialized so any backend hang (see `dialog` and `vr_prompt` module docs) is visible in
//! `logs/bootstrapper/*.log` rather than producing a silent "nothing happens" failure.

use std::env;

use logger::LogLevel;

use crate::vr_prompt;

/// Trims `token`, strips a single leading `-` (if present), and ASCII-lowercases the rest.
///
/// Used by argv scanning to compare a flag-shaped token against canonical lowercase forms
/// regardless of leading dashes or letter case, matching `FrooxEngine`'s normalized argv tokens.
/// Only one leading `-` is stripped -- `--log-level` normalizes to `-log-level`, not `log-level`.
pub(crate) fn normalize_flag_token(token: &str) -> String {
    let s = token.trim();
    if let Some(rest) = s.strip_prefix('-') {
        rest.to_ascii_lowercase()
    } else {
        s.to_ascii_lowercase()
    }
}

/// Parses bootstrapper args, extracting `--log-level` / `-l` for bootstrapper and Renderide.
///
/// Returns `(arguments to forward to Host, optional log level)`.
pub fn parse_args() -> (Vec<String>, Option<LogLevel>) {
    let args: Vec<String> = env::args().skip(1).collect();
    parse_host_args_tokens(&args)
}

/// Parses `args` as argv after the program name: strips `--log-level` / `-l` plus the following
/// token when present, and records the parsed [`LogLevel`] (if any).
///
/// If `--log-level` or `-l` appears without a trailing value, that flag is left in the returned
/// host list (same as ResoBoot-style forwarding).
///
/// When the flag appears multiple times, the **last** [`LogLevel::parse`] result wins (including `None` for unknown tokens).
pub fn parse_host_args_tokens(args: &[String]) -> (Vec<String>, Option<LogLevel>) {
    let mut host_args = Vec::new();
    let mut log_level = None;
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        let normalized = normalize_flag_token(arg);
        if (normalized == "-log-level" || normalized == "l") && i + 1 < args.len() {
            log_level = LogLevel::parse(&args[i + 1]);
            i += 2;
            continue;
        }
        host_args.push(arg.clone());
        i += 1;
    }
    (host_args, log_level)
}

/// Runs the desktop vs VR dialog (`prompt`) if required by
/// [`vr_prompt::should_prompt_vr_dialog`] and returns host argv augmented with the resulting
/// `-Screen` / `-Device SteamVR` flag.
///
/// `prompt` is invoked with no arguments and must return `Some(true)` for VR, `Some(false)` for
/// desktop, or [`None`] if the user cancelled. The dialog implementation is supplied by the
/// caller so this function -- and the bootstrapper library as a whole -- does not depend on
/// `rfd`; the production implementation lives in the bin-only `dialog` module
/// (`crates/bootstrapper/src/dialog.rs`). See `vr_prompt`'s module docs for why keeping `rfd`
/// out of the library is a hard requirement on Windows.
///
/// Returns [`None`] only when the dialog runs **and** `prompt` returns [`None`]; in every
/// bypass path (explicit output flag, `CI`, [`vr_prompt::ENV_SKIP_VR_DIALOG`], no Linux display)
/// the original `host_args` are returned unchanged and `prompt` is **not** called.
///
/// The caller **must** have initialized the global logger before invocation because the
/// production `prompt` emits before/after log lines and installs a watchdog that logs on timeout.
pub fn resolve_vr_choice<F>(host_args: Vec<String>, prompt: F) -> Option<Vec<String>>
where
    F: FnOnce() -> Option<bool>,
{
    if !vr_prompt::should_prompt_vr_dialog(&host_args) {
        return Some(host_args);
    }
    let vr = prompt()?;
    Some(vr_prompt::apply_host_vr_choice(host_args, vr))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn normalize_flag_token_no_leading_dash_lowercases() {
        assert_eq!(normalize_flag_token("Screen"), "screen");
    }

    #[test]
    fn normalize_flag_token_strips_single_dash() {
        assert_eq!(normalize_flag_token("-Screen"), "screen");
    }

    #[test]
    fn normalize_flag_token_strips_only_one_dash_for_double_dash() {
        assert_eq!(normalize_flag_token("--log-level"), "-log-level");
    }

    #[test]
    fn normalize_flag_token_trims_whitespace() {
        assert_eq!(normalize_flag_token("  -L  "), "l");
    }

    #[test]
    fn normalize_flag_token_empty_input() {
        assert_eq!(normalize_flag_token(""), "");
    }

    #[test]
    fn parse_host_args_tokens_empty() {
        let (host, level) = parse_host_args_tokens(&[]);
        assert!(host.is_empty());
        assert!(level.is_none());
    }

    #[test]
    fn parse_host_args_tokens_log_level_consumed() {
        let (host, level) =
            parse_host_args_tokens(&tokens(&["--log-level", "debug", "-Invisible"]));
        assert_eq!(host, vec!["-Invisible".to_string()]);
        assert_eq!(level, Some(LogLevel::Debug));
    }

    #[test]
    fn parse_host_args_tokens_short_flag_case_insensitive() {
        let (host, level) = parse_host_args_tokens(&tokens(&["-L", "trace", "x"]));
        assert_eq!(host, vec!["x".to_string()]);
        assert_eq!(level, Some(LogLevel::Trace));
    }

    #[test]
    fn parse_host_args_tokens_unknown_level_yields_none_but_consumes_pair() {
        let (host, level) = parse_host_args_tokens(&tokens(&["--log-level", "nope", "y"]));
        assert_eq!(host, vec!["y".to_string()]);
        assert!(level.is_none());
    }

    #[test]
    fn parse_host_args_tokens_trailing_log_flag_forwarded() {
        let (host, level) = parse_host_args_tokens(&tokens(&["-l"]));
        assert_eq!(host, vec!["-l".to_string()]);
        assert!(level.is_none());
    }

    #[test]
    fn parse_host_args_tokens_mid_list_flag() {
        let (host, level) = parse_host_args_tokens(&tokens(&[
            "-Invisible",
            "--log-level",
            "warn",
            "-Data",
            "x",
        ]));
        assert_eq!(
            host,
            vec![
                "-Invisible".to_string(),
                "-Data".to_string(),
                "x".to_string()
            ]
        );
        assert_eq!(level, Some(LogLevel::Warn));
    }

    #[test]
    fn parse_host_args_tokens_repeated_log_level_last_wins() {
        let (host, level) =
            parse_host_args_tokens(&tokens(&["--log-level", "debug", "-x", "-l", "error"]));
        assert_eq!(host, vec!["-x".to_string()]);
        assert_eq!(level, Some(LogLevel::Error));
    }

    #[test]
    fn parse_host_args_tokens_last_unknown_level_clears() {
        let (host, level) =
            parse_host_args_tokens(&tokens(&["--log-level", "debug", "-l", "nope"]));
        assert!(host.is_empty());
        assert!(level.is_none());
    }

    #[test]
    fn parse_host_args_tokens_mixed_l_and_long_form() {
        let (host, level) = parse_host_args_tokens(&tokens(&["-l", "info", "tail"]));
        assert_eq!(host, vec!["tail".to_string()]);
        assert_eq!(level, Some(LogLevel::Info));
    }

    #[test]
    fn parse_host_args_tokens_empty_value_after_flag_forwarded() {
        let (host, level) = parse_host_args_tokens(&tokens(&["--log-level"]));
        assert_eq!(host, vec!["--log-level".to_string()]);
        assert!(level.is_none());
    }

    /// Serializes env-mutating tests so parallel runs do not race on shared env state.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Restores a previously captured env var, removing it when `value` is [`None`].
    fn restore(key: &str, value: Option<std::ffi::OsString>) {
        if let Some(v) = value {
            // SAFETY: env mutation in test; serialized via ENV_LOCK / cargo test single-thread.
            unsafe {
                env::set_var(key, v);
            }
        } else {
            // SAFETY: env mutation in test; serialized via ENV_LOCK / cargo test single-thread.
            unsafe {
                env::remove_var(key);
            }
        }
    }

    /// Closure usable as the `prompt` argument to [`resolve_vr_choice`] in bypass-path tests:
    /// panics if invoked, asserting that the dialog must not be called.
    fn unreachable_prompt() -> Option<bool> {
        panic!("dialog prompt must not be invoked when bypass path is taken")
    }

    #[test]
    fn resolve_vr_choice_bypasses_dialog_on_skip_env() {
        let _g = ENV_LOCK.lock().expect("env lock");
        let prev_skip = env::var_os(vr_prompt::ENV_SKIP_VR_DIALOG);
        let prev_ci = env::var_os("CI");
        // SAFETY: env mutation in test; serialized via ENV_LOCK / cargo test single-thread.
        unsafe {
            env::set_var(vr_prompt::ENV_SKIP_VR_DIALOG, "1");
        }
        // SAFETY: env mutation in test; serialized via ENV_LOCK / cargo test single-thread.
        unsafe {
            env::set_var("CI", "1");
        }
        let out = resolve_vr_choice(vec!["-Invisible".to_string()], unreachable_prompt)
            .expect("bypass path must yield Some");
        assert_eq!(out, vec!["-Invisible".to_string()]);
        restore(vr_prompt::ENV_SKIP_VR_DIALOG, prev_skip);
        restore("CI", prev_ci);
    }

    #[test]
    fn resolve_vr_choice_preserves_explicit_screen_arg() {
        let _g = ENV_LOCK.lock().expect("env lock");
        let prev_skip = env::var_os(vr_prompt::ENV_SKIP_VR_DIALOG);
        let prev_ci = env::var_os("CI");
        // SAFETY: env mutation in test; serialized via ENV_LOCK / cargo test single-thread.
        unsafe {
            env::remove_var(vr_prompt::ENV_SKIP_VR_DIALOG);
        }
        // SAFETY: env mutation in test; serialized via ENV_LOCK / cargo test single-thread.
        unsafe {
            env::remove_var("CI");
        }
        let out = resolve_vr_choice(
            vec!["-Screen".to_string(), "-Invisible".to_string()],
            unreachable_prompt,
        )
        .expect("explicit output flag bypasses dialog");
        assert_eq!(out, vec!["-Screen".to_string(), "-Invisible".to_string()]);
        restore(vr_prompt::ENV_SKIP_VR_DIALOG, prev_skip);
        restore("CI", prev_ci);
    }

    /// When the dialog runs and the user cancels (`prompt` returns [`None`]), [`resolve_vr_choice`]
    /// propagates the cancellation as `None` so `main` can exit cleanly without launching the Host.
    #[test]
    fn resolve_vr_choice_returns_none_when_prompt_cancels() {
        let _g = ENV_LOCK.lock().expect("env lock");
        let prev_skip = env::var_os(vr_prompt::ENV_SKIP_VR_DIALOG);
        let prev_ci = env::var_os("CI");
        let prev_display = env::var_os("DISPLAY");
        let prev_wayland = env::var_os("WAYLAND_DISPLAY");
        // SAFETY: env mutation in test; serialized via ENV_LOCK / cargo test single-thread.
        unsafe {
            env::remove_var(vr_prompt::ENV_SKIP_VR_DIALOG);
            env::remove_var("CI");
            env::set_var("DISPLAY", ":0");
            env::remove_var("WAYLAND_DISPLAY");
        }
        let result = resolve_vr_choice(vec!["-Invisible".to_string()], || None);
        assert!(result.is_none(), "cancelled dialog must yield None");
        restore(vr_prompt::ENV_SKIP_VR_DIALOG, prev_skip);
        restore("CI", prev_ci);
        restore("DISPLAY", prev_display);
        restore("WAYLAND_DISPLAY", prev_wayland);
    }

    /// When the dialog runs and the user picks VR (`prompt` returns `Some(true)`), the Host argv is
    /// prepended with `-Device SteamVR` ahead of any forwarded tokens.
    #[test]
    fn resolve_vr_choice_applies_vr_device_when_prompt_confirms() {
        let _g = ENV_LOCK.lock().expect("env lock");
        let prev_skip = env::var_os(vr_prompt::ENV_SKIP_VR_DIALOG);
        let prev_ci = env::var_os("CI");
        let prev_display = env::var_os("DISPLAY");
        let prev_wayland = env::var_os("WAYLAND_DISPLAY");
        // SAFETY: env mutation in test; serialized via ENV_LOCK / cargo test single-thread.
        unsafe {
            env::remove_var(vr_prompt::ENV_SKIP_VR_DIALOG);
            env::remove_var("CI");
            env::set_var("DISPLAY", ":0");
            env::remove_var("WAYLAND_DISPLAY");
        }
        let out = resolve_vr_choice(vec!["-Invisible".to_string()], || Some(true))
            .expect("vr choice must yield host args");
        assert_eq!(
            out,
            vec![
                "-Device".to_string(),
                "SteamVR".to_string(),
                "-Invisible".to_string(),
            ]
        );
        restore(vr_prompt::ENV_SKIP_VR_DIALOG, prev_skip);
        restore("CI", prev_ci);
        restore("DISPLAY", prev_display);
        restore("WAYLAND_DISPLAY", prev_wayland);
    }
}
