//! Command-line parsing and the optional desktop vs VR dialog that precedes [`crate::run`].
//!
//! The desktop/VR dialog is resolved **after** the global logger has been initialized so that any
//! `rfd` hang (see `vr_prompt` module docs) is visible in `logs/bootstrapper/*.log` rather than
//! producing a silent "nothing happens" failure.

use std::env;

use logger::LogLevel;

use crate::vr_prompt;

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
        let arg_lower = arg.to_lowercase();
        if (arg_lower == "--log-level" || arg_lower == "-l") && i + 1 < args.len() {
            log_level = LogLevel::parse(&args[i + 1]);
            i += 2;
            continue;
        }
        host_args.push(arg.clone());
        i += 1;
    }
    (host_args, log_level)
}

/// Runs the desktop vs VR dialog if required by [`vr_prompt::should_prompt_vr_dialog`] and
/// returns host argv augmented with the resulting `-Screen` / `-Device SteamVR` flag.
///
/// Returns [`None`] only when the dialog runs **and** the user cancels or dismisses it; in every
/// bypass path (explicit output flag, `CI`, [`vr_prompt::ENV_SKIP_VR_DIALOG`], no Linux display)
/// the original `host_args` are returned unchanged.
///
/// The caller **must** have initialized the global logger before invocation because
/// [`vr_prompt::prompt_desktop_or_vr`] emits before/after log lines and installs a watchdog that
/// logs on timeout.
pub fn resolve_vr_choice(host_args: Vec<String>) -> Option<Vec<String>> {
    if !vr_prompt::should_prompt_vr_dialog(&host_args) {
        return Some(host_args);
    }
    let vr = vr_prompt::prompt_desktop_or_vr()?;
    Some(vr_prompt::apply_host_vr_choice(host_args, vr))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| (*s).to_string()).collect()
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
            env::set_var(key, v);
        } else {
            env::remove_var(key);
        }
    }

    #[test]
    fn resolve_vr_choice_bypasses_dialog_on_skip_env_without_invoking_rfd() {
        let _g = ENV_LOCK.lock().expect("env lock");
        let prev_skip = env::var_os(vr_prompt::ENV_SKIP_VR_DIALOG);
        let prev_ci = env::var_os("CI");
        env::set_var(vr_prompt::ENV_SKIP_VR_DIALOG, "1");
        env::set_var("CI", "1");
        let out =
            resolve_vr_choice(vec!["-Invisible".to_string()]).expect("bypass path must yield Some");
        assert_eq!(out, vec!["-Invisible".to_string()]);
        restore(vr_prompt::ENV_SKIP_VR_DIALOG, prev_skip);
        restore("CI", prev_ci);
    }

    #[test]
    fn resolve_vr_choice_preserves_explicit_screen_arg() {
        let _g = ENV_LOCK.lock().expect("env lock");
        let prev_skip = env::var_os(vr_prompt::ENV_SKIP_VR_DIALOG);
        let prev_ci = env::var_os("CI");
        env::remove_var(vr_prompt::ENV_SKIP_VR_DIALOG);
        env::remove_var("CI");
        let out = resolve_vr_choice(vec!["-Screen".to_string(), "-Invisible".to_string()])
            .expect("explicit output flag bypasses dialog");
        assert_eq!(out, vec!["-Screen".to_string(), "-Invisible".to_string()]);
        restore(vr_prompt::ENV_SKIP_VR_DIALOG, prev_skip);
        restore("CI", prev_ci);
    }
}
