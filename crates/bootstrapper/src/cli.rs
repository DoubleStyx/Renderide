//! Command-line parsing and optional desktop vs VR dialog before [`crate::run`].

use std::env;

use logger::LogLevel;

use crate::vr_prompt;
use crate::BootstrapOptions;

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

/// Parses argv, optionally prompts for desktop vs VR, and builds [`BootstrapOptions`] for [`crate::run`].
///
/// Uses [`logger::log_filename_timestamp`] for the log file basename.
pub fn prepare_run_inputs() -> BootstrapOptions {
    let (mut host_args, log_level) = parse_args();
    if vr_prompt::should_prompt_vr_dialog(&host_args) {
        if let Some(vr) = vr_prompt::prompt_desktop_or_vr() {
            host_args = vr_prompt::apply_host_vr_choice(host_args, vr);
        }
    }
    BootstrapOptions {
        host_args,
        log_level,
        log_timestamp: logger::log_filename_timestamp(),
    }
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

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn prepare_run_inputs_respects_skip_vr_dialog() {
        let _g = ENV_LOCK.lock().expect("env lock");
        std::env::set_var(vr_prompt::ENV_SKIP_VR_DIALOG, "1");
        std::env::set_var("CI", "1");
        let opts = prepare_run_inputs();
        assert!(!opts.log_timestamp.is_empty());
        assert!(!opts
            .host_args
            .iter()
            .any(|a| a == "-Screen" || a == "-Device"));
        std::env::remove_var(vr_prompt::ENV_SKIP_VR_DIALOG);
        std::env::remove_var("CI");
    }
}
