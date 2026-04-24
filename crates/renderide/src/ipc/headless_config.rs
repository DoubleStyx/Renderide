//! CLI parsing for the renderer's headless offscreen mode.
//!
//! Headless mode bypasses winit/OpenXR and the swapchain, initializes wgpu against any available
//! adapter (GPU on dev machines; often lavapipe when it is the only Vulkan ICD), and writes the
//! offscreen color attachment to a PNG once per interval. It is the back-end for the integration
//! test harness in `renderide-test`.

use std::env;
use std::path::PathBuf;

/// Default offscreen render target width/height in pixels.
pub const DEFAULT_HEADLESS_WIDTH: u32 = 256;
/// Default offscreen render target height in pixels.
pub const DEFAULT_HEADLESS_HEIGHT: u32 = 256;
/// Default interval (milliseconds) between consecutive PNG writes.
pub const DEFAULT_HEADLESS_INTERVAL_MS: u64 = 1000;
/// Default output PNG path when `--headless-output` is omitted.
pub const DEFAULT_HEADLESS_OUTPUT: &str = "renderide-headless.png";

/// Parsed headless-mode parameters from the process command line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadlessParams {
    /// Output PNG path (the renderer atomically writes `path.tmp` then renames to `path`).
    pub output_path: PathBuf,
    /// Offscreen render target width in pixels.
    pub width: u32,
    /// Offscreen render target height in pixels.
    pub height: u32,
    /// Interval between consecutive PNG writes (ms).
    pub interval_ms: u64,
}

impl Default for HeadlessParams {
    fn default() -> Self {
        Self {
            output_path: PathBuf::from(DEFAULT_HEADLESS_OUTPUT),
            width: DEFAULT_HEADLESS_WIDTH,
            height: DEFAULT_HEADLESS_HEIGHT,
            interval_ms: DEFAULT_HEADLESS_INTERVAL_MS,
        }
    }
}

/// Returns parsed headless params if `--headless` is present in `args`; otherwise [`None`].
pub fn parse_headless_params(args: &[String]) -> Option<HeadlessParams> {
    let mut headless = false;
    let mut params = HeadlessParams::default();

    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        let lower = arg.to_lowercase();
        if lower == "--headless" {
            headless = true;
            i += 1;
            continue;
        }
        if lower == "--headless-output" {
            if let Some(value) = args.get(i + 1) {
                params.output_path = PathBuf::from(value);
                i += 2;
                continue;
            }
        }
        if lower == "--headless-resolution" {
            if let Some(value) = args.get(i + 1) {
                if let Some((w, h)) = parse_wxh(value) {
                    params.width = w;
                    params.height = h;
                }
                i += 2;
                continue;
            }
        }
        if lower == "--headless-interval-ms" {
            if let Some(value) = args.get(i + 1) {
                if let Ok(ms) = value.parse::<u64>() {
                    if ms > 0 {
                        params.interval_ms = ms;
                    }
                }
                i += 2;
                continue;
            }
        }
        i += 1;
    }

    if headless {
        Some(params)
    } else {
        None
    }
}

/// Convenience wrapper that reads from [`std::env::args`].
pub fn get_headless_params() -> Option<HeadlessParams> {
    let args: Vec<String> = env::args().collect();
    parse_headless_params(&args)
}

/// Returns `true` if `--ignore-config` appears anywhere in `args` (case-insensitive).
pub fn parse_ignore_config(args: &[String]) -> bool {
    args.iter()
        .any(|a| a.eq_ignore_ascii_case("--ignore-config"))
}

/// Convenience wrapper that reads from [`std::env::args`].
pub fn get_ignore_config() -> bool {
    let args: Vec<String> = env::args().collect();
    parse_ignore_config(&args)
}

fn parse_wxh(value: &str) -> Option<(u32, u32)> {
    let (w_str, h_str) = value.split_once(['x', 'X'])?;
    let w: u32 = w_str.parse().ok()?;
    let h: u32 = h_str.parse().ok()?;
    Some((w.max(1), h.max(1)))
}

#[cfg(test)]
mod tests {
    use super::{parse_headless_params, parse_ignore_config, HeadlessParams};
    use std::path::PathBuf;

    fn s(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn parse_ignore_config_detects_flag() {
        assert!(parse_ignore_config(&s(&["renderide", "--ignore-config"])));
        assert!(parse_ignore_config(&s(&["renderide", "--IGNORE-CONFIG"])));
        assert!(!parse_ignore_config(&s(&["renderide", "--headless"])));
    }

    #[test]
    fn returns_none_when_flag_absent() {
        assert_eq!(parse_headless_params(&s(&["renderide"])), None);
        assert_eq!(
            parse_headless_params(&s(&["renderide", "--headless-output", "x.png"])),
            None,
            "no --headless flag means params are ignored"
        );
    }

    #[test]
    fn parses_defaults_when_only_flag_given() {
        let p = parse_headless_params(&s(&["renderide", "--headless"])).expect("present");
        assert_eq!(p, HeadlessParams::default());
    }

    #[test]
    fn parses_all_options() {
        let p = parse_headless_params(&s(&[
            "renderide",
            "--headless",
            "--headless-output",
            "/tmp/out.png",
            "--headless-resolution",
            "640x480",
            "--headless-interval-ms",
            "500",
        ]))
        .expect("present");
        assert_eq!(p.output_path, PathBuf::from("/tmp/out.png"));
        assert_eq!(p.width, 640);
        assert_eq!(p.height, 480);
        assert_eq!(p.interval_ms, 500);
    }

    #[test]
    fn rejects_zero_interval_falls_back_to_default() {
        let p = parse_headless_params(&s(&[
            "renderide",
            "--headless",
            "--headless-interval-ms",
            "0",
        ]))
        .expect("present");
        assert_eq!(p.interval_ms, super::DEFAULT_HEADLESS_INTERVAL_MS);
    }

    #[test]
    fn case_insensitive_flag_matching() {
        let p = parse_headless_params(&s(&[
            "renderide",
            "--HEADLESS",
            "--Headless-Resolution",
            "32x32",
        ]))
        .expect("present");
        assert_eq!(p.width, 32);
        assert_eq!(p.height, 32);
    }
}
