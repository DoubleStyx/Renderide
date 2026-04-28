//! Command-line interface for the golden-image harness.

#![expect(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "CLI tool: stdout/stderr is the user-facing interface"
)]

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::{Parser, Subcommand};

use crate::error::HarnessError;
use crate::host::{HarnessRunOutcome, HostHarness, HostHarnessConfig};

/// CLI entry point.
pub(crate) fn run() -> ExitCode {
    let cli = Cli::parse();
    init_logger();
    match dispatch(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            logger::error!("renderide-test failed: {err}");
            eprintln!("renderide-test: {err}");
            ExitCode::FAILURE
        }
    }
}

fn init_logger() {
    use logger::{LogComponent, LogLevel};
    let timestamp = logger::log_filename_timestamp();
    let _ = logger::init_for(
        LogComponent::Bootstrapper,
        &timestamp,
        LogLevel::Info,
        false,
    );
}

#[derive(Parser, Debug)]
#[command(
    name = "renderide-test",
    about = "Mock host harness for Renderide golden-image integration tests."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run the harness, then overwrite the golden PNG at `--out` with the produced image.
    Generate {
        /// Path of the golden image to write.
        #[arg(long, default_value = "crates/renderide-test/goldens/sphere.png")]
        out: PathBuf,
        /// Common harness options.
        #[command(flatten)]
        common: CommonOpts,
    },
    /// Run the harness and compare the produced PNG against the committed golden via SSIM.
    Check {
        /// Path of the golden image to compare against.
        #[arg(long, default_value = "crates/renderide-test/goldens/sphere.png")]
        golden: PathBuf,
        /// Minimum hybrid SSIM score required to pass (0.0 - 1.0).
        ///
        /// Default 0.95 absorbs cross-adapter variance (Intel vs lavapipe vs other). The
        /// flat-image sanity gate in [`crate::golden`] rejects clear-only frames regardless.
        #[arg(long, default_value_t = 0.90)]
        ssim_min: f64,
        /// Where to write the diff visualization on failure.
        #[arg(long, default_value = "target/golden-diff.png")]
        diff_out: PathBuf,
        /// Common harness options.
        #[command(flatten)]
        common: CommonOpts,
    },
    /// Run the harness for local debugging without comparison.
    Run {
        /// Common harness options.
        #[command(flatten)]
        common: CommonOpts,
    },
}

#[derive(Parser, Debug, Clone)]
struct CommonOpts {
    /// Path to the renderide binary to spawn (defaults to `target/{profile}/renderide`).
    #[arg(long)]
    renderer: Option<PathBuf>,
    /// Use the `dev-fast` profile renderer binary (`target/dev-fast/renderide`).
    #[arg(long, default_value_t = false, conflicts_with = "release")]
    dev_fast: bool,
    /// Use the release-mode renderer binary (`target/release/renderide`).
    #[arg(long, default_value_t = false, conflicts_with = "dev_fast")]
    release: bool,
    /// Output resolution (`WxH`) for the offscreen render target.
    #[arg(long, default_value = "256x256")]
    resolution: String,
    /// How long to wait for handshake / asset acks / a fresh PNG.
    #[arg(long, default_value_t = 30)]
    timeout_seconds: u64,
    /// Custom path for the renderer's PNG output (default: a tempfile under the OS temp dir).
    #[arg(long)]
    output: Option<PathBuf>,
    /// Renderer interval between consecutive offscreen renders (ms).
    #[arg(long, default_value_t = 1000)]
    interval_ms: u64,
    /// Print the renderer process's stdout/stderr instead of swallowing it.
    #[arg(long, default_value_t = false)]
    verbose_renderer: bool,
}

fn dispatch(cli: Cli) -> Result<(), HarnessError> {
    match cli.command {
        Command::Generate { out, common } => {
            let outcome = run_harness(&common)?;
            crate::golden::generate(&outcome.png_path, &out)?;
            logger::info!("Wrote golden to {}", out.display());
            println!("Wrote golden to {}", out.display());
            Ok(())
        }
        Command::Check {
            golden,
            ssim_min,
            diff_out,
            common,
        } => {
            let outcome = run_harness(&common)?;
            let score = crate::golden::check(&outcome.png_path, &golden, ssim_min, &diff_out)?;
            logger::info!("SSIM score {score:.4} >= threshold {ssim_min:.4}");
            println!("SSIM score {score:.4} >= threshold {ssim_min:.4}");
            Ok(())
        }
        Command::Run { common } => {
            let outcome = run_harness(&common)?;
            logger::info!("Produced PNG at {}", outcome.png_path.display());
            println!("Produced PNG at {}", outcome.png_path.display());
            Ok(())
        }
    }
}

fn run_harness(common: &CommonOpts) -> Result<HarnessRunOutcome, HarnessError> {
    let (width, height) = parse_resolution(&common.resolution);
    let timeout = Duration::from_secs(common.timeout_seconds);
    let renderer_path = match &common.renderer {
        Some(p) => p.clone(),
        None => resolve_renderer_path(BuildProfile::from_flags(common.release, common.dev_fast)),
    };
    let cfg = HostHarnessConfig {
        renderer_path,
        forced_output_path: common.output.clone(),
        width,
        height,
        interval_ms: common.interval_ms,
        timeout,
        verbose_renderer: common.verbose_renderer,
    };
    let mut harness = HostHarness::start(cfg)?;
    let outcome = harness.run()?;
    Ok(outcome)
}

fn parse_resolution(s: &str) -> (u32, u32) {
    if let Some((w_str, h_str)) = s.split_once(['x', 'X']) {
        if let (Ok(w), Ok(h)) = (w_str.parse::<u32>(), h_str.parse::<u32>()) {
            return (w.max(1), h.max(1));
        }
    }
    (256, 256)
}

/// Cargo build profile selecting which `target/<profile>/renderide` binary to spawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BuildProfile {
    /// `target/debug/renderide` — default `cargo build` profile.
    Debug,
    /// `target/release/renderide` — `cargo build --release`.
    Release,
    /// `target/dev-fast/renderide` — the project's `dev-fast` workspace profile.
    DevFast,
}

impl BuildProfile {
    /// Resolves the profile from the mutually-exclusive `--release` / `--dev-fast` CLI flags.
    fn from_flags(release: bool, dev_fast: bool) -> Self {
        if dev_fast {
            Self::DevFast
        } else if release {
            Self::Release
        } else {
            Self::Debug
        }
    }

    /// Subdirectory name under `target/` for this profile.
    fn target_dir(self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Release => "release",
            Self::DevFast => "dev-fast",
        }
    }
}

fn default_renderer_path(profile: BuildProfile) -> PathBuf {
    let exe = if cfg!(windows) {
        "renderide.exe"
    } else {
        "renderide"
    };
    PathBuf::from("target").join(profile.target_dir()).join(exe)
}

/// Picks a renderer next to this binary when no `--release` / `--dev-fast` flags are set, so
/// e.g. `target/dev-fast/renderide-test` uses `target/dev-fast/renderide` by default.
fn resolve_renderer_path(profile: BuildProfile) -> PathBuf {
    if profile != BuildProfile::Debug {
        return default_renderer_path(profile);
    }
    if let Some(p) = renderide_next_to_this_test_binary() {
        return p;
    }
    default_renderer_path(BuildProfile::Debug)
}

fn renderide_next_to_this_test_binary() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let name = exe.file_name()?.to_str()?;
    if name != "renderide-test" && name != "renderide-test.exe" {
        return None;
    }
    let profile_dir = exe.parent()?;
    let under_target = profile_dir.parent()?;
    if under_target.file_name() != Some(std::ffi::OsStr::new("target")) {
        return None;
    }
    let candidate = profile_dir.join(if cfg!(windows) {
        "renderide.exe"
    } else {
        "renderide"
    });
    candidate.is_file().then_some(candidate)
}

#[cfg(test)]
mod cli_tests {
    use std::path::PathBuf;

    use super::{default_renderer_path, parse_resolution, resolve_renderer_path, BuildProfile};

    #[test]
    fn parse_resolution_accepts_lowercase_and_uppercase_x() {
        assert_eq!(parse_resolution("128x64"), (128, 64));
        assert_eq!(parse_resolution("128X64"), (128, 64));
    }

    #[test]
    fn parse_resolution_invalid_falls_back_to_default() {
        assert_eq!(parse_resolution("not-a-resolution"), (256, 256));
        assert_eq!(parse_resolution(""), (256, 256));
    }

    #[test]
    fn parse_resolution_clamps_zero_dimensions_to_one() {
        assert_eq!(parse_resolution("0x0"), (1, 1));
        assert_eq!(parse_resolution("0x64"), (1, 64));
    }

    #[test]
    fn default_renderer_path_profiles_and_exe_name() {
        assert_eq!(
            default_renderer_path(BuildProfile::DevFast),
            PathBuf::from("target")
                .join("dev-fast")
                .join(expected_exe())
        );
        assert_eq!(
            default_renderer_path(BuildProfile::Release),
            PathBuf::from("target").join("release").join(expected_exe())
        );
        assert_eq!(
            default_renderer_path(BuildProfile::Debug),
            PathBuf::from("target").join("debug").join(expected_exe())
        );
    }

    #[test]
    fn resolve_renderer_path_matches_explicit_profiles() {
        assert_eq!(
            resolve_renderer_path(BuildProfile::Release),
            default_renderer_path(BuildProfile::Release)
        );
        assert_eq!(
            resolve_renderer_path(BuildProfile::DevFast),
            default_renderer_path(BuildProfile::DevFast)
        );
    }

    #[test]
    fn build_profile_from_flags_maps_correctly() {
        assert_eq!(BuildProfile::from_flags(false, false), BuildProfile::Debug);
        assert_eq!(BuildProfile::from_flags(true, false), BuildProfile::Release);
        assert_eq!(BuildProfile::from_flags(false, true), BuildProfile::DevFast);
        // dev_fast wins when both flags are set; clap rejects that combination at the CLI layer.
        assert_eq!(BuildProfile::from_flags(true, true), BuildProfile::DevFast);
    }

    fn expected_exe() -> &'static str {
        if cfg!(windows) {
            "renderide.exe"
        } else {
            "renderide"
        }
    }
}
