//! Command-line interface for the golden-image harness.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::{Parser, Subcommand};

use crate::error::HarnessError;
use crate::host::{HarnessRunOutcome, HostHarness, HostHarnessConfig};

/// CLI entry point.
pub fn run() -> ExitCode {
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
        #[arg(long, default_value_t = 0.95)]
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
    /// Output resolution (WxH) for the offscreen render target.
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
        None => default_renderer_path(common.release, common.dev_fast),
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

fn default_renderer_path(release: bool, dev_fast: bool) -> PathBuf {
    let profile = if dev_fast {
        "dev-fast"
    } else if release {
        "release"
    } else {
        "debug"
    };
    let exe = if cfg!(windows) {
        "renderide.exe"
    } else {
        "renderide"
    };
    PathBuf::from("target").join(profile).join(exe)
}
