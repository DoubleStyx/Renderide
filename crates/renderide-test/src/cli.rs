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
use crate::scene_dsl::output::default_output_root;
use crate::scene_dsl::runner::RunnerConfig;
use crate::scene_dsl::suite::{SuiteConfig, run_suite, select_cases, update_suite};

/// CLI entry point.
pub fn run() -> ExitCode {
    let cli = Cli::parse();
    if let Err(err) = crate::logging::init_renderer_test_logging() {
        eprintln!("renderide-test: failed to initialize logging: {err}");
        return ExitCode::FAILURE;
    }

    let exit_code = match dispatch(cli) {
        Ok(()) => {
            logger::info!("renderide-test completed successfully");
            ExitCode::SUCCESS
        }
        Err(err) => {
            logger::error!("renderide-test failed: {err}");
            eprintln!("renderide-test: {err}");
            ExitCode::FAILURE
        }
    };
    logger::flush();
    exit_code
}

#[derive(Parser, Debug)]
#[command(
    name = "renderide-test",
    about = "Mock host harness for Renderide golden-image integration tests.",
    disable_help_subcommand = true
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run the registered scene suite in parallel and fail if any case fails.
    CheckSuite {
        /// Suite harness options.
        #[command(flatten)]
        suite: SuiteOpts,
    },
    /// Run the registered scene suite and overwrite each selected golden with its capture.
    UpdateSuite {
        /// Suite harness options.
        #[command(flatten)]
        suite: SuiteOpts,
    },
}

#[derive(Parser, Debug, Clone)]
struct SuiteOpts {
    /// Path to the renderide binary to spawn (defaults to `target/{profile}/renderide`).
    #[arg(long)]
    renderer: Option<PathBuf>,
    /// Use the `dev-fast` profile renderer binary (`target/dev-fast/renderide`).
    #[arg(long, default_value_t = false, conflicts_with = "release")]
    dev_fast: bool,
    /// Use the release-mode renderer binary (`target/release/renderide`).
    #[arg(long, default_value_t = false, conflicts_with = "dev_fast")]
    release: bool,
    /// Case name to run. Repeat to run a subset; omitted means every registered case.
    #[arg(long = "case", value_name = "NAME")]
    case: Vec<String>,
    /// Maximum number of cases to run concurrently. Defaults to the selected case count.
    #[arg(long)]
    jobs: Option<usize>,
    /// Test-output root. Defaults to the workspace target/renderide-test-out directory.
    #[arg(long)]
    output_root: Option<PathBuf>,
    /// How long to wait for handshake / asset acks / a fresh PNG per case.
    #[arg(long, default_value_t = 180)]
    timeout_seconds: u64,
    /// Renderer interval between consecutive offscreen renders (ms).
    #[arg(long, default_value_t = 500)]
    interval_ms: u64,
    /// Print renderer processes' stdout/stderr instead of swallowing them.
    #[arg(long, default_value_t = false)]
    verbose_renderer: bool,
}

fn dispatch(cli: Cli) -> Result<(), HarnessError> {
    match cli.command {
        Command::CheckSuite { suite } => run_suite_command(&suite),
        Command::UpdateSuite { suite } => run_update_suite_command(&suite),
    }
}

fn run_suite_command(opts: &SuiteOpts) -> Result<(), HarnessError> {
    let config = suite_config_from_opts(opts)?;
    logger::info!(
        "Suite: running {} case(s) with jobs={}, renderer_path={}, output_root={}",
        config.cases.len(),
        config.jobs,
        config.runner.renderer_path.display(),
        config.runner.output_root.display()
    );

    let outcome = run_suite(config)?;
    print_suite_outcome(&outcome, "PASS", "passed");
    if outcome.passed() {
        Ok(())
    } else {
        Err(HarnessError::SuiteFailed {
            failed: outcome.report.failed,
            total: outcome.report.total,
            report_path: outcome.report_path,
        })
    }
}

fn run_update_suite_command(opts: &SuiteOpts) -> Result<(), HarnessError> {
    let config = suite_config_from_opts(opts)?;
    logger::info!(
        "Suite update: running {} case(s) with jobs={}, renderer_path={}, output_root={}",
        config.cases.len(),
        config.jobs,
        config.runner.renderer_path.display(),
        config.runner.output_root.display()
    );

    let outcome = update_suite(config)?;
    print_suite_outcome(&outcome, "UPDATE", "updated");
    if outcome.passed() {
        Ok(())
    } else {
        Err(HarnessError::SuiteFailed {
            failed: outcome.report.failed,
            total: outcome.report.total,
            report_path: outcome.report_path,
        })
    }
}

fn suite_config_from_opts(opts: &SuiteOpts) -> Result<SuiteConfig, HarnessError> {
    let cases = select_cases(&opts.case)?;
    let case_count = cases.len();
    let renderer_path = match &opts.renderer {
        Some(p) => p.clone(),
        None => resolve_renderer_path(BuildProfile::from_flags(opts.release, opts.dev_fast)),
    };
    let mut runner = RunnerConfig::with_defaults(renderer_path);
    runner.timeout = Duration::from_secs(opts.timeout_seconds);
    runner.interval_ms = opts.interval_ms;
    runner.verbose_renderer = opts.verbose_renderer;
    runner.output_root = opts.output_root.clone().unwrap_or_else(default_output_root);

    let jobs = opts.jobs.unwrap_or(case_count.max(1)).max(1);
    Ok(SuiteConfig {
        cases,
        runner,
        jobs,
    })
}

fn print_suite_outcome(
    outcome: &crate::scene_dsl::suite::SuiteRunOutcome,
    success_label: &str,
    result_label: &str,
) {
    for case in &outcome.report.cases {
        if case.passed {
            println!(
                "{} {} ({})",
                success_label,
                case.name,
                case.artifacts_dir.display()
            );
        } else {
            println!(
                "FAIL {} ({}) {}",
                case.name,
                case.artifacts_dir.display(),
                case.error.as_deref().unwrap_or("case failed")
            );
        }
    }
    println!("Suite report: {}", outcome.report_path.display());
    println!(
        "Suite result: {}/{} {}",
        outcome.report.passed, outcome.report.total, result_label
    );
}

/// Cargo build profile selecting which `target/<profile>/renderide` binary to spawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BuildProfile {
    /// `target/debug/renderide` -- default `cargo build` profile.
    Debug,
    /// `target/release/renderide` -- `cargo build --release`.
    Release,
    /// `target/dev-fast/renderide` -- the project's `dev-fast` workspace profile.
    DevFast,
}

impl BuildProfile {
    /// Resolves the profile from the mutually-exclusive `--release` / `--dev-fast` CLI flags.
    const fn from_flags(release: bool, dev_fast: bool) -> Self {
        if dev_fast {
            Self::DevFast
        } else if release {
            Self::Release
        } else {
            Self::Debug
        }
    }

    /// Subdirectory name under `target/` for this profile.
    const fn target_dir(self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Release => "release",
            Self::DevFast => "dev-fast",
        }
    }
}

fn default_renderer_path(profile: BuildProfile) -> PathBuf {
    let exe = if cfg!(windows) {
        "renderide-renderer.exe"
    } else {
        "renderide-renderer"
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
        "renderide-renderer.exe"
    } else {
        "renderide-renderer"
    });
    candidate.is_file().then_some(candidate)
}

#[cfg(test)]
mod cli_tests {
    use std::path::PathBuf;

    use clap::{CommandFactory, Parser, error::ErrorKind};

    use super::{BuildProfile, Cli, default_renderer_path, resolve_renderer_path};

    #[test]
    fn cli_exposes_only_suite_subcommands() {
        let names: Vec<_> = Cli::command()
            .get_subcommands()
            .map(|cmd| cmd.get_name().to_string())
            .collect();

        assert_eq!(names, ["check-suite", "update-suite"]);
    }

    #[test]
    fn help_subcommand_is_disabled_but_help_flag_still_works() {
        let help_subcommand = Cli::try_parse_from(["renderide-test", "help"])
            .expect_err("help subcommand should be disabled");
        assert_ne!(help_subcommand.kind(), ErrorKind::DisplayHelp);

        let help_flag = Cli::try_parse_from(["renderide-test", "--help"])
            .expect_err("--help should print clap help");
        assert_eq!(help_flag.kind(), ErrorKind::DisplayHelp);
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
            "renderide-renderer.exe"
        } else {
            "renderide-renderer"
        }
    }
}
