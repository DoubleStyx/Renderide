//! ResoBoot-compatible bootstrapper: launches Renderite Host, bridges clipboard/renderer over shared-memory queues.
//!
//! Queue backing paths follow [`interprocess::default_memory_dir`] on each OS unless
//! `RENDERIDE_INTERPROCESS_DIR` is set—the Host must use the same directory (same env or defaults).
//!
//! The binary entry point is [`run`]; use [`BootstrapOptions`] to supply Host arguments and logging.

#![warn(missing_docs)]

mod child_lifetime;
mod cleanup;
pub mod cli;
mod config;
mod constants;
mod error;
mod host;
pub mod ipc;
mod orchestration;
mod panic_hook;
mod paths;
mod protocol;
mod protocol_handlers;
mod renderer_stub;
mod vr_prompt;
mod wine_detect;

pub use error::BootstrapError;

/// Inputs for [`run`]: Host argv, optional verbosity, and log filename timestamp.
#[derive(Debug, Clone)]
pub struct BootstrapOptions {
    /// Arguments forwarded to Renderite Host (before `-Invisible` / `-shmprefix`).
    pub host_args: Vec<String>,
    /// Maximum level written to the bootstrapper log file; also forwarded to Renderide when set.
    pub log_level: Option<logger::LogLevel>,
    /// Filename segment from [`logger::log_filename_timestamp`] (without `.log`).
    pub log_timestamp: String,
}

/// Initializes logging under `logs/bootstrapper/` (or the directory in the `RENDERIDE_LOGS_ROOT`
/// environment variable), installs a panic hook, then runs the bootstrap sequence.
///
/// Panics are logged and swallowed with `Ok(())` to mirror the production ResoBoot behavior.
pub fn run(options: BootstrapOptions) -> Result<(), BootstrapError> {
    let shared_memory_prefix =
        config::generate_shared_memory_prefix(16).map_err(BootstrapError::Prefix)?;
    let resonite_config = config::ResoBootConfig::new(shared_memory_prefix, options.log_level)
        .map_err(BootstrapError::CurrentDir)?;

    let max_level = options.log_level.unwrap_or(logger::LogLevel::Trace);
    let log_path = logger::init_for(
        logger::LogComponent::Bootstrapper,
        &options.log_timestamp,
        max_level,
        false,
    )
    .map_err(BootstrapError::Logging)?;

    panic_hook::install(log_path);

    let ctx = orchestration::RunContext {
        host_args: options.host_args,
        log_timestamp: options.log_timestamp,
    };

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        orchestration::run(&resonite_config, ctx)
    }));

    match result {
        Ok(r) => r,
        Err(e) => {
            logger::error!("Exception in bootstrapper:\n{e:?}");
            logger::flush();
            Ok(())
        }
    }
}
