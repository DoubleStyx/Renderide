//! Full bootstrap sequence: IPC, Host spawn, watchdogs, queue loop, Wine cleanup.
//!
//! Shared-memory queue files use [`crate::ipc::interprocess_backing_dir`] unless overridden; see
//! [`crate::ipc::RENDERIDE_INTERPROCESS_DIR_ENV`].

use std::env;
use std::process::Child;
use std::sync::atomic::AtomicBool;
#[cfg(target_os = "macos")]
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::BootstrapError;
use crate::child_lifetime::ChildLifetimeGroup;
use crate::cleanup;
use crate::config::ResoBootConfig;
use crate::constants::initial_heartbeat_timeout;
use crate::host;
use crate::ipc::{BootstrapQueues, bootstrap_queue_base_names};
use crate::process_state::SharedChildSlot;
use crate::protocol;
use crate::watchdogs;

/// Paths and argv for a single bootstrap run (owned so a panic boundary can move it).
pub struct RunContext {
    /// Extra Host CLI args (before `-Invisible` / `-shmprefix` are appended).
    pub host_args: Vec<String>,
    /// Shared basename (no `.log`) for paths like `logs/host/{timestamp}.log` under [`logger::logs_root`].
    pub log_timestamp: String,
}

/// Logs Resonite / interprocess paths and queue names at bootstrap start.
fn log_run_intro(config: &ResoBootConfig) {
    if let Some(ref level) = config.renderide_log_level {
        logger::info!("Renderide log level: {}", level.as_arg());
    }

    logger::info!(
        "Bootstrapper start: version={} pid={} target={} {} arch={} exe={} cwd={}",
        env!("CARGO_PKG_VERSION"),
        std::process::id(),
        env::consts::OS,
        env::consts::FAMILY,
        env::consts::ARCH,
        env::current_exe()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|e| format!("<unavailable: {e}>")),
        env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|e| format!("<unavailable: {e}>")),
    );
    logger::info!("Shared memory prefix: {}", config.shared_memory_prefix);
    logger::info!(
        "Bootstrapper paths: current_directory={} renderite_directory={} renderite_executable={} runtime_config={} resonite_dir_override={}",
        config.current_directory.display(),
        config.renderite_directory.display(),
        config.renderite_executable.display(),
        config.runtime_config.display(),
        config
            .resonite_dir
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "<auto>".to_string()),
    );
    let backing = crate::ipc::interprocess_backing_dir();
    logger::info!(
        "Interprocess queue backing directory: {:?} (set {} to override; Host must match)",
        backing,
        crate::ipc::RENDERIDE_INTERPROCESS_DIR_ENV
    );
    for key in ["RENDERIDE_INTERPROCESS_DIR", "RENDERIDE_LOGS_ROOT"] {
        if let Ok(value) = env::var(key)
            && !value.trim().is_empty()
        {
            logger::info!("Bootstrapper env override: {key}={value}");
        }
    }
}

/// Appends `-shmprefix` and the generated prefix to Host argv.
pub fn assemble_host_args(mut host_args: Vec<String>, shared_memory_prefix: &str) -> Vec<String> {
    host_args.push("-shmprefix".to_string());
    host_args.push(shared_memory_prefix.to_string());
    host_args
}

/// Spawns the Host, raises priority, and starts stdout/stderr drainers into the host log file.
fn start_host_with_drainers(
    config: &ResoBootConfig,
    args: &[String],
    lifetime: &ChildLifetimeGroup,
    log_timestamp: &str,
) -> Result<Child, std::io::Error> {
    let mut child = host::spawn_host(config, args, lifetime)?;
    logger::info!("Process started. Id: {}", child.id());

    host::set_host_above_normal_priority(&child);

    logger::ensure_log_dir(logger::LogComponent::Host)?;
    let host_log_path = logger::log_file_path(logger::LogComponent::Host, log_timestamp);
    logger::info!("Host stdout/stderr log path: {}", host_log_path.display());

    if let Some(stdout) = child.stdout.take() {
        let _stdout_drainer =
            host::spawn_output_drainer(host_log_path.clone(), stdout, "[Host stdout]");
    }
    if let Some(stderr) = child.stderr.take() {
        let _stderr_drainer = host::spawn_output_drainer(host_log_path, stderr, "[Host stderr]");
    }

    Ok(child)
}

/// Installs Ctrl+C handler on macOS to set `cancel`.
#[cfg(target_os = "macos")]
fn install_macos_signal_handler(cancel: &Arc<AtomicBool>) {
    let c = Arc::clone(cancel);
    if let Err(e) = ctrlc::set_handler(move || {
        c.store(true, Ordering::SeqCst);
    }) {
        logger::warn!("macOS: could not install ctrlc (SIGINT/SIGTERM) handler: {e}");
    }
}

/// macOS child teardown, Wine queue cleanup, and final log line.
fn finalize(config: &ResoBootConfig, lifetime: &ChildLifetimeGroup) {
    #[cfg(target_os = "macos")]
    lifetime.shutdown_tracked_children();
    #[cfg(not(target_os = "macos"))]
    let _ = lifetime;

    if config.is_wine {
        cleanup::remove_wine_queue_backing_files(&config.shared_memory_prefix);
    }

    logger::info!("Bootstrapper end");
}

/// Runs the bootstrapper main loop after logging is initialized.
pub fn run(config: &ResoBootConfig, ctx: RunContext) -> Result<(), BootstrapError> {
    log_run_intro(config);

    let lifetime = ChildLifetimeGroup::new().map_err(BootstrapError::Io)?;
    let mut queues = BootstrapQueues::open(&config.shared_memory_prefix)?;

    let (incoming_name, outgoing_name) = bootstrap_queue_base_names(&config.shared_memory_prefix);
    logger::info!(
        "Queues: incoming={incoming_name} outgoing={outgoing_name} (capacity {})",
        crate::ipc::BOOTSTRAP_QUEUE_CAPACITY
    );

    let RunContext {
        host_args,
        log_timestamp,
    } = ctx;

    let args = assemble_host_args(host_args, &config.shared_memory_prefix);
    logger::info!("Host args: {:?}", args);

    let host_process = start_host_with_drainers(config, &args, &lifetime, &log_timestamp)
        .map_err(BootstrapError::Io)?;

    let host_child = SharedChildSlot::with_child(host_process);
    let renderer_child = SharedChildSlot::empty();

    let cancel = Arc::new(AtomicBool::new(false));

    #[cfg(target_os = "macos")]
    install_macos_signal_handler(&cancel);

    let heartbeat_deadline = Arc::new(Mutex::new(Instant::now() + initial_heartbeat_timeout()));
    let _watchdogs = watchdogs::spawn_watchdogs(
        config,
        Arc::clone(&cancel),
        Arc::clone(&heartbeat_deadline),
        host_child,
        renderer_child.clone(),
        log_timestamp,
    );

    protocol::queue_loop(
        &mut queues.incoming,
        &mut queues.outgoing,
        config,
        &cancel,
        &lifetime,
        &heartbeat_deadline,
        &renderer_child,
    );

    finalize(config, &lifetime);
    Ok(())
}

#[cfg(test)]
mod assemble_host_args_tests {
    use super::assemble_host_args;

    #[test]
    fn empty_argv_appends_shmprefix_and_prefix() {
        let out = assemble_host_args(vec![], "Ab12");
        assert_eq!(out, vec!["-shmprefix".to_string(), "Ab12".to_string()]);
    }

    #[test]
    fn preserves_order_and_appends_suffix() {
        let out = assemble_host_args(
            vec![
                "-Invisible".to_string(),
                "-Data".to_string(),
                "path".to_string(),
            ],
            "Z9",
        );
        assert_eq!(
            out,
            vec![
                "-Invisible".to_string(),
                "-Data".to_string(),
                "path".to_string(),
                "-shmprefix".to_string(),
                "Z9".to_string(),
            ]
        );
    }

    #[test]
    fn ends_with_shmprefix_then_prefix() {
        let prefix = "prefX";
        let out = assemble_host_args(vec!["a".into(), "b".into()], prefix);
        assert!(out.len() >= 2);
        assert_eq!(out[out.len() - 2], "-shmprefix");
        assert_eq!(out[out.len() - 1], prefix);
    }

    #[test]
    fn preserves_existing_shmprefix_arguments_and_appends_new_pair() {
        let out = assemble_host_args(
            vec![
                "-shmprefix".to_string(),
                "old".to_string(),
                "-Invisible".to_string(),
            ],
            "new",
        );

        assert_eq!(
            out,
            vec![
                "-shmprefix".to_string(),
                "old".to_string(),
                "-Invisible".to_string(),
                "-shmprefix".to_string(),
                "new".to_string(),
            ]
        );
    }

    #[test]
    fn accepts_empty_shared_memory_prefix_as_explicit_host_value() {
        let out = assemble_host_args(vec!["-Data".into()], "");

        assert_eq!(
            out,
            vec!["-Data".to_string(), "-shmprefix".to_string(), String::new()]
        );
    }
}
