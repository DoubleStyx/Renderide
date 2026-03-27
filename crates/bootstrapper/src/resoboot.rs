//! ResoBoot - main orchestration for the bootstrapper.
//! Sets up IPC queues, spawns Host, runs the queue loop, and cleans up.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use interprocess::{QueueFactory, QueueOptions};

use crate::config::ResoBootConfig;
use crate::host_spawner;
use crate::process_lifetime::ChildLifetimeGroup;
use crate::queue_commands;

const QUEUE_CAPACITY: i64 = 8192;

/// Main entry point. Runs the full bootstrap sequence.
pub fn run(host_args_from_cli: &[String], log_level: Option<logger::LogLevel>) {
    if let Some(ref level) = log_level {
        logger::info!("Renderide log level: {}", level.as_arg());
    }
    let config = ResoBootConfig::new(log_level);
    let logs_dir = config.current_directory.join("logs");
    let _ = fs::create_dir_all(&logs_dir);

    if let Err(e) = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(logs_dir.join("HostOutput.log"))
    {
        logger::warn!("Failed to reset HostOutput.log: {}", e);
    }
    if let Err(e) = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(logs_dir.join("Renderide.log"))
    {
        logger::warn!("Failed to reset Renderide.log: {}", e);
    }

    let lifetime = match ChildLifetimeGroup::new() {
        Ok(g) => g,
        Err(e) => {
            logger::error!("Failed to create child lifetime group: {}", e);
            return;
        }
    };

    logger::info!("Bootstrapper start");
    logger::info!("Shared memory prefix: {}", config.shared_memory_prefix);

    let incoming_name = format!("{}.bootstrapper_in", config.shared_memory_prefix);
    let outgoing_name = format!("{}.bootstrapper_out", config.shared_memory_prefix);
    logger::info!(
        "Queue names: incoming={} outgoing={}",
        incoming_name,
        outgoing_name
    );

    let queue_factory = QueueFactory::new();
    let mut incoming = match queue_factory.create_subscriber(QueueOptions::with_destroy(
        &incoming_name,
        QUEUE_CAPACITY,
        true,
    )) {
        Ok(s) => s,
        Err(e) => {
            logger::error!("Failed to create subscriber queue: {}", e);
            return;
        }
    };
    let mut outgoing = match queue_factory.create_publisher(QueueOptions::with_destroy(
        &outgoing_name,
        QUEUE_CAPACITY,
        true,
    )) {
        Ok(p) => p,
        Err(e) => {
            logger::error!("Failed to create publisher queue: {}", e);
            return;
        }
    };
    logger::info!("Queues created (Subscriber bootstrapper_in, Publisher bootstrapper_out)");

    let mut args: Vec<String> = host_args_from_cli.to_vec();
    args.push("-Invisible".to_string());
    args.push("-shmprefix".to_string());
    args.push(config.shared_memory_prefix.clone());
    args.push("-LoadAssembly".to_string());
    args.push("Libraries/ResoniteModLoader.dll".to_string());
    logger::info!("Host args: {:?}", args);

    let mut p = match host_spawner::spawn_host(&config, &args, &lifetime) {
        Ok(c) => c,
        Err(e) => {
            logger::error!("Failed to start process: {}", e);
            if e.kind() == std::io::ErrorKind::NotFound {
                logger::error!(
                    "Could not find Resonite installation. Set RESONITE_DIR or ensure Steam has Resonite installed.",
                );
            }
            return;
        }
    };

    logger::info!("Process started. Id: {}, HasExited: {}", p.id(), false);
    logger::info!(
        "Host must parse -shmprefix and create BootstrapperManager with matching queue names"
    );
    logger::info!(
        "Host sends first message to bootstrapper_in: renderer start args (-QueueName X -QueueCapacity Y)"
    );

    let log_path = logs_dir.join("HostOutput.log");
    if let Some(stdout) = p.stdout.take() {
        host_spawner::spawn_output_drainer(log_path.clone(), stdout, "[Host stdout]");
    }
    if let Some(stderr) = p.stderr.take() {
        host_spawner::spawn_output_drainer(log_path, stderr, "[Host stderr]");
    }

    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_clone = Arc::clone(&cancel);

    if !config.is_wine {
        logger::info!("Process watcher: will set cancel=true when Host process exits");
        std::thread::spawn(move || {
            let exit_status = loop {
                match p.try_wait() {
                    Ok(Some(status)) => break Some(status),
                    Ok(None) => {}
                    Err(e) => {
                        logger::error!("Process watcher try_wait error: {}", e);
                        break None;
                    }
                }
                std::thread::sleep(Duration::from_secs(1));
            };
            let _timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| {
                    let s = d.as_secs();
                    format!("{:02}:{:02}:{:02}", (s / 3600) % 24, (s / 60) % 60, s % 60)
                })
                .unwrap_or_else(|_| "?".to_string());
            let exit_info = exit_status
                .as_ref()
                .map(|s| format!(" (exit code: {:?})", s))
                .unwrap_or_default();
            let msg = format!(
                "Host process exited{}, triggering cancellation. Check HostOutput.log for host stdout/stderr.",
                exit_info
            );
            eprintln!("{}", msg);
            logger::info!("{}", msg);
            cancel_clone.store(true, Ordering::SeqCst);
        });
    } else {
        logger::info!("Wine mode: process watcher disabled (child is shell, not Host)");
    }

    queue_commands::queue_loop(&mut incoming, &mut outgoing, &config, &cancel, &lifetime);

    if config.is_wine {
        let shm_dir = PathBuf::from("/dev/shm");
        if shm_dir.exists()
            && let Ok(entries) = fs::read_dir(&shm_dir)
        {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name()
                    && name
                        .to_string_lossy()
                        .contains(&config.shared_memory_prefix)
                {
                    let _ = fs::remove_file(&path);
                }
            }
        }
    }

    logger::info!("Bootstrapper end");
}
