//! ResoBoot - main orchestration for the bootstrapper.
//! Sets up IPC queues, spawns Host, runs the queue loop, and cleans up.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use interprocess::{QueueFactory, QueueOptions};

use crate::config::ResoBootConfig;
use crate::host_spawner;
use crate::logger::Logger;
use crate::orphan;
use crate::queue_commands;

const QUEUE_CAPACITY: i64 = 8192;

/// Main entry point. Runs the full bootstrap sequence.
pub fn run(logger: &mut Logger) {
    let config = ResoBootConfig::new();
    let logs_dir = config.current_directory.join("logs");
    let _ = fs::create_dir_all(&logs_dir);

    if let Err(e) = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(logs_dir.join("HostOutput.log"))
    {
        logger.log(&format!("Failed to reset HostOutput.log: {}", e));
    }
    if let Err(e) = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(logs_dir.join("Renderide.log"))
    {
        logger.log(&format!("Failed to reset Renderide.log: {}", e));
    }

    orphan::kill_orphans(logger);

    logger.log("Bootstrapper start");
    logger.log(&format!("Shared memory prefix: {}", config.shared_memory_prefix));

    let incoming_name = format!("{}.bootstrapper_in", config.shared_memory_prefix);
    let outgoing_name = format!("{}.bootstrapper_out", config.shared_memory_prefix);
    logger.log(&format!("Queue names: incoming={} outgoing={}", incoming_name, outgoing_name));

    let queue_factory = QueueFactory::new();
    let mut incoming = queue_factory.create_subscriber(QueueOptions::with_destroy(
        &incoming_name,
        QUEUE_CAPACITY,
        true,
    ));
    let mut outgoing = queue_factory.create_publisher(QueueOptions::with_destroy(
        &outgoing_name,
        QUEUE_CAPACITY,
        true,
    ));
    logger.log("Queues created (Subscriber bootstrapper_in, Publisher bootstrapper_out)");

    let mut args: Vec<String> = env::args().skip(1).collect();
    args.push("-shmprefix".to_string());
    args.push(config.shared_memory_prefix.clone());
    logger.log(&format!("Host args: {:?}", args));

    let mut p = match host_spawner::spawn_host(&config, &args, logger) {
        Ok(c) => c,
        Err(e) => {
            logger.log(&format!("Failed to start process: {}", e));
            if e.kind() == std::io::ErrorKind::NotFound {
                logger.log(
                    "Could not find Resonite installation. Set RESONITE_DIR or ensure Steam has Resonite installed.",
                );
            }
            return;
        }
    };

    logger.log(&format!("Process started. Id: {}, HasExited: {}", p.id(), false));
    logger.log("Host must parse -shmprefix and create BootstrapperManager with matching queue names");
    logger.log("Host sends first message to bootstrapper_in: renderer start args (-QueueName X -QueueCapacity Y)");

    orphan::write_pid_file(p.id(), "host", logger);

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
        logger.log("Process watcher: will set cancel=true when Host process exits");
        std::thread::spawn(move || {
            while match p.try_wait() {
                Ok(None) => true,
                _ => false,
            } {
                std::thread::sleep(Duration::from_secs(1));
            }
            let timestamp = chrono::Local::now().format("%H:%M:%S");
            println!("{}\tMain process has exited, triggering cancellation", timestamp);
            cancel_clone.store(true, Ordering::SeqCst);
        });
    } else {
        logger.log("Wine mode: process watcher disabled (child is shell, not Host)");
    }

    queue_commands::queue_loop(&mut incoming, &mut outgoing, &config, &*cancel, logger);

    if config.is_wine {
        let shm_dir = PathBuf::from("/dev/shm");
        if shm_dir.exists() {
            if let Ok(entries) = fs::read_dir(&shm_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if let Some(name) = path.file_name() {
                        if name.to_string_lossy().contains(&config.shared_memory_prefix) {
                            let _ = fs::remove_file(&path);
                        }
                    }
                }
            }
        }
    }

    orphan::remove_pid_file();
    logger.log("Bootstrapper end");
}
