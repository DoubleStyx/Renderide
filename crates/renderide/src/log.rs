//! Log to logs/Renderide.log in repo root, cleared on each run.

/// Throttle interval for diagnostic logs (every N frames).
pub const DIAG_FRAME_INTERVAL: u64 = 50;
/// Heavier throttle for verbose logs (SPACE AUDIT, MATH SUMMARY, etc.).
pub const DIAG_FRAME_INTERVAL_THROTTLE: u64 = 250;
/// Flush log buffer every N writes to avoid sync disk I/O on every call.
const FLUSH_EVERY_N_WRITES: u32 = 50;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Mutex;

/// Log state: writer plus write counter for periodic flush.
struct LogState {
    writer: BufWriter<File>,
    write_count: u32,
}

static LOG: Mutex<Option<LogState>> = Mutex::new(None);

/// Path to Renderide.log in the logs folder at repo root (two levels up from crates/renderide).
pub fn log_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap_or_else(|| Path::new("."))
        .join("logs")
        .join("Renderide.log")
}

/// Initialize logging. Opens logs/Renderide.log in repo root and truncates it (clears previous run).
/// Call once at startup before any log writes.
pub fn init_log() {
    let path = log_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match File::options()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&path)
    {
        Ok(f) => {
            let _ = LOG.lock().map(|mut g| {
                *g = Some(LogState {
                    writer: BufWriter::new(f),
                    write_count: 0,
                })
            });
        }
        Err(e) => {
            let _ = LOG.lock().map(|mut g| *g = None);
            eprintln!("[Renderide] Failed to open {}: {}", path.display(), e);
        }
    }
}

/// Write a line to the log file. No-op if init failed.
/// Flushes periodically (every `FLUSH_EVERY_N_WRITES` calls) to avoid sync disk I/O on every write.
pub fn log_write(msg: &str) {
    if let Ok(mut g) = LOG.lock() {
        if let Some(ref mut state) = *g {
            let _ = writeln!(state.writer, "{}", msg);
            state.write_count += 1;
            if state.write_count >= FLUSH_EVERY_N_WRITES {
                state.write_count = 0;
                let _ = state.writer.flush();
            }
        }
    }
}
