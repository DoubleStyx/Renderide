//! Logger that writes timestamped lines to logs/Bootstrapper.log or stdout when file open fails.

use std::fs::File;
use std::io::{BufWriter, Write};

/// Logger that writes timestamped lines to a file or stdout.
/// Used for logs/Bootstrapper.log; falls back to stdout when the file cannot be opened.
pub struct Logger {
    inner: Option<BufWriter<File>>,
}

impl Logger {
    /// Creates a new logger. Opens the file at `path`, truncating it.
    /// If opening fails, logs will go to stdout instead.
    pub fn new(path: &str) -> Self {
        let inner = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .map(BufWriter::new)
            .ok();
        if inner.is_none() {
            eprintln!("Exception creating log file: {}", path);
        }
        Self { inner }
    }

    /// Writes a timestamped log line. Flushes after each write.
    pub fn log(&mut self, msg: &str) {
        let timestamp = chrono::Local::now().format("%H:%M:%S");
        let line = format!("{}\t{}", timestamp, msg);
        if let Some(ref mut w) = self.inner {
            let _ = writeln!(w, "{}", line);
            let _ = w.flush();
        } else {
            println!("{}", line);
        }
    }

    /// Flushes any buffered output.
    pub fn flush(&mut self) {
        if let Some(ref mut w) = self.inner {
            let _ = w.flush();
        }
    }
}
