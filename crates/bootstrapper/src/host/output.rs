//! Host stdout and stderr draining into bootstrapper-owned log files.

use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;

/// Drains a reader into a log file line-by-line with a prefix.
pub fn spawn_output_drainer(
    log_path: PathBuf,
    reader: impl Read + Send + 'static,
    prefix: &'static str,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut file = match fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
        {
            Ok(f) => f,
            Err(e) => {
                logger::warn!("Could not open host log {:?} for drainer: {}", log_path, e);
                return;
            }
        };
        let mut buf_reader = BufReader::new(reader);
        let mut line = String::new();
        while buf_reader.read_line(&mut line).is_ok_and(|n| n > 0) {
            let _ = writeln!(file, "{} {}", prefix, line.trim_end());
            let _ = file.flush();
            line.clear();
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn spawn_output_drainer_writes_prefixed_lines() {
        let log_path =
            std::env::temp_dir().join(format!("bootstrapper_drainer_{}.log", std::process::id()));
        let _ = fs::remove_file(&log_path);
        let input = b"line one\nline two\n";
        let handle = spawn_output_drainer(log_path.clone(), Cursor::new(input), "[P]");
        handle.join().expect("drain output");
        let out = fs::read_to_string(&log_path).expect("read log");
        assert!(out.contains("[P] line one"));
        assert!(out.contains("[P] line two"));
        let _ = fs::remove_file(&log_path);
    }

    #[test]
    fn spawn_output_drainer_trims_line_endings_only() {
        let log_path = std::env::temp_dir().join(format!(
            "bootstrapper_drainer_trim_{}.log",
            std::process::id()
        ));
        let _ = fs::remove_file(&log_path);
        let handle = spawn_output_drainer(log_path.clone(), Cursor::new(b"  padded  \r\n"), "[P]");
        handle.join().expect("drain output");
        let out = fs::read_to_string(&log_path).expect("read log");
        assert!(out.contains("[P]   padded"));
        assert!(!out.contains('\r'));
        let _ = fs::remove_file(&log_path);
    }
}
