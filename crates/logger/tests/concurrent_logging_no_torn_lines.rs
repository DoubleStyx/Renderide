//! Integration: many threads logging through [`logger::info!`] produce intact, well-formed lines.
//!
//! The mutex inside `Logger::file` and the per-thread reusable line buffer exist to keep
//! multi-thread output well-formed. This test exercises both end-to-end against a real file sink.

use std::collections::BTreeSet;
use std::time::{SystemTime, UNIX_EPOCH};

const THREADS: usize = 8;
const LINES_PER_THREAD: usize = 50;

/// Spawns [`THREADS`] worker threads, each emitting [`LINES_PER_THREAD`] tagged log lines, and
/// asserts that every line is well-formed and that no `(thread, sequence)` pair was lost or
/// duplicated due to torn writes.
#[test]
fn concurrent_threads_produce_intact_log_lines() {
    let dir = tempfile::tempdir().expect("tempdir");
    // SAFETY: env mutation in test; serialized via cargo test integration-binary isolation.
    unsafe {
        std::env::set_var("RENDERIDE_LOGS_ROOT", dir.path().as_os_str());
    }

    let ts = format!(
        "concurrent_{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );

    let log_path = logger::init_for(
        logger::LogComponent::Renderer,
        &ts,
        logger::LogLevel::Info,
        false,
    )
    .expect("init_for");

    let mut handles = Vec::with_capacity(THREADS);
    for thread_id in 0..THREADS {
        handles.push(std::thread::spawn(move || {
            for sequence in 0..LINES_PER_THREAD {
                logger::info!("concurrent_marker tid={thread_id} seq={sequence}");
            }
        }));
    }
    for h in handles {
        h.join().expect("thread join");
    }
    logger::flush();

    let contents = std::fs::read_to_string(&log_path).expect("read log");
    let lines: Vec<&str> = contents.split('\n').filter(|l| !l.is_empty()).collect();

    assert_eq!(
        lines.len(),
        THREADS * LINES_PER_THREAD,
        "expected {} lines, got {}",
        THREADS * LINES_PER_THREAD,
        lines.len()
    );

    let mut seen = BTreeSet::new();
    for line in &lines {
        assert!(line.starts_with('['), "missing leading `[` in {line:?}");
        assert!(line.contains("] INFO "), "missing level token in {line:?}");
        assert!(
            line.contains("concurrent_marker"),
            "missing message marker in {line:?}"
        );

        let tid = extract_field(line, "tid=").expect("tid field");
        let seq = extract_field(line, "seq=").expect("seq field");
        assert!(
            seen.insert((tid, seq)),
            "duplicate (tid, seq) pair in {line:?}"
        );
    }

    for thread_id in 0..THREADS {
        for sequence in 0..LINES_PER_THREAD {
            assert!(
                seen.contains(&(thread_id, sequence)),
                "missing tid={thread_id} seq={sequence}"
            );
        }
    }

    // SAFETY: env mutation in test; serialized via cargo test integration-binary isolation.
    unsafe {
        std::env::remove_var("RENDERIDE_LOGS_ROOT");
    }
}

/// Parses `key=<usize>` out of a log line, returning the integer value or `None` if the key is
/// absent or the value is not an unsigned integer terminated by whitespace or end-of-line.
fn extract_field(line: &str, key: &str) -> Option<usize> {
    let start = line.find(key)? + key.len();
    let tail = &line[start..];
    let end = tail
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(tail.len());
    tail[..end].parse().ok()
}
