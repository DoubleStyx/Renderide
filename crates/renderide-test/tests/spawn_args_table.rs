//! Exhaustive matrix verification for
//! [`renderide_test::host::scene_session::spawn::renderer_spawn_args`]: every variation should
//! emit the required flag set with values paired immediately after their flag.

use std::path::PathBuf;
use std::time::Duration;

use renderide_test::host::SceneSessionConfig;
use renderide_test::host::ipc_setup::DEFAULT_QUEUE_CAPACITY_BYTES;
use renderide_test::host::scene_session::spawn::renderer_spawn_args;

#[test]
fn spawn_args_table_covers_release_and_dev_paths() {
    let cases = [
        (
            "target/release/renderide",
            "target/headless.png",
            1920u32,
            1080u32,
            33u64,
            "queue-rel",
        ),
        (
            "target/dev-fast/renderide",
            "/tmp/headless.png",
            64,
            32,
            250,
            "queue-devfast",
        ),
        (
            "target/debug/renderide",
            "out/test.png",
            800,
            600,
            16,
            "queue-debug",
        ),
    ];

    for (renderer, output, width, height, interval, queue) in cases {
        let cfg = SceneSessionConfig {
            renderer_path: PathBuf::from(renderer),
            output_path: PathBuf::from(output),
            width,
            height,
            interval_ms: interval,
            timeout: Duration::from_secs(5),
            verbose_renderer: false,
        };
        let args = renderer_spawn_args(&cfg, queue);

        let resolution = format!("{width}x{height}");
        let interval_str = interval.to_string();
        let capacity = DEFAULT_QUEUE_CAPACITY_BYTES.to_string();

        let pairs: &[(&str, &str)] = &[
            ("--headless-output", output),
            ("--headless-resolution", resolution.as_str()),
            ("--headless-interval-ms", interval_str.as_str()),
            ("-QueueName", queue),
            ("-QueueCapacity", capacity.as_str()),
            ("-LogLevel", "debug"),
        ];

        assert_eq!(
            args.first().map(String::as_str),
            Some("--headless"),
            "case {queue}: missing leading --headless"
        );
        assert!(
            args.iter().any(|a| a == "--ignore-config"),
            "case {queue}: --ignore-config missing in {args:?}"
        );
        for (flag, expected_value) in pairs {
            let pos = args.iter().position(|a| a == flag);
            assert!(
                pos.is_some(),
                "case {queue}: flag {flag} missing in {args:?}"
            );
            let pos = pos.unwrap_or(0);
            assert!(
                pos + 1 < args.len(),
                "case {queue}: flag {flag} has no value slot"
            );
            assert_eq!(
                &args[pos + 1],
                expected_value,
                "case {queue}: flag {flag} value mismatch"
            );
        }
    }
}
