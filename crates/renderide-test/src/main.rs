//! Golden-image integration test harness binary entry point.
//!
//! Acts as a minimal mock of the `FrooxEngine` host: opens the same Cloudtoid IPC queue layout
//! that Resonite uses, spawns the renderer in `--headless` mode, drives the init handshake plus a
//! single-sphere scene over IPC, then reads the PNG that the renderer writes to disk and compares
//! it against a committed golden via `image-compare` SSIM.
//!
//! All harness logic lives in the library at [`renderide_test`]; this binary only dispatches.

use std::process::ExitCode;

fn main() -> ExitCode {
    renderide_test::cli::run()
}
