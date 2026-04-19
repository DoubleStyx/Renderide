//! Golden-image integration test harness for Renderide.
//!
//! Acts as a minimal mock of the FrooxEngine host: opens the same Cloudtoid IPC queue layout that
//! Resonite uses, spawns the renderer in `--headless` mode, drives the init handshake plus a
//! single-sphere scene over IPC, then reads the PNG that the renderer writes to disk and compares
//! it against a committed golden via `image-compare` SSIM.

#![warn(missing_docs)]

mod cli;
mod error;
mod golden;
mod host;
mod scene;

use std::process::ExitCode;

fn main() -> ExitCode {
    cli::run()
}
