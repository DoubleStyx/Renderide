//! Runtime-owned state holders that aggregate as fields on [`super::RendererRuntime`].
//!
//! Each submodule owns one compact, single-responsibility state struct. They are reachable from
//! the runtime root through the re-exports below so [`super::RendererRuntime`] can keep its
//! field-type imports as a flat list.

pub(super) mod config;
pub(super) mod diagnostics;
pub(super) mod ipc;
pub(super) mod tick;
pub(super) mod xr;

pub(crate) use config::DesktopFramePacingCaps;
pub(super) use config::RuntimeConfigState;
pub(super) use diagnostics::RuntimeDiagnosticsState;
pub(super) use ipc::RuntimeIpcState;
pub(super) use tick::RuntimeTickState;
pub(super) use xr::RuntimeXrStats;
