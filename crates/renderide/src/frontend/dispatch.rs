//! IPC command dispatch: routes incoming `RendererCommand`s to runtime / backend / scene handlers.
//!
//! Pulls IPC fan-out out of `crate::runtime/` so transport-shaped routing lives next to the queue
//! drain in `crate::frontend`. The dispatcher receives a `&mut RendererRuntime` from
//! `runtime::RendererRuntime::poll_ipc` and walks each `RendererCommand` variant; payload application
//! is delegated to the right runtime / backend / scene method.

pub(crate) mod command_dispatch;
pub(crate) mod commands;
pub(crate) mod frame_submit;
pub(crate) mod host_camera_apply;
pub(crate) mod ipc_init;
pub(crate) mod lights_ipc;
pub(crate) mod renderer_command_kind;
pub(crate) mod shader_material_ipc;
