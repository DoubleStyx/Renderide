//! Session: orchestrates IPC, scene, assets, and frame flow.

#![allow(clippy::module_inception)]

pub mod collect;
pub mod commands;
pub mod frame_data;
pub(crate) mod frame_perf;
pub mod init;
pub mod native_ui_routing_metrics;
pub mod session;
pub mod state;

pub use commands::{
    AssetContext, CommandContext, CommandDispatcher, CommandResult, FrameContext, SessionFlags,
};
pub use session::Session;
pub(crate) use session::SpaceCollectTimingSplit;
