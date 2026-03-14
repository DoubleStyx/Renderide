//! Command translation from host protocol to platonic renderer primitives.
//!
//! Maps RendererCommand variants into TranslatedCommand for consumption by the Session.

use crate::core::{RenderConfig, SessionConfig};
use crate::shared::{FrameSubmitData, MeshUploadData, RendererCommand};

/// Translated command: engine-agnostic action for the Session to apply.
pub enum TranslatedCommand {
    /// Session initialization from host.
    SessionInit(SessionConfig),
    /// Shutdown requested.
    SessionShutdown,
    /// Init finalization (host signals ready).
    InitFinalize,
    /// Frame data from host (render spaces, camera tasks).
    FrameSubmit(FrameSubmitData),
    /// Mesh upload from shared memory.
    MeshUpload(MeshUploadData),
    /// Mesh unload.
    MeshUnload(i32),
    /// Render config update.
    ConfigUpdate(RenderConfig),
    /// No-op (e.g. KeepAlive).
    NoOp,
    /// Unimplemented command (logged, ignored).
    Unimplemented(&'static str),
}

/// Stateless command translator: maps host commands to platonic actions.
#[derive(Default)]
pub struct CommandMapper;

impl CommandMapper {
    /// Translates a host command into a platonic command for the Session.
    pub fn translate(&self, cmd: RendererCommand) -> TranslatedCommand {
        match cmd {
            RendererCommand::renderer_init_data(x) => {
                TranslatedCommand::SessionInit(SessionConfig {
                    shared_memory_prefix: x.shared_memory_prefix,
                    output_device: x.output_device,
                })
            }
            RendererCommand::renderer_shutdown(_) | RendererCommand::renderer_shutdown_request(_) => {
                TranslatedCommand::SessionShutdown
            }
            RendererCommand::renderer_init_finalize_data(_) => TranslatedCommand::InitFinalize,
            RendererCommand::frame_submit_data(x) => TranslatedCommand::FrameSubmit(x),
            RendererCommand::mesh_upload_data(x) => TranslatedCommand::MeshUpload(x),
            RendererCommand::mesh_unload(x) => TranslatedCommand::MeshUnload(x.asset_id),
            RendererCommand::desktop_config(x) => TranslatedCommand::ConfigUpdate(RenderConfig {
                near_clip: 0.01,
                far_clip: 1024.0,
                desktop_fov: 75.0,
                vsync: x.v_sync,
            }),
            RendererCommand::keep_alive(_) => TranslatedCommand::NoOp,
            RendererCommand::renderer_init_progress_update(_) => TranslatedCommand::NoOp,
            RendererCommand::renderer_engine_ready(_) => TranslatedCommand::NoOp,
            RendererCommand::renderer_init_result(_) => TranslatedCommand::NoOp,
            RendererCommand::frame_start_data(_) => TranslatedCommand::NoOp,
            _ => TranslatedCommand::Unimplemented("texture/material/shader/etc"),
        }
    }
}
