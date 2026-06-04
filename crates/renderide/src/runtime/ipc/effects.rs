//! IPC dispatch effect application on [`RendererRuntime`].
//!
//! [`Self::handle_timed_ipc_command`] decodes one renderer command via
//! [`crate::frontend::dispatch`] and routes it through [`Self::apply_ipc_dispatch_effect`].
//! Post-handshake commands fan out to per-domain effect handlers under [`mod@self`]'s submodules.

use std::time::Instant;

use crate::diagnostics::crash_context::{self, InitState as CrashInitState};
use crate::frontend::InitState;
use crate::frontend::dispatch::command_dispatch::RunningCommandEffect;
use crate::frontend::dispatch::ipc_init::{self, IpcDispatchEffect};
use crate::ipc::TimedRendererCommand;
#[cfg(test)]
use crate::shared::RendererCommand;

use super::super::RendererRuntime;

mod auxiliary_assets;
mod init_capabilities;
mod materials_shaders;
mod texture_assets;
mod video_textures;

use init_capabilities::log_frame_start_data_trace;

impl RendererRuntime {
    /// Decodes and applies one IPC command according to the current init state.
    #[cfg(test)]
    pub(crate) fn handle_ipc_command(&mut self, cmd: RendererCommand) {
        self.handle_timed_ipc_command(TimedRendererCommand::received_now(cmd));
    }

    /// Decodes and applies one timestamped IPC command according to the current init state.
    pub(crate) fn handle_timed_ipc_command(&mut self, cmd: TimedRendererCommand) {
        let received_at = cmd.received_at;
        let effect = ipc_init::dispatch_ipc_command(self.frontend.init_state(), cmd.command);
        self.apply_ipc_dispatch_effect(effect, received_at);
    }

    /// Applies an init-routed command effect.
    pub(crate) fn apply_ipc_dispatch_effect(
        &mut self,
        effect: IpcDispatchEffect,
        received_at: Instant,
    ) {
        match effect {
            IpcDispatchEffect::Ignore => {}
            IpcDispatchEffect::ApplyInitData(d) => {
                self.apply_renderer_init_data(d);
                crash_context::set_init_state(CrashInitState::InitDataReceived);
            }
            IpcDispatchEffect::Finalize => {
                logger::info!("IPC init finalized; renderer entering running command dispatch");
                self.frontend.set_init_state(InitState::Finalized);
                crash_context::set_init_state(CrashInitState::Finalized);
                self.replay_deferred_pre_finalize_commands();
            }
            IpcDispatchEffect::DispatchRunning(effect) => {
                self.apply_running_command_effect_received_at(effect, received_at);
            }
            IpcDispatchEffect::DeferUntilFinalized(cmd) => {
                logger::trace!("IPC: deferring command until init finalized");
                self.ipc_state
                    .defer_pre_finalize_command(TimedRendererCommand::new(*cmd, received_at));
            }
            IpcDispatchEffect::FatalExpectedInitData { actual_tag } => {
                logger::error!(
                    "IPC: expected RendererInitData first, received RendererCommand::{actual_tag}\n{}",
                    crash_context::format_snapshot()
                );
                self.frontend.set_fatal_error(true);
            }
        }
    }

    /// Replays commands that arrived after init data and before init finalization.
    pub(crate) fn replay_deferred_pre_finalize_commands(&mut self) {
        let mut deferred = self.ipc_state.take_deferred_pre_finalize_commands();
        if deferred.is_empty() {
            return;
        }
        let mix = super::super::state::ipc::summarize_renderer_command_mix(
            deferred.iter().map(|cmd| &cmd.command),
        );
        logger::info!(
            "IPC init finalized; replaying {} deferred command(s) mix=[{}]",
            deferred.len(),
            mix
        );
        while let Some(cmd) = deferred.pop_front() {
            self.handle_timed_ipc_command(cmd);
            if self.frontend.fatal_error() {
                break;
            }
        }
    }

    /// Applies a decoded post-init command effect to runtime-owned domains.
    #[cfg(test)]
    pub(crate) fn apply_running_command_effect(&mut self, effect: RunningCommandEffect) {
        self.apply_running_command_effect_received_at(effect, Instant::now());
    }

    fn apply_running_command_effect_received_at(
        &mut self,
        effect: RunningCommandEffect,
        received_at: Instant,
    ) {
        match effect {
            RunningCommandEffect::KeepAlive => {}
            RunningCommandEffect::RequestShutdown => self.frontend.set_shutdown_requested(true),
            RunningCommandEffect::FrameSubmit(data) => {
                self.apply_frame_submit_data_received_at(data, received_at);
            }
            RunningCommandEffect::MeshUpload(d) => self.process_mesh_upload(d),
            RunningCommandEffect::MeshUnload(u) => self.backend.on_mesh_unload(u),
            effect @ (RunningCommandEffect::SetTexture2DFormat(_)
            | RunningCommandEffect::SetTexture2DProperties(_)
            | RunningCommandEffect::SetTexture2DData(_)
            | RunningCommandEffect::UnloadTexture2D(_)
            | RunningCommandEffect::SetTexture3DFormat(_)
            | RunningCommandEffect::SetTexture3DProperties(_)
            | RunningCommandEffect::SetTexture3DData(_)
            | RunningCommandEffect::UnloadTexture3D(_)
            | RunningCommandEffect::SetCubemapFormat(_)
            | RunningCommandEffect::SetCubemapProperties(_)
            | RunningCommandEffect::SetCubemapData(_)
            | RunningCommandEffect::UnloadCubemap(_)
            | RunningCommandEffect::SetRenderTextureFormat(_)
            | RunningCommandEffect::UnloadRenderTexture(_)) => {
                self.apply_texture_asset_effect(effect);
            }
            effect @ (RunningCommandEffect::SetDesktopTextureProperties(_)
            | RunningCommandEffect::DesktopTexturePropertiesUpdate(_)
            | RunningCommandEffect::UnloadDesktopTexture(_)
            | RunningCommandEffect::PointRenderBufferUpload(_)
            | RunningCommandEffect::PointRenderBufferUnload(_)
            | RunningCommandEffect::TrailRenderBufferUpload(_)
            | RunningCommandEffect::TrailRenderBufferUnload(_)
            | RunningCommandEffect::GaussianSplatConfig(_)
            | RunningCommandEffect::GaussianSplatUploadRaw(_)
            | RunningCommandEffect::GaussianSplatUploadEncoded(_)
            | RunningCommandEffect::UnloadGaussianSplat(_)
            | RunningCommandEffect::PointRenderBufferConsumed
            | RunningCommandEffect::TrailRenderBufferConsumed
            | RunningCommandEffect::GaussianSplatResult) => {
                self.apply_auxiliary_asset_effect(effect);
            }
            effect @ (RunningCommandEffect::VideoTextureLoad(_)
            | RunningCommandEffect::VideoTextureUpdate(_)
            | RunningCommandEffect::VideoTextureProperties(_)
            | RunningCommandEffect::VideoTextureStartAudioTrack(_)
            | RunningCommandEffect::UnloadVideoTexture(_)) => {
                self.apply_video_texture_effect(effect);
            }
            RunningCommandEffect::FreeSharedMemoryView { buffer_id } => {
                self.release_shared_memory_view(buffer_id);
            }
            RunningCommandEffect::SetWindowIcon(icon) => self.queue_window_icon_request(icon),
            effect @ (RunningCommandEffect::MaterialPropertyIdRequest(_)
            | RunningCommandEffect::MaterialsUpdateBatch(_)
            | RunningCommandEffect::UnloadMaterial { .. }
            | RunningCommandEffect::UnloadMaterialPropertyBlock { .. }
            | RunningCommandEffect::ShaderUpload(_)
            | RunningCommandEffect::ShaderUnload(_)) => self.apply_material_shader_effect(effect),
            RunningCommandEffect::FrameStartData(fs) => log_frame_start_data_trace(fs.as_ref()),
            RunningCommandEffect::LightsBufferRendererSubmission(sub) => {
                self.apply_lights_buffer_renderer_submission(sub);
            }
            RunningCommandEffect::LightsBufferRendererConsumed => {
                logger::trace!("runtime: lights_buffer_renderer_consumed from host (ignored)");
            }
            RunningCommandEffect::RenderTextureResult => {
                logger::trace!(
                    "runtime: render_texture_result from host (ignored; renderer is source)"
                );
            }
            RunningCommandEffect::RendererEngineReady => {
                logger::trace!(
                    "runtime: renderer_engine_ready from host; enabling strict frame lockstep"
                );
                self.frontend.on_renderer_engine_ready();
            }
            RunningCommandEffect::DesktopConfig(cfg) => self.apply_desktop_config(cfg),
            RunningCommandEffect::QualityConfig(cfg) => self.apply_quality_config(cfg),
            RunningCommandEffect::RenderDecouplingConfig(cfg) => {
                self.apply_render_decoupling_config(cfg);
            }
            RunningCommandEffect::Unhandled { tag } => self.note_unhandled_renderer_command(tag),
        }
    }

    fn note_unhandled_renderer_command(&mut self, tag: &'static str) {
        let count = self.record_unhandled_renderer_command(tag);
        if count == 1 {
            logger::warn!(
                "runtime: no handler for RendererCommand::{tag} (host sent unexpected command; further occurrences counted in diagnostics)"
            );
        } else {
            logger::trace!(
                "runtime: no handler for RendererCommand::{tag} occurrence_count={count}"
            );
        }
    }
}
