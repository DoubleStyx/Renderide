//! Config command handlers: desktop_config, resolution_config, quality_config, etc.

use crate::config::RenderConfig;
use crate::shared::RendererCommand;

use super::{CommandContext, CommandHandler, CommandResult};

/// Handles `desktop_config`. Updates view state and render config. Post-finalize only.
pub struct ConfigCommandHandler;

impl CommandHandler for ConfigCommandHandler {
    fn handle(&mut self, cmd: &RendererCommand, ctx: &mut CommandContext<'_>) -> CommandResult {
        if !ctx.session_flags.init_state.is_finalized() {
            return CommandResult::Ignored;
        }
        match cmd {
            RendererCommand::desktop_config(x) => {
                *ctx.render_config = RenderConfig {
                    near_clip: ctx.view_state.near_clip,
                    far_clip: ctx.view_state.far_clip,
                    desktop_fov: ctx.view_state.desktop_fov,
                    vsync: x.v_sync,
                    use_debug_uv: ctx.render_config.use_debug_uv,
                    use_pbr: ctx.render_config.use_pbr,
                    skinned_apply_mesh_root_transform: ctx
                        .render_config
                        .skinned_apply_mesh_root_transform,
                    skinned_use_root_bone: ctx.render_config.skinned_use_root_bone,
                    debug_skinned: ctx.render_config.debug_skinned,
                    debug_blendshapes: ctx.render_config.debug_blendshapes,
                    skinned_flip_handedness: ctx.render_config.skinned_flip_handedness,
                    rtao_enabled: ctx.render_config.rtao_enabled,
                    rtao_strength: ctx.render_config.rtao_strength,
                    ao_radius: ctx.render_config.ao_radius,
                };
                CommandResult::Handled
            }
            _ => CommandResult::Ignored,
        }
    }
}
