//! Lights buffer command handlers: lights_buffer_renderer_submission.
//!
//! Parses LightsBufferRendererSubmission, reads LightData array from shared memory,
//! and stores in the scene graph's light cache.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::shared::{LightData, RendererCommand};

use super::{CommandContext, CommandHandler, CommandResult};

/// Whether we have logged the first LightsBufferRendererSubmission (diagnostic).
static LIGHTS_BUFFER_SUBMISSION_LOGGED: AtomicBool = AtomicBool::new(false);

/// Handles `lights_buffer_renderer_submission`. Reads light data from shared memory
/// and stores it in the scene graph's light cache.
pub struct LightsBufferCommandHandler;

impl CommandHandler for LightsBufferCommandHandler {
    fn handle(&mut self, cmd: &RendererCommand, ctx: &mut CommandContext<'_>) -> CommandResult {
        let RendererCommand::lights_buffer_renderer_submission(data) = cmd else {
            return CommandResult::Ignored;
        };

        if data.lights.is_empty() || data.lights_count <= 0 {
            ctx.scene_graph
                .light_cache
                .store_full(data.lights_buffer_unique_id, Vec::new());
            return CommandResult::Handled;
        }

        let Some(shm) = ctx.assets.shared_memory.as_mut() else {
            logger::warn!(
                "LightsBufferRendererSubmission: no shared memory (buffer_id={})",
                data.lights_buffer_unique_id
            );
            return CommandResult::Handled;
        };

        match shm.access_with_context::<LightData>(&data.lights, "LightsBufferRendererSubmission") {
            Ok(lights) => {
                if !lights.is_empty()
                    && !LIGHTS_BUFFER_SUBMISSION_LOGGED.swap(true, Ordering::Relaxed)
                {
                    logger::info!(
                        "First LightsBufferRendererSubmission received: buffer_id={} count={}",
                        data.lights_buffer_unique_id,
                        lights.len()
                    );
                }
                logger::trace!(
                    "LightsBufferRendererSubmission: buffer_id={} count={} lights=[{}]",
                    data.lights_buffer_unique_id,
                    lights.len(),
                    lights
                        .iter()
                        .map(|l| format!(
                            "pos=({:.2},{:.2},{:.2}) color=({:.2},{:.2},{:.2}) intensity={:.2} range={:.2}",
                            l.point.x, l.point.y, l.point.z,
                            l.color.x, l.color.y, l.color.z,
                            l.intensity, l.range
                        ))
                        .collect::<Vec<_>>()
                        .join("; ")
                );
                ctx.scene_graph
                    .light_cache
                    .store_full(data.lights_buffer_unique_id, lights);
            }
            Err(e) => {
                logger::warn!("LightsBufferRendererSubmission: {}", e);
            }
        }

        CommandResult::Handled
    }
}
