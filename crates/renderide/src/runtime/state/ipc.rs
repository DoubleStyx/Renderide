//! Runtime-owned IPC scratch and counters.

use std::collections::VecDeque;

use hashbrown::HashMap;

use crate::frontend::dispatch::renderer_command_kind::renderer_command_variant_tag;
use crate::ipc::TimedRendererCommand;
use crate::shared::{RendererCommand, SetWindowIcon};

const DEFERRED_PRE_FINALIZE_WARN_THRESHOLD: usize = 1024;

/// IPC scratch state that is not part of transport ownership.
pub(in crate::runtime) struct RuntimeIpcState {
    /// In-flight shader uploads whose resolution is running on the rayon pool.
    pub(in crate::runtime) pending_shader_resolutions:
        Vec<crate::runtime::ipc::shader_material::PendingShaderResolution>,
    /// Host commands received after init data but before init finalization.
    deferred_pre_finalize_commands: VecDeque<TimedRendererCommand>,
    /// Host window-icon requests waiting for the app driver to apply them to the winit window.
    pending_window_icon_requests: VecDeque<SetWindowIcon>,
    /// Running counts of post-init renderer command variants seen without a running handler.
    unhandled_ipc_command_counts: HashMap<&'static str, u64>,
}

/// Summarizes a renderer-command collection as `Variant=count` pairs for diagnostic logs.
pub(in crate::runtime) fn summarize_renderer_command_mix<'a>(
    commands: impl IntoIterator<Item = &'a RendererCommand>,
) -> String {
    let mut counts: Vec<(&'static str, usize)> = Vec::new();
    for cmd in commands {
        let tag = renderer_command_variant_tag(cmd);
        if let Some((_, count)) = counts.iter_mut().find(|(existing, _)| *existing == tag) {
            *count += 1;
        } else {
            counts.push((tag, 1));
        }
    }
    let mut out = String::new();
    for (idx, (tag, count)) in counts.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push_str(tag);
        out.push('=');
        out.push_str(&count.to_string());
    }
    out
}

impl RuntimeIpcState {
    /// Creates empty IPC scratch state.
    pub(in crate::runtime) fn new() -> Self {
        Self {
            pending_shader_resolutions: Vec::new(),
            deferred_pre_finalize_commands: VecDeque::new(),
            pending_window_icon_requests: VecDeque::new(),
            unhandled_ipc_command_counts: HashMap::new(),
        }
    }

    /// Defers a host command received before init finalization.
    pub(in crate::runtime) fn defer_pre_finalize_command(&mut self, cmd: TimedRendererCommand) {
        self.deferred_pre_finalize_commands.push_back(cmd);
        let count = self.deferred_pre_finalize_commands.len();
        if count == DEFERRED_PRE_FINALIZE_WARN_THRESHOLD
            || (count > DEFERRED_PRE_FINALIZE_WARN_THRESHOLD
                && (count - DEFERRED_PRE_FINALIZE_WARN_THRESHOLD)
                    .is_multiple_of(DEFERRED_PRE_FINALIZE_WARN_THRESHOLD))
        {
            let mix = summarize_renderer_command_mix(
                self.deferred_pre_finalize_commands
                    .iter()
                    .map(|cmd| &cmd.command),
            );
            logger::warn!(
                "IPC: {count} commands queued while waiting for init finalization kinds=[{mix}]"
            );
        }
    }

    /// Drains deferred pre-finalize commands in host arrival order.
    pub(in crate::runtime) fn take_deferred_pre_finalize_commands(
        &mut self,
    ) -> VecDeque<TimedRendererCommand> {
        std::mem::take(&mut self.deferred_pre_finalize_commands)
    }

    /// Queues a host window-icon request for app-thread application.
    pub(in crate::runtime) fn queue_window_icon_request(&mut self, request: SetWindowIcon) {
        self.pending_window_icon_requests.push_back(request);
    }

    /// Drains host window-icon requests in arrival order.
    pub(in crate::runtime) fn take_pending_window_icon_requests(
        &mut self,
    ) -> VecDeque<SetWindowIcon> {
        std::mem::take(&mut self.pending_window_icon_requests)
    }

    /// Records one unhandled renderer command variant.
    pub(in crate::runtime) fn record_unhandled_renderer_command(
        &mut self,
        tag: &'static str,
    ) -> u64 {
        let count = self.unhandled_ipc_command_counts.entry(tag).or_insert(0);
        *count += 1;
        *count
    }

    /// Total number of unhandled post-handshake renderer commands.
    pub(in crate::runtime) fn unhandled_command_event_total(&self) -> u64 {
        self.unhandled_ipc_command_counts.values().copied().sum()
    }

    /// Number of pending host window-icon requests.
    #[cfg(test)]
    pub(in crate::runtime) fn pending_window_icon_request_count(&self) -> usize {
        self.pending_window_icon_requests.len()
    }

    /// Number of commands waiting for init finalization replay.
    #[cfg(test)]
    pub(in crate::runtime) fn deferred_pre_finalize_command_count(&self) -> usize {
        self.deferred_pre_finalize_commands.len()
    }
}

#[cfg(test)]
mod tests {
    use super::RuntimeIpcState;
    use crate::ipc::TimedRendererCommand;
    use crate::shared::{KeepAlive, QualityConfig, RendererCommand};

    fn timed(cmd: RendererCommand) -> TimedRendererCommand {
        TimedRendererCommand::received_now(cmd)
    }

    #[test]
    fn unhandled_command_total_sums_variant_counts() {
        let mut state = RuntimeIpcState::new();

        state.record_unhandled_renderer_command("Foo");
        state.record_unhandled_renderer_command("Foo");
        state.record_unhandled_renderer_command("Bar");

        assert_eq!(state.unhandled_command_event_total(), 3);
    }

    #[test]
    fn deferred_pre_finalize_commands_drain_fifo() {
        let mut state = RuntimeIpcState::new();

        state.defer_pre_finalize_command(timed(RendererCommand::QualityConfig(QualityConfig {
            per_pixel_lights: 1,
            ..Default::default()
        })));
        state.defer_pre_finalize_command(timed(RendererCommand::QualityConfig(QualityConfig {
            per_pixel_lights: 2,
            ..Default::default()
        })));

        assert_eq!(state.deferred_pre_finalize_command_count(), 2);
        let mut drained = state.take_deferred_pre_finalize_commands();
        assert_eq!(state.deferred_pre_finalize_command_count(), 0);

        match drained.pop_front() {
            Some(timed) => match timed.command {
                RendererCommand::QualityConfig(cfg) => assert_eq!(cfg.per_pixel_lights, 1),
                other => panic!("unexpected first deferred command: {other:?}"),
            },
            other => panic!("unexpected first deferred command: {other:?}"),
        }
        match drained.pop_front() {
            Some(timed) => match timed.command {
                RendererCommand::QualityConfig(cfg) => assert_eq!(cfg.per_pixel_lights, 2),
                other => panic!("unexpected second deferred command: {other:?}"),
            },
            other => panic!("unexpected second deferred command: {other:?}"),
        }
        assert!(drained.is_empty());
    }

    #[test]
    fn command_mix_summary_counts_tags() {
        let commands = [
            RendererCommand::KeepAlive(KeepAlive::default()),
            RendererCommand::QualityConfig(QualityConfig::default()),
            RendererCommand::QualityConfig(QualityConfig {
                per_pixel_lights: 2,
                ..Default::default()
            }),
        ];
        let summary = super::summarize_renderer_command_mix(commands.iter());
        assert!(summary.contains("KeepAlive=1"), "summary={summary}");
        assert!(summary.contains("QualityConfig=2"), "summary={summary}");
    }
}
