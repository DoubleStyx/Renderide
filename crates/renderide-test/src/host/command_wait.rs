//! Shared lockstep-pumping wait helper for host-side renderer command acknowledgements.
//!
//! The renderer may emit `FrameStartData` while the harness is waiting for an unrelated
//! acknowledgement on the background queue. Every wait path must therefore keep driving the
//! lockstep loop or the renderer can stall waiting for `FrameSubmitData`.

use std::time::{Duration, Instant};

use renderide_shared::ipc::HostDualQueueIpc;
use renderide_shared::shared::RendererCommand;

use crate::error::HarnessError;

use super::lockstep::LockstepDriver;

/// Poll cadence used while waiting for a command acknowledgement.
pub(super) const COMMAND_WAIT_POLL: Duration = Duration::from_millis(2);

/// Pumps lockstep until `match_command` extracts a value from a drained renderer command.
pub(super) fn wait_for_command<T>(
    queues: &mut HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
    timeout: Duration,
    timeout_error: impl FnOnce(Duration) -> HarnessError,
    mut match_command: impl FnMut(RendererCommand) -> Option<T>,
) -> Result<T, HarnessError> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        let tick = lockstep.tick(queues);
        for msg in tick.other_messages {
            if let Some(value) = match_command(msg) {
                return Ok(value);
            }
        }
        std::thread::sleep(COMMAND_WAIT_POLL);
    }
    Err(timeout_error(timeout))
}
