//! Graceful renderer shutdown: send `RendererShutdownRequest`, wait, escalate to kill.

use std::time::Instant;

use renderide_shared::ipc::HostDualQueueIpc;
use renderide_shared::shared::{RendererCommand, RendererShutdownRequest};

use crate::error::HarnessError;

use super::consts::timing;
use super::spawn::SpawnedRenderer;

/// Sends `RendererShutdownRequest` and waits for the child to exit voluntarily within
/// [`timing::SHUTDOWN_GRACE`]. Falls back to `kill()` on timeout. Always succeeds.
pub(super) fn request_shutdown_and_wait(
    queues: &mut HostDualQueueIpc,
    spawned: &mut SpawnedRenderer,
) -> Result<(), HarnessError> {
    let _ = queues.send_primary(RendererCommand::RendererShutdownRequest(
        RendererShutdownRequest {},
    ));
    logger::info!("Session: sent RendererShutdownRequest, waiting for child to exit");

    let deadline = Instant::now() + timing::SHUTDOWN_GRACE;
    while Instant::now() < deadline {
        if let Some(child) = spawned.child.as_mut() {
            if let Ok(Some(_status)) = child.try_wait() {
                spawned.child = None;
                return Ok(());
            }
        }
        std::thread::sleep(timing::SHUTDOWN_POLL);
    }
    logger::warn!(
        "Session: renderer did not exit within {:?}, killing",
        timing::SHUTDOWN_GRACE
    );
    if let Some(child) = spawned.child.as_mut() {
        let _ = child.kill();
        let _ = child.wait();
        spawned.child = None;
    }
    Ok(())
}
