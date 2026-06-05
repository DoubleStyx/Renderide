//! Reliable background IPC acknowledgement helpers for asset transfers.

use crate::ipc::DualQueueIpc;
use crate::shared::RendererCommand;

/// Sends a reliable background command when IPC is available.
pub(in crate::backend::asset_transfers) fn send_background_reliable(
    ipc: &mut Option<&mut DualQueueIpc>,
    command: RendererCommand,
    enqueue_failure: impl FnOnce() -> String,
) -> bool {
    let Some(ipc) = ipc.as_mut() else {
        return false;
    };
    let queued = ipc.send_background_reliable(command);
    if !queued {
        logger::warn!("{}", enqueue_failure());
    }
    queued
}

/// Enqueues a reliable background command when IPC is available.
pub(in crate::backend::asset_transfers) fn enqueue_background_reliable(
    ipc: &mut Option<&mut DualQueueIpc>,
    command: RendererCommand,
    enqueue_failure: impl FnOnce() -> String,
) -> bool {
    let Some(ipc) = ipc.as_mut() else {
        return false;
    };
    let queued = ipc.enqueue_background_reliable(command);
    if !queued {
        logger::warn!("{}", enqueue_failure());
    }
    queued
}
