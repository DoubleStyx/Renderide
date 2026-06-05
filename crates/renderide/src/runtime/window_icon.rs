//! Runtime-owned host window-icon request queue and acknowledgements.

use std::collections::VecDeque;

use crate::shared::{RendererCommand, SetWindowIcon, SetWindowIconResult};

use super::RendererRuntime;

const MAX_WINDOW_ICON_BYTES: i32 = 4 * 1024 * 1024;

impl RendererRuntime {
    /// Queues a host `SetWindowIcon` request for app-thread application.
    pub(crate) fn queue_window_icon_request(&mut self, request: SetWindowIcon) {
        if request.is_overlay {
            logger::warn!(
                "runtime: taskbar overlay window icons are unsupported request_id={}",
                request.request_id
            );
            self.send_window_icon_result(request.request_id, false);
            return;
        }
        self.ipc_state.queue_window_icon_request(request);
    }

    /// Drains pending host window-icon requests in arrival order.
    pub(crate) fn take_pending_window_icon_requests(&mut self) -> VecDeque<SetWindowIcon> {
        self.ipc_state.take_pending_window_icon_requests()
    }

    /// Loads a host window-icon BGRA32 payload from shared memory.
    pub(crate) fn load_window_icon_bgra(&mut self, request: &SetWindowIcon) -> Option<Vec<u8>> {
        if request.icon_data.length > MAX_WINDOW_ICON_BYTES {
            logger::warn!(
                "runtime: rejected host window icon request_id={} length={} cap={}",
                request.request_id,
                request.icon_data.length,
                MAX_WINDOW_ICON_BYTES
            );
            return None;
        }
        let Some(shm) = self.frontend.shared_memory_mut() else {
            logger::warn!(
                "runtime: cannot load host window icon request_id={} because shared memory is unavailable",
                request.request_id
            );
            return None;
        };
        let Some(bytes) = shm.access_copy::<u8>(&request.icon_data) else {
            logger::warn!(
                "runtime: failed to copy host window icon shared-memory payload request_id={} buffer_id={} offset={} length={}",
                request.request_id,
                request.icon_data.buffer_id,
                request.icon_data.offset,
                request.icon_data.length
            );
            return None;
        };
        Some(bytes)
    }

    /// Sends the host acknowledgement for a `SetWindowIcon` request.
    pub(crate) fn send_window_icon_result(&mut self, request_id: i32, success: bool) {
        let Some(ipc) = self.frontend.ipc_mut() else {
            return;
        };
        if !ipc.send_background_reliable(RendererCommand::SetWindowIconResult(
            SetWindowIconResult {
                request_id,
                success,
            },
        )) {
            logger::warn!(
                "runtime: SetWindowIconResult was not queued request_id={} success={}",
                request_id,
                success
            );
        }
    }

    /// Number of pending host window-icon requests.
    #[cfg(test)]
    pub(crate) fn pending_window_icon_request_count(&self) -> usize {
        self.ipc_state.pending_window_icon_request_count()
    }
}
