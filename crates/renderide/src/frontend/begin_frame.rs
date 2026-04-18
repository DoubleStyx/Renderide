//! Pure predicate for whether the renderer should emit [`crate::shared::FrameStartData`] this tick.
//!
//! [`RendererFrontend::should_send_begin_frame`](crate::frontend::RendererFrontend::should_send_begin_frame)
//! delegates here so the lock-step rules are unit-testable without constructing a full frontend.

/// Returns `true` when init is complete, IPC is connected, there is no fatal handshake error, and
/// either the host has submitted a frame since the last begin-frame (`last_frame_data_processed`)
/// or the bootstrap begin-frame is still pending (`last_frame_index < 0` and no bootstrap send yet).
pub(crate) fn begin_frame_allowed(
    init_finalized: bool,
    fatal_error: bool,
    ipc_connected: bool,
    last_frame_data_processed: bool,
    last_frame_index: i32,
    sent_bootstrap_frame_start: bool,
) -> bool {
    if !init_finalized || fatal_error || !ipc_connected {
        return false;
    }
    let bootstrap = last_frame_index < 0 && !sent_bootstrap_frame_start;
    last_frame_data_processed || bootstrap
}

#[cfg(test)]
mod tests {
    use super::begin_frame_allowed;

    #[test]
    fn not_finalized_blocks() {
        assert!(!begin_frame_allowed(false, false, true, true, 0, true,));
    }

    #[test]
    fn fatal_blocks() {
        assert!(!begin_frame_allowed(true, true, true, true, 0, true,));
    }

    #[test]
    fn no_ipc_blocks() {
        assert!(!begin_frame_allowed(true, false, false, true, 0, true,));
    }

    #[test]
    fn finalized_ipc_processed_allows() {
        assert!(begin_frame_allowed(true, false, true, true, 5, true,));
    }

    #[test]
    fn bootstrap_before_first_submit_allows_without_processed_flag() {
        assert!(begin_frame_allowed(true, false, true, false, -1, false,));
    }

    #[test]
    fn after_bootstrap_without_new_submit_blocks() {
        assert!(!begin_frame_allowed(true, false, true, false, -1, true,));
    }

    #[test]
    fn positive_frame_index_without_processed_blocks_unless_bootstrap() {
        assert!(!begin_frame_allowed(true, false, true, false, 3, true,));
    }
}
