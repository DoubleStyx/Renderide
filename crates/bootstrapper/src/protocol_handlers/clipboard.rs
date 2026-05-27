//! Clipboard command handlers.

use interprocess::Publisher;

use crate::protocol::LoopAction;

/// Reads the system clipboard and enqueues UTF-8 bytes, or an empty string on failure.
pub(super) fn handle_get_text(outgoing: &mut Publisher) -> LoopAction {
    logger::info!("Getting clipboard text");
    let text = match arboard::Clipboard::new().and_then(|mut c| c.get_text()) {
        Ok(t) => t,
        Err(e) => {
            logger::warn!("Clipboard read failed, returning empty string to Host: {e}");
            String::new()
        }
    };
    if !outgoing.try_enqueue(text.as_bytes()) {
        logger::warn!("Failed to enqueue GETTEXT response on bootstrapper_out");
    }
    LoopAction::Continue
}

/// Writes UTF-8 text to the system clipboard on a best-effort basis.
pub(super) fn handle_set_text(text: &str) -> LoopAction {
    logger::info!("Setting clipboard text");
    if let Ok(mut clipboard) = arboard::Clipboard::new() {
        let _ = clipboard.set_text(text);
    }
    LoopAction::Continue
}
