//! Reliable-background FIFO outbox used by [`super::dual_queue::DualQueueIpc`].
//!
//! When the renderer needs a background message to definitely reach the host (e.g. an asset
//! result that the host is waiting on), it parks the encoded bytes in this queue and retries
//! until the publisher accepts them. The outbox tracks message count and pending byte total so
//! the renderer can log backpressure summaries without re-scanning the queue.

use std::collections::VecDeque;

/// Maximum retained reliable background messages under peer backpressure.
pub(super) const RELIABLE_BACKGROUND_OUTBOX_MAX_MESSAGES: usize = 4096;

/// FIFO of encoded reliable-background payloads waiting on publisher capacity.
#[derive(Default)]
pub(super) struct ReliableBackgroundOutbox {
    payloads: VecDeque<Vec<u8>>,
    pending_bytes: usize,
}

impl ReliableBackgroundOutbox {
    /// Appends `payload` to the tail of the queue and grows [`Self::pending_bytes`].
    pub(super) fn enqueue(&mut self, payload: Vec<u8>) -> bool {
        if self.payloads.len() >= RELIABLE_BACKGROUND_OUTBOX_MAX_MESSAGES {
            return false;
        }
        self.pending_bytes = self.pending_bytes.saturating_add(payload.len());
        self.payloads.push_back(payload);
        true
    }

    /// Returns the head payload (if any) without consuming it.
    pub(super) fn front(&self) -> Option<&[u8]> {
        self.payloads.front().map(Vec::as_slice)
    }

    /// Drops the head payload after the publisher accepted it.
    pub(super) fn mark_front_sent(&mut self) {
        if let Some(payload) = self.payloads.pop_front() {
            self.pending_bytes = self.pending_bytes.saturating_sub(payload.len());
        }
    }

    /// Whether the outbox holds no pending payloads.
    pub(super) fn is_empty(&self) -> bool {
        self.payloads.is_empty()
    }

    /// Number of payloads currently waiting for publisher capacity.
    pub(super) fn len(&self) -> usize {
        self.payloads.len()
    }

    /// Sum of byte lengths of all pending payloads.
    pub(super) fn pending_bytes(&self) -> usize {
        self.pending_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_fifo_and_byte_count() {
        let mut outbox = ReliableBackgroundOutbox::default();

        assert!(outbox.enqueue(vec![1, 2, 3]));
        assert!(outbox.enqueue(vec![4, 5]));

        assert_eq!(outbox.len(), 2);
        assert_eq!(outbox.pending_bytes(), 5);
        assert_eq!(outbox.front(), Some(&[1, 2, 3][..]));

        outbox.mark_front_sent();

        assert_eq!(outbox.len(), 1);
        assert_eq!(outbox.pending_bytes(), 2);
        assert_eq!(outbox.front(), Some(&[4, 5][..]));

        outbox.mark_front_sent();

        assert!(outbox.is_empty());
        assert_eq!(outbox.pending_bytes(), 0);
    }

    #[test]
    fn empty_mark_sent_is_noop() {
        let mut outbox = ReliableBackgroundOutbox::default();

        outbox.mark_front_sent();

        assert!(outbox.is_empty());
        assert_eq!(outbox.len(), 0);
        assert_eq!(outbox.pending_bytes(), 0);
        assert_eq!(outbox.front(), None);
    }

    #[test]
    fn counts_zero_length_payloads_as_messages() {
        let mut outbox = ReliableBackgroundOutbox::default();

        assert!(outbox.enqueue(Vec::new()));
        assert!(outbox.enqueue(vec![1]));

        assert_eq!(outbox.len(), 2);
        assert_eq!(outbox.pending_bytes(), 1);
        assert_eq!(outbox.front(), Some(&[][..]));
        outbox.mark_front_sent();
        assert_eq!(outbox.front(), Some(&[1][..]));
        assert_eq!(outbox.pending_bytes(), 1);
    }

    #[test]
    fn rejects_payloads_after_message_cap() {
        let mut outbox = ReliableBackgroundOutbox::default();
        for _ in 0..RELIABLE_BACKGROUND_OUTBOX_MAX_MESSAGES {
            assert!(outbox.enqueue(vec![1]));
        }

        assert!(!outbox.enqueue(vec![2]));
        assert_eq!(outbox.len(), RELIABLE_BACKGROUND_OUTBOX_MAX_MESSAGES);
        assert_eq!(
            outbox.pending_bytes(),
            RELIABLE_BACKGROUND_OUTBOX_MAX_MESSAGES
        );
    }
}
