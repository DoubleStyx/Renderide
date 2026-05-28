//! Shared process slots used by watchdog and protocol threads.

use std::process::{Child, ExitStatus};
use std::sync::{Arc, Mutex};

/// Thread-safe storage for one child process handle.
#[derive(Clone, Default)]
pub(crate) struct SharedChildSlot {
    /// Synchronized child process handle storage.
    inner: Arc<Mutex<Option<Child>>>,
}

impl SharedChildSlot {
    /// Creates an empty process slot.
    pub(crate) fn empty() -> Self {
        Self {
            inner: Arc::new(Mutex::new(None)),
        }
    }

    /// Creates a process slot containing an already spawned child.
    pub(crate) fn with_child(child: Child) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Some(child))),
        }
    }

    /// Polls the stored process for exit without consuming the slot.
    pub(crate) fn poll_exit(&self) -> Result<ChildPoll, ChildSlotPoisoned> {
        let mut guard = self.inner.lock().map_err(|_poisoned| ChildSlotPoisoned)?;
        let poll = match guard.as_mut() {
            Some(child) => match child.try_wait() {
                Ok(Some(status)) => ChildPoll::Exited(status),
                Ok(None) => ChildPoll::Running,
                Err(e) => ChildPoll::WaitError(e),
            },
            None => ChildPoll::Missing,
        };
        drop(guard);
        Ok(poll)
    }

    /// Replaces the stored process and returns the previous process, if any.
    ///
    /// If the slot is poisoned, terminates the new process before returning an error.
    pub(crate) fn replace(&self, mut child: Child) -> Result<Option<Child>, ChildSlotPoisoned> {
        let Ok(mut guard) = self.inner.lock() else {
            let _ = child.kill();
            let _ = child.wait();
            return Err(ChildSlotPoisoned);
        };
        Ok(guard.replace(child))
    }

    /// Removes and returns the stored process.
    pub(crate) fn take(&self) -> Result<Option<Child>, ChildSlotPoisoned> {
        let mut guard = self.inner.lock().map_err(|_poisoned| ChildSlotPoisoned)?;
        Ok(guard.take())
    }
}

/// Result of polling a child slot for process exit.
pub(crate) enum ChildPoll {
    /// No child has been registered.
    Missing,
    /// The child is still running.
    Running,
    /// The child exited with the given status.
    Exited(ExitStatus),
    /// `Child::try_wait` failed.
    WaitError(std::io::Error),
}

/// Marker returned when a child-slot mutex is poisoned.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ChildSlotPoisoned;

#[cfg(test)]
mod tests {
    use std::process::Command;

    use super::{ChildPoll, SharedChildSlot};

    #[test]
    fn empty_slot_polls_as_missing() {
        let slot = SharedChildSlot::empty();
        assert!(matches!(slot.poll_exit(), Ok(ChildPoll::Missing)));
    }

    #[cfg(unix)]
    #[test]
    fn replace_returns_previous_child() {
        let slot = SharedChildSlot::empty();
        let first = Command::new("true").spawn().expect("spawn first");
        let second = Command::new("true").spawn().expect("spawn second");

        assert!(slot.replace(first).expect("replace first").is_none());
        let mut previous = slot.replace(second).expect("replace second");
        assert!(previous.is_some());
        if let Some(child) = previous.as_mut() {
            let _ = child.wait();
        }
        if let Ok(Some(mut child)) = slot.take() {
            let _ = child.wait();
        }
    }
}
