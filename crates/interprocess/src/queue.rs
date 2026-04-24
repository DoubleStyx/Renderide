//! Factory for [`crate::Publisher`] / [`crate::Subscriber`].

use crate::error::OpenError;
use crate::options::QueueOptions;
use crate::Publisher;
use crate::Subscriber;

/// Stateless builder for [`Subscriber`] and [`Publisher`] (mirrors the managed `QueueFactory` type).
///
/// The managed API constructs a factory once and then opens endpoints; this Rust type carries no
/// fields so it can be copied freely and used as a namespace for constructor helpers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QueueFactory;

impl QueueFactory {
    /// Creates a factory with no state; matches the managed `QueueFactory` usage pattern.
    pub fn new() -> Self {
        Self
    }

    /// Opens a subscriber for `options`.
    pub fn create_subscriber(&self, options: QueueOptions) -> Result<Subscriber, OpenError> {
        Subscriber::new(options)
    }

    /// Opens a publisher for `options`.
    pub fn create_publisher(&self, options: QueueOptions) -> Result<Publisher, OpenError> {
        Publisher::new(options)
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::QueueFactory;
    use crate::options::QueueOptions;

    #[test]
    fn queue_factory_default_matches_new() {
        assert_eq!(QueueFactory, QueueFactory::new());
        assert_eq!(size_of::<QueueFactory>(), 0);
    }

    #[test]
    fn queue_factory_creates_publisher_and_subscriber() {
        let dir = tempdir().expect("tempdir");
        let opts = QueueOptions::with_path("qf_queue", dir.path(), 4096).expect("valid options");
        let factory = QueueFactory::new();
        let mut publisher = factory.create_publisher(opts.clone()).expect("publisher");
        let mut subscriber = factory.create_subscriber(opts).expect("subscriber");

        assert!(publisher.try_enqueue(b"via_factory"));
        assert_eq!(
            subscriber.try_dequeue().as_deref(),
            Some(b"via_factory".as_slice())
        );
    }

    #[test]
    fn queue_factory_subscriber_before_publisher_roundtrip() {
        let dir = tempdir().expect("tempdir");
        let name = format!("qf_sub_first_{}", std::process::id());
        let opts = QueueOptions::with_path(&name, dir.path(), 4096).expect("valid options");
        let factory = QueueFactory::new();
        let mut subscriber = factory.create_subscriber(opts.clone()).expect("subscriber");
        let mut publisher = factory.create_publisher(opts).expect("publisher");

        assert!(publisher.try_enqueue(b"sub_first"));
        assert_eq!(
            subscriber.try_dequeue().as_deref(),
            Some(b"sub_first".as_slice())
        );
    }
}
