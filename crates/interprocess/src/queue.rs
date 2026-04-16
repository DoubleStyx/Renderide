//! Factory for [`crate::Publisher`] / [`crate::Subscriber`].

use crate::error::OpenError;
use crate::options::QueueOptions;
use crate::Publisher;
use crate::Subscriber;

/// Builds [`Subscriber`] and [`Publisher`] instances for the same option type as the managed API.
#[derive(PartialEq, Eq)]
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

impl Default for QueueFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::QueueFactory;
    use crate::options::QueueOptions;

    #[test]
    fn queue_factory_default_matches_new() {
        assert!(QueueFactory == QueueFactory::new());
        assert_eq!(std::mem::size_of::<QueueFactory>(), 0);
    }

    #[test]
    fn queue_factory_creates_publisher_and_subscriber() {
        let dir =
            std::env::temp_dir().join(format!("interprocess_qfactory_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let opts = QueueOptions::with_path("qf_queue", &dir, 4096).expect("valid options");
        let factory = QueueFactory::new();
        let mut publisher = factory.create_publisher(opts.clone()).expect("publisher");
        let mut subscriber = factory.create_subscriber(opts).expect("subscriber");

        assert!(publisher.try_enqueue(b"via_factory"));
        assert_eq!(
            subscriber.try_dequeue().as_deref(),
            Some(b"via_factory".as_slice())
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
