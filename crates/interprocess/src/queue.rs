//! Factory for [`crate::Publisher`] / [`crate::Subscriber`].

use crate::error::OpenError;
use crate::options::QueueOptions;
use crate::Publisher;
use crate::Subscriber;

/// Builds [`Subscriber`] and [`Publisher`] instances for the same option type as the managed API.
pub struct QueueFactory;

impl QueueFactory {
    /// Creates an empty factory.
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
