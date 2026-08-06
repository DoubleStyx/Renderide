//! Bounded off-thread bind-group destruction for render-path replacements. -xlinka

use std::sync::Arc;
use std::sync::mpsc::{SyncSender, TrySendError, sync_channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const DROP_QUEUE_CAPACITY: usize = 16_384;
const DROP_GRACE: Duration = Duration::from_millis(100);

struct DeferredBindGroupDrop {
    ready_at: Instant,
    bind_group: Arc<wgpu::BindGroup>,
}

pub(crate) struct DeferredBindGroupDrops {
    sender: Option<SyncSender<DeferredBindGroupDrop>>,
    worker: Option<JoinHandle<()>>,
}

impl DeferredBindGroupDrops {
    pub(crate) fn new() -> Self {
        let (sender, receiver) = sync_channel::<DeferredBindGroupDrop>(DROP_QUEUE_CAPACITY);
        let worker = std::thread::Builder::new()
            .name("renderide-bind-group-drop".to_string())
            .spawn(move || {
                crate::profiling::register_worker_thread();
                while let Ok(item) = receiver.recv() {
                    let delay = item.ready_at.saturating_duration_since(Instant::now());
                    if !delay.is_zero() {
                        std::thread::sleep(delay);
                    }
                    profiling::scope!("gpu_resource::deferred_bind_group_drop");
                    drop(item.bind_group);
                }
            });
        match worker {
            Ok(worker) => Self {
                sender: Some(sender),
                worker: Some(worker),
            },
            Err(_) => Self {
                sender: None,
                worker: None,
            },
        }
    }

    pub(crate) fn defer(&self, bind_group: Arc<wgpu::BindGroup>) {
        let item = DeferredBindGroupDrop {
            ready_at: Instant::now() + DROP_GRACE,
            bind_group,
        };
        let Some(sender) = self.sender.as_ref() else {
            drop(item);
            return;
        };
        if let Err(TrySendError::Full(item) | TrySendError::Disconnected(item)) =
            sender.try_send(item)
        {
            drop(item);
        }
    }
}

impl Default for DeferredBindGroupDrops {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for DeferredBindGroupDrops {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
