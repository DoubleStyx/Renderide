//! Awake-equivalent initialization: connection parameters and singleton.
//!
//! The host passes `-QueueName <name> -QueueCapacity <capacity>` when launching the renderer.
//!
//! Extension point for connection params, IPC init.

use std::env;
use std::sync::atomic::Ordering;

use thiserror::Error;

/// Error returned when init fails.
#[derive(Debug, Error)]
pub enum InitError {
    #[error("Only one RenderingManager can exist")]
    SingletonAlreadyExists,
    #[error("Could not get queue parameters")]
    NoConnectionParams,
    #[error("IPC connect: {0}")]
    IpcConnect(String),
}

/// Default queue capacity (8 MiB), matching MessagingManager.DEFAULT_CAPACITY.
pub const DEFAULT_QUEUE_CAPACITY: i64 = 8_388_608;

/// Parsed connection parameters for IPC with the host.
#[derive(Clone, Debug)]
pub struct ConnectionParams {
    pub queue_name: String,
    pub queue_capacity: i64,
}

/// Parse -QueueName and -QueueCapacity from command line args.
pub fn get_connection_parameters() -> Option<ConnectionParams> {
    let args: Vec<String> = env::args().collect();
    if args.is_empty() {
        return None;
    }

    let mut queue_name = None;
    let mut queue_capacity = None;

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        let next_i = i + 1;
        if next_i >= args.len() {
            break;
        }

        let arg_lower = arg.to_lowercase();
        if arg_lower.ends_with("queuename") {
            if queue_name.is_some() {
                return None;
            }
            queue_name = Some(args[next_i].clone());
            i = next_i;
        } else if arg_lower.ends_with("queuecapacity") {
            if queue_capacity.is_some_and(|c| c > 0) {
                return None;
            }
            queue_capacity = args[next_i].parse().ok().filter(|&c| c > 0);
            i = next_i;
        }

        i += 1;

        if let (Some(name), Some(cap)) = (queue_name.as_ref(), queue_capacity.as_ref())
            && *cap > 0
        {
            return Some(ConnectionParams {
                queue_name: name.clone(),
                queue_capacity: *cap,
            });
        }
    }

    queue_name.and_then(|name| {
        queue_capacity
            .filter(|&c| c > 0)
            .map(|cap| ConnectionParams {
                queue_name: name,
                queue_capacity: cap,
            })
    })
}

/// Singleton guard: only one RenderingManager init is allowed.
static INITIALIZED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Take the singleton init flag. Returns true if we were the first to init.
pub fn take_singleton_init() -> bool {
    !INITIALIZED.swap(true, Ordering::SeqCst)
}

/// Sends the renderer init result to the host. Called by InitCommandHandler.
pub fn send_renderer_init_result(receiver: &mut crate::ipc::receiver::CommandReceiver) {
    use crate::shared::{HeadOutputDevice, RendererCommand, RendererInitResult, TextureFormat};

    let result = RendererInitResult {
        actual_output_device: HeadOutputDevice::screen,
        renderer_identifier: Some("Renderide 0.1.0 (wgpu)".to_string()),
        main_window_handle_ptr: 0,
        stereo_rendering_mode: Some("None".to_string()),
        max_texture_size: 8192,
        is_gpu_texture_pot_byte_aligned: true,
        supported_texture_formats: vec![TextureFormat::rgba32],
    };
    receiver.send(RendererCommand::renderer_init_result(result));
}
