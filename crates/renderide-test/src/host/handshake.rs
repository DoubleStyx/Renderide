//! Init handshake: sends `RendererInitData` → awaits `RendererInitResult` → sends
//! `RendererInitFinalizeData`.
//!
//! Field defaults come from the production C# host (`FrooxEngine/RenderSystem.cs:197-222`):
//! `shared_memory_prefix` always non-null, `unique_session_id` random per session,
//! `main_process_id = std::process::id()`, `window_title` always set, `output_device = Screen`.

use std::time::{Duration, Instant};

use renderide_shared::ipc::HostDualQueueIpc;
use renderide_shared::shared::{
    Guid, HeadOutputDevice, RendererCommand, RendererInitData, RendererInitFinalizeData,
    RendererInitResult,
};

use crate::error::HarnessError;

use super::lockstep::LockstepDriver;

/// Default deadline for receiving `RendererInitResult` after sending `RendererInitData`.
pub(super) const DEFAULT_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(15);

/// Outcome of [`run_handshake`].
///
/// The init result is exposed for future callers that want to log GPU caps or condition behavior
/// on `actual_output_device`; the current harness only verifies the handshake completed.
#[expect(
    dead_code,
    reason = "exposed for future callers; current harness only checks handshake completion"
)]
#[derive(Clone, Debug)]
pub(super) struct HandshakeOutcome {
    /// The renderer's reply to our `RendererInitData`.
    pub init_result: RendererInitResult,
}

/// Drives the three-step init handshake. Pumps the lockstep loop while waiting for the result so
/// any `FrameStartData` the renderer sends before finalize gets answered (the renderer's
/// `frontend::begin_frame::begin_frame_allowed` only sends `FrameStartData` after `Finalized`,
/// but we still want to keep `…S` drained so `KeepAlive`s don't pile up).
pub(super) fn run_handshake(
    queues: &mut HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
    shared_memory_prefix: &str,
    timeout: Duration,
) -> Result<HandshakeOutcome, HarnessError> {
    let init = RendererInitData {
        shared_memory_prefix: Some(shared_memory_prefix.to_string()),
        unique_session_id: random_guid(),
        main_process_id: std::process::id() as i32,
        debug_frame_pacing: false,
        output_device: HeadOutputDevice::Screen,
        window_title: Some("renderide-test".to_string()),
        set_window_icon: None,
        splash_screen_override: None,
    };
    logger::info!(
        "Handshake: sending RendererInitData (prefix={shared_memory_prefix}, session_id_a=0x{:08x})",
        init.unique_session_id.a as u32
    );
    if !queues.send_primary(RendererCommand::RendererInitData(init)) {
        return Err(HarnessError::QueueOptions(
            "send_primary(RendererInitData) returned false (queue full?)".to_string(),
        ));
    }

    let deadline = Instant::now() + timeout;
    let init_result = wait_for_init_result(queues, lockstep, deadline)?;
    logger::info!(
        "Handshake: received RendererInitResult (device={:?}, max_texture_size={}, identifier={:?})",
        init_result.actual_output_device,
        init_result.max_texture_size,
        init_result.renderer_identifier
    );

    if !queues.send_primary(RendererCommand::RendererInitFinalizeData(
        RendererInitFinalizeData {},
    )) {
        return Err(HarnessError::QueueOptions(
            "send_primary(RendererInitFinalizeData) returned false (queue full?)".to_string(),
        ));
    }
    logger::info!("Handshake: sent RendererInitFinalizeData; lockstep open");
    Ok(HandshakeOutcome { init_result })
}

fn wait_for_init_result(
    queues: &mut HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
    deadline: Instant,
) -> Result<RendererInitResult, HarnessError> {
    while Instant::now() < deadline {
        let tick = lockstep.tick(queues);
        for msg in tick.other_messages {
            if let RendererCommand::RendererInitResult(r) = msg {
                return Ok(r);
            }
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    Err(HarnessError::HandshakeTimeout(deadline.elapsed()))
}

/// Synthesizes a `Guid` from a 128-bit RNG using the platform's nanosecond timer + process id +
/// a per-call atomic counter for entropy.
fn random_guid() -> Guid {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let pid = std::process::id() as u64;
    let now_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mix1 = pid
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(now_ns)
        .wrapping_add(n);
    let mix2 = mix1.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    Guid {
        a: (mix1 as i32),
        b: ((mix1 >> 32) as i16),
        c: ((mix1 >> 48) as i16),
        d: (mix2 & 0xff) as u8,
        e: ((mix2 >> 8) & 0xff) as u8,
        f: ((mix2 >> 16) & 0xff) as u8,
        g: ((mix2 >> 24) & 0xff) as u8,
        h: ((mix2 >> 32) & 0xff) as u8,
        i: ((mix2 >> 40) & 0xff) as u8,
        j: ((mix2 >> 48) & 0xff) as u8,
        k: ((mix2 >> 56) & 0xff) as u8,
    }
}

#[cfg(test)]
mod tests {
    use super::random_guid;

    #[test]
    fn random_guid_varies_between_calls() {
        let a = random_guid();
        let b = random_guid();
        assert!(a.a != b.a || a.b != b.b);
    }
}
