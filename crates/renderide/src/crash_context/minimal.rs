//! Allocation-free crash context formatting for native crash handlers.

use super::state::{refresh_uptime, snapshot};

/// Writes a compact, allocation-free snapshot for native crash handlers.
pub(crate) fn write_minimal_snapshot(out: &mut [u8]) -> usize {
    refresh_uptime();
    let s = snapshot();
    let mut w = 0usize;
    push(out, &mut w, b"CRASH_CONTEXT uptime_ms=");
    push_u64(out, &mut w, s.uptime_ms);
    push(out, &mut w, b" tick=");
    push_u64(out, &mut w, s.tick_sequence);
    push(out, &mut w, b" phase=");
    push(out, &mut w, s.tick_phase.as_str().as_bytes());
    push(out, &mut w, b" cpu_phase=");
    push(out, &mut w, s.cpu_render_phase.as_str().as_bytes());
    push(out, &mut w, b" mode=");
    push(out, &mut w, s.render_mode.as_str().as_bytes());
    push(out, &mut w, b" target=");
    push(out, &mut w, s.target_mode.as_str().as_bytes());
    push(out, &mut w, b" init=");
    push(out, &mut w, s.init_state.as_str().as_bytes());
    push(out, &mut w, b" last_host_frame=");
    push_i64(out, &mut w, s.last_host_frame_index);
    push(out, &mut w, b" prepared_views=");
    push_u64(out, &mut w, u64::from(s.prepared_view_count));
    push(out, &mut w, b" ipc_drop=");
    push_u64(out, &mut w, u64::from(s.primary_ipc_drop_streak));
    push(out, &mut w, b"/");
    push_u64(out, &mut w, u64::from(s.background_ipc_drop_streak));
    push(out, &mut w, b" driver_backlog=");
    push_u64(out, &mut w, u64::from(s.driver_backlog));
    push(out, &mut w, b" graph_error=");
    push(out, &mut w, s.last_graph_error.as_str().as_bytes());
    push(out, &mut w, b" driver_stage=");
    push(out, &mut w, s.driver_stage.as_str().as_bytes());
    push(out, &mut w, b" openxr_call=");
    push(out, &mut w, s.openxr_call.as_str().as_bytes());
    push(out, &mut w, b" xr_finalize=");
    push(out, &mut w, s.xr_finalize_kind.as_str().as_bytes());
    push(out, &mut w, b" xr_image=");
    push_optional_u32(out, &mut w, s.xr_finalize_image_index);
    push(out, &mut w, b" xr_frame_seq=");
    push_u64(out, &mut w, s.xr_finalize_frame_seq);
    push(out, &mut w, b" xr_command_buffers=");
    push_u64(out, &mut w, u64::from(s.xr_finalize_command_buffers));
    push(out, &mut w, b" xr_extent=");
    push_optional_extent(out, &mut w, s.xr_finalize_extent);
    push(out, &mut w, b" xr_predicted_time_ns=");
    push_optional_i64(out, &mut w, s.xr_finalize_predicted_display_time_nanos);
    push(out, &mut w, b"\n");
    w
}

fn push(out: &mut [u8], w: &mut usize, bytes: &[u8]) {
    let remaining = out.len().saturating_sub(*w);
    let n = bytes.len().min(remaining);
    if n > 0 {
        out[*w..*w + n].copy_from_slice(&bytes[..n]);
        *w += n;
    }
}

fn push_u64(out: &mut [u8], w: &mut usize, mut value: u64) {
    if value == 0 {
        push(out, w, b"0");
        return;
    }
    let mut tmp = [0u8; 20];
    let mut i = 0usize;
    while value > 0 {
        tmp[i] = b'0' + (value % 10) as u8;
        value /= 10;
        i += 1;
    }
    while i > 0 {
        i -= 1;
        push(out, w, &tmp[i..=i]);
    }
}

fn push_i64(out: &mut [u8], w: &mut usize, value: i64) {
    if value < 0 {
        push(out, w, b"-");
        push_u64(out, w, value.unsigned_abs());
    } else {
        push_u64(out, w, value as u64);
    }
}

fn push_optional_u32(out: &mut [u8], w: &mut usize, value: Option<u32>) {
    if let Some(value) = value {
        push_u64(out, w, u64::from(value));
    } else {
        push(out, w, b"none");
    }
}

fn push_optional_i64(out: &mut [u8], w: &mut usize, value: Option<i64>) {
    if let Some(value) = value {
        push_i64(out, w, value);
    } else {
        push(out, w, b"none");
    }
}

fn push_optional_extent(out: &mut [u8], w: &mut usize, value: Option<(u32, u32)>) {
    if let Some((width, height)) = value {
        push_u64(out, w, u64::from(width));
        push(out, w, b"x");
        push_u64(out, w, u64::from(height));
    } else {
        push(out, w, b"none");
    }
}
