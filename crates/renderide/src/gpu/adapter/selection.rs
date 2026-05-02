//! Adapter enumeration, scoring, and selection.
//!
//! Pure scoring policy ([`power_preference_score`], [`pick_adapter_index`]) plus
//! the IO-bearing wrappers ([`build_wgpu_instance`], [`select_adapter`]) that drive
//! [`crate::gpu::GpuContext`] construction. Kept separate from device creation so the
//! ranking rules can be exercised by unit tests without a live wgpu device.

use super::super::context::GpuError;
use super::super::instance_limits::instance_flags_for_gpu_init;

/// Lower scores rank earlier. Stable across systems so Vulkan ICD reordering does not flip the
/// chosen adapter.
///
/// [`wgpu::PowerPreference::None`] is treated as [`wgpu::PowerPreference::HighPerformance`] so that
/// callers without an explicit preference still get the discrete GPU on hybrid systems -- matches
/// Renderide's `[debug] power_preference` default.
pub(crate) fn power_preference_score(
    device_type: wgpu::DeviceType,
    power_preference: wgpu::PowerPreference,
) -> u8 {
    use wgpu::DeviceType::*;
    let prefer_low_power = power_preference == wgpu::PowerPreference::LowPower;
    match device_type {
        DiscreteGpu => {
            if prefer_low_power {
                1
            } else {
                0
            }
        }
        IntegratedGpu => {
            if prefer_low_power {
                0
            } else {
                1
            }
        }
        VirtualGpu => 2,
        Cpu => 3,
        Other => 4,
    }
}

/// Returns the index of the best compatible adapter, or [`None`] if none pass `is_compatible`.
///
/// Ranking uses [`power_preference_score`]; ties break on enumeration order so the result is
/// deterministic given the same adapter list.
fn pick_adapter_index<F>(
    adapters: &[wgpu::Adapter],
    is_compatible: F,
    power_preference: wgpu::PowerPreference,
) -> Option<usize>
where
    F: Fn(&wgpu::Adapter) -> bool,
{
    adapters
        .iter()
        .enumerate()
        .filter(|(_, a)| is_compatible(a))
        .min_by_key(|(i, a)| {
            (
                power_preference_score(a.get_info().device_type, power_preference),
                *i,
            )
        })
        .map(|(i, _)| i)
}

/// Logs every enumerated adapter at info level so users can see what wgpu found and why one was chosen.
fn log_adapter_candidates(adapters: &[wgpu::Adapter]) {
    if adapters.is_empty() {
        logger::warn!("wgpu adapter candidates: <none enumerated>");
        return;
    }
    for a in adapters {
        let info = a.get_info();
        logger::info!(
            "wgpu adapter candidate: {} type={:?} backend={:?} vendor=0x{:04x} device=0x{:04x}",
            info.name,
            info.device_type,
            info.backend,
            info.vendor,
            info.device,
        );
    }
}

/// Builds the [`wgpu::Instance`] used by both windowed and headless paths and returns the
/// derived [`wgpu::InstanceFlags`] for logging.
pub(crate) fn build_wgpu_instance(
    gpu_validation_layers: bool,
) -> (wgpu::Instance, wgpu::InstanceFlags) {
    let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
    instance_desc.backends = wgpu::Backends::all();
    instance_desc.flags = instance_flags_for_gpu_init(gpu_validation_layers);
    let instance_desc = instance_desc.with_env();
    let instance_flags = instance_desc.flags;
    (wgpu::Instance::new(instance_desc), instance_flags)
}

/// Enumerates adapters, logs all candidates, and returns the best match for `power_preference`.
///
/// When `surface` is [`Some`], adapters that cannot present to it are filtered out. Errors are
/// returned as [`GpuError::Adapter`] with messages distinguishing the windowed and headless paths.
pub(crate) async fn select_adapter(
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface<'_>>,
    power_preference: wgpu::PowerPreference,
) -> Result<wgpu::Adapter, GpuError> {
    let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
    log_adapter_candidates(&adapters);
    let chosen = match surface {
        Some(s) => pick_adapter_index(&adapters, |a| a.is_surface_supported(s), power_preference),
        None => pick_adapter_index(&adapters, |_| true, power_preference),
    }
    .ok_or_else(|| adapter_not_found_error(surface, adapters.len()))?;
    let adapter = adapters
        .into_iter()
        .nth(chosen)
        .ok_or_else(|| GpuError::Adapter("adapter index out of range".into()))?;
    let info = adapter.get_info();
    let label = if surface.is_some() {
        "wgpu adapter selected"
    } else {
        "wgpu adapter selected (headless)"
    };
    logger::info!(
        "{label}: {} type={:?} backend={:?} (preference={:?})",
        info.name,
        info.device_type,
        info.backend,
        power_preference,
    );
    Ok(adapter)
}

/// Builds the user-facing adapter-selection failure for the active path.
fn adapter_not_found_error(
    surface: Option<&wgpu::Surface<'_>>,
    candidate_count: usize,
) -> GpuError {
    if surface.is_some() {
        GpuError::Adapter(format!(
            "no surface-compatible adapter found among {candidate_count} candidate(s)"
        ))
    } else {
        GpuError::Adapter(
            "no headless adapter found. Install graphics drivers or verify that a supported \
             wgpu backend is available."
                .into(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_performance_preference_ranks_discrete_before_integrated() {
        assert!(
            power_preference_score(
                wgpu::DeviceType::DiscreteGpu,
                wgpu::PowerPreference::HighPerformance,
            ) < power_preference_score(
                wgpu::DeviceType::IntegratedGpu,
                wgpu::PowerPreference::HighPerformance,
            )
        );
        assert_eq!(
            power_preference_score(wgpu::DeviceType::DiscreteGpu, wgpu::PowerPreference::None),
            power_preference_score(
                wgpu::DeviceType::DiscreteGpu,
                wgpu::PowerPreference::HighPerformance,
            )
        );
    }

    #[test]
    fn low_power_preference_ranks_integrated_before_discrete() {
        assert!(
            power_preference_score(
                wgpu::DeviceType::IntegratedGpu,
                wgpu::PowerPreference::LowPower
            ) < power_preference_score(
                wgpu::DeviceType::DiscreteGpu,
                wgpu::PowerPreference::LowPower,
            )
        );
    }

    #[test]
    fn fallback_device_type_scores_are_stable() {
        assert_eq!(
            power_preference_score(
                wgpu::DeviceType::VirtualGpu,
                wgpu::PowerPreference::LowPower
            ),
            2
        );
        assert_eq!(
            power_preference_score(
                wgpu::DeviceType::Cpu,
                wgpu::PowerPreference::HighPerformance
            ),
            3
        );
        assert_eq!(
            power_preference_score(wgpu::DeviceType::Other, wgpu::PowerPreference::None),
            4
        );
    }

    #[test]
    fn headless_adapter_error_reports_driver_backend_guidance() {
        let error = adapter_not_found_error(None, 3).to_string();

        assert!(error.contains("no headless adapter found"));
        assert!(error.contains("supported wgpu backend"));
    }
}
