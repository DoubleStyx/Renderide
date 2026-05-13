//! Adapter enumeration, scoring, and selection.
//!
//! Pure scoring policy ([`power_preference_score`], adapter attempt ranking) plus
//! the IO-bearing wrappers ([`build_wgpu_instance`], [`select_adapter`]) that drive
//! [`crate::gpu::GpuContext`] construction. Kept separate from device creation so the
//! ranking rules can be exercised by unit tests without a live wgpu device.

use super::super::context::GpuError;
use super::super::instance_setup::instance_flags_for_gpu_init;

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
    use wgpu::DeviceType::{Cpu, DiscreteGpu, IntegratedGpu, Other, VirtualGpu};
    let prefer_low_power = power_preference == wgpu::PowerPreference::LowPower;
    match device_type {
        DiscreteGpu => u8::from(prefer_low_power),
        IntegratedGpu => u8::from(!prefer_low_power),
        VirtualGpu => 2,
        Cpu => 3,
        Other => 4,
    }
}

/// Returns compatible adapter indices in the order they should be attempted.
///
/// Ranking uses [`power_preference_score`]; ties break on enumeration order so the result is
/// deterministic given the same adapter list.
fn ranked_compatible_adapter_indices<F, G>(
    adapter_count: usize,
    is_compatible: F,
    device_type: G,
    power_preference: wgpu::PowerPreference,
) -> Vec<usize>
where
    F: Fn(usize) -> bool,
    G: Fn(usize) -> wgpu::DeviceType,
{
    let mut indices = (0..adapter_count)
        .filter(|&i| is_compatible(i))
        .collect::<Vec<_>>();
    indices
        .sort_unstable_by_key(|&i| (power_preference_score(device_type(i), power_preference), i));
    indices
}

/// Logs every enumerated adapter at info level so users can see what wgpu found and why one was chosen.
fn log_adapter_candidates(active_backends: wgpu::Backends, adapters: &[wgpu::Adapter]) {
    if adapters.is_empty() {
        logger::warn!("wgpu adapter candidates: <none enumerated> for {active_backends:?}");
        return;
    }
    for a in adapters {
        let info = a.get_info();
        logger::info!(
            "wgpu adapter candidate: {} type={:?} backend={:?} vendor=0x{:04x} device=0x{:04x} active_backends={:?}",
            info.name,
            info.device_type,
            info.backend,
            info.vendor,
            info.device,
            active_backends,
        );
    }
}

/// Builds the [`wgpu::Instance`] used by both windowed and headless paths and returns the
/// derived [`wgpu::InstanceFlags`] and active [`wgpu::Backends`] for logging.
///
/// `requested_backends` comes from the renderer config. [`wgpu::InstanceDescriptor::with_env`]
/// is still applied afterward, preserving `WGPU_BACKEND` as the final override.
pub(crate) fn build_wgpu_instance(
    gpu_validation_layers: bool,
    requested_backends: wgpu::Backends,
) -> (wgpu::Instance, wgpu::InstanceFlags, wgpu::Backends) {
    let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
    instance_desc.backends = requested_backends;
    instance_desc.flags = instance_flags_for_gpu_init(gpu_validation_layers);
    let instance_desc = instance_desc.with_env();
    let instance_flags = instance_desc.flags;
    let active_backends = instance_desc.backends;
    (
        wgpu::Instance::new(instance_desc),
        instance_flags,
        active_backends,
    )
}

/// Enumerates adapters, logs all candidates, and returns the best match for `power_preference`.
///
/// When `surface` is [`Some`], adapters that cannot present to it are filtered out. Errors are
/// returned as [`GpuError::Adapter`] with messages distinguishing the windowed and headless paths.
pub(crate) async fn select_adapter(
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface<'_>>,
    power_preference: wgpu::PowerPreference,
    active_backends: wgpu::Backends,
) -> Result<wgpu::Adapter, GpuError> {
    let adapters = select_adapters(instance, surface, power_preference, active_backends).await?;
    let adapter = adapters
        .into_iter()
        .next()
        .ok_or_else(|| GpuError::Adapter("adapter list unexpectedly empty".into()))?;
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

/// Enumerates adapters, logs all candidates, and returns every compatible adapter in attempt order.
///
/// When `surface` is [`Some`], adapters that cannot present to it are filtered out. Callers that
/// need runtime validation beyond [`wgpu::Adapter::is_surface_supported`] can try each returned
/// adapter until device creation and surface configuration both succeed.
pub(crate) async fn select_adapters(
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface<'_>>,
    power_preference: wgpu::PowerPreference,
    active_backends: wgpu::Backends,
) -> Result<Vec<wgpu::Adapter>, GpuError> {
    let adapters = instance.enumerate_adapters(active_backends).await;
    log_adapter_candidates(active_backends, &adapters);
    let ranked_indices = match surface {
        Some(s) => ranked_compatible_adapter_indices(
            adapters.len(),
            |i| adapters[i].is_surface_supported(s),
            |i| adapters[i].get_info().device_type,
            power_preference,
        ),
        None => ranked_compatible_adapter_indices(
            adapters.len(),
            |_| true,
            |i| adapters[i].get_info().device_type,
            power_preference,
        ),
    };
    if ranked_indices.is_empty() {
        return Err(adapter_not_found_error(
            surface,
            adapters.len(),
            active_backends,
        ));
    }
    Ok(ranked_indices
        .into_iter()
        .map(|i| adapters[i].clone())
        .collect())
}

/// Builds the user-facing adapter-selection failure for the active path.
fn adapter_not_found_error(
    surface: Option<&wgpu::Surface<'_>>,
    candidate_count: usize,
    active_backends: wgpu::Backends,
) -> GpuError {
    if surface.is_some() {
        GpuError::Adapter(format!(
            "no surface-compatible adapter found among {candidate_count} candidate(s) for {active_backends:?}"
        ))
    } else {
        GpuError::Adapter(format!(
            "no headless adapter found. Install graphics drivers or verify that a supported \
             wgpu backend is available. active_backends={active_backends:?}"
        ))
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
        let error = adapter_not_found_error(None, 3, wgpu::Backends::VULKAN).to_string();

        assert!(error.contains("no headless adapter found"));
        assert!(error.contains("supported wgpu backend"));
    }

    #[test]
    fn high_performance_ranked_attempt_order_prefers_discrete() {
        let types = [
            wgpu::DeviceType::IntegratedGpu,
            wgpu::DeviceType::DiscreteGpu,
            wgpu::DeviceType::Cpu,
        ];

        let ranked = ranked_compatible_adapter_indices(
            types.len(),
            |_| true,
            |i| types[i],
            wgpu::PowerPreference::HighPerformance,
        );

        assert_eq!(ranked, vec![1, 0, 2]);
    }

    #[test]
    fn low_power_ranked_attempt_order_prefers_integrated() {
        let types = [
            wgpu::DeviceType::DiscreteGpu,
            wgpu::DeviceType::IntegratedGpu,
            wgpu::DeviceType::Cpu,
        ];

        let ranked = ranked_compatible_adapter_indices(
            types.len(),
            |_| true,
            |i| types[i],
            wgpu::PowerPreference::LowPower,
        );

        assert_eq!(ranked, vec![1, 0, 2]);
    }

    #[test]
    fn ranked_attempt_order_excludes_incompatible_candidates() {
        let types = [
            wgpu::DeviceType::DiscreteGpu,
            wgpu::DeviceType::IntegratedGpu,
            wgpu::DeviceType::VirtualGpu,
        ];
        let compatible = [false, true, true];

        let ranked = ranked_compatible_adapter_indices(
            types.len(),
            |i| compatible[i],
            |i| types[i],
            wgpu::PowerPreference::HighPerformance,
        );

        assert_eq!(ranked, vec![1, 2]);
    }

    #[test]
    fn ranked_attempt_order_keeps_enumeration_order_for_ties() {
        let types = [
            wgpu::DeviceType::DiscreteGpu,
            wgpu::DeviceType::DiscreteGpu,
            wgpu::DeviceType::DiscreteGpu,
        ];

        let ranked = ranked_compatible_adapter_indices(
            types.len(),
            |_| true,
            |i| types[i],
            wgpu::PowerPreference::HighPerformance,
        );

        assert_eq!(ranked, vec![0, 1, 2]);
    }
}
