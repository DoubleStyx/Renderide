//! wgpu feature negotiation for the OpenXR-selected Vulkan device.
//!
//! `MULTIVIEW` is the only mandatory feature: every other entry in [`OPTIONAL_WGPU_FEATURES`] is
//! enabled when the adapter reports it, dropped silently otherwise. The fallback paths for the
//! optional features are documented at their use sites in
//! [`crate::gpu::GpuContext::set_swapchain_msaa_requested_stereo`] (MSAA array) and the
//! reflection-probe / timestamp pipelines.

use wgpu::wgt;

/// Entry in the optional-feature table consulted by [`negotiate_wgpu_features`].
struct OptionalWgpuFeature {
    /// wgpu feature bit. Enabled when [`wgt::Features`] reports it on the adapter; absent
    /// otherwise.
    wgt: wgt::Features,
}

/// Optional wgpu features negotiated against the OpenXR-selected adapter.
///
/// Order matches the imperative chain it replaces. Adding a new optional feature here makes it
/// available to the OpenXR path the same way the desktop GPU init negotiates its own list.
///
/// `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` unlocks hardware-reported MSAA sample counts so the
/// device exposes the real tiers instead of the WebGPU baseline. `MULTISAMPLE_ARRAY` is required
/// for multisampled 2D array color/depth textures used by the stereo (single-pass multiview) MSAA
/// path; absence is silently handled by falling back to `sample_count = 1` in
/// [`crate::gpu::GpuContext::set_swapchain_msaa_requested_stereo`].
const OPTIONAL_WGPU_FEATURES: &[OptionalWgpuFeature] = &[
    OptionalWgpuFeature {
        wgt: wgt::Features::TEXTURE_COMPRESSION_BC,
    },
    OptionalWgpuFeature {
        wgt: wgt::Features::TEXTURE_COMPRESSION_ETC2,
    },
    OptionalWgpuFeature {
        wgt: wgt::Features::TEXTURE_COMPRESSION_ASTC,
    },
    OptionalWgpuFeature {
        wgt: wgt::Features::FLOAT32_FILTERABLE,
    },
    OptionalWgpuFeature {
        wgt: wgt::Features::RG11B10UFLOAT_RENDERABLE,
    },
    OptionalWgpuFeature {
        wgt: wgt::Features::DEPTH32FLOAT_STENCIL8,
    },
    OptionalWgpuFeature {
        wgt: wgt::Features::TIMESTAMP_QUERY,
    },
    OptionalWgpuFeature {
        wgt: wgt::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
    },
    OptionalWgpuFeature {
        wgt: wgt::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
    },
    OptionalWgpuFeature {
        wgt: wgt::Features::MULTISAMPLE_ARRAY,
    },
    OptionalWgpuFeature {
        wgt: wgt::Features::SHADER_BARYCENTRICS,
    },
];

/// Returns the negotiated wgpu feature set: mandatory [`wgt::Features::MULTIVIEW`] plus every
/// [`OPTIONAL_WGPU_FEATURES`] entry that the adapter advertises.
pub(super) fn negotiate_wgpu_features(adapter_features: wgt::Features) -> wgt::Features {
    let mask = OPTIONAL_WGPU_FEATURES
        .iter()
        .fold(wgt::Features::empty(), |acc, e| acc | e.wgt);
    wgt::Features::MULTIVIEW | (adapter_features & mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Aggregate every entry in the optional table; used as the expected mask when the adapter
    /// advertises every optional feature.
    fn expected_optional_mask() -> wgt::Features {
        OPTIONAL_WGPU_FEATURES
            .iter()
            .fold(wgt::Features::empty(), |acc, e| acc | e.wgt)
    }

    #[test]
    fn empty_adapter_returns_only_multiview() {
        let result = negotiate_wgpu_features(wgt::Features::empty());
        assert_eq!(result, wgt::Features::MULTIVIEW);
    }

    #[test]
    fn all_features_adapter_returns_multiview_plus_full_optional_mask() {
        let result = negotiate_wgpu_features(wgt::Features::all());
        assert_eq!(result, wgt::Features::MULTIVIEW | expected_optional_mask());
    }

    #[test]
    fn unrelated_adapter_features_are_dropped() {
        // A feature not in OPTIONAL_WGPU_FEATURES (e.g. MAPPABLE_PRIMARY_BUFFERS) must not appear
        // in the negotiated set, even if the adapter advertises it.
        let adapter = wgt::Features::MAPPABLE_PRIMARY_BUFFERS
            | wgt::Features::TEXTURE_COMPRESSION_BC
            | wgt::Features::FLOAT32_FILTERABLE;
        let result = negotiate_wgpu_features(adapter);
        assert!(result.contains(wgt::Features::MULTIVIEW));
        assert!(result.contains(wgt::Features::TEXTURE_COMPRESSION_BC));
        assert!(result.contains(wgt::Features::FLOAT32_FILTERABLE));
        assert!(!result.contains(wgt::Features::MAPPABLE_PRIMARY_BUFFERS));
    }

    #[test]
    fn multiview_is_always_present_regardless_of_adapter() {
        // The renderer requires multiview even when wgpu-hal reports the adapter does not
        // advertise it; the device creation path will fail at logical device creation if the
        // physical device truly cannot satisfy it, which is the correct failure mode.
        let result = negotiate_wgpu_features(wgt::Features::empty());
        assert!(result.contains(wgt::Features::MULTIVIEW));
    }

    #[test]
    fn every_listed_entry_round_trips_through_negotiation() {
        for entry in OPTIONAL_WGPU_FEATURES {
            let result = negotiate_wgpu_features(entry.wgt);
            assert!(
                result.contains(entry.wgt),
                "negotiate_wgpu_features dropped advertised entry {:?}",
                entry.wgt
            );
        }
    }

    #[test]
    fn negotiated_mask_matches_legacy_imperative_aggregation() {
        // The pre-table chain in `create_vulkan_logical_device_openxr` consisted of the OR of
        // these flag groups. Locking that exact aggregation in a test guards against later
        // table edits that silently drop a feature.
        let legacy = wgt::Features::TEXTURE_COMPRESSION_BC
            | wgt::Features::TEXTURE_COMPRESSION_ETC2
            | wgt::Features::TEXTURE_COMPRESSION_ASTC
            | wgt::Features::FLOAT32_FILTERABLE
            | wgt::Features::RG11B10UFLOAT_RENDERABLE
            | wgt::Features::DEPTH32FLOAT_STENCIL8
            | wgt::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | wgt::Features::MULTISAMPLE_ARRAY
            | wgt::Features::TIMESTAMP_QUERY
            | wgt::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
            | wgt::Features::SHADER_BARYCENTRICS;
        let result = negotiate_wgpu_features(wgt::Features::all());
        assert_eq!(result, wgt::Features::MULTIVIEW | legacy);
    }
}
