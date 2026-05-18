//! Adapter feature negotiation.
//!
//! Picks the subset of [`wgpu::Features`] Renderide can use against a given adapter. Pure
//! data: no instance, surface, or device side effects.

/// Intersects [`wgpu::Adapter::features`] with the feature bits Renderide requires for rendering.
///
/// Always requests the subset of `TIMESTAMP_QUERY | TIMESTAMP_QUERY_INSIDE_ENCODERS` that the
/// adapter supports, regardless of Cargo features. The debug HUD's frame-bracket GPU timing
/// uses encoder-level `write_timestamp` calls on the driver thread; the `tracy`-gated
/// [`crate::profiling::GpuProfilerHandle`] consumes the same features for its pass-level path.
/// Either feature being absent is gracefully tolerated: the frame-bracket falls back to
/// callback-latency reporting and [`crate::profiling::GpuProfilerHandle::try_new`] returns
/// [`None`].
pub(crate) fn adapter_render_features_intersection(adapter: &wgpu::Adapter) -> wgpu::Features {
    let optional = optional_render_feature_mask();
    let timestamp = crate::profiling::timestamp_query_features_if_supported(adapter);
    adapter.features() & optional | timestamp
}

/// Optional adapter features requested for the normal wgpu-owned device path.
fn optional_render_feature_mask() -> wgpu::Features {
    let compression = wgpu::Features::TEXTURE_COMPRESSION_BC
        | wgpu::Features::TEXTURE_COMPRESSION_ETC2
        | wgpu::Features::TEXTURE_COMPRESSION_ASTC;
    let optional_float32_filterable = wgpu::Features::FLOAT32_FILTERABLE;
    let optional_rg11b10_renderable = wgpu::Features::RG11B10UFLOAT_RENDERABLE;
    let adapter_format_features = wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
    let optional_depth32_stencil8 = wgpu::Features::DEPTH32FLOAT_STENCIL8;
    let multisample_array = wgpu::Features::MULTISAMPLE_ARRAY;
    let shader_barycentrics = wgpu::Features::SHADER_BARYCENTRICS;
    compression
        | optional_float32_filterable
        | optional_rg11b10_renderable
        | adapter_format_features
        | optional_depth32_stencil8
        | multisample_array
        | shader_barycentrics
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optional_render_features_include_shader_barycentrics() {
        assert!(optional_render_feature_mask().contains(wgpu::Features::SHADER_BARYCENTRICS));
    }
}
