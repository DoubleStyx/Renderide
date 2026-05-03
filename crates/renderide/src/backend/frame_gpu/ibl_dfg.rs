//! Frame-global image-based-lighting DFG LUT generation and texture upload.

use std::sync::Arc;

use glam::Vec3;

/// Edge length in texels for the precomputed IBL DFG lookup table.
const IBL_DFG_LUT_SIZE: u32 = 128;
/// GGX samples per texel for the generated runtime DFG resource.
const IBL_DFG_SAMPLE_COUNT: u32 = 1024;
/// Two 32-bit float channels: multiscatter DFG x/y.
const IBL_DFG_CHANNELS: u32 = 2;
/// Pi.
const PI: f32 = std::f32::consts::PI;

/// Creates the frame-global DFG lookup texture used by split-sum indirect specular.
pub(super) fn create_ibl_dfg_lut(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (Arc<wgpu::Texture>, Arc<wgpu::TextureView>) {
    profiling::scope!("frame_gpu::create_ibl_dfg_lut");
    let lut = generate_ibl_dfg_lut_rg32f();
    let texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
        label: Some("frame_ibl_dfg_lut"),
        size: wgpu::Extent3d {
            width: IBL_DFG_LUT_SIZE,
            height: IBL_DFG_LUT_SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rg32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    }));
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: texture.as_ref(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&lut),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(IBL_DFG_LUT_SIZE * IBL_DFG_CHANNELS * size_of::<f32>() as u32),
            rows_per_image: Some(IBL_DFG_LUT_SIZE),
        },
        wgpu::Extent3d {
            width: IBL_DFG_LUT_SIZE,
            height: IBL_DFG_LUT_SIZE,
            depth_or_array_layers: 1,
        },
    );
    let view = Arc::new(texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("frame_ibl_dfg_lut_view"),
        format: Some(wgpu::TextureFormat::Rg32Float),
        dimension: Some(wgpu::TextureViewDimension::D2),
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(1),
        base_array_layer: 0,
        array_layer_count: Some(1),
    }));
    (texture, view)
}

/// Generates multiscatter DFG terms in RG32F row-major order.
fn generate_ibl_dfg_lut_rg32f() -> Vec<f32> {
    let len = (IBL_DFG_LUT_SIZE * IBL_DFG_LUT_SIZE * IBL_DFG_CHANNELS) as usize;
    let mut out = Vec::with_capacity(len);
    for y in 0..IBL_DFG_LUT_SIZE {
        let perceptual_roughness = (y as f32 + 0.5) / IBL_DFG_LUT_SIZE as f32;
        let linear_roughness = perceptual_roughness * perceptual_roughness;
        for x in 0..IBL_DFG_LUT_SIZE {
            let no_v = (x as f32 + 0.5) / IBL_DFG_LUT_SIZE as f32;
            let dfg = dfv_multiscatter(no_v, linear_roughness, IBL_DFG_SAMPLE_COUNT);
            out.push(dfg[0]);
            out.push(dfg[1]);
        }
    }
    out
}

/// Multiscatter DFG integral factored for runtime `mix(dfg.x, dfg.y, f0)`.
fn dfv_multiscatter(no_v: f32, linear_roughness: f32, sample_count: u32) -> [f32; 2] {
    let no_v = saturate(no_v);
    let v = Vec3::new((1.0 - no_v * no_v).max(0.0).sqrt(), 0.0, no_v);
    let mut r = [0.0_f32; 2];
    for i in 0..sample_count {
        let u = hammersley(i, sample_count);
        let h = hemisphere_importance_sample_dggx(u, linear_roughness);
        let l = 2.0 * v.dot(h) * h - v;
        let vo_h = saturate(v.dot(h));
        let no_l = saturate(l.z);
        let no_h = saturate(h.z);
        if no_l > 0.0 && no_h > 0.0 {
            let visibility = visibility_smith_ggx_correlated(no_v, no_l, linear_roughness);
            let v_term = visibility * no_l * (vo_h / no_h);
            let fc = pow5(1.0 - vo_h);
            r[0] += v_term * fc;
            r[1] += v_term;
        }
    }
    let scale = 4.0 / (sample_count.max(1) as f32);
    [r[0] * scale, r[1] * scale]
}

/// GGX importance sample for `D(a) * cos(theta)` in tangent space.
fn hemisphere_importance_sample_dggx(u: [f32; 2], linear_roughness: f32) -> Vec3 {
    let a = linear_roughness.max(0.0);
    let phi = 2.0 * PI * u[0];
    let denominator = (1.0 + (a + 1.0) * ((a - 1.0) * u[1])).max(1e-7);
    let cos_theta2 = ((1.0 - u[1]) / denominator).clamp(0.0, 1.0);
    let cos_theta = cos_theta2.sqrt();
    let sin_theta = (1.0 - cos_theta2).max(0.0).sqrt();
    Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta)
}

/// Height-correlated Smith-GGX visibility.
fn visibility_smith_ggx_correlated(no_v: f32, no_l: f32, linear_roughness: f32) -> f32 {
    let a2 = linear_roughness * linear_roughness;
    let ggx_l = no_v * ((no_l - no_l * a2) * no_l + a2).max(0.0).sqrt();
    let ggx_v = no_l * ((no_v - no_v * a2) * no_v + a2).max(0.0).sqrt();
    0.5 / (ggx_v + ggx_l).max(1e-7)
}

/// Hammersley low-discrepancy 2D sample.
fn hammersley(i: u32, sample_count: u32) -> [f32; 2] {
    [
        i as f32 / (sample_count.max(1) as f32),
        radical_inverse_vdc(i),
    ]
}

/// Base-2 Van der Corput radical inverse.
fn radical_inverse_vdc(bits: u32) -> f32 {
    bits.reverse_bits() as f32 * 2.328_306_4e-10
}

/// `(1 - x)^5` helper used by Schlick Fresnel factoring.
fn pow5(x: f32) -> f32 {
    let x2 = x * x;
    x2 * x2 * x
}

/// Clamps `v` to `[0, 1]`.
fn saturate(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texel(lut: &[f32], x: u32, y: u32) -> [f32; 2] {
        let i = ((y * IBL_DFG_LUT_SIZE + x) * IBL_DFG_CHANNELS) as usize;
        [lut[i], lut[i + 1]]
    }

    #[test]
    fn generated_lut_has_expected_shape_and_finite_values() {
        let lut = generate_ibl_dfg_lut_rg32f();

        assert_eq!(
            lut.len(),
            (IBL_DFG_LUT_SIZE * IBL_DFG_LUT_SIZE * IBL_DFG_CHANNELS) as usize
        );
        assert!(lut.iter().all(|v| v.is_finite()));
        assert!(lut.iter().all(|v| *v >= 0.0));
    }

    #[test]
    fn generated_lut_is_deterministic_at_selected_texels() {
        let a = generate_ibl_dfg_lut_rg32f();
        let b = generate_ibl_dfg_lut_rg32f();

        for (x, y) in [(0, 0), (32, 32), (64, 64), (127, 127)] {
            let a = texel(&a, x, y);
            let b = texel(&b, x, y);
            assert_eq!(a[0].to_bits(), b[0].to_bits());
            assert_eq!(a[1].to_bits(), b[1].to_bits());
        }
    }

    #[test]
    fn generated_lut_matches_expected_energy_trends() {
        let lut = generate_ibl_dfg_lut_rg32f();
        let smooth_facing = texel(&lut, 127, 0);
        let smooth_grazing = texel(&lut, 0, 0);
        let rough_facing = texel(&lut, 127, 127);

        assert!(smooth_facing[0] < 0.01);
        assert!(smooth_facing[1] > 0.95);
        assert!(smooth_grazing[0] > smooth_facing[0]);
        assert!(rough_facing[1] < smooth_facing[1]);
    }

    #[test]
    fn dfv_multiscatter_outputs_unit_energy_for_smooth_facing_mirror() {
        let dfg = dfv_multiscatter(1.0, 0.0, IBL_DFG_SAMPLE_COUNT);

        assert!(dfg[0] < 1e-6);
        assert!((dfg[1] - 1.0).abs() < 1e-5);
    }
}
