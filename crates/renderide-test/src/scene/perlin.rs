//! Pure-Rust 2D Perlin noise + RGBA texture generator used by integration cases.
//!
//! Implements the classic Perlin permutation table with smoothstep fade. Output is fully
//! deterministic for a given seed: the permutation table is built from the seed via a
//! splitmix64-style hash, so identical seeds always yield identical textures across machines.

use image::RgbaImage;
use serde::{Deserialize, Serialize};

/// A 256-entry Perlin permutation table (duplicated to 512 entries for convenient wrap-free indexing).
#[derive(Clone, Debug)]
pub struct PerlinNoise2D {
    perm: [u8; 512],
}

impl PerlinNoise2D {
    /// Builds a permutation table from `seed`.
    pub fn new(seed: u64) -> Self {
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut base: [u8; 256] = [0; 256];
        for (i, slot) in base.iter_mut().enumerate() {
            *slot = i as u8;
        }
        // Fisher-Yates shuffle driven by a splitmix64 PRNG seeded from `seed`.
        for i in (1..256).rev() {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let j = (z as usize) % (i + 1);
            base.swap(i, j);
        }
        let mut perm = [0u8; 512];
        perm[..256].copy_from_slice(&base);
        perm[256..].copy_from_slice(&base);
        Self { perm }
    }

    /// Samples 2D Perlin noise at `(x, y)`. Output range is `[-1.0, 1.0]` (in practice a
    /// little narrower; `~[-0.71, 0.71]`).
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let xi = x.floor() as i32 & 255;
        let yi = y.floor() as i32 & 255;
        let xf = x - x.floor();
        let yf = y - y.floor();
        let u = fade(xf);
        let v = fade(yf);

        let xi = xi as usize;
        let yi = yi as usize;
        let yi1 = (yi + 1) & 255;
        let xi1 = (xi + 1) & 255;
        let py0 = self.perm[yi] as usize;
        let py1 = self.perm[yi1] as usize;
        let a = self.perm[(xi + py0) & 255];
        let b = self.perm[(xi + py1) & 255];
        let c = self.perm[(xi1 + py0) & 255];
        let d = self.perm[(xi1 + py1) & 255];

        let g_aa = grad(a, xf, yf);
        let g_ba = grad(c, xf - 1.0, yf);
        let g_ab = grad(b, xf, yf - 1.0);
        let g_bb = grad(d, xf - 1.0, yf - 1.0);

        lerp(v, lerp(u, g_aa, g_ba), lerp(u, g_ab, g_bb))
    }

    /// Samples fractal Brownian motion with `octaves` doublings of frequency. Each octave
    /// halves amplitude. Output is normalized to `[-1.0, 1.0]`.
    pub fn fbm(&self, x: f32, y: f32, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
        let mut sum = 0.0f32;
        let mut amp = 1.0f32;
        let mut freq = 1.0f32;
        let mut total_amp = 0.0f32;
        for _ in 0..octaves {
            sum += amp * self.sample(x * freq, y * freq);
            total_amp += amp;
            amp *= gain;
            freq *= lacunarity;
        }
        if total_amp > 0.0 {
            sum / total_amp
        } else {
            0.0
        }
    }
}

fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn lerp(t: f32, a: f32, b: f32) -> f32 {
    a + t * (b - a)
}

fn grad(hash: u8, x: f32, y: f32) -> f32 {
    // Use the low 3 bits of `hash` to pick one of 8 gradient directions in 2D.
    let h = hash & 7;
    let (gx, gy) = match h {
        0 => (1.0, 1.0),
        1 => (-1.0, 1.0),
        2 => (1.0, -1.0),
        3 => (-1.0, -1.0),
        4 => (1.0, 0.0),
        5 => (-1.0, 0.0),
        6 => (0.0, 1.0),
        _ => (0.0, -1.0),
    };
    gx * x + gy * y
}

/// Configuration for [`generate_perlin_rgba`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PerlinTextureSpec {
    /// Texture width in pixels.
    pub width: u32,
    /// Texture height in pixels.
    pub height: u32,
    /// PRNG seed driving the permutation table.
    pub seed: u64,
    /// Number of fBm octaves to sum.
    pub octaves: u32,
    /// Frequency growth per octave (`2.0` is classic).
    pub lacunarity: f32,
    /// Amplitude decay per octave (`0.5` is classic).
    pub gain: f32,
    /// Tile scale: how many noise units the full texture spans.
    pub scale: f32,
    /// RGB tint applied to the noise mass; alpha is always `255`.
    pub tint: [u8; 3],
}

impl Default for PerlinTextureSpec {
    fn default() -> Self {
        Self {
            width: 256,
            height: 256,
            seed: 0x00C0_FFEE,
            octaves: 5,
            lacunarity: 2.0,
            gain: 0.5,
            scale: 4.0,
            tint: [240, 200, 96],
        }
    }
}

/// Renders a deterministic Perlin-fBm RGBA8 texture from `spec`.
///
/// Noise output is mapped from `[-1, 1]` to `[0, 1]` and modulated by `spec.tint` before
/// being stored as the RGB channels; alpha is always `255`. Identical specs produce identical
/// textures across machines (no floating-point reductions over hardware-dependent intrinsics).
pub fn generate_perlin_rgba(spec: &PerlinTextureSpec) -> RgbaImage {
    let noise = PerlinNoise2D::new(spec.seed);
    let mut img = RgbaImage::new(spec.width, spec.height);
    let inv_w = spec.scale / spec.width as f32;
    let inv_h = spec.scale / spec.height as f32;
    for y in 0..spec.height {
        for x in 0..spec.width {
            let nx = x as f32 * inv_w;
            let ny = y as f32 * inv_h;
            let n = noise
                .fbm(nx, ny, spec.octaves, spec.lacunarity, spec.gain)
                .clamp(-1.0, 1.0);
            let m = n * 0.5 + 0.5; // [0, 1]
            let r = (m * spec.tint[0] as f32).round() as u8;
            let g = (m * spec.tint[1] as f32).round() as u8;
            let b = (m * spec.tint[2] as f32).round() as u8;
            img.put_pixel(x, y, image::Rgba([r, g, b, 255]));
        }
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_returns_finite_values_in_range() {
        let n = PerlinNoise2D::new(42);
        for i in 0..200 {
            let x = i as f32 * 0.137;
            let y = (i as f32 * 0.219).sin();
            let v = n.sample(x, y);
            assert!(v.is_finite());
            assert!((-1.5..=1.5).contains(&v), "out-of-range sample {v}");
        }
    }

    #[test]
    fn same_seed_produces_identical_texture() {
        let spec = PerlinTextureSpec {
            width: 64,
            height: 64,
            seed: 7,
            ..PerlinTextureSpec::default()
        };
        let a = generate_perlin_rgba(&spec);
        let b = generate_perlin_rgba(&spec);
        assert_eq!(a.as_raw(), b.as_raw());
    }

    #[test]
    fn different_seeds_produce_different_textures() {
        let a = generate_perlin_rgba(&PerlinTextureSpec {
            width: 64,
            height: 64,
            seed: 1,
            ..PerlinTextureSpec::default()
        });
        let b = generate_perlin_rgba(&PerlinTextureSpec {
            width: 64,
            height: 64,
            seed: 2,
            ..PerlinTextureSpec::default()
        });
        assert_ne!(a.as_raw(), b.as_raw());
    }

    #[test]
    fn texture_is_not_flat() {
        let img = generate_perlin_rgba(&PerlinTextureSpec {
            width: 64,
            height: 64,
            seed: 11,
            ..PerlinTextureSpec::default()
        });
        let raw = img.as_raw();
        let mut min = 255u8;
        let mut max = 0u8;
        for i in 0..raw.len() / 4 {
            let r = raw[i * 4];
            min = min.min(r);
            max = max.max(r);
        }
        assert!(
            max - min > 32,
            "expected non-flat noise; got R range [{min}, {max}]"
        );
    }
}
