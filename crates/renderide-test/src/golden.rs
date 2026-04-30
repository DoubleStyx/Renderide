//! Golden image management: file copy on `generate`, perceptual diff on `check`.

use std::path::Path;

use image::RgbaImage;

use crate::error::HarnessError;

/// Maximum per-channel value range (inclusive) still treated as a flat / clear-only image.
///
/// A real shaded frame (e.g. world normals on a sphere) spans many levels per channel.
const FLAT_CHANNEL_RANGE_MAX: u8 = 1;

/// Copies the freshly produced PNG at `actual` over the golden image at `golden_path`.
///
/// Refuses to overwrite the golden if the capture is flat (no geometry).
pub fn generate(actual: &Path, golden_path: &Path) -> Result<(), HarnessError> {
    let img = load_rgba(actual)?;
    reject_flat_image(&img, actual)?;
    if let Some(parent) = golden_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::copy(actual, golden_path)?;
    Ok(())
}

/// Compares `actual` against `golden`, returning the SSIM-Y score on success.
///
/// On failure (score below `threshold`), writes a side-by-side diff visualization to `diff_out`
/// and returns [`HarnessError::GoldenMismatch`].
pub fn check(
    actual: &Path,
    golden: &Path,
    threshold: f64,
    diff_out: &Path,
) -> Result<f64, HarnessError> {
    let actual_img = load_rgba(actual)?;
    let golden_img = load_rgba(golden).map_err(|e| match e {
        HarnessError::PngRead { .. } => HarnessError::GoldenMissing(golden.to_path_buf()),
        other => other,
    })?;

    if actual_img.dimensions() != golden_img.dimensions() {
        write_actual_for_debug(&actual_img, diff_out)?;
        return Err(HarnessError::ImageCompare(format!(
            "dimensions differ: actual {:?} vs golden {:?}",
            actual_img.dimensions(),
            golden_img.dimensions()
        )));
    }

    reject_flat_image(&actual_img, actual)?;
    reject_flat_image(&golden_img, golden)?;

    let result = image_compare::rgba_hybrid_compare(&actual_img, &golden_img)
        .map_err(|e| HarnessError::ImageCompare(format!("{e:?}")))?;
    let score = result.score;

    if score < threshold {
        if let Some(parent) = diff_out.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let diff_img = result.image.to_color_map();
        diff_img
            .save(diff_out)
            .map_err(|e| HarnessError::PngWrite {
                path: diff_out.to_path_buf(),
                source: image::ImageError::IoError(std::io::Error::other(format!("{e:?}"))),
            })?;
        return Err(HarnessError::GoldenMismatch {
            score,
            threshold,
            diff_path: diff_out.to_path_buf(),
        });
    }

    Ok(score)
}

/// Returns [`HarnessError::FlatImage`] when every channel's min/max spread is at most
/// [`FLAT_CHANNEL_RANGE_MAX`] (clear color or single flat fill).
fn reject_flat_image(img: &RgbaImage, path: &Path) -> Result<(), HarnessError> {
    if let Some(color) = flat_sample_rgba_if_nearly_uniform(img) {
        return Err(HarnessError::FlatImage {
            path: path.to_path_buf(),
            color,
        });
    }
    Ok(())
}

/// If the image is nearly uniform per channel, returns a representative RGBA; otherwise [`None`].
fn flat_sample_rgba_if_nearly_uniform(img: &RgbaImage) -> Option<[u8; 4]> {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return None;
    }
    let mut min_c = [255u8; 4];
    let mut max_c = [0u8; 4];
    for p in img.pixels() {
        let c = p.0;
        for i in 0..4 {
            min_c[i] = min_c[i].min(c[i]);
            max_c[i] = max_c[i].max(c[i]);
        }
    }
    (max_c[0].saturating_sub(min_c[0]) <= FLAT_CHANNEL_RANGE_MAX
        && max_c[1].saturating_sub(min_c[1]) <= FLAT_CHANNEL_RANGE_MAX
        && max_c[2].saturating_sub(min_c[2]) <= FLAT_CHANNEL_RANGE_MAX
        && max_c[3].saturating_sub(min_c[3]) <= FLAT_CHANNEL_RANGE_MAX)
        .then(|| {
            let px = img.get_pixel(0, 0).0;
            [px[0], px[1], px[2], px[3]]
        })
}

fn load_rgba(path: &Path) -> Result<RgbaImage, HarnessError> {
    let img = image::open(path).map_err(|e| HarnessError::PngRead {
        path: path.to_path_buf(),
        source: e,
    })?;
    Ok(img.to_rgba8())
}

fn write_actual_for_debug(actual: &RgbaImage, diff_out: &Path) -> Result<(), HarnessError> {
    if let Some(parent) = diff_out.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    actual.save(diff_out).map_err(|e| HarnessError::PngWrite {
        path: diff_out.to_path_buf(),
        source: e,
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{check, flat_sample_rgba_if_nearly_uniform, generate};
    use crate::error::HarnessError;
    use image::RgbaImage;

    /// Builds a deterministic non-flat RGBA gradient. The gradient spans every channel by more
    /// than [`super::FLAT_CHANNEL_RANGE_MAX`], so the flat-image gate accepts it.
    fn non_flat_gradient(width: u32, height: u32) -> RgbaImage {
        let mut img = RgbaImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let r = (x.wrapping_mul(251) / width.max(1)) as u8;
                let g = (y.wrapping_mul(239) / height.max(1)) as u8;
                let b = ((x ^ y).wrapping_mul(53) & 0xff) as u8;
                img.put_pixel(x, y, image::Rgba([r, g, b, 255]));
            }
        }
        img
    }

    #[test]
    fn flat_detects_single_fill_color() {
        let mut img = RgbaImage::new(4, 4);
        for p in img.pixels_mut() {
            *p = image::Rgba([39u8, 63, 97, 255]);
        }
        assert!(flat_sample_rgba_if_nearly_uniform(&img).is_some());
    }

    #[test]
    fn not_flat_when_channel_spans_more_than_epsilon() {
        let mut img = RgbaImage::new(2, 2);
        img.put_pixel(0, 0, image::Rgba([0, 0, 0, 255]));
        img.put_pixel(1, 0, image::Rgba([10, 0, 0, 255]));
        assert!(flat_sample_rgba_if_nearly_uniform(&img).is_none());
    }

    #[test]
    fn check_rejects_dimension_mismatch_before_ssim() {
        let dir = tempfile::tempdir().expect("tempdir");
        let actual = dir.path().join("actual.png");
        let golden = dir.path().join("golden.png");
        let diff_out = dir.path().join("diff.png");

        non_flat_gradient(8, 8).save(&actual).expect("write actual");
        non_flat_gradient(12, 12)
            .save(&golden)
            .expect("write golden");

        let err = check(&actual, &golden, 0.95, &diff_out).expect_err("dimension mismatch");
        match err {
            HarnessError::ImageCompare(msg) => {
                assert!(
                    msg.contains("dimensions differ"),
                    "unexpected message: {msg}"
                );
            }
            other => panic!("expected ImageCompare, got {other:?}"),
        }
    }

    #[test]
    fn generate_writes_to_golden_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let actual = dir.path().join("actual.png");
        let golden = dir.path().join("golden.png");
        non_flat_gradient(8, 8).save(&actual).expect("save actual");

        generate(&actual, &golden).expect("generate");

        assert!(golden.exists(), "golden path was not written");
        let actual_bytes = std::fs::read(&actual).expect("read actual");
        let golden_bytes = std::fs::read(&golden).expect("read golden");
        assert_eq!(actual_bytes, golden_bytes);
    }

    #[test]
    fn generate_creates_missing_parent_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let actual = dir.path().join("actual.png");
        let golden = dir.path().join("nested").join("subdir").join("golden.png");
        non_flat_gradient(8, 8).save(&actual).expect("save actual");

        generate(&actual, &golden).expect("generate");

        assert!(golden.exists());
    }

    #[test]
    fn generate_rejects_flat_capture() {
        let dir = tempfile::tempdir().expect("tempdir");
        let actual = dir.path().join("actual.png");
        let golden = dir.path().join("golden.png");

        let mut flat = RgbaImage::new(8, 8);
        for p in flat.pixels_mut() {
            *p = image::Rgba([7, 7, 7, 255]);
        }
        flat.save(&actual).expect("save flat");

        let err = generate(&actual, &golden).expect_err("flat capture must be rejected");
        match err {
            HarnessError::FlatImage { color, .. } => assert_eq!(color, [7, 7, 7, 255]),
            other => panic!("expected FlatImage, got {other:?}"),
        }
        assert!(!golden.exists(), "flat image must not overwrite golden");
    }

    #[test]
    fn check_returns_score_one_for_identical_images() {
        let dir = tempfile::tempdir().expect("tempdir");
        let actual = dir.path().join("actual.png");
        let golden = dir.path().join("golden.png");
        let diff_out = dir.path().join("diff.png");

        let img = non_flat_gradient(16, 16);
        img.save(&actual).expect("save actual");
        img.save(&golden).expect("save golden");

        let score = check(&actual, &golden, 0.95, &diff_out).expect("check identical");
        assert!(score >= 0.99, "expected near-1.0 SSIM, got {score}");
        assert!(!diff_out.exists(), "diff must not be written on success");
    }

    #[test]
    fn check_maps_missing_golden_to_golden_missing_variant() {
        let dir = tempfile::tempdir().expect("tempdir");
        let actual = dir.path().join("actual.png");
        let golden = dir.path().join("golden.png");
        let diff_out = dir.path().join("diff.png");

        non_flat_gradient(8, 8).save(&actual).expect("save actual");

        let err = check(&actual, &golden, 0.95, &diff_out).expect_err("missing golden");
        assert!(
            matches!(&err, HarnessError::GoldenMissing(p) if p == &golden),
            "expected GoldenMissing, got {err:?}"
        );
    }

    #[test]
    fn check_writes_diff_on_mismatch() {
        let dir = tempfile::tempdir().expect("tempdir");
        let actual = dir.path().join("actual.png");
        let golden = dir.path().join("golden.png");
        let diff_out = dir.path().join("diff.png");

        let mut a = RgbaImage::new(16, 16);
        let mut g = RgbaImage::new(16, 16);
        for y in 0..16 {
            for x in 0..16 {
                a.put_pixel(x, y, image::Rgba([(x * 16) as u8, 0, 0, 255]));
                g.put_pixel(x, y, image::Rgba([0, (y * 16) as u8, 255, 255]));
            }
        }
        a.save(&actual).expect("save actual");
        g.save(&golden).expect("save golden");

        let err = check(&actual, &golden, 0.999, &diff_out).expect_err("mismatch");
        match err {
            HarnessError::GoldenMismatch {
                score,
                threshold,
                diff_path,
            } => {
                assert!(score < threshold);
                assert_eq!(diff_path, diff_out);
            }
            other => panic!("expected GoldenMismatch, got {other:?}"),
        }
        let diff_meta = std::fs::metadata(&diff_out).expect("diff metadata");
        assert!(diff_meta.len() > 0, "diff file should not be empty");
    }
}
