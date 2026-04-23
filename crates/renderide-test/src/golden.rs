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
pub(crate) fn generate(actual: &Path, golden_path: &Path) -> Result<(), HarnessError> {
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
pub(crate) fn check(
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
    if max_c[0].saturating_sub(min_c[0]) <= FLAT_CHANNEL_RANGE_MAX
        && max_c[1].saturating_sub(min_c[1]) <= FLAT_CHANNEL_RANGE_MAX
        && max_c[2].saturating_sub(min_c[2]) <= FLAT_CHANNEL_RANGE_MAX
        && max_c[3].saturating_sub(min_c[3]) <= FLAT_CHANNEL_RANGE_MAX
    {
        let px = img.get_pixel(0, 0).0;
        Some([px[0], px[1], px[2], px[3]])
    } else {
        None
    }
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
    use super::flat_sample_rgba_if_nearly_uniform;
    use image::RgbaImage;

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
        use crate::error::HarnessError;

        let dir = tempfile::tempdir().expect("tempdir");
        let actual = dir.path().join("actual.png");
        let golden = dir.path().join("golden.png");
        let diff_out = dir.path().join("diff.png");

        let mut small = RgbaImage::new(2, 2);
        for p in small.pixels_mut() {
            *p = image::Rgba([10, 20, 30, 255]);
        }
        small.save(&actual).expect("write actual");

        let mut large = RgbaImage::new(3, 3);
        for p in large.pixels_mut() {
            *p = image::Rgba([10, 20, 30, 255]);
        }
        large.save(&golden).expect("write golden");

        let err = super::check(&actual, &golden, 0.95, &diff_out).expect_err("dimension mismatch");
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
}
