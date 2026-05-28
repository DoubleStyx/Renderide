//! Shared PNG loading and diff-writing helpers for the golden-image harness.

use std::path::Path;

use image::RgbaImage;

use crate::error::HarnessError;

/// Loads `path` as an RGBA image.
pub(crate) fn load_rgba(path: &Path) -> Result<RgbaImage, HarnessError> {
    let img = image::open(path).map_err(|e| HarnessError::PngRead {
        path: path.to_path_buf(),
        source: e,
    })?;
    Ok(img.to_rgba8())
}

/// Saves an RGBA image to `path`, creating the parent directory if necessary.
pub(crate) fn save_rgba(img: &RgbaImage, path: &Path) -> Result<(), HarnessError> {
    ensure_parent_dir(path)?;
    img.save(path).map_err(|e| HarnessError::PngWrite {
        path: path.to_path_buf(),
        source: e,
    })
}

/// Writes an image-compare color-map diff for `actual` and `golden`.
pub(crate) fn write_diff_image(
    actual: &RgbaImage,
    golden: &RgbaImage,
    diff_path: &Path,
) -> Result<(), HarnessError> {
    ensure_parent_dir(diff_path)?;
    let result = image_compare::rgba_hybrid_compare(actual, golden)
        .map_err(|e| HarnessError::ImageCompare(format!("{e:?}")))?;
    let diff_img = result.image.to_color_map();
    diff_img
        .save(diff_path)
        .map_err(|e| HarnessError::PngWrite {
            path: diff_path.to_path_buf(),
            source: image::ImageError::IoError(std::io::Error::other(format!("{e:?}"))),
        })?;
    Ok(())
}

/// Ensures the parent directory for `path` exists.
fn ensure_parent_dir(path: &Path) -> Result<(), HarnessError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}
