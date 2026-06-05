//! Small GLB fixture importer for renderide-test visual cases.

use std::path::{Path, PathBuf};

use glam::Vec3;
use gltf::image::Format;
use gltf::mesh::Mode;

use crate::error::HarnessError;

use super::mesh::{Mesh, Vertex};

/// Imported GLB payload converted to the harness mesh and RGBA8 texture formats.
#[derive(Clone, Debug)]
pub struct ImportedGltfScene {
    /// First triangle-list mesh primitive converted to Renderide's clockwise vertex layout.
    pub mesh: Mesh,
    /// Base-color texture pixels converted to RGBA8.
    pub texture_rgba: Vec<u8>,
    /// Texture dimensions in pixels.
    pub texture_size: (u32, u32),
}

/// Returns the default optional GLB fixture path used by the scene registry.
pub fn default_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join("gltf")
        .join("static_textured_mesh.glb")
}

/// Loads a static textured GLB fixture into the same CPU-side asset types as procedural cases.
pub fn load_static_textured_mesh(path: &Path) -> Result<ImportedGltfScene, HarnessError> {
    if !path.is_file() {
        return Err(fixture_error(path, "fixture file is missing"));
    }

    let (document, buffers, images) =
        gltf::import(path).map_err(|e| fixture_error(path, format!("import: {e}")))?;

    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            if primitive.mode() != Mode::Triangles {
                continue;
            }
            let imported_mesh = import_primitive_mesh(path, &buffers, &primitive)?;
            let image_index = primitive
                .material()
                .pbr_metallic_roughness()
                .base_color_texture()
                .map(|info| info.texture().source().index())
                .unwrap_or(0);
            let image = images.get(image_index).ok_or_else(|| {
                fixture_error(
                    path,
                    format!("primitive references missing image {image_index}"),
                )
            })?;
            let texture_rgba = convert_image_rgba8(path, image)?;
            return Ok(ImportedGltfScene {
                mesh: imported_mesh,
                texture_rgba,
                texture_size: (image.width, image.height),
            });
        }
    }

    Err(fixture_error(
        path,
        "fixture contains no triangle-list mesh primitive",
    ))
}

fn import_primitive_mesh(
    path: &Path,
    buffers: &[gltf::buffer::Data],
    primitive: &gltf::Primitive<'_>,
) -> Result<Mesh, HarnessError> {
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
    let positions = reader
        .read_positions()
        .ok_or_else(|| fixture_error(path, "triangle primitive has no POSITION attribute"))?
        .collect::<Vec<_>>();
    if positions.is_empty() {
        return Err(fixture_error(path, "triangle primitive has no vertices"));
    }

    let normals = reader
        .read_normals()
        .map(|values| values.collect::<Vec<_>>())
        .unwrap_or_default();
    let texcoords = reader
        .read_tex_coords(0)
        .map(|values| values.into_f32().collect::<Vec<_>>())
        .unwrap_or_default();

    let vertices = positions
        .iter()
        .enumerate()
        .map(|(index, position)| Vertex {
            position: *position,
            normal: normals
                .get(index)
                .copied()
                .unwrap_or_else(|| fallback_normal(*position)),
            uv: texcoords.get(index).copied().unwrap_or([0.0, 0.0]),
        })
        .collect::<Vec<_>>();

    let mut indices = reader
        .read_indices()
        .map(|values| values.into_u32().collect::<Vec<_>>())
        .unwrap_or_else(|| (0..vertices.len() as u32).collect::<Vec<_>>());
    if !indices.len().is_multiple_of(3) {
        return Err(fixture_error(
            path,
            format!("triangle primitive has {} indices", indices.len()),
        ));
    }
    if let Some(index) = indices
        .iter()
        .copied()
        .find(|index| *index as usize >= vertices.len())
    {
        return Err(fixture_error(
            path,
            format!("triangle primitive references missing vertex {index}"),
        ));
    }
    for triangle in indices.chunks_exact_mut(3) {
        triangle.swap(1, 2);
    }

    Ok(Mesh { vertices, indices })
}

fn fallback_normal(position: [f32; 3]) -> [f32; 3] {
    Vec3::from_array(position)
        .try_normalize()
        .unwrap_or(Vec3::Z)
        .to_array()
}

fn convert_image_rgba8(path: &Path, image: &gltf::image::Data) -> Result<Vec<u8>, HarnessError> {
    let pixels = match image.format {
        Format::R8 => expand_u8_channels(&image.pixels, 1),
        Format::R8G8 => expand_u8_channels(&image.pixels, 2),
        Format::R8G8B8 => expand_u8_channels(&image.pixels, 3),
        Format::R8G8B8A8 => image.pixels.clone(),
        Format::R16 => expand_u16_channels(&image.pixels, 1),
        Format::R16G16 => expand_u16_channels(&image.pixels, 2),
        Format::R16G16B16 => expand_u16_channels(&image.pixels, 3),
        Format::R16G16B16A16 => expand_u16_channels(&image.pixels, 4),
        _ => {
            return Err(fixture_error(
                path,
                format!("unsupported texture format {:?}", image.format),
            ));
        }
    };
    let expected_len = image.width as usize * image.height as usize * 4;
    if pixels.len() != expected_len {
        return Err(fixture_error(
            path,
            format!(
                "texture converted to {} bytes, expected {expected_len}",
                pixels.len()
            ),
        ));
    }
    Ok(pixels)
}

fn expand_u8_channels(bytes: &[u8], channels: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() / channels * 4);
    for pixel in bytes.chunks_exact(channels) {
        let r = pixel[0];
        let g = pixel.get(1).copied().unwrap_or(r);
        let b = pixel.get(2).copied().unwrap_or(r);
        let a = pixel.get(3).copied().unwrap_or(255);
        out.extend_from_slice(&[r, g, b, a]);
    }
    out
}

fn expand_u16_channels(bytes: &[u8], channels: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() / (channels * 2) * 4);
    for pixel in bytes.chunks_exact(channels * 2) {
        let channel = |index: usize| -> u8 {
            pixel
                .get(index * 2..index * 2 + 2)
                .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]) >> 8)
                .unwrap_or(255) as u8
        };
        let r = channel(0);
        let g = if channels > 1 { channel(1) } else { r };
        let b = if channels > 2 { channel(2) } else { r };
        let a = if channels > 3 { channel(3) } else { 255 };
        out.extend_from_slice(&[r, g, b, a]);
    }
    out
}

fn fixture_error(path: &Path, message: impl Into<String>) -> HarnessError {
    HarnessError::GltfFixture {
        path: path.to_path_buf(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_fixture_path_is_under_manifest_dir() {
        let path = default_fixture_path();
        assert!(path.ends_with("fixtures/gltf/static_textured_mesh.glb"));
    }

    #[test]
    fn expands_rgb8_texture_to_rgba8() {
        let path = Path::new("fixture.glb");
        let image = gltf::image::Data {
            pixels: vec![10, 20, 30, 40, 50, 60],
            format: Format::R8G8B8,
            width: 2,
            height: 1,
        };
        let rgba = convert_image_rgba8(path, &image).expect("convert");
        assert_eq!(rgba, vec![10, 20, 30, 255, 40, 50, 60, 255]);
    }
}
