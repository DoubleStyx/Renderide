//! Build-time procedural skybox mesh extraction.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use super::shader::BuildError;

/// Source asset for the procedural skybox mesh.
const PROCEDURAL_SKYBOX_MESH_PATH: &str = "assets/models/skybox.glb";

/// Generated Rust file containing the procedural skybox vertex data.
const GENERATED_SKYBOX_MESH_RS: &str = "procedural_skybox_mesh.rs";

/// Extracts the procedural skybox GLB mesh into generated Rust constants.
pub fn emit_procedural_skybox_mesh(manifest_dir: &Path, out_dir: &Path) -> Result<(), BuildError> {
    let source_path = manifest_dir.join(PROCEDURAL_SKYBOX_MESH_PATH);
    println!("cargo:rerun-if-changed={}", source_path.display());

    let vertices = read_procedural_skybox_vertices(&source_path)?;
    let generated = render_generated_skybox_mesh(&vertices)?;
    fs::write(out_dir.join(GENERATED_SKYBOX_MESH_RS), generated)?;
    Ok(())
}

/// Loads and validates the source GLB as a flat triangle-list position stream.
fn read_procedural_skybox_vertices(path: &Path) -> Result<Vec<[f32; 3]>, BuildError> {
    if !path.is_file() {
        return Err(BuildError::Message(format!(
            "procedural skybox mesh source is missing: {}",
            path.display()
        )));
    }

    let (document, buffers, _) = gltf::import(path)
        .map_err(|e| BuildError::Message(format!("import {}: {e}", path.display())))?;
    let mut vertices = Vec::new();
    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                return Err(BuildError::Message(format!(
                    "procedural skybox mesh {} primitive {} is {:?}, expected triangles",
                    mesh.index(),
                    primitive.index(),
                    primitive.mode()
                )));
            }
            append_primitive_vertices(path, &buffers, &primitive, &mut vertices)?;
        }
    }

    validate_vertex_stream(path, &vertices)?;
    Ok(vertices)
}

/// Appends one glTF primitive as a flat triangle-list position stream.
fn append_primitive_vertices(
    path: &Path,
    buffers: &[gltf::buffer::Data],
    primitive: &gltf::Primitive<'_>,
    out: &mut Vec<[f32; 3]>,
) -> Result<(), BuildError> {
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
    let positions = reader
        .read_positions()
        .ok_or_else(|| {
            BuildError::Message(format!(
                "procedural skybox mesh primitive {} in {} has no POSITION attribute",
                primitive.index(),
                path.display()
            ))
        })?
        .collect::<Vec<_>>();

    if let Some(indices) = reader.read_indices() {
        let indices = indices.into_u32().collect::<Vec<_>>();
        if !indices.len().is_multiple_of(3) {
            return Err(BuildError::Message(format!(
                "procedural skybox mesh primitive {} in {} has {} indices, not a triangle list",
                primitive.index(),
                path.display(),
                indices.len()
            )));
        }
        for index in indices {
            let Some(position) = positions.get(index as usize) else {
                return Err(BuildError::Message(format!(
                    "procedural skybox mesh primitive {} in {} references missing vertex {}",
                    primitive.index(),
                    path.display(),
                    index
                )));
            };
            out.push(*position);
        }
    } else {
        if !positions.len().is_multiple_of(3) {
            return Err(BuildError::Message(format!(
                "procedural skybox mesh primitive {} in {} has {} positions, not a triangle list",
                primitive.index(),
                path.display(),
                positions.len()
            )));
        }
        out.extend(positions);
    }

    Ok(())
}

/// Validates the generated vertex stream before Rust code emission.
fn validate_vertex_stream(path: &Path, vertices: &[[f32; 3]]) -> Result<(), BuildError> {
    if vertices.is_empty() {
        return Err(BuildError::Message(format!(
            "procedural skybox mesh {} did not produce any vertices",
            path.display()
        )));
    }
    if !vertices.len().is_multiple_of(3) {
        return Err(BuildError::Message(format!(
            "procedural skybox mesh {} produced {} vertices, not a triangle list",
            path.display(),
            vertices.len()
        )));
    }
    if vertices.len() > u32::MAX as usize {
        return Err(BuildError::Message(format!(
            "procedural skybox mesh {} produced too many vertices: {}",
            path.display(),
            vertices.len()
        )));
    }

    for (index, vertex) in vertices.iter().enumerate() {
        if !vertex.iter().all(|component| component.is_finite()) {
            return Err(BuildError::Message(format!(
                "procedural skybox mesh {} has a non-finite vertex at index {}",
                path.display(),
                index
            )));
        }
        if !vertex
            .iter()
            .any(|component| component.abs() > f32::EPSILON)
        {
            return Err(BuildError::Message(format!(
                "procedural skybox mesh {} has a zero-position vertex at index {}",
                path.display(),
                index
            )));
        }
    }

    Ok(())
}

/// Renders the generated Rust module consumed by the renderer.
fn render_generated_skybox_mesh(vertices: &[[f32; 3]]) -> Result<String, BuildError> {
    let vertex_count = u32::try_from(vertices.len()).map_err(|_out_of_range| {
        BuildError::Message(format!(
            "procedural skybox mesh has too many vertices: {}",
            vertices.len()
        ))
    })?;
    let mut out = String::new();
    let _ = writeln!(out, "// Generated by `build.rs` - do not edit.");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "/// Triangle-list vertices for the procedural skybox mesh."
    );
    let _ = writeln!(
        out,
        "static PROCEDURAL_SKYBOX_VERTICES: [[f32; 3]; {}] = [",
        vertices.len()
    );
    for vertex in vertices {
        let _ = writeln!(
            out,
            "    [f32::from_bits({}), f32::from_bits({}), f32::from_bits({})],",
            f32_bits_literal(vertex[0]),
            f32_bits_literal(vertex[1]),
            f32_bits_literal(vertex[2])
        );
    }
    let _ = writeln!(out, "];");
    let _ = writeln!(out);
    let _ = writeln!(out, "/// Vertex count for the procedural skybox mesh.");
    let _ = writeln!(
        out,
        "const PROCEDURAL_SKYBOX_VERTEX_COUNT: u32 = {vertex_count};"
    );
    Ok(out)
}

/// Formats one `f32` value as a stable hexadecimal bit-pattern literal.
fn f32_bits_literal(value: f32) -> String {
    let bits = value.to_bits();
    format!("0x{:04X}_{:04X}", bits >> 16, bits & 0xffff)
}
