//! Final pass that assembles the [`wgpu::BindGroupEntry`] list for an embedded material bind group.

use std::sync::Arc;

use super::super::embedded_material_bind_error::EmbeddedMaterialBindError;
use super::super::layout::StemMaterialLayout;

/// Assembles a [`wgpu::BindGroupEntry`] list matching the reflected material entry order.
///
/// `keepalive_views` and `keepalive_samplers` are positional: the i-th texture binding consumes
/// the i-th view, and the i-th sampler binding consumes the i-th sampler. The caller is
/// responsible for keeping those vectors alive for the lifetime of the returned entries (the
/// returned slice borrows them).
pub(super) fn build_embedded_bind_group_entries<'a>(
    layout: &'a Arc<StemMaterialLayout>,
    uniform_buf: &'a Arc<wgpu::Buffer>,
    keepalive_views: &'a [Arc<wgpu::TextureView>],
    keepalive_samplers: &'a [Arc<wgpu::Sampler>],
) -> Result<Vec<wgpu::BindGroupEntry<'a>>, EmbeddedMaterialBindError> {
    profiling::scope!("materials::embedded_build_bind_entries");
    let mut view_i = 0usize;
    let mut samp_i = 0usize;
    let mut entries: Vec<wgpu::BindGroupEntry<'a>> =
        Vec::with_capacity(layout.reflected.material_entries.len());
    for entry in &layout.reflected.material_entries {
        let b = entry.binding;
        match entry.ty {
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                ..
            } => {
                entries.push(wgpu::BindGroupEntry {
                    binding: b,
                    resource: uniform_buf.as_entire_binding(),
                });
            }
            wgpu::BindingType::Texture { .. } => {
                let tv = keepalive_views
                    .get(view_i)
                    .ok_or_else(|| format!("internal: texture view index {view_i}"))?;
                view_i += 1;
                entries.push(wgpu::BindGroupEntry {
                    binding: b,
                    resource: wgpu::BindingResource::TextureView(tv.as_ref()),
                });
            }
            wgpu::BindingType::Sampler(_) => {
                let s = keepalive_samplers
                    .get(samp_i)
                    .ok_or_else(|| format!("internal: sampler index {samp_i}"))?;
                samp_i += 1;
                entries.push(wgpu::BindGroupEntry {
                    binding: b,
                    resource: wgpu::BindingResource::Sampler(s.as_ref()),
                });
            }
            _ => {
                return Err(EmbeddedMaterialBindError::from(format!(
                    "unsupported binding type for @binding({b})"
                )));
            }
        }
    }
    Ok(entries)
}

/// Pairs a sampler binding to its preceding texture binding (sampler at `N+1` follows texture at `N`).
#[inline]
pub(super) fn sampler_pairs_texture_binding(sampler_binding: u32) -> u32 {
    sampler_binding.saturating_sub(1)
}
