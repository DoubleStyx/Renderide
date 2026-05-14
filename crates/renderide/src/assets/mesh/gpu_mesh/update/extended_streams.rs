//! Lazy creation of tangent / UV1-UV3 vertex streams for meshes that hit extended embedded shaders.

use crate::materials::EmbeddedTangentFallbackMode;
use crate::shared::VertexAttributeType;

use super::super::upload::{
    ExtendedVertexUploadSource, UvVertexUploadSource, upload_default_extended_vertex_streams,
    upload_default_raw_tangent_vertex_stream, upload_default_tangent_vertex_stream,
    upload_default_uv_vertex_stream, upload_default_wide_uv_vertex_stream,
    upload_extended_vertex_streams, upload_raw_tangent_vertex_stream, upload_tangent_vertex_stream,
    upload_uv_vertex_stream, upload_wide_uv_vertex_stream,
};
use super::super::{ExtendedVertexStreamSource, GpuMesh, extended_vertex_stream_bytes};

impl GpuMesh {
    /// Creates tangent / UV1-3 streams the first time an embedded shader needs all of them.
    pub(crate) fn ensure_extended_vertex_streams(
        &mut self,
        device: &wgpu::Device,
        tangent_fallback_mode: EmbeddedTangentFallbackMode,
    ) -> bool {
        profiling::scope!("asset::mesh_ensure_extended_vertex_streams");
        if self.extended_vertex_streams_ready() {
            if self.tangent_fallback_needs_upgrade(tangent_fallback_mode) {
                return self.ensure_tangent_vertex_stream(device, tangent_fallback_mode);
            }
            return true;
        }

        let old_bytes = extended_vertex_stream_bytes(self);
        let vc_usize = self.vertex_count as usize;
        let (tangent_buffer, uv1_buffer, uv2_buffer, uv3_buffer) =
            if let Some(source) = self.extended_vertex_stream_source.as_ref() {
                upload_extended_vertex_streams(
                    device,
                    self.asset_id,
                    ExtendedVertexUploadSource {
                        vertex_slice: source.vertex_bytes.as_ref(),
                        index_slice: source.index_bytes.as_ref(),
                        vertex_count: vc_usize,
                        vertex_stride: self.vertex_stride as usize,
                        vertex_attributes: source.vertex_attributes.as_ref(),
                        index_format: source.index_format,
                        submeshes: source.submeshes.as_ref(),
                    },
                    tangent_fallback_mode.generate_missing(),
                )
            } else {
                upload_default_extended_vertex_streams(device, self.asset_id, vc_usize)
            };

        if tangent_buffer.is_none()
            || uv1_buffer.is_none()
            || uv2_buffer.is_none()
            || uv3_buffer.is_none()
        {
            return false;
        }

        // pay the 40 bytes/vertex only for meshes that hit extended shaders.
        self.tangent_buffer = tangent_buffer;
        self.tangent_fallback_mode = self.tangent_fallback_mode.max(tangent_fallback_mode);
        self.uv1_buffer = uv1_buffer;
        self.uv2_buffer = uv2_buffer;
        self.uv3_buffer = uv3_buffer;
        let new_bytes = extended_vertex_stream_bytes(self);
        self.resident_bytes = self
            .resident_bytes
            .saturating_sub(old_bytes)
            .saturating_add(new_bytes);
        self.drop_extended_vertex_stream_source_if_complete();
        true
    }

    /// Creates a tangent stream the first time an embedded shader declares `@location(4)`.
    pub(crate) fn ensure_tangent_vertex_stream(
        &mut self,
        device: &wgpu::Device,
        tangent_fallback_mode: EmbeddedTangentFallbackMode,
    ) -> bool {
        profiling::scope!("asset::mesh_ensure_tangent_vertex_stream");
        if self.tangent_vertex_stream_ready()
            && !self.tangent_fallback_needs_upgrade(tangent_fallback_mode)
        {
            return true;
        }

        let vc_usize = self.vertex_count as usize;
        let tangent_buffer = if let Some(source) = self.extended_vertex_stream_source.as_ref() {
            upload_tangent_vertex_stream(
                device,
                self.asset_id,
                ExtendedVertexUploadSource {
                    vertex_slice: source.vertex_bytes.as_ref(),
                    index_slice: source.index_bytes.as_ref(),
                    vertex_count: vc_usize,
                    vertex_stride: self.vertex_stride as usize,
                    vertex_attributes: source.vertex_attributes.as_ref(),
                    index_format: source.index_format,
                    submeshes: source.submeshes.as_ref(),
                },
                tangent_fallback_mode.generate_missing(),
            )
        } else {
            upload_default_tangent_vertex_stream(device, self.asset_id, vc_usize)
        };

        let Some(tangent_buffer) = tangent_buffer else {
            return false;
        };
        let old_bytes = self
            .tangent_buffer
            .as_ref()
            .map_or(0, |buffer| buffer.size());
        self.tangent_buffer = Some(tangent_buffer);
        self.tangent_fallback_mode = self.tangent_fallback_mode.max(tangent_fallback_mode);
        let new_bytes = self
            .tangent_buffer
            .as_ref()
            .map_or(0, |buffer| buffer.size());
        self.resident_bytes = self
            .resident_bytes
            .saturating_sub(old_bytes)
            .saturating_add(new_bytes);
        self.drop_extended_vertex_stream_source_if_complete();
        true
    }

    /// Creates a raw tangent payload stream for UI shaders that use `TANGENT` as data.
    pub(crate) fn ensure_raw_tangent_vertex_stream(&mut self, device: &wgpu::Device) -> bool {
        profiling::scope!("asset::mesh_ensure_raw_tangent_vertex_stream");
        if self.raw_tangent_vertex_stream_ready() {
            return true;
        }

        let vc_usize = self.vertex_count as usize;
        let raw_tangent_buffer = if let Some(source) = self.extended_vertex_stream_source.as_ref() {
            upload_raw_tangent_vertex_stream(
                device,
                self.asset_id,
                ExtendedVertexUploadSource {
                    vertex_slice: source.vertex_bytes.as_ref(),
                    index_slice: source.index_bytes.as_ref(),
                    vertex_count: vc_usize,
                    vertex_stride: self.vertex_stride as usize,
                    vertex_attributes: source.vertex_attributes.as_ref(),
                    index_format: source.index_format,
                    submeshes: source.submeshes.as_ref(),
                },
            )
        } else {
            upload_default_raw_tangent_vertex_stream(device, self.asset_id, vc_usize)
        };

        let Some(raw_tangent_buffer) = raw_tangent_buffer else {
            return false;
        };
        let old_bytes = self
            .raw_tangent_buffer
            .as_ref()
            .map_or(0, |buffer| buffer.size());
        self.raw_tangent_buffer = Some(raw_tangent_buffer);
        let new_bytes = self
            .raw_tangent_buffer
            .as_ref()
            .map_or(0, |buffer| buffer.size());
        self.resident_bytes = self
            .resident_bytes
            .saturating_sub(old_bytes)
            .saturating_add(new_bytes);
        self.drop_extended_vertex_stream_source_if_complete();
        true
    }

    /// Creates a UV1 stream the first time an embedded shader needs it without other extended streams.
    pub(crate) fn ensure_uv1_vertex_stream(&mut self, device: &wgpu::Device) -> bool {
        profiling::scope!("asset::mesh_ensure_uv1_vertex_stream");
        if self.uv1_vertex_stream_ready() {
            return true;
        }

        let old_bytes = self.uv1_buffer.as_ref().map_or(0, |buffer| buffer.size());
        let vc_usize = self.vertex_count as usize;
        let uv1_buffer = if let Some(source) = self.extended_vertex_stream_source.as_ref() {
            upload_uv_vertex_stream(
                device,
                self.asset_id,
                UvVertexUploadSource {
                    vertex_slice: source.vertex_bytes.as_ref(),
                    vertex_count: vc_usize,
                    vertex_stride: self.vertex_stride as usize,
                    vertex_attributes: source.vertex_attributes.as_ref(),
                    target: VertexAttributeType::UV1,
                    label: "uv1",
                },
            )
        } else {
            upload_default_uv_vertex_stream(device, self.asset_id, vc_usize, "uv1")
        };

        let Some(uv1_buffer) = uv1_buffer else {
            return false;
        };
        self.uv1_buffer = Some(uv1_buffer);
        let new_bytes = self.uv1_buffer.as_ref().map_or(0, |buffer| buffer.size());
        self.resident_bytes = self
            .resident_bytes
            .saturating_sub(old_bytes)
            .saturating_add(new_bytes);
        self.drop_extended_vertex_stream_source_if_complete();
        true
    }

    /// Creates a UV2 stream the first time an embedded shader declares `@location(6)`.
    pub(crate) fn ensure_uv2_vertex_stream(&mut self, device: &wgpu::Device) -> bool {
        self.ensure_extra_uv_vertex_stream(
            device,
            VertexAttributeType::UV2,
            "uv2",
            Self::uv2_vertex_stream_ready,
        )
    }

    /// Creates a UV3 stream the first time an embedded shader declares `@location(7)`.
    pub(crate) fn ensure_uv3_vertex_stream(&mut self, device: &wgpu::Device) -> bool {
        self.ensure_extra_uv_vertex_stream(
            device,
            VertexAttributeType::UV3,
            "uv3",
            Self::uv3_vertex_stream_ready,
        )
    }

    /// Creates the packed UV0-UV7 stream the first time a shader needs wide UV inputs.
    pub(crate) fn ensure_wide_uv_vertex_stream(&mut self, device: &wgpu::Device) -> bool {
        profiling::scope!("asset::mesh_ensure_wide_uv_vertex_stream");
        if self.wide_uv_vertex_stream_ready() {
            return true;
        }

        let old_bytes = self
            .wide_uv_buffer
            .as_ref()
            .map_or(0, |buffer| buffer.size());
        let vc_usize = self.vertex_count as usize;
        let wide_uv_buffer = if let Some(source) = self.extended_vertex_stream_source.as_ref() {
            upload_wide_uv_vertex_stream(
                device,
                self.asset_id,
                UvVertexUploadSource {
                    vertex_slice: source.vertex_bytes.as_ref(),
                    vertex_count: vc_usize,
                    vertex_stride: self.vertex_stride as usize,
                    vertex_attributes: source.vertex_attributes.as_ref(),
                    target: VertexAttributeType::UV0,
                    label: "wide_uv",
                },
            )
        } else {
            upload_default_wide_uv_vertex_stream(device, self.asset_id, vc_usize)
        };

        let Some(wide_uv_buffer) = wide_uv_buffer else {
            return false;
        };
        self.wide_uv_buffer = Some(wide_uv_buffer);
        let new_bytes = self
            .wide_uv_buffer
            .as_ref()
            .map_or(0, |buffer| buffer.size());
        self.resident_bytes = self
            .resident_bytes
            .saturating_sub(old_bytes)
            .saturating_add(new_bytes);
        self.drop_extended_vertex_stream_source_if_complete();
        true
    }

    fn ensure_extra_uv_vertex_stream(
        &mut self,
        device: &wgpu::Device,
        target: VertexAttributeType,
        label: &str,
        ready: fn(&Self) -> bool,
    ) -> bool {
        profiling::scope!("asset::mesh_ensure_extra_uv_vertex_stream");
        if ready(self) {
            return true;
        }

        let vc_usize = self.vertex_count as usize;
        let buffer = if let Some(source) = self.extended_vertex_stream_source.as_ref() {
            upload_uv_vertex_stream(
                device,
                self.asset_id,
                UvVertexUploadSource {
                    vertex_slice: source.vertex_bytes.as_ref(),
                    vertex_count: vc_usize,
                    vertex_stride: self.vertex_stride as usize,
                    vertex_attributes: source.vertex_attributes.as_ref(),
                    target,
                    label,
                },
            )
        } else {
            upload_default_uv_vertex_stream(device, self.asset_id, vc_usize, label)
        };

        let Some(buffer) = buffer else {
            return false;
        };
        let slot = match target {
            VertexAttributeType::UV2 => &mut self.uv2_buffer,
            VertexAttributeType::UV3 => &mut self.uv3_buffer,
            _ => return false,
        };
        let old_bytes = slot.as_ref().map_or(0, |buffer| buffer.size());
        *slot = Some(buffer);
        let new_bytes = slot.as_ref().map_or(0, |buffer| buffer.size());
        self.resident_bytes = self
            .resident_bytes
            .saturating_sub(old_bytes)
            .saturating_add(new_bytes);
        self.drop_extended_vertex_stream_source_if_complete();
        true
    }

    fn drop_extended_vertex_stream_source_if_complete(&mut self) {
        if self.can_drop_extended_vertex_stream_source() {
            self.extended_vertex_stream_source = None;
        }
    }

    pub(super) fn can_drop_extended_vertex_stream_source(&self) -> bool {
        !self
            .extended_vertex_stream_source
            .as_ref()
            .is_some_and(|source| self.should_keep_extended_vertex_stream_source(source))
    }

    pub(super) fn should_keep_extended_vertex_stream_source(
        &self,
        source: &ExtendedVertexStreamSource,
    ) -> bool {
        self.should_keep_extended_vertex_stream_source_for_compact_streams(source)
            || self.should_keep_extended_vertex_stream_source_for_wide_uv(source)
            || self.should_keep_extended_vertex_stream_source_for_tangent_upgrade_from(
                source.can_generate_missing_tangents,
            )
    }

    fn should_keep_extended_vertex_stream_source_for_compact_streams(
        &self,
        source: &ExtendedVertexStreamSource,
    ) -> bool {
        !self.extended_vertex_streams_ready() && source.has_compact_extended_payload
    }

    fn should_keep_extended_vertex_stream_source_for_wide_uv(
        &self,
        source: &ExtendedVertexStreamSource,
    ) -> bool {
        self.wide_uv_buffer.is_none() && source.has_wide_uv_payload
    }

    fn tangent_fallback_needs_upgrade(&self, requested: EmbeddedTangentFallbackMode) -> bool {
        requested > self.tangent_fallback_mode
    }

    fn should_keep_extended_vertex_stream_source_for_tangent_upgrade_from(
        &self,
        can_generate_missing_tangents: bool,
    ) -> bool {
        should_keep_tangent_upgrade_source(
            self.tangent_buffer.is_some(),
            self.tangent_fallback_mode,
            can_generate_missing_tangents,
        )
    }
}

pub(super) fn should_keep_tangent_upgrade_source(
    tangent_ready: bool,
    tangent_fallback_mode: EmbeddedTangentFallbackMode,
    can_generate_missing_tangents: bool,
) -> bool {
    tangent_ready
        && tangent_fallback_mode < EmbeddedTangentFallbackMode::GenerateMissing
        && can_generate_missing_tangents
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_tangent_stream_keeps_source_when_generated_upgrade_is_possible() {
        assert!(should_keep_tangent_upgrade_source(
            true,
            EmbeddedTangentFallbackMode::PreserveHostOrDefault,
            true
        ));
    }

    #[test]
    fn generated_or_unusable_tangent_streams_drop_lazy_source() {
        assert!(!should_keep_tangent_upgrade_source(
            false,
            EmbeddedTangentFallbackMode::PreserveHostOrDefault,
            true
        ));
        assert!(!should_keep_tangent_upgrade_source(
            true,
            EmbeddedTangentFallbackMode::GenerateMissing,
            true
        ));
        assert!(!should_keep_tangent_upgrade_source(
            true,
            EmbeddedTangentFallbackMode::PreserveHostOrDefault,
            false
        ));
    }
}
