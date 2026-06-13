//! Typed ownership set for GPU resources that must outlive recorded command buffers.
//!
//! The driver thread may submit command buffers after the main render thread has moved on to the
//! next frame. Any resource referenced by those command buffers must stay owned until the driver
//! reaches `Queue::submit`; this container gives submit paths one explicit payload for that
//! ownership.

use std::sync::Arc;

/// GPU resources retained until their owning submit batch has reached the driver thread.
#[derive(Default)]
pub(crate) struct GpuRetainedResources {
    /// Buffers retained until submit.
    buffers: Vec<wgpu::Buffer>,
    /// Textures retained until submit.
    textures: Vec<wgpu::Texture>,
    /// Texture views retained until submit.
    texture_views: Vec<wgpu::TextureView>,
    /// Samplers retained until submit.
    samplers: Vec<wgpu::Sampler>,
    /// Bind groups retained until submit.
    bind_groups: Vec<wgpu::BindGroup>,
    /// Render pipelines retained until submit.
    render_pipelines: Vec<wgpu::RenderPipeline>,
    /// Compute pipelines retained until submit.
    compute_pipelines: Vec<wgpu::ComputePipeline>,
}

impl GpuRetainedResources {
    /// Creates an empty retained-resource set.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Returns true when no resources are retained.
    pub(crate) fn is_empty(&self) -> bool {
        self.buffers.is_empty()
            && self.textures.is_empty()
            && self.texture_views.is_empty()
            && self.samplers.is_empty()
            && self.bind_groups.is_empty()
            && self.render_pipelines.is_empty()
            && self.compute_pipelines.is_empty()
    }

    /// Appends another retained-resource set.
    pub(crate) fn append(&mut self, mut other: Self) {
        self.buffers.append(&mut other.buffers);
        self.textures.append(&mut other.textures);
        self.texture_views.append(&mut other.texture_views);
        self.samplers.append(&mut other.samplers);
        self.bind_groups.append(&mut other.bind_groups);
        self.render_pipelines.append(&mut other.render_pipelines);
        self.compute_pipelines.append(&mut other.compute_pipelines);
    }

    /// Retains one buffer.
    pub(crate) fn retain_buffer(&mut self, buffer: wgpu::Buffer) {
        self.buffers.push(buffer);
    }

    /// Retains multiple buffers.
    pub(crate) fn retain_buffers(&mut self, buffers: impl IntoIterator<Item = wgpu::Buffer>) {
        self.buffers.extend(buffers);
    }

    /// Retains one texture.
    pub(crate) fn retain_texture(&mut self, texture: wgpu::Texture) {
        self.textures.push(texture);
    }

    /// Retains one texture view.
    pub(crate) fn retain_texture_view(&mut self, view: wgpu::TextureView) {
        self.texture_views.push(view);
    }

    /// Retains multiple texture views.
    pub(crate) fn retain_texture_views(
        &mut self,
        views: impl IntoIterator<Item = wgpu::TextureView>,
    ) {
        self.texture_views.extend(views);
    }

    /// Retains one sampler.
    pub(crate) fn retain_sampler(&mut self, sampler: wgpu::Sampler) {
        self.samplers.push(sampler);
    }

    /// Retains one bind group.
    pub(crate) fn retain_bind_group(&mut self, bind_group: wgpu::BindGroup) {
        self.bind_groups.push(bind_group);
    }

    /// Returns a new retained set with one additional buffer.
    pub(crate) fn with_buffer(mut self, buffer: wgpu::Buffer) -> Self {
        self.retain_buffer(buffer);
        self
    }

    /// Returns a new retained set with additional buffers.
    pub(crate) fn with_buffers(mut self, buffers: Vec<wgpu::Buffer>) -> Self {
        self.retain_buffers(buffers);
        self
    }

    /// Returns a new retained set with one additional bind group.
    pub(crate) fn with_bind_group(mut self, bind_group: wgpu::BindGroup) -> Self {
        self.retain_bind_group(bind_group);
        self
    }

    /// Returns a new retained set with one additional shared texture.
    pub(crate) fn with_shared_texture(mut self, texture: Arc<wgpu::Texture>) -> Self {
        self.retain_texture(texture.as_ref().clone());
        self
    }

    /// Returns a new retained set with one additional shared texture view.
    pub(crate) fn with_shared_texture_view(mut self, view: Arc<wgpu::TextureView>) -> Self {
        self.retain_texture_view(view.as_ref().clone());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::GpuRetainedResources;

    #[test]
    fn retained_resources_start_empty() {
        assert!(GpuRetainedResources::new().is_empty());
    }

    #[test]
    fn appending_empty_resource_sets_keeps_empty() {
        let mut resources = GpuRetainedResources::new();
        resources.append(GpuRetainedResources::new());

        assert!(resources.is_empty());
    }
}
