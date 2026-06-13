//! Retired frame-global GPU resources retained until driver-thread submit completion.

use std::sync::Arc;

/// Frame-global GPU handles that must outlive already-recorded command buffers.
#[derive(Default)]
pub(in crate::backend) struct RetiredFrameGpuResources {
    bind_groups: Vec<Arc<wgpu::BindGroup>>,
    buffers: Vec<wgpu::Buffer>,
    shared_buffers: Vec<Arc<wgpu::Buffer>>,
    textures: Vec<Arc<wgpu::Texture>>,
    texture_views: Vec<Arc<wgpu::TextureView>>,
    #[cfg(test)]
    markers: Vec<Arc<()>>,
}

impl RetiredFrameGpuResources {
    /// Returns whether this batch holds no GPU handles.
    pub(in crate::backend) fn is_empty(&self) -> bool {
        self.bind_groups.is_empty()
            && self.buffers.is_empty()
            && self.shared_buffers.is_empty()
            && self.textures.is_empty()
            && self.texture_views.is_empty()
            && {
                #[cfg(test)]
                {
                    self.markers.is_empty()
                }
                #[cfg(not(test))]
                {
                    true
                }
            }
    }

    /// Retains a bind group.
    pub(in crate::backend) fn push_bind_group(&mut self, bind_group: Arc<wgpu::BindGroup>) {
        self.bind_groups.push(bind_group);
    }

    /// Retains an owned buffer handle.
    pub(super) fn push_buffer(&mut self, buffer: wgpu::Buffer) {
        self.buffers.push(buffer);
    }

    /// Retains a shared buffer handle.
    pub(super) fn push_shared_buffer(&mut self, buffer: Arc<wgpu::Buffer>) {
        self.shared_buffers.push(buffer);
    }

    /// Retains a texture handle.
    pub(super) fn push_texture(&mut self, texture: Arc<wgpu::Texture>) {
        self.textures.push(texture);
    }

    /// Retains a texture-view handle.
    pub(super) fn push_texture_view(&mut self, texture_view: Arc<wgpu::TextureView>) {
        self.texture_views.push(texture_view);
    }

    /// Retains multiple texture-view handles.
    pub(super) fn extend_texture_views(
        &mut self,
        texture_views: impl IntoIterator<Item = Arc<wgpu::TextureView>>,
    ) {
        self.texture_views.extend(texture_views);
    }

    /// Appends another retired-resource batch.
    pub(in crate::backend) fn append(&mut self, mut other: Self) {
        self.bind_groups.append(&mut other.bind_groups);
        self.buffers.append(&mut other.buffers);
        self.shared_buffers.append(&mut other.shared_buffers);
        self.textures.append(&mut other.textures);
        self.texture_views.append(&mut other.texture_views);
        #[cfg(test)]
        self.markers.append(&mut other.markers);
    }

    /// Converts this batch into a submit-done callback that releases the retained handles.
    pub(in crate::backend) fn into_submit_callback(
        self,
    ) -> Option<Box<dyn FnOnce() + Send + 'static>> {
        if self.is_empty() {
            None
        } else {
            Some(Box::new(move || drop(self)))
        }
    }

    /// Builds a test-only resource batch that can prove callback retention without a GPU device.
    #[cfg(test)]
    pub(crate) fn marker_for_tests(marker: Arc<()>) -> Self {
        Self {
            markers: vec![marker],
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::RetiredFrameGpuResources;

    #[test]
    fn empty_batch_produces_no_submit_callback() {
        assert!(
            RetiredFrameGpuResources::default()
                .into_submit_callback()
                .is_none()
        );
    }

    #[test]
    fn submit_callback_holds_resources_until_run() {
        let marker = Arc::new(());
        let resources = RetiredFrameGpuResources::marker_for_tests(Arc::clone(&marker));

        assert_eq!(Arc::strong_count(&marker), 2);
        let callback = resources
            .into_submit_callback()
            .expect("retired marker should produce a callback");
        assert_eq!(Arc::strong_count(&marker), 2);

        callback();

        assert_eq!(Arc::strong_count(&marker), 1);
    }
}
