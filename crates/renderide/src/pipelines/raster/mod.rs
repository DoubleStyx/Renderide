//! Raster pipeline family builders (mesh materials, UI, etc.).

pub(crate) mod debug_world_normals;

pub use debug_world_normals::{DebugWorldNormalsFamily, SHADER_PERM_MULTIVIEW_STEREO};

#[cfg(test)]
mod wgpu_pipeline_tests {
    use std::sync::Arc;

    use crate::materials::MaterialPipelineDesc;
    use crate::pipelines::raster::debug_world_normals::{
        build_debug_world_normals_wgsl, create_debug_world_normals_render_pipeline,
    };
    use crate::pipelines::ShaderPermutation;

    use super::{DebugWorldNormalsFamily, SHADER_PERM_MULTIVIEW_STEREO};

    async fn device_with_adapter() -> Option<Arc<wgpu::Device>> {
        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = wgpu::Backends::all();
        instance_desc.flags = wgpu::InstanceFlags::empty();
        let instance = wgpu::Instance::new(instance_desc);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok()?;
        let (device, _) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("pipelines_wgpu_pipeline_tests"),
                required_features: wgpu::Features::empty(),
                ..Default::default()
            })
            .await
            .ok()?;
        Some(Arc::new(device))
    }

    /// Headless GPU stack; run `cargo test -p renderide pipelines_wgpu -- --ignored --test-threads=1`.
    #[test]
    #[ignore = "wgpu/GPU stack (may SIGSEGV vs parallel harness); run with --ignored"]
    fn debug_world_normals_pipeline_build_smoke() {
        let Some(device) = pollster::block_on(device_with_adapter()) else {
            logger::warn!("skipping debug_world_normals_pipeline_build_smoke: no wgpu adapter");
            return;
        };
        let desc = MaterialPipelineDesc {
            surface_format: wgpu::TextureFormat::Bgra8UnormSrgb,
            depth_stencil_format: None,
            sample_count: 1,
            multiview_mask: None,
        };
        DebugWorldNormalsFamily::per_draw_bind_group_layout(&device).expect("per_draw layout");
        let w0 = build_debug_world_normals_wgsl(ShaderPermutation(0)).expect("wgsl0");
        let w1 = build_debug_world_normals_wgsl(SHADER_PERM_MULTIVIEW_STEREO).expect("wgsl1");
        assert_ne!(w0, w1);
        assert!(!w0.is_empty());
        let sm0 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("debug_world_normals_pipeline_smoke"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(w0.as_str())),
        });
        let _pipe = create_debug_world_normals_render_pipeline(&device, &sm0, &desc, &w0)
            .expect("render pipeline");
    }
}
