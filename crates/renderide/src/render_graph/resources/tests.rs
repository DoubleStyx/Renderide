//! Unit tests for render-graph resource vocabulary.

use super::*;

#[test]
fn handle_indices_match_wrapped_ids() {
    assert_eq!(TextureHandle(7).index(), 7);
    assert_eq!(SubresourceHandle(8).index(), 8);
    assert_eq!(BufferHandle(9).index(), 9);
    assert_eq!(ImportedTextureHandle(10).index(), 10);
    assert_eq!(ImportedBufferHandle(11).index(), 11);
}

#[test]
fn subresource_constructors_target_single_mip_or_layer() {
    let parent = TextureHandle(3);
    let mip = TransientSubresourceDesc::single_mip(parent, "mip2", 2);
    assert_eq!(mip.parent, parent);
    assert_eq!(mip.base_mip_level, 2);
    assert_eq!(mip.mip_level_count, 1);
    assert_eq!(mip.base_array_layer, 0);
    assert_eq!(mip.array_layer_count, 1);

    let layer = TransientSubresourceDesc::single_layer(parent, "layer4", 4);
    assert_eq!(layer.parent, parent);
    assert_eq!(layer.base_mip_level, 0);
    assert_eq!(layer.base_array_layer, 4);
    assert_eq!(layer.array_layer_count, 1);
}

#[test]
fn transient_extent_fixed_extent_only_for_concrete_sizes() {
    assert_eq!(
        TransientExtent::Custom {
            width: 10,
            height: 20
        }
        .fixed_extent(),
        Some((10, 20, 1))
    );
    assert_eq!(
        TransientExtent::MultiLayer {
            width: 10,
            height: 20,
            layers: 3,
        }
        .fixed_extent(),
        Some((10, 20, 3))
    );
    assert_eq!(TransientExtent::Backbuffer.fixed_extent(), None);
    assert_eq!(
        TransientExtent::BackbufferScaledMip {
            max_dim: 512,
            mip: 2
        }
        .fixed_extent(),
        None
    );
}

#[test]
fn transient_format_and_count_policies_resolve_without_gpu() {
    assert_eq!(
        TransientTextureFormat::Fixed(wgpu::TextureFormat::Rgba8Unorm).resolve(
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Depth24Plus,
            wgpu::TextureFormat::Rgba16Float,
        ),
        wgpu::TextureFormat::Rgba8Unorm
    );
    assert_eq!(
        TransientTextureFormat::FrameColor.resolve(
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Depth24Plus,
            wgpu::TextureFormat::Rgba16Float,
        ),
        wgpu::TextureFormat::Bgra8Unorm
    );
    assert_eq!(
        TransientTextureFormat::FrameDepthStencil.resolve(
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Depth24Plus,
            wgpu::TextureFormat::Rgba16Float,
        ),
        wgpu::TextureFormat::Depth24Plus
    );
    assert_eq!(
        TransientTextureFormat::SceneColorHdr.resolve(
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Depth24Plus,
            wgpu::TextureFormat::Rgba16Float,
        ),
        wgpu::TextureFormat::Rgba16Float
    );

    assert_eq!(TransientArrayLayers::Fixed(0).resolve(false), 1);
    assert_eq!(TransientArrayLayers::Fixed(4).resolve(true), 4);
    assert_eq!(TransientArrayLayers::Frame.resolve(false), 1);
    assert_eq!(TransientArrayLayers::Frame.resolve(true), 2);
    assert_eq!(TransientSampleCount::Fixed(0).resolve(8), 1);
    assert_eq!(TransientSampleCount::Fixed(4).resolve(8), 4);
    assert_eq!(TransientSampleCount::Frame.resolve(0), 1);
    assert_eq!(TransientSampleCount::Frame.resolve(8), 8);
}

#[test]
fn transient_texture_constructors_set_frame_sampled_policies() {
    let fixed = TransientTextureDesc::texture_2d(
        "fixed",
        wgpu::TextureFormat::Rgba8Unorm,
        TransientExtent::Backbuffer,
        0,
        wgpu::TextureUsages::COPY_DST,
    );
    assert_eq!(fixed.sample_count, TransientSampleCount::Fixed(0));
    assert_eq!(fixed.array_layers, TransientArrayLayers::Fixed(1));
    assert!(fixed.alias);

    let frame_layers = fixed.with_array_layers(0).with_frame_array_layers();
    assert_eq!(frame_layers.array_layers, TransientArrayLayers::Frame);

    let frame_color = TransientTextureDesc::frame_color_sampled_texture_2d(
        "frame_color",
        TransientExtent::Backbuffer,
        wgpu::TextureUsages::TEXTURE_BINDING,
    );
    assert_eq!(frame_color.format, TransientTextureFormat::FrameColor);
    assert_eq!(frame_color.sample_count, TransientSampleCount::Frame);

    let frame_depth = TransientTextureDesc::frame_depth_stencil_sampled_texture_2d(
        "frame_depth",
        TransientExtent::Backbuffer,
        wgpu::TextureUsages::RENDER_ATTACHMENT,
    );
    assert_eq!(
        frame_depth.format,
        TransientTextureFormat::FrameDepthStencil
    );
    assert_eq!(frame_depth.sample_count, TransientSampleCount::Frame);
}

#[test]
fn storage_access_read_write_flags_cover_all_variants() {
    assert!(StorageAccess::ReadOnly.reads());
    assert!(!StorageAccess::ReadOnly.writes());
    assert!(!StorageAccess::WriteOnly.reads());
    assert!(StorageAccess::WriteOnly.writes());
    assert!(StorageAccess::ReadWrite.reads());
    assert!(StorageAccess::ReadWrite.writes());
}

#[test]
fn texture_access_flags_and_usages_cover_all_variants() {
    let load_color = TextureAccess::ColorAttachment {
        load: wgpu::LoadOp::Load,
        store: wgpu::StoreOp::Store,
        resolve_to: None,
    };
    assert!(load_color.reads());
    assert!(load_color.writes());
    assert!(load_color.is_attachment());
    assert_eq!(load_color.usage(), wgpu::TextureUsages::RENDER_ATTACHMENT);

    let clear_depth = TextureAccess::DepthAttachment {
        depth: wgpu::Operations {
            load: wgpu::LoadOp::Clear(1.0),
            store: wgpu::StoreOp::Store,
        },
        stencil: Some(wgpu::Operations {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Discard,
        }),
    };
    assert!(clear_depth.reads());
    assert!(clear_depth.writes());
    assert!(clear_depth.is_attachment());

    let sampled = TextureAccess::Sampled {
        stages: wgpu::ShaderStages::FRAGMENT,
    };
    assert!(sampled.reads());
    assert!(!sampled.writes());
    assert_eq!(sampled.usage(), wgpu::TextureUsages::TEXTURE_BINDING);

    let storage_write = TextureAccess::Storage {
        stages: wgpu::ShaderStages::COMPUTE,
        access: StorageAccess::WriteOnly,
    };
    assert!(!storage_write.reads());
    assert!(storage_write.writes());
    assert_eq!(storage_write.usage(), wgpu::TextureUsages::STORAGE_BINDING);

    assert!(TextureAccess::CopySrc.reads());
    assert!(!TextureAccess::CopySrc.writes());
    assert_eq!(
        TextureAccess::CopySrc.usage(),
        wgpu::TextureUsages::COPY_SRC
    );
    assert!(!TextureAccess::CopyDst.reads());
    assert!(TextureAccess::CopyDst.writes());
    assert_eq!(
        TextureAccess::CopyDst.usage(),
        wgpu::TextureUsages::COPY_DST
    );
    assert!(!TextureAccess::Present.reads());
    assert!(TextureAccess::Present.writes());
}

#[test]
fn buffer_access_flags_and_usages_cover_all_variants() {
    let uniform = BufferAccess::Uniform {
        stages: wgpu::ShaderStages::VERTEX,
        dynamic_offset: true,
    };
    assert!(uniform.reads());
    assert!(!uniform.writes());
    assert_eq!(uniform.usage(), wgpu::BufferUsages::UNIFORM);

    let storage_read_write = BufferAccess::Storage {
        stages: wgpu::ShaderStages::COMPUTE,
        access: StorageAccess::ReadWrite,
    };
    assert!(storage_read_write.reads());
    assert!(storage_read_write.writes());
    assert_eq!(storage_read_write.usage(), wgpu::BufferUsages::STORAGE);

    for (access, usage) in [
        (BufferAccess::Index, wgpu::BufferUsages::INDEX),
        (BufferAccess::Vertex, wgpu::BufferUsages::VERTEX),
        (BufferAccess::Indirect, wgpu::BufferUsages::INDIRECT),
        (BufferAccess::CopySrc, wgpu::BufferUsages::COPY_SRC),
    ] {
        assert!(access.reads(), "{access:?}");
        assert!(!access.writes(), "{access:?}");
        assert_eq!(access.usage(), usage);
    }
    assert!(!BufferAccess::CopyDst.reads());
    assert!(BufferAccess::CopyDst.writes());
    assert_eq!(BufferAccess::CopyDst.usage(), wgpu::BufferUsages::COPY_DST);
}

#[test]
fn resource_access_delegates_texture_and_buffer_metadata() {
    let texture =
        ResourceAccess::texture(TextureHandle(1).into(), TextureAccess::CopyDst, false, true);
    assert!(!texture.reads());
    assert!(texture.writes());
    assert_eq!(texture.texture_usage(), Some(wgpu::TextureUsages::COPY_DST));
    assert_eq!(texture.buffer_usage(), None);
    assert!(!texture.resource.is_imported());
    assert_eq!(texture.resource.transient_texture(), Some(TextureHandle(1)));

    let buffer = ResourceAccess::buffer(
        ImportedBufferHandle(2).into(),
        BufferAccess::CopySrc,
        true,
        false,
    );
    assert!(buffer.reads());
    assert!(!buffer.writes());
    assert_eq!(buffer.texture_usage(), None);
    assert_eq!(buffer.buffer_usage(), Some(wgpu::BufferUsages::COPY_SRC));
    assert!(buffer.resource.is_imported());
    assert_eq!(buffer.resource.transient_buffer(), None);
}

#[test]
fn backend_buffer_labels_are_stable() {
    assert_eq!(BackendFrameBufferKind::Lights.label(), "lights");
    assert_eq!(
        BackendFrameBufferKind::ClusterLightCounts.label(),
        "cluster_light_counts"
    );
    assert_eq!(
        BackendFrameBufferKind::ClusterLightIndices.label(),
        "cluster_light_indices"
    );
    assert_eq!(BackendFrameBufferKind::PerDrawSlab.label(), "per_draw_slab");
    assert_eq!(
        BackendFrameBufferKind::FrameUniforms.label(),
        "frame_uniforms"
    );
    assert_eq!(HistorySlotId::HI_Z.name(), "hi_z");
    assert_eq!(HistorySlotId::new("custom").name(), "custom");
}
