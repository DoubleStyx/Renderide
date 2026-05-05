//! Raster template and frame-policy tests.

use super::common::*;

#[test]
fn frameglobal_runs_before_perview_by_default() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let bb = b.import_texture(backbuffer_import());
    b.add_raster_pass(Box::new(TestRasterPass::new("per-view", bb)));
    b.add_compute_pass(Box::new(
        TestComputePass::new("frame").frame_global().cull_exempt(),
    ));
    let g = b.build()?;
    assert_eq!(g.pass_info[0].name, "frame");
    assert_eq!(g.pass_info[1].name, "per-view");
    Ok(())
}

#[test]
fn group_order_respects_group_after_declarations() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let bb = b.import_texture(backbuffer_import());
    let a_group = b.group("a", GroupScope::PerView);
    let z_group = b.group("z", GroupScope::PerView);
    b.group_after(z_group, a_group);
    b.add_raster_pass_to_group(z_group, Box::new(TestRasterPass::new("z", bb)));
    b.add_compute_pass_to_group(a_group, Box::new(TestComputePass::new("a").cull_exempt()));
    let g = b.build()?;
    assert_eq!(g.pass_info[0].name, "a");
    assert_eq!(g.pass_info[1].name, "z");
    Ok(())
}

#[test]
fn multiview_mask_propagates_into_template() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let bb = b.import_texture(backbuffer_import());
    let mut pass = TestRasterPass::new("mv", bb);
    pass.multiview_mask = std::num::NonZeroU32::new(3);
    b.add_raster_pass(Box::new(pass));
    let g = b.build()?;
    assert_eq!(g.pass_info[0].multiview_mask.unwrap().get(), 3);
    let mv = g.pass_info[0]
        .raster_template
        .as_ref()
        .and_then(|template| template.multiview_mask);
    assert!(mv.is_some());
    assert_eq!(mv.unwrap().get(), 3);
    Ok(())
}

#[test]
fn raster_template_records_color_depth_and_resolve_targets() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let color = b.import_texture(backbuffer_import());
    let resolve = b.import_texture(backbuffer_import());
    let depth = b.import_texture(depth_import());
    let mut pass = TestRasterPass::new("templated", color);
    pass.resolve = Some(resolve.into());
    pass.depth = Some(depth.into());
    b.add_raster_pass(Box::new(pass));
    let g = b.build()?;
    assert!(g.pass_info[0].raster_template.is_some());
    let template = g.pass_info[0].raster_template.as_ref().unwrap();
    assert_eq!(template.color_attachments.len(), 1);
    assert_eq!(
        template.color_attachments[0].target,
        TextureAttachmentTarget::Resource(TextureResourceHandle::Imported(color))
    );
    assert_eq!(
        template.color_attachments[0].resolve_to,
        Some(TextureAttachmentResolve::Always(
            TextureResourceHandle::Imported(resolve)
        ))
    );
    assert_eq!(
        template.depth_stencil_attachment.as_ref().map(|d| d.target),
        Some(TextureAttachmentTarget::Resource(
            TextureResourceHandle::Imported(depth)
        ))
    );
    Ok(())
}

#[test]
fn raster_template_records_frame_sampled_targets() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let color = b.import_texture(backbuffer_import());
    let resolve = b.import_texture(backbuffer_import());
    let depth = b.import_texture(depth_import());
    let msaa_color = b.create_texture(frame_sampled_tex_desc("msaa-color"));
    let msaa_depth = b.create_texture(TransientTextureDesc::frame_sampled_texture_2d(
        "msaa-depth",
        wgpu::TextureFormat::Depth32Float,
        TransientExtent::Custom {
            width: 64,
            height: 64,
        },
        wgpu::TextureUsages::empty(),
    ));
    let mut pass = TestRasterPass::new("frame-sampled", color);
    pass.frame_sampled_color = Some((color.into(), msaa_color.into(), Some(resolve.into())));
    pass.frame_sampled_depth = Some((depth.into(), msaa_depth.into()));
    b.add_raster_pass(Box::new(pass));
    let g = b.build()?;
    assert!(g.pass_info[0].raster_template.is_some());
    let template = g.pass_info[0].raster_template.as_ref().unwrap();
    assert_eq!(
        template.color_attachments[0].target,
        TextureAttachmentTarget::FrameSampled {
            single_sample: TextureResourceHandle::Imported(color),
            multisampled: TextureResourceHandle::Transient(msaa_color),
        }
    );
    assert_eq!(
        template.color_attachments[0].resolve_to,
        Some(TextureAttachmentResolve::FrameMultisampled(
            TextureResourceHandle::Imported(resolve)
        ))
    );
    assert_eq!(
        template.depth_stencil_attachment.as_ref().map(|d| d.target),
        Some(TextureAttachmentTarget::FrameSampled {
            single_sample: TextureResourceHandle::Imported(depth),
            multisampled: TextureResourceHandle::Transient(msaa_depth),
        })
    );
    Ok(())
}

#[test]
fn buffer_aliasing_uses_size_and_usage_key() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let a = b.create_buffer(TransientBufferDesc {
        label: "a",
        size_policy: BufferSizePolicy::Fixed(64),
        base_usage: wgpu::BufferUsages::empty(),
        alias: true,
    });
    let c = b.create_buffer(TransientBufferDesc {
        label: "c",
        size_policy: BufferSizePolicy::Fixed(64),
        base_usage: wgpu::BufferUsages::empty(),
        alias: true,
    });
    let out = b.import_buffer(ImportedBufferDecl {
        label: "history",
        source: BufferImportSource::PingPong(HistorySlotId::HI_Z),
        initial_access: BufferAccess::CopyDst,
        final_access: BufferAccess::CopyDst,
    });
    let mut p0 = TestComputePass::new("write-a");
    p0.buffer_writes.push(a);
    let mut p1 = TestComputePass::new("export-a");
    p1.buffer_reads.push(a);
    p1.imported_buffer_writes.push(out);
    let mut p2 = TestComputePass::new("write-c");
    p2.buffer_writes.push(c);
    let mut p3 = TestComputePass::new("export-c");
    p3.buffer_reads.push(c);
    p3.imported_buffer_writes.push(out);
    b.add_compute_pass(Box::new(p0));
    let p1_id = b.add_compute_pass(Box::new(p1));
    let p2_id = b.add_compute_pass(Box::new(p2));
    b.add_compute_pass(Box::new(p3));
    b.add_edge(p1_id, p2_id);
    let g = b.build()?;
    assert_eq!(
        g.transient_buffers[a.index()].physical_slot,
        g.transient_buffers[c.index()].physical_slot
    );
    Ok(())
}

#[test]
fn texture_aliasing_keys_on_sample_count_policy() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let frame_sampled = b.create_texture(frame_sampled_tex_desc("frame-sampled"));
    let fixed = b.create_texture(tex_desc("fixed"));
    let bb = b.import_texture(backbuffer_import());
    let mut p0 = TestComputePass::new("write-frame-sampled");
    p0.texture_writes.push(frame_sampled);
    let mut p1 = TestRasterPass::new("export-frame-sampled", bb);
    p1.texture_reads.push(frame_sampled);
    let mut p2 = TestComputePass::new("write-fixed");
    p2.texture_writes.push(fixed);
    let mut p3 = TestRasterPass::new("export-fixed", bb);
    p3.texture_reads.push(fixed);
    b.add_compute_pass(Box::new(p0));
    let p1_id = b.add_raster_pass(Box::new(p1));
    b.add_compute_pass(Box::new(p2));
    b.add_raster_pass(Box::new(p3));
    b.add_edge(p1_id, PassId(2));
    let g = b.build()?;
    assert_ne!(
        g.transient_textures[frame_sampled.index()].physical_slot,
        g.transient_textures[fixed.index()].physical_slot
    );
    Ok(())
}

#[test]
fn frame_sample_count_policy_resolves_current_frame_value() {
    use crate::render_graph::resources::TransientSampleCount;
    assert_eq!(TransientSampleCount::Fixed(0).resolve(4), 1);
    assert_eq!(TransientSampleCount::Fixed(2).resolve(4), 2);
    assert_eq!(TransientSampleCount::Frame.resolve(0), 1);
    assert_eq!(TransientSampleCount::Frame.resolve(4), 4);
}

#[test]
fn frame_texture_format_and_layer_policies_resolve_current_frame_values() {
    assert_eq!(
        TransientTextureFormat::Fixed(wgpu::TextureFormat::Rgba8Unorm).resolve(
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Depth24PlusStencil8,
            wgpu::TextureFormat::Rgba16Float,
        ),
        wgpu::TextureFormat::Rgba8Unorm
    );
    assert_eq!(
        TransientTextureFormat::FrameColor.resolve(
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Depth24PlusStencil8,
            wgpu::TextureFormat::Rgba16Float,
        ),
        wgpu::TextureFormat::Bgra8UnormSrgb
    );
    assert_eq!(
        TransientTextureFormat::SceneColorHdr.resolve(
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Depth24PlusStencil8,
            wgpu::TextureFormat::Rg11b10Ufloat,
        ),
        wgpu::TextureFormat::Rg11b10Ufloat
    );
    use crate::render_graph::resources::TransientArrayLayers;
    assert_eq!(TransientArrayLayers::Fixed(0).resolve(true), 1);
    assert_eq!(TransientArrayLayers::Fixed(3).resolve(false), 3);
    assert_eq!(TransientArrayLayers::Frame.resolve(false), 1);
    assert_eq!(TransientArrayLayers::Frame.resolve(true), 2);
}

/// Verifies that the FrameSchedule is the single source of truth and contains the expected steps
/// in the expected phase order (frame-global before per-view).
#[test]
fn schedule_orders_frame_global_before_per_view() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let bb = b.import_texture(backbuffer_import());
    b.add_raster_pass(Box::new(TestRasterPass::new("per-view-a", bb)));
    b.add_raster_pass(Box::new(TestRasterPass::new("per-view-b", bb)));
    b.add_compute_pass(Box::new(
        TestComputePass::new("frame").frame_global().cull_exempt(),
    ));
    let g = b.build()?;
    // FrameSchedule is the single source of truth.

    assert_eq!(
        g.schedule.frame_global_steps().count(),
        1,
        "expected one frame-global pass"
    );
    assert_eq!(
        g.schedule.per_view_steps().count(),
        2,
        "expected two per-view passes"
    );
    // Validate structural invariants.
    g.schedule.validate().expect("schedule validates");
    Ok(())
}
