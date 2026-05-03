//! Builder unit tests: dependency synthesis, culling, aliasing, and group ordering.

use super::GraphBuilder;
use crate::render_graph::context::{ComputePassCtx, RasterPassCtx};
use crate::render_graph::error::{GraphBuildError, RenderPassError, SetupError};
use crate::render_graph::ids::PassId;
use crate::render_graph::pass::{
    ComputePass, GroupScope, PassBuilder, PassMergeHint, PassPhase, RasterPass,
};
use crate::render_graph::resources::{
    BufferAccess, BufferHandle, BufferImportSource, BufferSizePolicy, FrameTargetRole,
    HistorySlotId, ImportedBufferDecl, ImportedBufferHandle, ImportedTextureDecl,
    ImportedTextureHandle, StorageAccess, SubresourceHandle, TextureAccess,
    TextureAttachmentResolve, TextureAttachmentTarget, TextureHandle, TextureResourceHandle,
    TransientBufferDesc, TransientExtent, TransientSubresourceDesc, TransientTextureDesc,
    TransientTextureFormat,
};

// -----------------------------------------------------------------------------
// Test pass helpers
// -----------------------------------------------------------------------------

/// Minimal compute test pass.
struct TestComputePass {
    name: &'static str,
    phase: PassPhase,
    texture_reads: Vec<TextureHandle>,
    texture_writes: Vec<TextureHandle>,
    subresource_reads: Vec<SubresourceHandle>,
    subresource_writes: Vec<SubresourceHandle>,
    buffer_reads: Vec<BufferHandle>,
    buffer_writes: Vec<BufferHandle>,
    imported_texture_writes: Vec<ImportedTextureHandle>,
    imported_buffer_writes: Vec<ImportedBufferHandle>,
    cull_exempt: bool,
}

impl TestComputePass {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            phase: PassPhase::PerView,
            texture_reads: Vec::new(),
            texture_writes: Vec::new(),
            subresource_reads: Vec::new(),
            subresource_writes: Vec::new(),
            buffer_reads: Vec::new(),
            buffer_writes: Vec::new(),
            imported_texture_writes: Vec::new(),
            imported_buffer_writes: Vec::new(),
            cull_exempt: false,
        }
    }

    fn frame_global(mut self) -> Self {
        self.phase = PassPhase::FrameGlobal;
        self
    }

    fn cull_exempt(mut self) -> Self {
        self.cull_exempt = true;
        self
    }
}

impl ComputePass for TestComputePass {
    fn name(&self) -> &str {
        self.name
    }

    fn phase(&self) -> PassPhase {
        self.phase
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.compute();
        if self.cull_exempt {
            b.cull_exempt();
        }
        for &h in &self.texture_reads {
            b.read_texture(
                h,
                TextureAccess::Sampled {
                    stages: wgpu::ShaderStages::COMPUTE,
                },
            );
        }
        for &h in &self.texture_writes {
            b.write_texture(h, TextureAccess::CopyDst);
        }
        for &h in &self.subresource_reads {
            b.read_texture_subresource(
                h,
                TextureAccess::Sampled {
                    stages: wgpu::ShaderStages::COMPUTE,
                },
            );
        }
        for &h in &self.subresource_writes {
            b.write_texture_subresource(h, TextureAccess::CopyDst);
        }
        for &h in &self.buffer_reads {
            b.read_buffer(
                h,
                BufferAccess::Storage {
                    stages: wgpu::ShaderStages::COMPUTE,
                    access: StorageAccess::ReadOnly,
                },
            );
        }
        for &h in &self.buffer_writes {
            b.write_buffer(h, BufferAccess::CopyDst);
        }
        for &h in &self.imported_texture_writes {
            b.import_texture(h, TextureAccess::Present);
        }
        for &h in &self.imported_buffer_writes {
            b.import_buffer(h, BufferAccess::CopyDst);
        }
        Ok(())
    }

    fn record(&self, _ctx: &mut ComputePassCtx<'_, '_, '_>) -> Result<(), RenderPassError> {
        Ok(())
    }
}

/// Minimal raster test pass.
struct TestRasterPass {
    name: &'static str,
    color: TextureResourceHandle,
    texture_reads: Vec<TextureHandle>,
    imported_texture_writes: Vec<ImportedTextureHandle>,
    multiview_mask: Option<std::num::NonZeroU32>,
    depth: Option<TextureResourceHandle>,
    resolve: Option<TextureResourceHandle>,
    frame_sampled_color: Option<(
        TextureResourceHandle,
        TextureResourceHandle,
        Option<TextureResourceHandle>,
    )>,
    frame_sampled_depth: Option<(TextureResourceHandle, TextureResourceHandle)>,
}

impl TestRasterPass {
    fn new(name: &'static str, color: impl Into<TextureResourceHandle>) -> Self {
        Self {
            name,
            color: color.into(),
            texture_reads: Vec::new(),
            imported_texture_writes: Vec::new(),
            multiview_mask: None,
            depth: None,
            resolve: None,
            frame_sampled_color: None,
            frame_sampled_depth: None,
        }
    }
}

impl RasterPass for TestRasterPass {
    fn name(&self) -> &str {
        self.name
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        {
            let mut r = b.raster();
            if let Some((single, msaa, res)) = self.frame_sampled_color {
                r.frame_sampled_color(
                    single,
                    msaa,
                    wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                        store: wgpu::StoreOp::Store,
                    },
                    res,
                );
            } else {
                r.color(
                    self.color,
                    wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    self.resolve,
                );
            }
            if let Some((single, msaa)) = self.frame_sampled_depth {
                r.frame_sampled_depth(
                    single,
                    msaa,
                    wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.5),
                        store: wgpu::StoreOp::Store,
                    },
                    None,
                );
            } else if let Some(d) = self.depth {
                r.depth(
                    d,
                    wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.5),
                        store: wgpu::StoreOp::Store,
                    },
                    None,
                );
            }
            if let Some(mask) = self.multiview_mask {
                r.multiview(mask);
            }
        }
        for &h in &self.texture_reads {
            b.read_texture(
                h,
                TextureAccess::Sampled {
                    stages: wgpu::ShaderStages::FRAGMENT,
                },
            );
        }
        for &h in &self.imported_texture_writes {
            b.import_texture(h, TextureAccess::Present);
        }
        Ok(())
    }

    fn record(
        &self,
        _ctx: &mut RasterPassCtx<'_, '_>,
        _rpass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), RenderPassError> {
        Ok(())
    }
}

// -----------------------------------------------------------------------------
// Helper descriptors
// -----------------------------------------------------------------------------

fn tex_desc(label: &'static str) -> TransientTextureDesc {
    TransientTextureDesc::texture_2d(
        label,
        wgpu::TextureFormat::Rgba8Unorm,
        TransientExtent::Custom {
            width: 64,
            height: 64,
        },
        1,
        wgpu::TextureUsages::empty(),
    )
}

fn frame_sampled_tex_desc(label: &'static str) -> TransientTextureDesc {
    TransientTextureDesc::frame_sampled_texture_2d(
        label,
        wgpu::TextureFormat::Rgba8Unorm,
        TransientExtent::Custom {
            width: 64,
            height: 64,
        },
        wgpu::TextureUsages::empty(),
    )
}

fn mip_chain_tex_desc(label: &'static str, mip_levels: u32) -> TransientTextureDesc {
    TransientTextureDesc {
        label,
        format: TransientTextureFormat::Fixed(wgpu::TextureFormat::R32Float),
        extent: TransientExtent::Custom {
            width: 256,
            height: 256,
        },
        mip_levels,
        sample_count: crate::render_graph::resources::TransientSampleCount::Fixed(1),
        dimension: wgpu::TextureDimension::D2,
        array_layers: crate::render_graph::resources::TransientArrayLayers::Fixed(1),
        base_usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        alias: false,
    }
}

fn backbuffer_import() -> ImportedTextureDecl {
    ImportedTextureDecl {
        label: "backbuffer",
        source: crate::render_graph::resources::ImportSource::Frame(
            FrameTargetRole::ColorAttachment,
        ),
        initial_access: TextureAccess::ColorAttachment {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
            resolve_to: None,
        },
        final_access: TextureAccess::Present,
    }
}

fn depth_import() -> ImportedTextureDecl {
    ImportedTextureDecl {
        label: "depth",
        source: crate::render_graph::resources::ImportSource::Frame(
            FrameTargetRole::DepthAttachment,
        ),
        initial_access: TextureAccess::DepthAttachment {
            depth: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
            stencil: None,
        },
        final_access: TextureAccess::Sampled {
            stages: wgpu::ShaderStages::COMPUTE,
        },
    }
}

fn buffer_import_readback() -> ImportedBufferDecl {
    ImportedBufferDecl {
        label: "readback",
        source: BufferImportSource::External,
        initial_access: BufferAccess::CopyDst,
        final_access: BufferAccess::CopyDst,
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn linear_chain_schedules_in_order() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let tex = b.create_texture(tex_desc("color"));
    let bb = b.import_texture(backbuffer_import());
    let mut a = TestComputePass::new("a");
    a.texture_writes.push(tex);
    let mut c = TestRasterPass::new("c", bb);
    c.texture_reads.push(tex);
    b.add_compute_pass(Box::new(a));
    b.add_raster_pass(Box::new(c));
    let g = b.build()?;
    assert_eq!(g.pass_count(), 2);
    assert_eq!(g.pass_info[0].name, "a");
    assert_eq!(g.pass_info[1].name, "c");
    Ok(())
}

#[test]
fn parallel_passes_single_level() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let out_a = b.import_texture(backbuffer_import());
    let out_b = b.import_buffer(buffer_import_readback());
    b.add_raster_pass(Box::new(TestRasterPass::new("a", out_a)));
    let mut b_pass = TestComputePass::new("b");
    b_pass.imported_buffer_writes.push(out_b);
    b.add_compute_pass(Box::new(b_pass));
    let g = b.build()?;
    assert_eq!(g.compile_stats.topo_levels, 1);
    assert_eq!(g.pass_count(), 2);
    Ok(())
}

#[test]
fn cycle_detected_through_handle_rw_conflict() {
    let mut b = GraphBuilder::new();
    let tex = b.create_texture(tex_desc("color"));
    let bb = b.import_texture(backbuffer_import());
    let mut a = TestRasterPass::new("a", bb);
    a.texture_reads.push(tex);
    let mut c = TestComputePass::new("c");
    c.texture_writes.push(tex);
    let a_id = b.add_raster_pass(Box::new(a));
    let c_id = b.add_compute_pass(Box::new(c));
    b.add_edge(a_id, c_id);
    assert!(matches!(
        b.build(),
        Err(GraphBuildError::MissingDependency { .. })
    ));
}

#[test]
fn read_without_writer_errors_with_handle_and_access() {
    let mut b = GraphBuilder::new();
    let tex = b.create_texture(tex_desc("orphan"));
    let mut p = TestComputePass::new("reader");
    p.texture_reads.push(tex);
    b.add_compute_pass(Box::new(p));
    assert!(matches!(
        b.build(),
        Err(GraphBuildError::MissingDependency { .. })
    ));
}

#[test]
fn aliased_handles_share_slot_when_lifetimes_disjoint() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let a = b.create_texture(tex_desc("a"));
    let c = b.create_texture(tex_desc("c"));
    let bb = b.import_texture(backbuffer_import());
    let mut p0 = TestComputePass::new("write-a");
    p0.texture_writes.push(a);
    let mut p1 = TestRasterPass::new("export-a", bb);
    p1.texture_reads.push(a);
    let mut p2 = TestComputePass::new("write-c");
    p2.texture_writes.push(c);
    let mut p3 = TestRasterPass::new("export-c", bb);
    p3.texture_reads.push(c);
    b.add_compute_pass(Box::new(p0));
    let p1_id = b.add_raster_pass(Box::new(p1));
    let p2_id = b.add_compute_pass(Box::new(p2));
    b.add_raster_pass(Box::new(p3));
    b.add_edge(p1_id, p2_id);
    let g = b.build()?;
    assert_eq!(
        g.transient_textures[a.index()].physical_slot,
        g.transient_textures[c.index()].physical_slot
    );
    assert_eq!(g.compile_stats.transient_texture_slots, 1);
    Ok(())
}

#[test]
fn aliased_handles_do_not_share_when_desc_alias_false() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let mut d0 = tex_desc("a");
    let mut d1 = tex_desc("c");
    d0.alias = false;
    d1.alias = false;
    let a = b.create_texture(d0);
    let c = b.create_texture(d1);
    let bb = b.import_texture(backbuffer_import());
    let mut p0 = TestComputePass::new("write-a");
    p0.texture_writes.push(a);
    let mut p1 = TestRasterPass::new("export-a", bb);
    p1.texture_reads.push(a);
    let mut p2 = TestComputePass::new("write-c");
    p2.texture_writes.push(c);
    let mut p3 = TestRasterPass::new("export-c", bb);
    p3.texture_reads.push(c);
    b.add_compute_pass(Box::new(p0));
    let p1_id = b.add_raster_pass(Box::new(p1));
    let p2_id = b.add_compute_pass(Box::new(p2));
    b.add_raster_pass(Box::new(p3));
    b.add_edge(p1_id, p2_id);
    let g = b.build()?;
    assert_ne!(
        g.transient_textures[a.index()].physical_slot,
        g.transient_textures[c.index()].physical_slot
    );
    Ok(())
}

#[test]
fn usage_union_promotes_transient_to_storage_when_sampled_and_stored() -> Result<(), GraphBuildError>
{
    let mut b = GraphBuilder::new();
    let tex = b.create_texture(tex_desc("scratch"));
    let bb = b.import_texture(backbuffer_import());
    let mut p0 = TestComputePass::new("write");
    p0.texture_writes.push(tex);
    let mut p1 = TestRasterPass::new("export", bb);
    p1.texture_reads.push(tex);
    b.add_compute_pass(Box::new(p0));
    b.add_raster_pass(Box::new(p1));
    let g = b.build()?;
    let usage = g.transient_textures[tex.index()].usage;
    assert!(usage.contains(wgpu::TextureUsages::COPY_DST));
    assert!(usage.contains(wgpu::TextureUsages::TEXTURE_BINDING));
    Ok(())
}

#[test]
fn dead_pass_culled_when_output_unused() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let tex = b.create_texture(tex_desc("dead"));
    let mut p = TestComputePass::new("dead");
    p.texture_writes.push(tex);
    b.add_compute_pass(Box::new(p));
    let g = b.build()?;
    assert_eq!(g.pass_count(), 0);
    assert_eq!(g.compile_stats.culled_count, 1);
    Ok(())
}

#[test]
fn dead_pass_retained_when_marked_exempt() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    b.add_compute_pass(Box::new(TestComputePass::new("side-effect").cull_exempt()));
    let g = b.build()?;
    assert_eq!(g.pass_count(), 1);
    Ok(())
}

#[test]
fn raster_pass_without_attachments_rejected() {
    /// A raster pass that calls `b.raster()` but doesn't add any attachment.
    struct RasterNoAttachment;
    impl RasterPass for RasterNoAttachment {
        fn name(&self) -> &str {
            "bad"
        }
        fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
            b.raster(); // no color or depth attachment declared
            Ok(())
        }
        fn record(
            &self,
            _ctx: &mut RasterPassCtx<'_, '_>,
            _rpass: &mut wgpu::RenderPass<'_>,
        ) -> Result<(), RenderPassError> {
            Ok(())
        }
    }
    let mut b = GraphBuilder::new();
    b.add_raster_pass(Box::new(RasterNoAttachment));
    assert!(matches!(
        b.build(),
        Err(GraphBuildError::Setup {
            source: SetupError::RasterWithoutAttachments,
            ..
        })
    ));
}

#[test]
fn compute_pass_with_attachment_rejected() {
    /// A compute pass that illegally declares a color attachment.
    struct BadComputePass(ImportedTextureHandle);
    impl ComputePass for BadComputePass {
        fn name(&self) -> &str {
            "bad"
        }
        fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
            b.compute();
            b.import_texture(
                self.0,
                TextureAccess::ColorAttachment {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                    resolve_to: None,
                },
            );
            Ok(())
        }
        fn record(&self, _ctx: &mut ComputePassCtx<'_, '_, '_>) -> Result<(), RenderPassError> {
            Ok(())
        }
    }
    let mut b = GraphBuilder::new();
    let bb = b.import_texture(backbuffer_import());
    b.add_compute_pass(Box::new(BadComputePass(bb)));
    assert!(matches!(
        b.build(),
        Err(GraphBuildError::Setup {
            source: SetupError::NonRasterPassHasAttachment,
            ..
        })
    ));
}

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

/// Pass that declares a non-default merge hint via [`PassBuilder::merge_hint`]. Used to verify
/// the hint roundtrips into [`crate::render_graph::compiled::CompiledPassInfo`].
struct MergeHintPass {
    name: &'static str,
    hint: PassMergeHint,
    out: ImportedTextureHandle,
}

impl ComputePass for MergeHintPass {
    fn name(&self) -> &str {
        self.name
    }

    fn phase(&self) -> PassPhase {
        PassPhase::PerView
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.compute();
        b.merge_hint(self.hint);
        b.import_texture(self.out, TextureAccess::Present);
        Ok(())
    }

    fn record(&self, _ctx: &mut ComputePassCtx<'_, '_, '_>) -> Result<(), RenderPassError> {
        Ok(())
    }
}

#[test]
fn create_subresource_assigns_sequential_handles_and_preserves_desc() {
    let mut b = GraphBuilder::new();
    let parent = b.create_texture(mip_chain_tex_desc("mip-chain-parent", 4));
    let h0 = b.create_subresource(TransientSubresourceDesc::single_mip(parent, "mip0", 0));
    let h1 = b.create_subresource(TransientSubresourceDesc::single_mip(parent, "mip1", 1));
    let h2 = b.create_subresource(TransientSubresourceDesc::single_mip(parent, "mip2", 2));
    assert_eq!(h0, SubresourceHandle(0));
    assert_eq!(h1, SubresourceHandle(1));
    assert_eq!(h2, SubresourceHandle(2));
    // Cull-exempt compute pass so the builder keeps the parent alive even with no import edge.
    b.add_compute_pass(Box::new(
        TestComputePass::new("keep-parent")
            .frame_global()
            .cull_exempt(),
    ));
    let g = b.build().expect("graph builds");
    assert_eq!(g.subresources.len(), 3);
    assert_eq!(g.subresources[0].base_mip_level, 0);
    assert_eq!(g.subresources[1].base_mip_level, 1);
    assert_eq!(g.subresources[2].base_mip_level, 2);
    assert!(g.subresources.iter().all(|s| s.parent == parent));
}

#[test]
fn overlapping_subresource_write_orders_matching_read() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let parent = b.create_texture(mip_chain_tex_desc("mip-chain", 2));
    let mip0 = b.create_subresource(TransientSubresourceDesc::single_mip(parent, "mip0", 0));

    let mut write_mip0 = TestComputePass::new("write-mip0");
    write_mip0.subresource_writes.push(mip0);
    let mut read_mip0 = TestComputePass::new("read-mip0").cull_exempt();
    read_mip0.subresource_reads.push(mip0);

    b.add_compute_pass(Box::new(write_mip0));
    b.add_compute_pass(Box::new(read_mip0));

    let g = b.build()?;
    assert_eq!(g.pass_info[0].name, "write-mip0");
    assert_eq!(g.pass_info[1].name, "read-mip0");
    assert_eq!(g.compile_stats.topo_levels, 2);
    Ok(())
}

#[test]
fn non_overlapping_subresources_do_not_create_cross_edges() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let parent = b.create_texture(mip_chain_tex_desc("mip-chain", 2));
    let mip0 = b.create_subresource(TransientSubresourceDesc::single_mip(parent, "mip0", 0));
    let mip1 = b.create_subresource(TransientSubresourceDesc::single_mip(parent, "mip1", 1));

    let mut write_mip0 = TestComputePass::new("write-mip0");
    write_mip0.subresource_writes.push(mip0);
    let mut write_mip1 = TestComputePass::new("write-mip1");
    write_mip1.subresource_writes.push(mip1);
    let mut read_mip0 = TestComputePass::new("read-mip0").cull_exempt();
    read_mip0.subresource_reads.push(mip0);
    let mut read_mip1 = TestComputePass::new("read-mip1").cull_exempt();
    read_mip1.subresource_reads.push(mip1);

    b.add_compute_pass(Box::new(write_mip0));
    b.add_compute_pass(Box::new(write_mip1));
    b.add_compute_pass(Box::new(read_mip0));
    b.add_compute_pass(Box::new(read_mip1));

    let g = b.build()?;
    let names: Vec<&str> = g.pass_info.iter().map(|info| info.name.as_str()).collect();
    assert_eq!(
        names,
        vec!["write-mip0", "write-mip1", "read-mip0", "read-mip1"]
    );
    assert_eq!(
        g.compile_stats.topo_levels, 2,
        "independent mip chains should share writer and reader waves"
    );
    Ok(())
}

#[test]
fn subresource_reads_without_overlapping_writer_error() {
    let mut b = GraphBuilder::new();
    let parent = b.create_texture(mip_chain_tex_desc("mip-chain", 2));
    let mip0 = b.create_subresource(TransientSubresourceDesc::single_mip(parent, "mip0", 0));
    let mip1 = b.create_subresource(TransientSubresourceDesc::single_mip(parent, "mip1", 1));

    let mut write_mip0 = TestComputePass::new("write-mip0");
    write_mip0.subresource_writes.push(mip0);
    let mut read_mip1 = TestComputePass::new("read-mip1").cull_exempt();
    read_mip1.subresource_reads.push(mip1);

    b.add_compute_pass(Box::new(write_mip0));
    b.add_compute_pass(Box::new(read_mip1));

    assert!(matches!(
        b.build(),
        Err(GraphBuildError::MissingDependency { .. })
    ));
}

#[test]
fn subresource_access_extends_parent_texture_lifetime_and_usage() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let parent = b.create_texture(mip_chain_tex_desc("mip-chain", 2));
    let mip0 = b.create_subresource(TransientSubresourceDesc::single_mip(parent, "mip0", 0));

    let mut write_mip0 = TestComputePass::new("write-mip0");
    write_mip0.subresource_writes.push(mip0);
    let mut read_mip0 = TestComputePass::new("read-mip0").cull_exempt();
    read_mip0.subresource_reads.push(mip0);

    b.add_compute_pass(Box::new(write_mip0));
    b.add_compute_pass(Box::new(read_mip0));

    let g = b.build()?;
    let compiled = &g.transient_textures[parent.index()];
    assert!(compiled.lifetime.is_some());
    assert_ne!(compiled.physical_slot, usize::MAX);
    assert!(compiled.usage.contains(wgpu::TextureUsages::COPY_DST));
    assert!(
        compiled
            .usage
            .contains(wgpu::TextureUsages::TEXTURE_BINDING)
    );
    Ok(())
}

#[test]
fn invalid_subresource_range_is_rejected() {
    let mut b = GraphBuilder::new();
    let parent = b.create_texture(mip_chain_tex_desc("mip-chain", 1));
    b.create_subresource(TransientSubresourceDesc::single_mip(parent, "mip4", 4));
    b.add_compute_pass(Box::new(TestComputePass::new("keep").cull_exempt()));

    assert!(matches!(
        b.build(),
        Err(GraphBuildError::InvalidSubresource { .. })
    ));
}

#[test]
fn merge_hint_roundtrips_from_pass_builder_to_compiled_pass_info() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let bb = b.import_texture(backbuffer_import());
    let hint = PassMergeHint {
        attachment_reuse: true,
        tile_memory_preferred: true,
    };
    b.add_compute_pass(Box::new(MergeHintPass {
        name: "merge-hint-pass",
        hint,
        out: bb,
    }));
    let g = b.build()?;
    // Exactly one retained pass; its compiled info should carry our hint.
    let info = g
        .pass_info
        .iter()
        .find(|info| info.name == "merge-hint-pass")
        .expect("merge-hint-pass is retained");
    assert_eq!(info.merge_hint, hint);
    // Passes that do not call `merge_hint` default to the zero hint, i.e. no-op on every backend.
    assert!(!PassMergeHint::default().attachment_reuse);
    assert!(!PassMergeHint::default().tile_memory_preferred);
    Ok(())
}
