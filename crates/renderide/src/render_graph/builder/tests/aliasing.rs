//! Resource lifetime, aliasing, and culling tests.

use super::common::*;

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
