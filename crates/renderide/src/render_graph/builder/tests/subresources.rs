//! Texture subresource and merge-hint tests.

use super::common::*;

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
