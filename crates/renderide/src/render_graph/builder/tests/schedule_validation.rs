//! Builder scheduling and validation tests.

use super::common::*;

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
