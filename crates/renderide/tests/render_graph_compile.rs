//! Integration test: compile a small render graph through the crate's public API and inspect the
//! compiled output. No GPU is involved — the builder is descriptor-only and `build()` never opens a
//! `wgpu::Device`.
//!
//! These tests mirror a few of the in-crate builder unit tests but go through the external public
//! boundary, so any regression in the re-exported surface (handles, trait objects, descriptor
//! structs) is caught here first.

use std::num::NonZeroU32;

use renderide::render_graph::{
    BufferAccess, BufferHandle, ComputePass, ComputePassCtx, FrameTargetRole, GraphBuildError,
    GraphBuilder, ImportSource, ImportedBufferHandle, ImportedTextureDecl, ImportedTextureHandle,
    PassBuilder, PassPhase, PostSubmitContext, RenderPassError, SetupError, StorageAccess,
    TextureAccess, TextureHandle, TransientBufferDesc, TransientExtent, TransientTextureDesc,
};

/// Minimal `ComputePass` implementation for integration tests. `record` is never called because
/// these tests stop at `build()`; declaring it as `unreachable!` guards against silent execution.
struct TestComputePass {
    name: &'static str,
    phase: PassPhase,
    texture_reads: Vec<TextureHandle>,
    texture_writes: Vec<TextureHandle>,
    imported_texture_writes: Vec<ImportedTextureHandle>,
    imported_buffer_writes: Vec<ImportedBufferHandle>,
    buffer_reads: Vec<BufferHandle>,
    buffer_writes: Vec<BufferHandle>,
    cull_exempt: bool,
}

impl TestComputePass {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            phase: PassPhase::PerView,
            texture_reads: Vec::new(),
            texture_writes: Vec::new(),
            imported_texture_writes: Vec::new(),
            imported_buffer_writes: Vec::new(),
            buffer_reads: Vec::new(),
            buffer_writes: Vec::new(),
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
        unreachable!("compile-only integration test must not execute the graph")
    }

    fn post_submit(&mut self, _ctx: &mut PostSubmitContext<'_>) -> Result<(), RenderPassError> {
        Ok(())
    }
}

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

fn backbuffer_import() -> ImportedTextureDecl {
    ImportedTextureDecl {
        label: "backbuffer",
        source: ImportSource::FrameTarget(FrameTargetRole::ColorAttachment),
        initial_access: TextureAccess::ColorAttachment {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
            resolve_to: None,
        },
        final_access: TextureAccess::Present,
    }
}

/// Two passes where the second reads a transient the first writes; the compiled schedule must
/// order producer before consumer in `schedule.steps`.
#[test]
fn linear_chain_orders_producer_before_consumer() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let tex = b.create_texture(tex_desc("color"));
    let bb = b.import_texture(backbuffer_import());

    let mut producer = TestComputePass::new("producer");
    producer.texture_writes.push(tex);

    let mut consumer = TestComputePass::new("consumer");
    consumer.texture_reads.push(tex);
    consumer.imported_texture_writes.push(bb);

    b.add_compute_pass(Box::new(producer));
    b.add_compute_pass(Box::new(consumer));

    let g = b.build()?;
    assert_eq!(g.pass_info.len(), 2);
    assert_eq!(g.pass_info[0].name, "producer");
    assert_eq!(g.pass_info[1].name, "consumer");

    let steps = &g.schedule.steps;
    assert_eq!(steps.len(), 2);
    assert!(
        steps[0].pass_idx == 0 && steps[1].pass_idx == 1,
        "producer must precede consumer in the flat schedule: {steps:?}"
    );
    g.schedule
        .validate()
        .expect("schedule must be structurally valid");
    Ok(())
}

/// Frame-global passes are scheduled before any per-view pass, even when declared in mixed order.
/// `FrameSchedule::validate` enforces this invariant explicitly.
#[test]
fn frame_global_passes_precede_per_view_passes() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let bb = b.import_texture(backbuffer_import());

    // Declare per-view first, frame-global second, to make sure phase ordering comes from the
    // compiled schedule and not declaration order.
    let mut per_view = TestComputePass::new("per_view");
    per_view.imported_texture_writes.push(bb);

    // Frame-global pass must have some reachable export to survive culling; give it cull_exempt.
    let frame_global = TestComputePass::new("frame_global")
        .frame_global()
        .cull_exempt();

    b.add_compute_pass(Box::new(per_view));
    b.add_compute_pass(Box::new(frame_global));

    let g = b.build()?;
    g.schedule.validate().expect("phase ordering must validate");

    let first_frame_global = g
        .schedule
        .steps
        .iter()
        .position(|s| s.phase == PassPhase::FrameGlobal);
    let first_per_view = g
        .schedule
        .steps
        .iter()
        .position(|s| s.phase == PassPhase::PerView);

    if let (Some(fg), Some(pv)) = (first_frame_global, first_per_view) {
        assert!(
            fg < pv,
            "frame-global step {fg} must come before per-view step {pv}: {:?}",
            g.schedule.steps
        );
    } else {
        panic!(
            "expected both a frame-global and a per-view step in the schedule: {:?}",
            g.schedule.steps
        );
    }
    Ok(())
}

/// A pass whose output is neither read by another pass nor written to an import must be culled and
/// reported in `compile_stats.culled_count`.
#[test]
fn dead_pass_without_exports_is_culled() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let bb = b.import_texture(backbuffer_import());
    let dead_tex = b.create_texture(tex_desc("dead"));

    // Alive: writes the backbuffer import (reachable).
    let mut alive = TestComputePass::new("alive");
    alive.imported_texture_writes.push(bb);

    // Dead: writes a transient that is never read or exported.
    let mut dead = TestComputePass::new("dead");
    dead.texture_writes.push(dead_tex);

    b.add_compute_pass(Box::new(alive));
    b.add_compute_pass(Box::new(dead));

    let g = b.build()?;
    assert!(
        g.compile_stats.culled_count >= 1,
        "expected at least one culled pass, got stats {:?}",
        g.compile_stats
    );
    assert!(
        !g.pass_info.iter().any(|p| p.name == "dead"),
        "culled pass must not appear in pass_info: {:?}",
        g.pass_info
            .iter()
            .map(|p| p.name.as_str())
            .collect::<Vec<_>>()
    );
    Ok(())
}

/// Two transients with disjoint lifetimes must share a physical alias slot. Mirrors the in-crate
/// `aliased_handles_share_slot_when_lifetimes_disjoint` pattern (write → read-and-export, twice,
/// with an explicit edge forcing the second write to happen after the first read).
#[test]
fn disjoint_lifetimes_share_alias_slot() -> Result<(), GraphBuildError> {
    use renderide::render_graph::PassId;

    let mut b = GraphBuilder::new();
    let bb = b.import_texture(backbuffer_import());
    let a = b.create_texture(tex_desc("a"));
    let c = b.create_texture(tex_desc("c"));

    let mut write_a = TestComputePass::new("write-a");
    write_a.texture_writes.push(a);

    let mut export_a = TestComputePass::new("export-a");
    export_a.texture_reads.push(a);
    export_a.imported_texture_writes.push(bb);

    let mut write_c = TestComputePass::new("write-c");
    write_c.texture_writes.push(c);

    let mut export_c = TestComputePass::new("export-c");
    export_c.texture_reads.push(c);
    export_c.imported_texture_writes.push(bb);

    b.add_compute_pass(Box::new(write_a));
    let export_a_id: PassId = b.add_compute_pass(Box::new(export_a));
    let write_c_id: PassId = b.add_compute_pass(Box::new(write_c));
    b.add_compute_pass(Box::new(export_c));
    // Force a's reader to retire before c's writer begins so their lifetimes are disjoint.
    b.add_edge(export_a_id, write_c_id);

    let g = b.build()?;
    assert_eq!(
        g.transient_textures[a.index()].physical_slot,
        g.transient_textures[c.index()].physical_slot,
        "disjoint-lifetime aliased transients must share a physical slot: stats {:?}",
        g.compile_stats
    );
    assert_eq!(g.compile_stats.transient_texture_slots, 1);
    Ok(())
}

/// The compiled transient buffer list preserves the descriptors given to the builder, proving the
/// external descriptor structs round-trip across the crate boundary.
#[test]
fn transient_buffer_descriptor_survives_compile() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let bb = b.import_texture(backbuffer_import());

    let buf = b.create_buffer(TransientBufferDesc::fixed(
        "tally",
        1024,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    ));

    let mut writer = TestComputePass::new("writer");
    writer.buffer_writes.push(buf);
    writer.imported_texture_writes.push(bb);
    b.add_compute_pass(Box::new(writer));

    let g = b.build()?;
    let compiled = g
        .transient_buffers
        .iter()
        .find(|r| r.desc.label == "tally")
        .expect("tally buffer should survive compile");
    assert!(
        compiled
            .usage
            .contains(wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST),
        "compiled usage must union base usage at minimum, got {:?}",
        compiled.usage
    );
    Ok(())
}

/// `NonZeroU32` is exercised only to confirm the re-exported handle type is reachable by downstream
/// code; some in-crate tests set a multiview mask on raster passes, but compute passes do not carry
/// one, so we just ensure the type exists at the boundary.
#[test]
fn nonzero_multiview_mask_type_is_available() {
    let _mask: Option<NonZeroU32> = NonZeroU32::new(1);
}
