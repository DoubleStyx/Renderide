//! Render-graph v2 validation and blackboard declaration tests.

use super::common::*;
use crate::render_graph::blackboard::BlackboardSlot;
use crate::render_graph::validation::RenderGraphValidationMode;

/// Test blackboard slot used by validation-mode coverage.
struct ValidationSlot;

impl BlackboardSlot for ValidationSlot {
    type Value = u32;
}

/// Compute pass that requires a blackboard seed and writes an imported buffer.
struct RequiredBlackboardReadPass {
    /// Imported output used to keep the pass retained after culling.
    output: ImportedBufferHandle,
}

impl ComputePass for RequiredBlackboardReadPass {
    fn name(&self) -> &str {
        "required-blackboard-read"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.compute();
        b.read_blackboard::<ValidationSlot>();
        b.import_buffer(self.output, BufferAccess::CopyDst);
        Ok(())
    }

    fn record(&self, _ctx: &mut ComputePassCtx<'_, '_, '_>) -> Result<(), RenderPassError> {
        Ok(())
    }
}

#[test]
fn strict_validation_rejects_required_blackboard_read_without_seed() {
    let mut b = GraphBuilder::with_validation_mode(RenderGraphValidationMode::Strict);
    let output = b.import_buffer(buffer_import_readback());
    b.add_compute_pass(Box::new(RequiredBlackboardReadPass { output }));

    assert!(matches!(
        b.build(),
        Err(GraphBuildError::Validation { report }) if report.len() == 1
    ));
}

#[test]
fn declared_blackboard_seed_satisfies_strict_validation() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::with_validation_mode(RenderGraphValidationMode::Strict);
    b.seed_blackboard::<ValidationSlot>("test seed");
    let output = b.import_buffer(buffer_import_readback());
    b.add_compute_pass(Box::new(RequiredBlackboardReadPass { output }));

    let g = b.build()?;
    assert_eq!(g.compile_stats.validation_diagnostics, 0);
    assert_eq!(g.pass_info[0].blackboard_accesses.len(), 1);
    Ok(())
}

#[test]
fn warn_validation_records_diagnostic_and_builds() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::with_validation_mode(RenderGraphValidationMode::Warn);
    let output = b.import_buffer(buffer_import_readback());
    b.add_compute_pass(Box::new(RequiredBlackboardReadPass { output }));

    let g = b.build()?;
    assert_eq!(g.compile_stats.validation_diagnostics, 1);
    assert_eq!(g.validation_report.len(), 1);
    Ok(())
}
