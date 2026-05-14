//! Cooperative asset-integration queues and wall-clock-bounded draining.

mod drain;
mod gpu_context;
mod queue;
mod retired;
mod step;
mod summary;
mod video_poll;

#[cfg(test)]
mod tests;

pub use drain::{drain_asset_tasks, drain_asset_tasks_unbounded};
pub(crate) use queue::AssetIntegratorDiagnosticSnapshot;
pub use queue::{AssetIntegrator, AssetTaskLane};
pub use retired::RetiredAssetResource;
pub use step::{AssetTask, ShaderRouteTask, StepResult};
pub use summary::AssetIntegrationDrainSummary;
