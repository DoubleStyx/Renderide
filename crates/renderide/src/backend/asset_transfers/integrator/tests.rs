use std::time::{Duration, Instant};

use super::super::limits::MAX_ASSET_INTEGRATION_QUEUE_TASKS;
use super::super::texture_task::TextureUploadTask;
use super::drain::{MIN_HIGH_PRIORITY_EMERGENCY_BUDGET, high_priority_emergency_deadline};
use super::queue::ASSET_INTEGRATION_QUEUE_WARN_THRESHOLD;
use super::*;
use crate::materials::RasterPipelineKind;
use crate::shared::MaterialsUpdateBatch;
use crate::shared::{SetTexture2DData, SetTexture2DFormat};

fn texture_task(high_priority: bool) -> AssetTask {
    AssetTask::Texture(TextureUploadTask::new(
        SetTexture2DData {
            high_priority,
            ..Default::default()
        },
        SetTexture2DFormat::default(),
        wgpu::TextureFormat::Rgba8Unorm,
    ))
}

fn task_is_high_priority(task: &AssetTask) -> bool {
    match task {
        AssetTask::MaterialUpdate(_)
        | AssetTask::ShaderRoute(_)
        | AssetTask::PointRenderBuffer(_)
        | AssetTask::TrailRenderBuffer(_) => false,
        AssetTask::Mesh(task) => task.high_priority(),
        AssetTask::Texture(task) => task.high_priority(),
        AssetTask::Texture3d(task) => task.high_priority(),
        AssetTask::Cubemap(task) => task.high_priority(),
    }
}

#[test]
fn asset_integration_summary_totals_and_budget_flag() {
    let summary = AssetIntegrationDrainSummary {
        main_before: 1,
        high_priority_before: 2,
        render_before: 3,
        normal_priority_before: 3,
        particle_before: 4,
        main_after: 5,
        high_priority_after: 1,
        render_after: 2,
        normal_priority_after: 4,
        particle_after: 3,
        normal_priority_budget_exhausted: true,
        ..Default::default()
    };

    assert_eq!(summary.total_before(), 13);
    assert_eq!(summary.total_after(), 15);
    assert!(summary.budget_exhausted());
}

#[test]
fn high_priority_emergency_deadline_extends_normal_budget() {
    let start = Instant::now();
    let normal_deadline = start + Duration::from_millis(3);

    let deadline = high_priority_emergency_deadline(start, normal_deadline);

    assert_eq!(
        deadline.checked_duration_since(normal_deadline),
        Some(Duration::from_millis(3))
    );
}

#[test]
fn high_priority_emergency_deadline_has_minimum_budget_when_normal_deadline_elapsed() {
    let start = Instant::now();
    let normal_deadline = match start.checked_sub(Duration::from_millis(5)) {
        Some(deadline) => deadline,
        None => start,
    };

    let deadline = high_priority_emergency_deadline(start, normal_deadline);

    assert_eq!(
        deadline.checked_duration_since(start),
        Some(MIN_HIGH_PRIORITY_EMERGENCY_BUDGET)
    );
}

#[test]
fn pop_next_prefers_high_priority_queue() {
    let mut integrator = AssetIntegrator::default();
    assert!(integrator.enqueue(texture_task(false), false));
    assert!(integrator.enqueue(texture_task(true), true));

    let first = integrator.pop_next().unwrap();
    let second = integrator.pop_next().unwrap();

    assert!(task_is_high_priority(&first));
    assert!(!task_is_high_priority(&second));
    assert_eq!(integrator.total_queued(), 0);
}

#[test]
fn push_front_preserves_priority_lane() {
    let mut integrator = AssetIntegrator::default();
    integrator.push_front(texture_task(false), false);
    integrator.push_front(texture_task(true), true);

    assert_eq!(integrator.high_priority.len(), 1);
    assert_eq!(integrator.normal_priority.len(), 1);
    assert!(task_is_high_priority(
        integrator.pop_next().as_ref().unwrap()
    ));
}

#[test]
fn enqueue_accepts_beyond_warning_threshold() {
    let mut integrator = AssetIntegrator::default();
    for _ in 0..=ASSET_INTEGRATION_QUEUE_WARN_THRESHOLD {
        assert!(integrator.enqueue(texture_task(false), false));
    }

    assert_eq!(
        integrator.total_queued(),
        ASSET_INTEGRATION_QUEUE_WARN_THRESHOLD + 1
    );
    assert_eq!(
        integrator.peak_queued(),
        ASSET_INTEGRATION_QUEUE_WARN_THRESHOLD + 1
    );
}

#[test]
fn enqueue_material_update_returns_batch_when_queue_is_full() {
    let mut integrator = AssetIntegrator::default();
    fill_integrator_to_capacity(&mut integrator);

    let batch = MaterialsUpdateBatch {
        update_batch_id: 42,
        ..Default::default()
    };
    let returned = integrator
        .enqueue_material_update(batch)
        .expect("full queue should return material batch");

    assert_eq!(returned.update_batch_id, 42);
    assert_eq!(integrator.total_queued(), MAX_ASSET_INTEGRATION_QUEUE_TASKS);
}

#[test]
fn enqueue_shader_route_returns_route_when_queue_is_full() {
    let mut integrator = AssetIntegrator::default();
    fill_integrator_to_capacity(&mut integrator);

    let route = ShaderRouteTask {
        asset_id: 77,
        pipeline: RasterPipelineKind::Null,
        shader_asset_name: Some(String::from("ExampleShader")),
        shader_variant_bits: Some(5),
    };
    let returned = integrator
        .enqueue_shader_route(route)
        .expect("full queue should return shader route");

    assert_eq!(returned.asset_id, 77);
    assert_eq!(returned.shader_asset_name.as_deref(), Some("ExampleShader"));
    assert_eq!(returned.shader_variant_bits, Some(5));
    assert_eq!(integrator.total_queued(), MAX_ASSET_INTEGRATION_QUEUE_TASKS);
}

fn fill_integrator_to_capacity(integrator: &mut AssetIntegrator) {
    for _ in 0..MAX_ASSET_INTEGRATION_QUEUE_TASKS {
        assert!(integrator.enqueue(texture_task(false), false));
    }
    assert_eq!(integrator.total_queued(), MAX_ASSET_INTEGRATION_QUEUE_TASKS);
}
