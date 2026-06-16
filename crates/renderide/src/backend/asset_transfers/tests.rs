use super::*;
use crate::shared::{
    PointRenderBufferUpload, TextureFilterMode, TextureWrapMode, TrailRenderBufferUpload,
    VideoTextureProperties,
};

#[test]
fn video_texture_properties_default_preserves_asset_id() {
    let queue = AssetTransferQueue::new();

    let props = queue.catalogs.video_texture_properties_or_default(42);

    assert_eq!(props.asset_id, 42);
    assert_eq!(props.filter_mode, TextureFilterMode::Point);
    assert_eq!(props.wrap_u, TextureWrapMode::Repeat);
    assert_eq!(props.wrap_v, TextureWrapMode::Repeat);
}

#[test]
fn video_texture_properties_default_uses_cached_properties() {
    let mut queue = AssetTransferQueue::new();
    queue.catalogs.video_texture_properties.insert(
        7,
        VideoTextureProperties {
            asset_id: 7,
            filter_mode: TextureFilterMode::Trilinear,
            aniso_level: 8,
            wrap_u: TextureWrapMode::Mirror,
            wrap_v: TextureWrapMode::Clamp,
        },
    );

    let props = queue.catalogs.video_texture_properties_or_default(7);

    assert_eq!(props.asset_id, 7);
    assert_eq!(props.filter_mode, TextureFilterMode::Trilinear);
    assert_eq!(props.aniso_level, 8);
    assert_eq!(props.wrap_u, TextureWrapMode::Mirror);
    assert_eq!(props.wrap_v, TextureWrapMode::Clamp);
}

#[test]
fn point_render_buffer_uploads_keep_newest_pending_generation() {
    let mut queue = AssetTransferQueue::new();

    let first = queue.retain_latest_point_render_buffer_upload(PointRenderBufferUpload {
        asset_id: 5,
        count: 1,
        ..Default::default()
    });
    let second = queue.retain_latest_point_render_buffer_upload(PointRenderBufferUpload {
        asset_id: 5,
        count: 2,
        ..Default::default()
    });

    assert!(!first.replaced_pending_upload);
    assert!(second.replaced_pending_upload);
    assert!(!queue.point_render_buffer_generation_is_current(5, first.generation));
    assert!(queue.point_render_buffer_generation_is_current(5, second.generation));
    let pending = queue
        .take_pending_point_render_buffer_upload(5)
        .expect("pending point upload");
    assert_eq!(pending.upload.count, 2);
    assert_eq!(pending.generation, second.generation);
    assert!(queue.take_pending_point_render_buffer_upload(5).is_none());
}

#[test]
fn point_render_buffer_pending_upload_waits_behind_active_build() {
    let mut queue = AssetTransferQueue::new();
    let first = queue.retain_latest_point_render_buffer_upload(PointRenderBufferUpload {
        asset_id: 5,
        count: 1,
        ..Default::default()
    });

    assert!(queue.mark_point_render_buffer_build_active(5));
    let claimed = queue
        .take_pending_point_render_buffer_upload(5)
        .expect("claimed point upload");
    let second = queue.retain_latest_point_render_buffer_upload(PointRenderBufferUpload {
        asset_id: 5,
        count: 2,
        ..Default::default()
    });

    assert_eq!(claimed.generation, first.generation);
    assert!(!second.replaced_pending_upload);
    assert!(!queue.has_pending_asset_work());

    queue.clear_point_render_buffer_build_active(5);

    assert!(queue.has_pending_asset_work());
    assert_eq!(queue.work_snapshot().startable_particle_uploads, 1);
}

#[test]
fn ready_particle_completion_counts_as_pending_work() {
    let mut queue = AssetTransferQueue::new();
    let generation = queue.begin_point_render_buffer_generation(11);
    let tx = queue.point_render_buffer_build_sender();

    tx.send(PointBuildResult {
        asset_id: 11,
        generation,
        result: Err(
            crate::particles::ParticleRenderBufferError::WorkerPanicked {
                kind: "point",
                asset_id: 11,
            },
        ),
    })
    .expect("ready point build result");

    assert!(queue.has_ready_particle_build_results());
    assert!(queue.has_pending_asset_work());
}

#[test]
fn pending_video_load_counts_as_deferred_asset_work() {
    let mut queue = AssetTransferQueue::new();
    queue.pending.pending_video_texture_loads.insert(
        3,
        crate::shared::VideoTextureLoad {
            asset_id: 3,
            ..Default::default()
        },
    );

    let snapshot = queue.work_snapshot();

    assert_eq!(snapshot.deferred_video_loads, 1);
    assert!(snapshot.has_pending_work());
    assert!(queue.has_pending_asset_work());
}

#[test]
fn mesh_upload_generation_marks_superseded_work_stale() {
    let mut queue = AssetTransferQueue::new();

    let first = queue.begin_mesh_upload_generation(5);
    let second = queue.begin_mesh_upload_generation(5);

    assert_ne!(first, second);
    assert!(!queue.mesh_upload_generation_is_current(5, first));
    assert!(queue.mesh_upload_generation_is_current(5, second));
    assert_eq!(queue.current_mesh_upload_generation(5), Some(second));
}

#[test]
fn mesh_upload_generation_invalidation_marks_existing_work_stale() {
    let mut queue = AssetTransferQueue::new();

    let generation = queue.begin_mesh_upload_generation(7);
    let invalidating_generation = queue.invalidate_mesh_upload_generation(7);

    assert_ne!(generation, invalidating_generation);
    assert!(!queue.mesh_upload_generation_is_current(7, generation));
    assert!(queue.mesh_upload_generation_is_current(7, invalidating_generation));
}

#[test]
fn trail_render_buffer_upload_after_worker_claim_enqueues_new_pending_upload() {
    let mut queue = AssetTransferQueue::new();
    let first = queue.retain_latest_trail_render_buffer_upload(TrailRenderBufferUpload {
        asset_id: 9,
        trails_count: 1,
        ..Default::default()
    });
    let claimed = queue
        .take_pending_trail_render_buffer_upload(9)
        .expect("claimed trail upload");
    let second = queue.retain_latest_trail_render_buffer_upload(TrailRenderBufferUpload {
        asset_id: 9,
        trails_count: 2,
        ..Default::default()
    });

    assert_eq!(claimed.generation, first.generation);
    assert!(!second.replaced_pending_upload);
    let pending = queue
        .take_pending_trail_render_buffer_upload(9)
        .expect("new pending trail upload");
    assert_eq!(pending.upload.trails_count, 2);
    assert_eq!(pending.generation, second.generation);
}

#[test]
fn trail_render_buffer_pending_replacement_stays_newest_during_active_build() {
    let mut queue = AssetTransferQueue::new();

    queue.retain_latest_trail_render_buffer_upload(TrailRenderBufferUpload {
        asset_id: 9,
        trails_count: 1,
        ..Default::default()
    });
    assert!(queue.mark_trail_render_buffer_build_active(9));
    queue.take_pending_trail_render_buffer_upload(9);
    let first_pending = queue.retain_latest_trail_render_buffer_upload(TrailRenderBufferUpload {
        asset_id: 9,
        trails_count: 2,
        ..Default::default()
    });
    let newest = queue.retain_latest_trail_render_buffer_upload(TrailRenderBufferUpload {
        asset_id: 9,
        trails_count: 3,
        ..Default::default()
    });

    assert!(!first_pending.replaced_pending_upload);
    assert!(newest.replaced_pending_upload);
    let pending = queue
        .take_pending_trail_render_buffer_upload(9)
        .expect("newest trail upload");
    assert_eq!(pending.upload.trails_count, 3);
}

#[test]
fn cancel_point_render_buffer_generation_reports_dropped_pending_upload() {
    let mut queue = AssetTransferQueue::new();
    let retained = queue.retain_latest_point_render_buffer_upload(PointRenderBufferUpload {
        asset_id: 17,
        count: 1,
        ..Default::default()
    });

    assert!(queue.cancel_point_render_buffer_generation(17));
    assert!(!queue.point_render_buffer_generation_is_current(17, retained.generation));
    assert!(queue.take_pending_point_render_buffer_upload(17).is_none());
    assert!(!queue.cancel_point_render_buffer_generation(17));
}
