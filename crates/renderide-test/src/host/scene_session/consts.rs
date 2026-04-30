//! Centralized constants for the scene-session orchestration.
//!
//! Grouped by concern so cross-concern relationships (e.g. timing floors that depend on each
//! other) stay visible in one place. Each value carries a `///` line explaining the *why* of
//! its number.

/// Asset and buffer ids used by the harness.
///
/// These never collide with anything the renderer allocates internally because the renderer
/// treats the shared memory only as host-driven input.
pub(super) mod asset_ids {
    /// Sphere mesh asset id; chosen `>0` to keep clear of any renderer-internal sentinel.
    pub(in crate::host::scene_session) const SPHERE_MESH: i32 = 2;
    /// Sphere material asset id; same rationale as [`SPHERE_MESH`].
    pub(in crate::host::scene_session) const SPHERE_MATERIAL: i32 = 4;
    /// Buffer id for the sphere mesh shared-memory region.
    pub(in crate::host::scene_session) const SPHERE_MESH_BUFFER: i32 = 0;
    /// Buffer id for the scene-state shared-memory region (pose updates, additions, mesh
    /// states, packed material ids).
    pub(in crate::host::scene_session) const SCENE_STATE_BUFFER: i32 = 1;
    /// Render-space id for the sole render space the harness submits.
    pub(in crate::host::scene_session) const RENDER_SPACE: i32 = 1;
}

/// Procedural sphere tessellation that stands in for "a real scene".
///
/// Values must match the golden image's vertex layout — changing them invalidates the committed
/// `goldens/sphere.png`.
pub(super) mod sphere_tessellation {
    /// Number of latitude bands; `16` produces enough silhouette smoothness for SSIM stability.
    pub(in crate::host::scene_session) const LATITUDE_BANDS: u32 = 16;
    /// Number of longitude bands; `24` keeps triangle count small while preserving the silhouette.
    pub(in crate::host::scene_session) const LONGITUDE_BANDS: u32 = 24;
}

/// Wall-clock timing parameters governing PNG readback, lockstep pumping, and shutdown.
pub(super) mod timing {
    use std::time::Duration;

    /// Floor on the post-submit wait before any PNG mtime is accepted.
    ///
    /// Covers slow software rendering (e.g. lavapipe on CI) where one renderer interval is not
    /// enough for apply-then-render to write a fresh PNG.
    pub(in crate::host::scene_session) const MIN_WALL_AFTER_SUBMIT_FLOOR: Duration =
        Duration::from_millis(1500);

    /// PNG mtime must remain unchanged for this duration before the file is treated as stable.
    ///
    /// Guards against accepting a mid-write PNG whose contents still mutate.
    pub(in crate::host::scene_session) const STABILITY_WINDOW: Duration =
        Duration::from_millis(200);

    /// Interval between informational "still waiting" log lines during readback.
    pub(in crate::host::scene_session) const LOG_INTERVAL: Duration = Duration::from_secs(2);

    /// Sleep between consecutive PNG-stability polls.
    pub(in crate::host::scene_session) const POLL_INTERVAL: Duration = Duration::from_millis(20);

    /// Sleep between scene-submission pump polls (`ensure_scene_submitted`).
    pub(in crate::host::scene_session) const SCENE_SUBMIT_POLL: Duration = Duration::from_millis(2);

    /// Slack added on top of `MIN_WALL_AFTER_SUBMIT_FLOOR` when computing the readback deadline.
    pub(in crate::host::scene_session) const PNG_DEADLINE_SLACK: Duration = Duration::from_secs(2);

    /// Grace period for the renderer to exit voluntarily after `RendererShutdownRequest`.
    pub(in crate::host::scene_session) const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

    /// Sleep between try-wait checks during shutdown.
    pub(in crate::host::scene_session) const SHUTDOWN_POLL: Duration = Duration::from_millis(50);
}
