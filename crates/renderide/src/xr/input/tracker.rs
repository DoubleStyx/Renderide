//! OpenXR Vive tracker role actions, enumeration, and IPC tracker-state conversion.

use std::ptr;

use glam::{Quat, Vec3};
use hashbrown::{HashMap, HashSet};
use openxr as xr;

use super::pose::pose_from_location;
use crate::shared::TrackerState;
use crate::xr::session::openxr_tracking_pose_to_host;

const UNKNOWN_BATTERY_LEVEL: f32 = -1.0;

/// One body-tracker role exposed by `XR_HTCX_vive_tracker_interaction`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::xr::input) struct TrackerRole {
    /// Stable output order for host-facing tracker arrays.
    pub(in crate::xr::input) order: u8,
    /// Stable role key from the OpenXR role path.
    pub(in crate::xr::input) key: &'static str,
    /// Top-level OpenXR user path for this role.
    pub(in crate::xr::input) user_path: &'static str,
    /// Renderide action id for this role's grip pose.
    pub(in crate::xr::input) action_id: &'static str,
    /// Human-readable action label shown in OpenXR binding UIs.
    pub(in crate::xr::input) localized_name: &'static str,
}

/// Body tracker roles forwarded to the host.
pub(in crate::xr::input) const TRACKER_ROLES: &[TrackerRole] = &[
    TrackerRole {
        order: 0,
        key: "waist",
        user_path: "/user/vive_tracker_htcx/role/waist",
        action_id: "tracker_waist_grip_pose",
        localized_name: "Waist tracker grip pose",
    },
    TrackerRole {
        order: 1,
        key: "chest",
        user_path: "/user/vive_tracker_htcx/role/chest",
        action_id: "tracker_chest_grip_pose",
        localized_name: "Chest tracker grip pose",
    },
    TrackerRole {
        order: 2,
        key: "left_foot",
        user_path: "/user/vive_tracker_htcx/role/left_foot",
        action_id: "tracker_left_foot_grip_pose",
        localized_name: "Left foot tracker grip pose",
    },
    TrackerRole {
        order: 3,
        key: "right_foot",
        user_path: "/user/vive_tracker_htcx/role/right_foot",
        action_id: "tracker_right_foot_grip_pose",
        localized_name: "Right foot tracker grip pose",
    },
    TrackerRole {
        order: 4,
        key: "left_knee",
        user_path: "/user/vive_tracker_htcx/role/left_knee",
        action_id: "tracker_left_knee_grip_pose",
        localized_name: "Left knee tracker grip pose",
    },
    TrackerRole {
        order: 5,
        key: "right_knee",
        user_path: "/user/vive_tracker_htcx/role/right_knee",
        action_id: "tracker_right_knee_grip_pose",
        localized_name: "Right knee tracker grip pose",
    },
    TrackerRole {
        order: 6,
        key: "left_elbow",
        user_path: "/user/vive_tracker_htcx/role/left_elbow",
        action_id: "tracker_left_elbow_grip_pose",
        localized_name: "Left elbow tracker grip pose",
    },
    TrackerRole {
        order: 7,
        key: "right_elbow",
        user_path: "/user/vive_tracker_htcx/role/right_elbow",
        action_id: "tracker_right_elbow_grip_pose",
        localized_name: "Right elbow tracker grip pose",
    },
    TrackerRole {
        order: 8,
        key: "left_shoulder",
        user_path: "/user/vive_tracker_htcx/role/left_shoulder",
        action_id: "tracker_left_shoulder_grip_pose",
        localized_name: "Left shoulder tracker grip pose",
    },
    TrackerRole {
        order: 9,
        key: "right_shoulder",
        user_path: "/user/vive_tracker_htcx/role/right_shoulder",
        action_id: "tracker_right_shoulder_grip_pose",
        localized_name: "Right shoulder tracker grip pose",
    },
    TrackerRole {
        order: 10,
        key: "left_wrist",
        user_path: "/user/vive_tracker_htcx/role/left_wrist",
        action_id: "tracker_left_wrist_grip_pose",
        localized_name: "Left wrist tracker grip pose",
    },
    TrackerRole {
        order: 11,
        key: "right_wrist",
        user_path: "/user/vive_tracker_htcx/role/right_wrist",
        action_id: "tracker_right_wrist_grip_pose",
        localized_name: "Right wrist tracker grip pose",
    },
    TrackerRole {
        order: 12,
        key: "left_ankle",
        user_path: "/user/vive_tracker_htcx/role/left_ankle",
        action_id: "tracker_left_ankle_grip_pose",
        localized_name: "Left ankle tracker grip pose",
    },
    TrackerRole {
        order: 13,
        key: "right_ankle",
        user_path: "/user/vive_tracker_htcx/role/right_ankle",
        action_id: "tracker_right_ankle_grip_pose",
        localized_name: "Right ankle tracker grip pose",
    },
];

/// Pose action handle for one Vive tracker role.
pub(in crate::xr::input) struct TrackerPoseAction {
    /// Body role this action samples.
    pub(in crate::xr::input) role: &'static TrackerRole,
    /// Role-specific grip pose action.
    pub(in crate::xr::input) action: xr::Action<xr::Posef>,
}

/// Pose space for one Vive tracker role.
pub(in crate::xr::input) struct TrackerPoseSpace {
    /// Body role this space samples.
    pub(in crate::xr::input) role: &'static TrackerRole,
    /// Role-specific grip pose space.
    pub(in crate::xr::input) space: xr::Space,
}

struct KnownTracker {
    role: &'static TrackerRole,
}

/// Stable tracker inventory keyed by OpenXR persistent tracker path.
#[derive(Default)]
pub(in crate::xr::input) struct TrackerCache {
    known: HashMap<String, KnownTracker>,
}

impl TrackerCache {
    /// Enumerates current Vive trackers, updates the known set, and returns host-facing states.
    pub(in crate::xr::input) fn sample(
        &mut self,
        instance: &xr::Instance,
        stage: &xr::Space,
        predicted_time: xr::Time,
        spaces: &[TrackerPoseSpace],
    ) -> Vec<TrackerState> {
        profiling::scope!("xr::trackers_sample");
        let current_ids = match self.refresh_current_trackers(instance) {
            Ok(current_ids) => current_ids,
            Err(error) => {
                logger::trace!("OpenXR Vive tracker enumeration failed: {error:?}");
                HashSet::new()
            }
        };
        self.known_tracker_states(stage, predicted_time, spaces, &current_ids)
    }

    fn refresh_current_trackers(
        &mut self,
        instance: &xr::Instance,
    ) -> Result<HashSet<String>, xr::sys::Result> {
        let paths = enumerate_vive_tracker_paths(instance)?;
        let mut current_ids = HashSet::with_capacity(paths.len());
        for paths in paths {
            let Some(role_path) = paths.role else {
                continue;
            };
            let Some(role) = role_from_openxr_path(instance, role_path)? else {
                continue;
            };
            let persistent_id = instance.path_to_string(paths.persistent)?;
            current_ids.insert(persistent_id.clone());
            self.known.insert(persistent_id, KnownTracker { role });
        }
        Ok(current_ids)
    }

    fn known_tracker_states(
        &self,
        stage: &xr::Space,
        predicted_time: xr::Time,
        spaces: &[TrackerPoseSpace],
        current_ids: &HashSet<String>,
    ) -> Vec<TrackerState> {
        self.sorted_known_trackers()
            .into_iter()
            .map(|(id, tracker)| {
                let pose = current_ids
                    .contains(id)
                    .then(|| locate_tracker_pose(stage, predicted_time, spaces, tracker.role))
                    .flatten();
                tracker_state(id, pose)
            })
            .collect()
    }

    fn sorted_known_trackers(&self) -> Vec<(&String, &KnownTracker)> {
        let mut known: Vec<(&String, &KnownTracker)> = self.known.iter().collect();
        known.sort_by(|(a_id, a), (b_id, b)| {
            a.role.order.cmp(&b.role.order).then_with(|| a_id.cmp(b_id))
        });
        known
    }
}

/// Creates optional pose spaces for all role-specific tracker actions.
pub(in crate::xr::input) fn create_tracker_pose_spaces(
    session: &xr::Session<xr::Vulkan>,
    actions: &[TrackerPoseAction],
) -> Vec<TrackerPoseSpace> {
    let mut spaces = Vec::with_capacity(actions.len());
    for tracker_action in actions {
        match tracker_action
            .action
            .create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)
        {
            Ok(space) => spaces.push(TrackerPoseSpace {
                role: tracker_action.role,
                space,
            }),
            Err(error) => {
                logger::warn!(
                    "OpenXR Vive tracker space creation failed for {}: {error:?}",
                    tracker_action.role.key
                );
            }
        }
    }
    spaces
}

fn role_from_openxr_path(
    instance: &xr::Instance,
    role_path: xr::Path,
) -> Result<Option<&'static TrackerRole>, xr::sys::Result> {
    let role_path = instance.path_to_string(role_path)?;
    Ok(role_for_user_path(&role_path))
}

fn role_for_user_path(path: &str) -> Option<&'static TrackerRole> {
    TRACKER_ROLES.iter().find(|role| role.user_path == path)
}

fn locate_tracker_pose(
    stage: &xr::Space,
    predicted_time: xr::Time,
    spaces: &[TrackerPoseSpace],
    role: &TrackerRole,
) -> Option<(Vec3, Quat)> {
    let space = spaces.iter().find(|space| space.role == role)?;
    match space.space.locate(stage, predicted_time) {
        Ok(location) => pose_from_location(&location)
            .map(|(position, rotation)| openxr_tracking_pose_to_host(position, rotation)),
        Err(error) => {
            logger::trace!(
                "OpenXR Vive tracker locate failed for {}: {error:?}",
                role.key
            );
            None
        }
    }
}

fn tracker_state(unique_id: &str, pose: Option<(Vec3, Quat)>) -> TrackerState {
    let (is_tracking, position, rotation) = match pose {
        Some((position, rotation)) => (true, position, rotation),
        None => (false, Vec3::ZERO, Quat::IDENTITY),
    };
    TrackerState {
        unique_id: Some(unique_id.to_string()),
        is_tracking,
        position,
        rotation,
        battery_level: UNKNOWN_BATTERY_LEVEL,
        battery_charging: false,
    }
}

fn enumerate_vive_tracker_paths(
    instance: &xr::Instance,
) -> Result<Vec<xr::ViveTrackerPathsHTCX>, xr::sys::Result> {
    let Some(ext) = instance.exts().htcx_vive_tracker_interaction.as_ref() else {
        return Ok(Vec::new());
    };

    let mut count = 0;
    // SAFETY: `instance` is a live OpenXR instance and this first call follows the extension's
    // two-call enumeration pattern with zero capacity and a null output pointer.
    unsafe {
        xr_result((ext.enumerate_vive_tracker_paths)(
            instance.as_raw(),
            0,
            &raw mut count,
            ptr::null_mut(),
        ))?;
    }
    if count == 0 {
        return Ok(Vec::new());
    }

    let mut raw_paths = Vec::with_capacity(count as usize);
    raw_paths.resize_with(count as usize, empty_vive_tracker_paths);
    let mut written = count;
    // SAFETY: `raw_paths` contains `count` initialized `XrViveTrackerPathsHTCX` structs with
    // correct `type` tags, and its pointer remains valid for the duration of the call.
    unsafe {
        xr_result((ext.enumerate_vive_tracker_paths)(
            instance.as_raw(),
            count,
            &raw mut written,
            raw_paths.as_mut_ptr(),
        ))?;
    }
    raw_paths.truncate(written as usize);
    Ok(raw_paths.into_iter().map(Into::into).collect())
}

fn empty_vive_tracker_paths() -> xr::sys::ViveTrackerPathsHTCX {
    xr::sys::ViveTrackerPathsHTCX {
        ty: xr::sys::StructureType::VIVE_TRACKER_PATHS_HTCX,
        next: ptr::null_mut(),
        persistent_path: xr::Path::NULL,
        role_path: xr::Path::NULL,
    }
}

fn xr_result(result: xr::sys::Result) -> Result<(), xr::sys::Result> {
    if result.into_raw() >= 0 {
        Ok(())
    } else {
        Err(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_lookup_accepts_body_roles() {
        let role = role_for_user_path("/user/vive_tracker_htcx/role/left_foot").expect("left foot");
        assert_eq!(role.key, "left_foot");
        assert_eq!(role.action_id, "tracker_left_foot_grip_pose");
        assert_eq!(role.localized_name, "Left foot tracker grip pose");
    }

    #[test]
    fn role_lookup_rejects_non_body_roles() {
        assert!(role_for_user_path("/user/vive_tracker_htcx/role/camera").is_none());
        assert!(role_for_user_path("/user/vive_tracker_htcx/role/keyboard").is_none());
        assert!(role_for_user_path("/user/vive_tracker_htcx/role/handheld_object").is_none());
    }

    #[test]
    fn tracker_state_uses_persistent_id_and_unknown_battery() {
        let state = tracker_state("/devices/htc/tracker/abc", None);
        assert_eq!(state.unique_id.as_deref(), Some("/devices/htc/tracker/abc"));
        assert!(!state.is_tracking);
        assert_eq!(state.position, Vec3::ZERO);
        assert_eq!(state.rotation, Quat::IDENTITY);
        assert_eq!(state.battery_level, UNKNOWN_BATTERY_LEVEL);
        assert!(!state.battery_charging);
    }

    #[test]
    fn tracker_state_preserves_tracked_pose() {
        let position = Vec3::new(1.0, 2.0, 3.0);
        let rotation = Quat::from_rotation_y(0.5);
        let state = tracker_state("tracker", Some((position, rotation)));
        assert!(state.is_tracking);
        assert_eq!(state.position, position);
        assert_eq!(state.rotation, rotation);
    }

    #[test]
    fn known_trackers_are_sorted_by_role_then_id() {
        let mut cache = TrackerCache::default();
        cache.known.insert(
            "z".to_string(),
            KnownTracker {
                role: role_for_user_path("/user/vive_tracker_htcx/role/right_foot")
                    .expect("right foot"),
            },
        );
        cache.known.insert(
            "a".to_string(),
            KnownTracker {
                role: role_for_user_path("/user/vive_tracker_htcx/role/waist").expect("waist"),
            },
        );
        cache.known.insert(
            "b".to_string(),
            KnownTracker {
                role: role_for_user_path("/user/vive_tracker_htcx/role/waist").expect("waist"),
            },
        );

        let states = cache.sorted_known_trackers();
        let ids: Vec<&str> = states.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, ["a", "b", "z"]);
    }
}
