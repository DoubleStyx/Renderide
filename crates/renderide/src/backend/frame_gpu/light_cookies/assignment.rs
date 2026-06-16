use hashbrown::HashMap;

use crate::assets::texture::HostTextureAssetKind;
use crate::backend::light_gpu::{
    LIGHT_COOKIE_KIND_DIRECTIONAL_2D, LIGHT_COOKIE_KIND_POINT_CUBE, LIGHT_COOKIE_KIND_SPOT_2D,
    LightCookieBinding,
};
use crate::shared::LightType;

use super::POINT_COOKIE_FACE_COUNT;

/// Fallback metadata row for white cookie sampling.
pub(super) const LIGHT_COOKIE_FALLBACK_RECT_INDEX: u32 = 0;
/// Maximum resident 2D cookies.
pub(super) const COOKIE_2D_RECT_CAP: u32 = 64;
/// Maximum resident point-light cookie cubemaps.
pub(super) const POINT_COOKIE_CUBEMAP_CAP: u32 = 16;
/// First metadata row reserved for point-light cubemap face cookies.
pub(super) const POINT_COOKIE_RECT_BASE: u32 =
    LIGHT_COOKIE_FALLBACK_RECT_INDEX + 1 + COOKIE_2D_RECT_CAP;
/// Total light-cookie metadata rows bound for all atlas kinds.
pub(super) const LIGHT_COOKIE_RECT_CAPACITY: usize =
    (POINT_COOKIE_RECT_BASE + POINT_COOKIE_CUBEMAP_CAP * POINT_COOKIE_FACE_COUNT) as usize;

/// One requested light-cookie source assigned to atlas metadata rows.
#[derive(Clone, Copy, Debug)]
pub(super) struct LightCookieRequest {
    /// Packed host texture handle.
    pub(super) packed_id: i32,
    /// Unpacked host asset id.
    pub(super) asset_id: i32,
    /// Unpacked host texture kind.
    pub(super) kind: HostTextureAssetKind,
    /// 2D atlas rect index or first point face rect index.
    pub(super) layer: u32,
}

/// One unpacked light-cookie handle ready for atlas assignment.
#[derive(Clone, Copy, Debug)]
pub(super) struct LightCookieAssignment {
    /// Host light type requesting the cookie.
    pub(super) light_type: LightType,
    /// Packed host texture handle.
    pub(super) packed_id: i32,
    /// Unpacked host asset id.
    pub(super) asset_id: i32,
    /// Unpacked host texture kind.
    pub(super) kind: HostTextureAssetKind,
    /// Packed 2D cookie wrap modes.
    pub(super) wrap_bits: u32,
}

/// Atlas metadata slot state for a packed host texture handle.
#[derive(Clone, Copy, Debug)]
struct LightCookieSlot {
    /// Atlas metadata row assigned to this packed handle.
    layer: u32,
    /// Whether this slot is referenced by the current frame's packed lights.
    requested_this_frame: bool,
}

/// Mutable cookie assignment state shared by light packing and atlas encoding.
#[derive(Debug)]
pub(super) struct LightCookieAtlasState {
    /// Persistent 2D-cookie slots keyed by packed texture handle.
    two_d_slots: HashMap<i32, LightCookieSlot>,
    /// Persistent point-cookie slots keyed by packed texture handle.
    point_slots: HashMap<i32, LightCookieSlot>,
    /// Unique 2D-cookie requests for the current frame.
    two_d_requests: Vec<LightCookieRequest>,
    /// Unique point-cookie requests for the current frame.
    point_requests: Vec<LightCookieRequest>,
    /// One-shot guard for 2D-cookie atlas overflow.
    two_d_overflow_logged: bool,
    /// One-shot guard for point-cookie atlas overflow.
    point_overflow_logged: bool,
}

impl LightCookieAtlasState {
    /// Creates an empty assignment table.
    pub(super) fn new() -> Self {
        Self {
            two_d_slots: HashMap::new(),
            point_slots: HashMap::new(),
            two_d_requests: Vec::new(),
            point_requests: Vec::new(),
            two_d_overflow_logged: false,
            point_overflow_logged: false,
        }
    }

    /// Marks all slots unrequested and clears current-frame request lists.
    pub(super) fn begin_frame(&mut self) {
        for slot in self.two_d_slots.values_mut() {
            slot.requested_this_frame = false;
        }
        for slot in self.point_slots.values_mut() {
            slot.requested_this_frame = false;
        }
        self.two_d_requests.clear();
        self.point_requests.clear();
    }

    /// Assigns a cookie atlas binding for one resolved light.
    pub(super) fn assign(&mut self, assignment: LightCookieAssignment) -> LightCookieBinding {
        match (assignment.light_type, assignment.kind) {
            (
                LightType::Spot,
                HostTextureAssetKind::Texture2D
                | HostTextureAssetKind::RenderTexture
                | HostTextureAssetKind::VideoTexture,
            ) => self.assign_2d(
                assignment.packed_id,
                assignment.asset_id,
                assignment.kind,
                LIGHT_COOKIE_KIND_SPOT_2D,
                assignment.wrap_bits,
            ),
            (
                LightType::Directional,
                HostTextureAssetKind::Texture2D
                | HostTextureAssetKind::RenderTexture
                | HostTextureAssetKind::VideoTexture,
            ) => self.assign_2d(
                assignment.packed_id,
                assignment.asset_id,
                assignment.kind,
                LIGHT_COOKIE_KIND_DIRECTIONAL_2D,
                assignment.wrap_bits,
            ),
            (LightType::Point, HostTextureAssetKind::Cubemap) => {
                self.assign_point(assignment.packed_id, assignment.asset_id, assignment.kind)
            }
            _ => LightCookieBinding::NONE,
        }
    }

    /// Assigns a 2D cookie metadata row.
    fn assign_2d(
        &mut self,
        packed_id: i32,
        asset_id: i32,
        kind: HostTextureAssetKind,
        cookie_kind: u32,
        wrap_bits: u32,
    ) -> LightCookieBinding {
        let Some(layer) = assign_cookie_layer(
            &mut self.two_d_slots,
            packed_id,
            LIGHT_COOKIE_FALLBACK_RECT_INDEX + 1,
            POINT_COOKIE_RECT_BASE,
            1,
            &mut self.two_d_overflow_logged,
            "2D",
        ) else {
            return LightCookieBinding::NONE;
        };
        if let Some(slot) = self.two_d_slots.get_mut(&packed_id)
            && !slot.requested_this_frame
        {
            slot.requested_this_frame = true;
            self.two_d_requests.push(LightCookieRequest {
                packed_id,
                asset_id,
                kind,
                layer,
            });
        }
        LightCookieBinding {
            kind: cookie_kind,
            layer,
            wrap_bits,
        }
    }

    /// Assigns six metadata rows for a point-light cubemap cookie.
    fn assign_point(
        &mut self,
        packed_id: i32,
        asset_id: i32,
        kind: HostTextureAssetKind,
    ) -> LightCookieBinding {
        let Some(layer) = assign_cookie_layer(
            &mut self.point_slots,
            packed_id,
            POINT_COOKIE_RECT_BASE,
            LIGHT_COOKIE_RECT_CAPACITY as u32,
            POINT_COOKIE_FACE_COUNT,
            &mut self.point_overflow_logged,
            "point",
        ) else {
            return LightCookieBinding::NONE;
        };
        if let Some(slot) = self.point_slots.get_mut(&packed_id)
            && !slot.requested_this_frame
        {
            slot.requested_this_frame = true;
            self.point_requests.push(LightCookieRequest {
                packed_id,
                asset_id,
                kind,
                layer,
            });
        }
        LightCookieBinding {
            kind: LIGHT_COOKIE_KIND_POINT_CUBE,
            layer,
            wrap_bits: 0,
        }
    }

    /// Returns whether any current-frame request needs atlas synchronization.
    pub(super) fn has_requests(&self) -> bool {
        !(self.two_d_requests.is_empty() && self.point_requests.is_empty())
    }

    /// Snapshot of requests for encoder recording without holding the state lock.
    pub(super) fn requests(&self) -> (Vec<LightCookieRequest>, Vec<LightCookieRequest>) {
        (self.two_d_requests.clone(), self.point_requests.clone())
    }
}

/// Assigns or reuses one atlas metadata row block.
fn assign_cookie_layer(
    slots: &mut HashMap<i32, LightCookieSlot>,
    packed_id: i32,
    first_layer: u32,
    layer_count: u32,
    layer_stride: u32,
    overflow_logged: &mut bool,
    label: &str,
) -> Option<u32> {
    if let Some(slot) = slots.get(&packed_id) {
        return Some(slot.layer);
    }
    let last_start = layer_count.checked_sub(layer_stride)?;
    let mut layer = first_layer;
    while layer <= last_start {
        if !slots.values().any(|slot| slot.layer == layer) {
            slots.insert(
                packed_id,
                LightCookieSlot {
                    layer,
                    requested_this_frame: false,
                },
            );
            return Some(layer);
        }
        layer = layer.saturating_add(layer_stride);
    }
    let reusable = slots
        .iter()
        .find_map(|(&id, slot)| (!slot.requested_this_frame).then_some((id, slot.layer)));
    if let Some((old_id, layer)) = reusable {
        slots.remove(&old_id);
        slots.insert(
            packed_id,
            LightCookieSlot {
                layer,
                requested_this_frame: false,
            },
        );
        return Some(layer);
    }
    if !*overflow_logged {
        logger::warn!(
            "light-cookie {label} atlas full; additional {label} cookies will be ignored"
        );
        *overflow_logged = true;
    }
    None
}
