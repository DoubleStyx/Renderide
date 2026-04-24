//! Synthesises per-finger [`HandState`] data from controller input.
//!
//! Without this, the host would receive no hand-tracking data and `HandPoser`
//! (`references_external/FrooxEngine/HandPoser.cs:501-517`) would reset every finger to its
//! `OriginalRotation`, leaving the avatar playing the desktop idle pose while the user is in VR.
//!
//! The presets below are transcribed from `references_external/FrooxEngine/FingerPosePresets.cs`
//! (`Idle` and `Fist`). Segment layout matches the host's unpack loop at
//! `references_external/FrooxEngine/VR_Manager.cs:385-394`: 24 entries indexed by
//! `BodyNode - LeftThumbMetacarpal`, ordered Thumb(Met,Prox,Dist,Tip), Index(Met,Prox,Inter,Dist,Tip),
//! Middle(..), Ring(..), Pinky(..). Right-hand [`HandState`] values reuse the same indexing but hold
//! right-hand data; the host mirrors via `bodyNode.GetSide(chirality)`.
//!
//! The curl blend is deliberately conservative: metacarpals are left at idle and we set
//! [`HandState::tracks_metacarpals`] to `false`, so the host overrides non-thumb metacarpals to the
//! avatar's own rest pose (see `HandPoser.cs:535`). Thumb is held at idle; index curl follows the
//! trigger analog; middle/ring/pinky follow the squeeze (grip) analog.

#![expect(
    clippy::unreadable_literal,
    reason = "preset literals are transcribed verbatim from FingerPosePresets.cs; \
              keep 1:1 for diff-ability against the C# source"
)]

use glam::{Quat, Vec3};

use crate::shared::{Chirality, HandState, VRControllerState};

/// Number of finger segments in a [`HandState`]. Equal to
/// `BodyNode::LeftPinkyTip - BodyNode::LeftThumbMetacarpal + 1 = 42 - 19 + 1`.
const SEGMENT_COUNT: usize = 24;

/// Stable IPC identifier for the left synthesised hand. The host keys [`HandState`] by `unique_id`
/// in `VR_Manager._hands` (`VR_Manager.cs:373`); reusing the same string across frames lets the
/// host's `Hand` instance persist.
const LEFT_HAND_ID: &str = "renderide_left_hand";
/// Stable IPC identifier for the right synthesised hand. See [`LEFT_HAND_ID`].
const RIGHT_HAND_ID: &str = "renderide_right_hand";

/// Which finger a [`HandState`] segment index (0..24) belongs to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FingerKind {
    /// Thumb: segments 0..=3 (Metacarpal, Proximal, Distal, Tip — no Intermediate).
    Thumb,
    /// Index finger: segments 4..=8.
    Index,
    /// Middle finger: segments 9..=13.
    Middle,
    /// Ring finger: segments 14..=18.
    Ring,
    /// Pinky: segments 19..=23.
    Pinky,
}

/// Returns which finger the segment at `index` (0..24) belongs to, matching the
/// `BodyNode::LeftThumbMetacarpal..=LeftPinkyTip` layout.
fn finger_kind_for_segment(index: usize) -> FingerKind {
    match index {
        0..=3 => FingerKind::Thumb,
        4..=8 => FingerKind::Index,
        9..=13 => FingerKind::Middle,
        14..=18 => FingerKind::Ring,
        19..=23 => FingerKind::Pinky,
        _ => FingerKind::Pinky,
    }
}

/// Idle pose positions for the left hand (24 segments). From `FingerPosePresets.cs:18-41`.
const IDLE_POS_LEFT: [[f32; 3]; SEGMENT_COUNT] = [
    [0.032114964, -0.013815194, 0.025049219],
    [0.06472655, -0.014863592, 0.04888261],
    [0.085127905, -0.0175668, 0.074058294],
    [0.10977111, -0.020164132, 0.09177886],
    [0.026866147, 0.000632452, 0.015001934],
    [0.03552094, 0.018421926, 0.086695716],
    [0.039913304, 0.014596581, 0.13023789],
    [0.04362595, 0.00970161, 0.15826792],
    [0.045708552, 0.004660368, 0.1804279],
    [0.007119526, 0.0021774252, 0.016318731],
    [0.007958223, 0.015812714, 0.085955404],
    [0.005875932, 0.013537288, 0.12895343],
    [0.007925333, 0.0040506124, 0.16077219],
    [0.009474924, -0.005749941, 0.18468782],
    [-0.006545117, 0.0005134335, 0.016347691],
    [-0.0144026885, 0.008917116, 0.08123935],
    [-0.021652648, 0.003262639, 0.12088381],
    [-0.02449421, -0.0067691803, 0.14767332],
    [-0.025248077, -0.019218326, 0.16631646],
    [-0.018981365, -0.0024781534, 0.01521358],
    [-0.03322176, -0.0032678149, 0.07651995],
    [-0.041671474, -0.009380937, 0.10488309],
    [-0.04358247, -0.017903566, 0.12083498],
    [-0.046035293, -0.02837658, 0.13528991],
];

/// Idle pose rotations for the left hand (24 segments, `[x, y, z, w]`). From
/// `FingerPosePresets.cs:18-41`. Entry 5 (`LeftIndexFinger_Proximal`) uses the C# expression
/// `MathF.PI * -52f / 165f` in the original; the f32 value is reproduced verbatim.
const IDLE_ROT_LEFT: [[f32; 4]; SEGMENT_COUNT] = [
    [0.27903825, -0.35667107, 0.5716526, -0.6842053],
    [-0.19714294, 0.27205768, -0.6480588, 0.68347573],
    [0.2759179, -0.3647225, 0.6079112, -0.64906913],
    [-0.27591783, 0.36472252, -0.6079112, 0.6490691],
    [-0.078234985, 0.08158678, 0.1385611, 0.98388195],
    [-0.04951689, -0.044297803, -0.12382177, -0.9900171],
    [-0.09054808, -0.05826731, -0.07661092, -0.9912299],
    [-0.11424534, -0.0379187, -0.070717074, -0.9902068],
    [-0.11424534, -0.03791874, -0.07071708, -0.99020666],
    [-0.101975575, 0.012416848, -0.0062540215, 0.9946898],
    [-0.021931179, 0.02829326, -0.16934969, -0.98490566],
    [-0.14726357, -0.008155652, -0.15507816, -0.9768304],
    [-0.19537397, 0.00020374882, -0.15417777, -0.9685341],
    [-0.23407964, -0.00598288, -0.15406176, -0.95991457],
    [-0.0803036, -0.052303787, 0.043042615, 0.9944662],
    [-0.05435912, 0.099860825, -0.16207728, -0.9802061],
    [-0.16581841, 0.080944, -0.17411849, -0.9672824],
    [-0.28148022, 0.072037935, -0.18077147, -0.93962806],
    [-0.28148016, 0.07203797, -0.18077146, -0.9396281],
    [-0.034600668, -0.114388935, 0.15665136, 0.98039705],
    [-0.06295043, 0.16357478, -0.25207797, -0.9517022],
    [-0.21643831, 0.12078378, -0.27558473, -0.9287727],
    [-0.2743513, 0.15363924, -0.26272792, -0.9121953],
    [-0.27435127, 0.15363923, -0.26272795, -0.9121953],
];

/// Idle pose positions for the right hand (24 segments). From `FingerPosePresets.cs:42-65`.
const IDLE_POS_RIGHT: [[f32; 3]; SEGMENT_COUNT] = [
    [-0.032747645, -0.014222979, 0.02505552],
    [-0.06477464, -0.016079383, 0.049621053],
    [-0.083759174, -0.019294739, 0.075823866],
    [-0.10661364, -0.022266746, 0.09574591],
    [-0.026866153, 0.00063245, 0.015001944],
    [-0.035505224, 0.018309893, 0.08672529],
    [-0.039789163, 0.01391685, 0.13022469],
    [-0.043620188, 0.0077228546, 0.15798062],
    [-0.045881294, 0.0012847185, 0.17975795],
    [-0.0071195387, 0.0021773104, 0.016318731],
    [-0.007973376, 0.016047278, 0.085908964],
    [-0.0066327956, 0.013078213, 0.1288943],
    [-0.009434605, 0.0023156404, 0.16024621],
    [-0.0121380035, -0.009670734, 0.18303739],
    [0.0065451185, 0.00051343645, 0.016347695],
    [0.014490828, 0.0090670725, 0.081209116],
    [0.020563135, 0.0024371147, 0.12090023],
    [0.022103736, -0.008921146, 0.1472631],
    [0.021454168, -0.022775173, 0.16489148],
    [0.01898137, -0.0024781493, 0.015213581],
    [0.032782484, -0.004373392, 0.07659627],
    [0.039951764, -0.01097393, 0.10520153],
    [0.040844504, -0.019955635, 0.120990306],
    [0.04168672, -0.031573057, 0.13473694],
];

/// Idle pose rotations for the right hand (24 segments, `[x, y, z, w]`). From
/// `FingerPosePresets.cs:42-65`.
const IDLE_ROT_RIGHT: [[f32; 4]; SEGMENT_COUNT] = [
    [0.26789832, 0.3524874, -0.58295935, -0.68127936],
    [-0.17208317, -0.25975865, 0.6496945, 0.69340456],
    [-0.24061818, -0.33929846, 0.61731255, 0.6677608],
    [-0.24061815, -0.33929846, 0.61731255, 0.6677609],
    [0.07749181, 0.08134778, 0.13846457, -0.9839742],
    [0.055702735, -0.04241033, -0.121478446, 0.99012196],
    [0.113620095, -0.05884121, -0.0754011, 0.9889099],
    [0.1458252, -0.04010107, -0.06851033, 0.9861202],
    [0.14582519, -0.04010112, -0.06851033, 0.9861203],
    [0.10365219, 0.012517784, -0.0062292945, -0.99451536],
    [0.031353656, 0.021133391, -0.1680567, 0.98505193],
    [0.16879009, -0.016700286, -0.15316582, 0.9735355],
    [0.24419676, -0.016012125, -0.15098418, 0.95776576],
    [0.28243104, -0.022057567, -0.15022011, 0.9471958],
    [0.081733055, -0.052710798, 0.047065213, -0.994146],
    [0.06745499, 0.088334866, -0.1750841, 0.9782599],
    [0.19286993, 0.065328054, -0.18660586, 0.96109915],
    [0.32311887, 0.051316313, -0.19176209, 0.92530435],
    [0.3231189, 0.051316313, -0.19176205, 0.92530435],
    [0.02579043, -0.11239051, 0.1609593, -0.9802017],
    [0.07509767, 0.14515463, -0.25997773, 0.9516839],
    [0.23698953, 0.0987691, -0.28174534, 0.9245],
    [0.3220941, 0.1215931, -0.26720595, 0.90003973],
    [0.32209414, 0.12159315, -0.26720595, 0.9000397],
];

/// Fist pose positions for the left hand (24 segments). From `FingerPosePresets.cs:67-90`.
const FIST_POS_LEFT: [[f32; 3]; SEGMENT_COUNT] = [
    [0.026763892, -0.0063486164, 0.01744702],
    [0.06192772, -0.008122697, 0.037271187],
    [0.06832408, -0.0105211735, 0.06906241],
    [0.0545115, -0.017816544, 0.095216505],
    [0.02152614, 0.003795074, 0.012808277],
    [0.031072546, 0.014069826, 0.08584704],
    [0.038632516, -0.02815175, 0.09168206],
    [0.03938142, -0.024447203, 0.06365971],
    [0.034943886, -0.0020633936, 0.06395769],
    [0.0068064122, 0.005786924, 0.01653391],
    [0.0076076197, 0.011343788, 0.08727571],
    [0.013980356, -0.030851126, 0.0933844],
    [0.01455404, -0.029297113, 0.060159665],
    [0.010745675, -0.004100561, 0.06474727],
    [-0.006858267, 0.004123045, 0.016562855],
    [-0.013476995, 0.003979577, 0.08213312],
    [-0.009529918, -0.035460234, 0.08958446],
    [-0.005206241, -0.036921382, 0.06146362],
    [-0.0085884165, -0.014979601, 0.06466259],
    [-0.019290796, 0.0010885983, 0.015426192],
    [-0.033806834, -0.0069735213, 0.076140136],
    [-0.030296907, -0.035037875, 0.085772604],
    [-0.025613647, -0.039969206, 0.069127135],
    [-0.027790949, -0.022555828, 0.065042846],
];

/// Fist pose rotations for the left hand (24 segments, `[x, y, z, w]`). From
/// `FingerPosePresets.cs:67-90`.
const FIST_ROT_LEFT: [[f32; 4]; SEGMENT_COUNT] = [
    [0.26891267, -0.42705113, 0.4962459, -0.70643705],
    [-0.0299031, 0.10131291, -0.59843856, 0.7941744],
    [-0.2432386, 0.1075707, 0.5974826, -0.7564906],
    [-0.24323854, 0.10757066, 0.5974827, -0.75649065],
    [-0.026341768, 0.08023323, 0.14190729, 0.9862713],
    [-0.65643066, -0.041254308, -0.085990064, -0.7483333],
    [-0.99288625, -0.098439865, -0.019689739, 0.06402316],
    [-0.6972107, -0.08586568, 0.05203204, 0.70980066],
    [0.6972108, 0.08586566, -0.05203202, -0.70980054],
    [-0.044604156, 0.011658788, -0.030069094, 0.9984841],
    [0.6538034, 0.041070007, 0.06577383, 0.75268066],
    [-0.99700874, -0.07316653, -0.01031061, 0.022671701],
    [-0.6387442, -0.058438186, 0.045063965, 0.76587224],
    [0.60749865, 0.0565829, -0.047372527, -0.7908854],
    [-0.013118888, -0.045193814, 0.011488465, 0.998826],
    [-0.63827336, 0.015155507, -0.09480049, -0.76379985],
    [-0.99475336, -0.06331103, -0.07434321, -0.03050795],
    [-0.6478517, -0.094741255, 0.0058441795, 0.7558296],
    [0.64785177, 0.09474128, -0.005844188, -0.75582945],
    [0.02653118, -0.12396325, 0.12497428, 0.9840278],
    [-0.5812628, 0.030625414, -0.14322409, -0.8004265],
    [-0.9788216, -0.06903386, -0.12256045, -0.14873403],
    [-0.7693869, -0.14622758, -0.03941031, 0.6205709],
    [-0.7693869, -0.14622758, -0.03941031, 0.6205709],
];

/// Fist pose positions for the right hand (24 segments). From `FingerPosePresets.cs:91-114`.
const FIST_POS_RIGHT: [[f32; 3]; SEGMENT_COUNT] = [
    [-0.027928837, -0.0065499977, 0.018533913],
    [-0.062129922, -0.00902483, 0.039906792],
    [-0.06678565, -0.011821985, 0.07196686],
    [-0.055600565, -0.018444896, 0.09951828],
    [-0.021514188, 0.0038021517, 0.012803375],
    [-0.03106215, 0.014058205, 0.085844465],
    [-0.038642187, -0.028236508, 0.09108152],
    [-0.039264888, -0.02384913, 0.063155666],
    [-0.03482183, -0.0014804602, 0.06400136],
    [-0.006806411, 0.005786925, 0.016533913],
    [-0.007436211, 0.01125816, 0.0872841],
    [-0.014006828, -0.030997515, 0.09272742],
    [-0.014592892, -0.029072523, 0.059522375],
    [-0.01066705, -0.0039459467, 0.064386584],
    [0.0068582585, 0.004123048, 0.016562866],
    [0.013470863, 0.0037285234, 0.082132846],
    [0.009370286, -0.03586018, 0.08865387],
    [0.0049451236, -0.036629677, 0.06052139],
    [0.008437558, -0.014785886, 0.0642316],
    [0.019294508, 0.0011314582, 0.015428751],
    [0.033720512, -0.007342328, 0.07610809],
    [0.029749613, -0.03583622, 0.084157854],
    [0.025065007, -0.039703965, 0.06723678],
    [0.02759617, -0.022073507, 0.06451594],
];

/// Fist pose rotations for the right hand (24 segments, `[x, y, z, w]`). From
/// `FingerPosePresets.cs:91-114`.
const FIST_ROT_RIGHT: [[f32; 4]; SEGMENT_COUNT] = [
    [0.25109518, 0.41529992, -0.5051993, -0.71361816],
    [-0.009018947, -0.08332496, 0.6021248, 0.7939908],
    [0.20381555, 0.07913755, 0.6027624, 0.7673814],
    [0.20381561, 0.07913753, 0.6027624, 0.7673813],
    [0.026214743, 0.080225475, 0.14191456, -0.9862743],
    [0.66164654, -0.04163931, -0.085522085, 0.74375814],
    [0.99204785, -0.09829777, -0.018667007, -0.07635586],
    [0.6886448, -0.08510799, 0.052603662, -0.71816283],
    [-0.6886448, 0.08510799, -0.052603662, 0.71816283],
    [0.043970812, 0.010536079, -0.0280599, -0.9985832],
    [0.65961385, -0.042108648, -0.06782819, 0.74735266],
    [0.9966953, -0.075386494, -0.01097153, -0.028200207],
    [0.6343669, -0.060378086, 0.046287294, -0.7692793],
    [-0.60298824, 0.058472194, -0.04867275, 0.7941141],
    [0.011215502, -0.04517185, 0.011574637, -0.9988492],
    [0.6472192, 0.016161142, -0.097422145, 0.7558806],
    [0.99477315, -0.06460124, -0.07686661, 0.018566478],
    [0.638573, -0.09755458, 0.0053005847, -0.7633346],
    [-0.638573, 0.09755457, -0.00530056, 0.7633345],
    [-0.030213088, -0.12350305, 0.122525044, -0.9842875],
    [0.6036873, 0.02883644, -0.14749046, 0.7829283],
    [0.98244506, -0.073423475, -0.123744294, 0.118735366],
    [0.7436073, -0.1501721, -0.036904518, -0.6504881],
    [0.7436072, -0.15017207, -0.03690451, -0.650488],
];

/// Controller-derived inputs used to drive the idle↔fist blend.
struct ControllerCurlInputs {
    /// Which hand this controller drives.
    side: Chirality,
    /// Wrist position to report on [`HandState::wrist_position`]. Derived from `hand_position` if
    /// the runtime bound a hand; otherwise falls back to the grip-space position.
    wrist_position: Vec3,
    /// Wrist rotation to report on [`HandState::wrist_rotation`].
    wrist_rotation: Quat,
    /// Grip/squeeze analog in 0..=1. Drives middle, ring, and pinky curl.
    grip: f32,
    /// Trigger analog in 0..=1. Drives index finger curl.
    trigger: f32,
}

/// Extracts the curl-driving inputs from a [`VRControllerState`] variant.
///
/// Returns `None` when the controller is not tracked (we do not want to feed the host random
/// hand poses). For controllers whose grip is a boolean (Vive wand, WMR), the boolean is
/// coerced to `0.0` / `1.0`.
fn extract_curl_inputs(controller: &VRControllerState) -> Option<ControllerCurlInputs> {
    match controller {
        VRControllerState::TouchControllerState(s) => {
            if !s.is_tracking {
                return None;
            }
            let (wrist_position, wrist_rotation) = if s.has_bound_hand {
                (s.hand_position, s.hand_rotation)
            } else {
                (s.position, s.rotation)
            };
            Some(ControllerCurlInputs {
                side: s.side,
                wrist_position,
                wrist_rotation,
                grip: s.grip.clamp(0.0, 1.0),
                trigger: s.trigger.clamp(0.0, 1.0),
            })
        }
        VRControllerState::IndexControllerState(s) => {
            if !s.is_tracking {
                return None;
            }
            let (wrist_position, wrist_rotation) = if s.has_bound_hand {
                (s.hand_position, s.hand_rotation)
            } else {
                (s.position, s.rotation)
            };
            Some(ControllerCurlInputs {
                side: s.side,
                wrist_position,
                wrist_rotation,
                grip: s.grip.clamp(0.0, 1.0),
                trigger: s.trigger.clamp(0.0, 1.0),
            })
        }
        VRControllerState::ViveControllerState(s) => {
            if !s.is_tracking {
                return None;
            }
            let (wrist_position, wrist_rotation) = if s.has_bound_hand {
                (s.hand_position, s.hand_rotation)
            } else {
                (s.position, s.rotation)
            };
            Some(ControllerCurlInputs {
                side: s.side,
                wrist_position,
                wrist_rotation,
                grip: if s.grip { 1.0 } else { 0.0 },
                trigger: s.trigger.clamp(0.0, 1.0),
            })
        }
        VRControllerState::WindowsMRControllerState(s) => {
            if !s.is_tracking {
                return None;
            }
            let (wrist_position, wrist_rotation) = if s.has_bound_hand {
                (s.hand_position, s.hand_rotation)
            } else {
                (s.position, s.rotation)
            };
            Some(ControllerCurlInputs {
                side: s.side,
                wrist_position,
                wrist_rotation,
                grip: if s.grip { 1.0 } else { 0.0 },
                trigger: s.trigger.clamp(0.0, 1.0),
            })
        }
        VRControllerState::CosmosControllerState(_)
        | VRControllerState::GenericControllerState(_)
        | VRControllerState::HPReverbControllerState(_)
        | VRControllerState::PicoNeo2ControllerState(_) => {
            // These variants are not produced by the current OpenXR dispatch
            // (`crate::xr::input::state::dispatch_openxr_profile_to_host_state`). If they start
            // being emitted, add the analogous extractor here.
            None
        }
    }
}

/// Returns the idle↔fist blend factor for a given segment index.
///
/// - Thumb and metacarpals are held at idle (`0.0`). Non-thumb metacarpals are overridden on the
///   host anyway when [`HandState::tracks_metacarpals`] is `false`, so their blend does not matter.
/// - Index curl follows `trigger`.
/// - Middle, ring, and pinky curl follow `grip`.
fn blend_factor_for_segment(index: usize, grip: f32, trigger: f32) -> f32 {
    match finger_kind_for_segment(index) {
        FingerKind::Thumb => 0.0,
        FingerKind::Index => trigger,
        FingerKind::Middle | FingerKind::Ring | FingerKind::Pinky => grip,
    }
}

/// Builds a [`HandState`] for one controller by blending the idle and fist presets. Returns
/// `None` if the controller is untracked or not a variant we drive hands for.
fn synthesize_one_hand(controller: &VRControllerState) -> Option<HandState> {
    let inputs = extract_curl_inputs(controller)?;
    let (pos_idle, rot_idle, pos_fist, rot_fist, unique_id) = match inputs.side {
        Chirality::Left => (
            &IDLE_POS_LEFT,
            &IDLE_ROT_LEFT,
            &FIST_POS_LEFT,
            &FIST_ROT_LEFT,
            LEFT_HAND_ID,
        ),
        Chirality::Right => (
            &IDLE_POS_RIGHT,
            &IDLE_ROT_RIGHT,
            &FIST_POS_RIGHT,
            &FIST_ROT_RIGHT,
            RIGHT_HAND_ID,
        ),
    };
    let mut segment_positions = Vec::with_capacity(SEGMENT_COUNT);
    let mut segment_rotations = Vec::with_capacity(SEGMENT_COUNT);
    for i in 0..SEGMENT_COUNT {
        let t = blend_factor_for_segment(i, inputs.grip, inputs.trigger);
        let pi = Vec3::from_array(pos_idle[i]);
        let pf = Vec3::from_array(pos_fist[i]);
        let ri = Quat::from_array(rot_idle[i]);
        let rf = Quat::from_array(rot_fist[i]);
        segment_positions.push(pi.lerp(pf, t));
        segment_rotations.push(ri.slerp(rf, t));
    }
    Some(HandState {
        unique_id: Some(unique_id.to_string()),
        priority: 0,
        chirality: inputs.side,
        is_device_active: true,
        is_tracking: true,
        tracks_metacarpals: false,
        confidence: 1.0,
        wrist_position: inputs.wrist_position,
        wrist_rotation: inputs.wrist_rotation,
        segment_positions,
        segment_rotations,
    })
}

/// Produces one [`HandState`] per tracked VR controller in `controllers`, blending the idle and
/// fist presets using the controller's grip and trigger analogs.
///
/// Call this every XR frame after building the per-controller [`VRControllerState`] slice; the
/// returned vector belongs on [`crate::shared::VRInputsState::hands`].
pub fn synthesize_hand_states(controllers: &[VRControllerState]) -> Vec<HandState> {
    controllers.iter().filter_map(synthesize_one_hand).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::{BodyNode, TouchControllerModel, TouchControllerState};

    fn touch_controller(
        side: Chirality,
        is_tracking: bool,
        grip: f32,
        trigger: f32,
    ) -> VRControllerState {
        VRControllerState::TouchControllerState(TouchControllerState {
            model: TouchControllerModel::QuestAndRiftS,
            start: false,
            button_yb: false,
            button_xa: false,
            button_yb_touch: false,
            button_xa_touch: false,
            thumbrest_touch: false,
            grip,
            grip_click: false,
            joystick_raw: glam::Vec2::ZERO,
            joystick_touch: false,
            joystick_click: false,
            trigger,
            trigger_touch: false,
            trigger_click: false,
            device_id: None,
            device_model: None,
            side,
            body_node: match side {
                Chirality::Left => BodyNode::LeftController,
                Chirality::Right => BodyNode::RightController,
            },
            is_device_active: true,
            is_tracking,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            has_bound_hand: false,
            hand_position: Vec3::ZERO,
            hand_rotation: Quat::IDENTITY,
            battery_level: 1.0,
            battery_charging: false,
        })
    }

    #[test]
    fn produces_one_hand_per_tracked_controller() {
        let controllers = vec![
            touch_controller(Chirality::Left, true, 0.0, 0.0),
            touch_controller(Chirality::Right, true, 0.0, 0.0),
        ];
        let hands = synthesize_hand_states(&controllers);
        assert_eq!(hands.len(), 2);
        assert_eq!(hands[0].chirality, Chirality::Left);
        assert_eq!(hands[1].chirality, Chirality::Right);
    }

    #[test]
    fn skips_untracked_controllers() {
        let controllers = vec![
            touch_controller(Chirality::Left, false, 0.0, 0.0),
            touch_controller(Chirality::Right, true, 0.0, 0.0),
        ];
        let hands = synthesize_hand_states(&controllers);
        assert_eq!(hands.len(), 1);
        assert_eq!(hands[0].chirality, Chirality::Right);
    }

    #[test]
    fn segment_arrays_have_host_expected_length() {
        let hands = synthesize_hand_states(&[touch_controller(Chirality::Left, true, 0.5, 0.5)]);
        let hand = &hands[0];
        assert_eq!(hand.segment_positions.len(), SEGMENT_COUNT);
        assert_eq!(hand.segment_rotations.len(), SEGMENT_COUNT);
        assert!(hand.is_tracking);
        assert!(!hand.tracks_metacarpals);
    }

    #[test]
    fn trigger_bends_index_but_not_other_fingers() {
        let idle =
            synthesize_hand_states(&[touch_controller(Chirality::Left, true, 0.0, 0.0)]).remove(0);
        let full_trigger =
            synthesize_hand_states(&[touch_controller(Chirality::Left, true, 0.0, 1.0)]).remove(0);
        let index_delta = (full_trigger.segment_rotations[6].to_array()[3]
            - idle.segment_rotations[6].to_array()[3])
            .abs();
        let middle_delta = (full_trigger.segment_rotations[11].to_array()[3]
            - idle.segment_rotations[11].to_array()[3])
            .abs();
        let thumb_delta = (full_trigger.segment_rotations[1].to_array()[3]
            - idle.segment_rotations[1].to_array()[3])
            .abs();
        assert!(
            index_delta > 0.05,
            "trigger should bend the index finger proximal joint (delta={index_delta})"
        );
        assert!(
            middle_delta < 1e-5,
            "trigger must not move the middle finger (delta={middle_delta})"
        );
        assert!(
            thumb_delta < 1e-5,
            "trigger must not move the thumb (delta={thumb_delta})"
        );
    }

    #[test]
    fn grip_bends_middle_ring_pinky_but_not_index_or_thumb() {
        let idle =
            synthesize_hand_states(&[touch_controller(Chirality::Left, true, 0.0, 0.0)]).remove(0);
        let full_grip =
            synthesize_hand_states(&[touch_controller(Chirality::Left, true, 1.0, 0.0)]).remove(0);
        let middle_delta = (full_grip.segment_rotations[11].to_array()[3]
            - idle.segment_rotations[11].to_array()[3])
            .abs();
        let ring_delta = (full_grip.segment_rotations[16].to_array()[3]
            - idle.segment_rotations[16].to_array()[3])
            .abs();
        let pinky_delta = (full_grip.segment_rotations[21].to_array()[3]
            - idle.segment_rotations[21].to_array()[3])
            .abs();
        let index_delta = (full_grip.segment_rotations[6].to_array()[3]
            - idle.segment_rotations[6].to_array()[3])
            .abs();
        let thumb_delta = (full_grip.segment_rotations[1].to_array()[3]
            - idle.segment_rotations[1].to_array()[3])
            .abs();
        assert!(
            middle_delta > 0.05,
            "grip should bend middle (delta={middle_delta})"
        );
        assert!(
            ring_delta > 0.05,
            "grip should bend ring (delta={ring_delta})"
        );
        assert!(
            pinky_delta > 0.05,
            "grip should bend pinky (delta={pinky_delta})"
        );
        assert!(
            index_delta < 1e-5,
            "grip must not move the index finger (delta={index_delta})"
        );
        assert!(
            thumb_delta < 1e-5,
            "grip must not move the thumb (delta={thumb_delta})"
        );
    }

    #[test]
    fn left_and_right_hands_differ() {
        let hands = synthesize_hand_states(&[
            touch_controller(Chirality::Left, true, 0.5, 0.5),
            touch_controller(Chirality::Right, true, 0.5, 0.5),
        ]);
        let left_index_met_x = hands[0].segment_positions[4].x;
        let right_index_met_x = hands[1].segment_positions[4].x;
        assert!(
            (left_index_met_x - right_index_met_x).abs() > 1e-4,
            "left/right hand index metacarpals must use different preset data"
        );
        assert!(
            left_index_met_x.signum() != right_index_met_x.signum(),
            "left hand metacarpal x should be positive, right hand negative \
             (left={left_index_met_x}, right={right_index_met_x})"
        );
        assert_eq!(
            hands[0].unique_id.as_deref(),
            Some(LEFT_HAND_ID),
            "left hand should use stable LEFT_HAND_ID"
        );
        assert_eq!(
            hands[1].unique_id.as_deref(),
            Some(RIGHT_HAND_ID),
            "right hand should use stable RIGHT_HAND_ID"
        );
    }

    #[test]
    fn thumb_metacarpal_always_at_idle_pose() {
        // Thumb is never blended, so thumb metacarpal position (segment 0) should always match the
        // idle preset regardless of grip/trigger.
        let hands = synthesize_hand_states(&[touch_controller(Chirality::Left, true, 1.0, 1.0)]);
        let expected = Vec3::from_array(IDLE_POS_LEFT[0]);
        let actual = hands[0].segment_positions[0];
        assert!(
            (actual - expected).length() < 1e-6,
            "thumb metacarpal should stay at idle when grip=1, trigger=1"
        );
    }
}
