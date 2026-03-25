//! Host contract for native WGSL UI materials (`UI_Unlit`, `UI_TextUnlit`).
//!
//! Shader asset IDs are assigned by the host at runtime. Configure them under `[rendering]` in
//! `configuration.ini` (see [`crate::config::RenderConfig`]) or via environment variables so the
//! renderer maps `set_shader` batches to [`NativeUiShaderFamily`].
//!
//! Material **property IDs** are also host-assigned (`MaterialProperty` indices on the host).
//! When a property id is `-1`, that channel is skipped and the GPU uniform uses a documented default.
//! Populate ids from host logs or a future `material_property_id_request` / `MaterialPropertyIdResult` flow.

/// Identifies which native UI WGSL program to use for a host shader asset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NativeUiShaderFamily {
    /// Resonite `UI/Unlit` ([`third_party/Resonite.UnityShaders/.../UI_Unlit.shader`](../../../../third_party/Resonite.UnityShaders/Assets/Shaders/UI/UI_Unlit.shader)).
    UiUnlit,
    /// Resonite `UI/Text/Unlit`.
    UiTextUnlit,
}

/// Resolves `shader_asset_id` to a native UI family using configured allowlist ids.
pub fn native_ui_family_for_shader(
    shader_asset_id: i32,
    ui_unlit_id: i32,
    ui_text_unlit_id: i32,
) -> Option<NativeUiShaderFamily> {
    if ui_unlit_id >= 0 && shader_asset_id == ui_unlit_id {
        return Some(NativeUiShaderFamily::UiUnlit);
    }
    if ui_text_unlit_id >= 0 && shader_asset_id == ui_text_unlit_id {
        return Some(NativeUiShaderFamily::UiTextUnlit);
    }
    None
}

/// Infers [`NativeUiShaderFamily`] from the host shader upload string (`ShaderUpload.file`: path or label).
///
/// Matches path fragments such as `UI/Unlit`, `UI_Unlit`, `UI/Text/Unlit` (text is checked before unlit).
pub fn native_ui_family_from_shader_path_hint(hint: &str) -> Option<NativeUiShaderFamily> {
    let h = hint.to_ascii_lowercase();
    if h.contains("ui/text") && (h.contains("unlit") || h.contains("textunlit")) {
        return Some(NativeUiShaderFamily::UiTextUnlit);
    }
    if h.contains("ui/unlit") || h.contains("ui_unlit") || h.contains("uiunlit") {
        return Some(NativeUiShaderFamily::UiUnlit);
    }
    None
}

/// Resolves native UI shader family using configured allowlist ids, then the shader registry path hint.
pub fn resolve_native_ui_shader_family(
    shader_asset_id: i32,
    native_ui_unlit_shader_id: i32,
    native_ui_text_unlit_shader_id: i32,
    registry: &super::AssetRegistry,
) -> Option<NativeUiShaderFamily> {
    native_ui_family_for_shader(
        shader_asset_id,
        native_ui_unlit_shader_id,
        native_ui_text_unlit_shader_id,
    )
    .or_else(|| {
        registry
            .get_shader(shader_asset_id)
            .and_then(|s| s.wgsl_source.as_deref())
            .and_then(native_ui_family_from_shader_path_hint)
    })
}

/// Property id map for `UI_Unlit` material batches. `-1` = omit (use GPU default).
#[derive(Clone, Debug)]
pub struct UiUnlitPropertyIds {
    /// `_Tint` (float4 linear color).
    pub tint: i32,
    /// `_OverlayTint` (float4).
    pub overlay_tint: i32,
    /// `_Cutoff` (float).
    pub cutoff: i32,
    /// `_Rect` min/max extents (float4), used with rect clip flag.
    pub rect: i32,
    /// `_MainTex_ST` scale.xy offset.zw (float4) or two floats — we expect float4 from host.
    pub main_tex_st: i32,
    /// `_MaskTex_ST` (float4).
    pub mask_tex_st: i32,
    /// `_MainTex` texture (`set_texture` packed id).
    pub main_tex: i32,
    /// `_MaskTex` texture.
    pub mask_tex: i32,
    /// Keyword-style flags sent as floats (0/1) when the host uses dedicated property ids.
    pub alphaclip: i32,
    pub rectclip: i32,
    pub overlay: i32,
    pub texture_normalmap: i32,
    pub texture_lerpcolor: i32,
    pub mask_texture_mul: i32,
    pub mask_texture_clip: i32,
}

impl Default for UiUnlitPropertyIds {
    fn default() -> Self {
        Self {
            tint: -1,
            overlay_tint: -1,
            cutoff: -1,
            rect: -1,
            main_tex_st: -1,
            mask_tex_st: -1,
            main_tex: -1,
            mask_tex: -1,
            alphaclip: -1,
            rectclip: -1,
            overlay: -1,
            texture_normalmap: -1,
            texture_lerpcolor: -1,
            mask_texture_mul: -1,
            mask_texture_clip: -1,
        }
    }
}

/// Property id map for `UI_TextUnlit`.
#[derive(Clone, Debug)]
pub struct UiTextUnlitPropertyIds {
    pub tint_color: i32,
    pub overlay_tint: i32,
    pub outline_color: i32,
    pub background_color: i32,
    pub range: i32,
    pub face_dilate: i32,
    pub face_softness: i32,
    pub outline_size: i32,
    pub rect: i32,
    pub font_atlas: i32,
    pub raster: i32,
    pub sdf: i32,
    pub msdf: i32,
    pub outline: i32,
    pub rectclip: i32,
    pub overlay: i32,
}

impl Default for UiTextUnlitPropertyIds {
    fn default() -> Self {
        Self {
            tint_color: -1,
            overlay_tint: -1,
            outline_color: -1,
            background_color: -1,
            range: -1,
            face_dilate: -1,
            face_softness: -1,
            outline_size: -1,
            rect: -1,
            font_atlas: -1,
            raster: -1,
            sdf: -1,
            msdf: -1,
            outline: -1,
            rectclip: -1,
            overlay: -1,
        }
    }
}

/// GPU-packed flags for `UI_Unlit` (single u32 in uniform block).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UiUnlitFlags {
    pub alphaclip: bool,
    pub rectclip: bool,
    pub overlay: bool,
    pub texture_normalmap: bool,
    pub texture_lerpcolor: bool,
    pub mask_texture_mul: bool,
    pub mask_texture_clip: bool,
}

impl UiUnlitFlags {
    /// Packs flags into a little-endian bitfield for WGSL `u32`.
    pub fn to_bits(self) -> u32 {
        let mut b = 0u32;
        if self.alphaclip {
            b |= 1;
        }
        if self.rectclip {
            b |= 2;
        }
        if self.overlay {
            b |= 4;
        }
        if self.texture_normalmap {
            b |= 8;
        }
        if self.texture_lerpcolor {
            b |= 16;
        }
        if self.mask_texture_mul {
            b |= 32;
        }
        if self.mask_texture_clip {
            b |= 64;
        }
        b
    }
}

/// CPU-side uniform data for `UI_Unlit` before upload (matches WGSL `UiUnlitMaterialUniform`).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UiUnlitMaterialUniform {
    pub tint: [f32; 4],
    pub overlay_tint: [f32; 4],
    pub main_tex_st: [f32; 4],
    pub mask_tex_st: [f32; 4],
    pub rect: [f32; 4],
    pub cutoff: f32,
    pub flags: u32,
    pub pad_tail: [u32; 2],
}

/// CPU-side uniform for `UI_TextUnlit`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UiTextUnlitMaterialUniform {
    pub tint_color: [f32; 4],
    pub overlay_tint: [f32; 4],
    pub outline_color: [f32; 4],
    pub background_color: [f32; 4],
    pub range_xy: [f32; 4],
    pub face_dilate: f32,
    pub face_softness: f32,
    pub outline_size: f32,
    pub pad_scalar: f32,
    pub rect: [f32; 4],
    /// Lower bits: mode 0=raster,1=sdf,2=msdf; bit 8 outline; bit 9 rectclip; bit 10 overlay.
    pub flags: u32,
    pub pad_flags: u32,
    pub pad_tail: [u32; 2],
}

fn float4(
    store: &super::MaterialPropertyStore,
    block: i32,
    pid: i32,
    default: [f32; 4],
) -> [f32; 4] {
    if pid < 0 {
        return default;
    }
    match store.get(block, pid) {
        Some(super::MaterialPropertyValue::Float4(v)) => *v,
        _ => default,
    }
}

fn float1(store: &super::MaterialPropertyStore, block: i32, pid: i32, default: f32) -> f32 {
    if pid < 0 {
        return default;
    }
    match store.get(block, pid) {
        Some(super::MaterialPropertyValue::Float(v)) => *v,
        _ => default,
    }
}

fn flag_f(store: &super::MaterialPropertyStore, block: i32, pid: i32) -> bool {
    if pid < 0 {
        return false;
    }
    matches!(
        store.get(block, pid),
        Some(super::MaterialPropertyValue::Float(f)) if *f >= 0.5
    )
}

/// Builds GPU uniform and texture handles for `UI_Unlit` from the property store.
pub fn ui_unlit_material_uniform(
    store: &super::MaterialPropertyStore,
    block_id: i32,
    ids: &UiUnlitPropertyIds,
) -> (UiUnlitMaterialUniform, i32, i32) {
    let tint = float4(store, block_id, ids.tint, [1.0, 1.0, 1.0, 1.0]);
    let overlay_tint = float4(store, block_id, ids.overlay_tint, [1.0, 1.0, 1.0, 0.73]);
    let cutoff = float1(store, block_id, ids.cutoff, 0.98);
    let main_tex_st = float4(store, block_id, ids.main_tex_st, [1.0, 1.0, 0.0, 0.0]);
    let mask_tex_st = float4(store, block_id, ids.mask_tex_st, [1.0, 1.0, 0.0, 0.0]);
    let rect = float4(store, block_id, ids.rect, [0.0, 0.0, 1.0, 1.0]);
    let flags = UiUnlitFlags {
        alphaclip: flag_f(store, block_id, ids.alphaclip),
        rectclip: flag_f(store, block_id, ids.rectclip),
        overlay: flag_f(store, block_id, ids.overlay),
        texture_normalmap: flag_f(store, block_id, ids.texture_normalmap),
        texture_lerpcolor: flag_f(store, block_id, ids.texture_lerpcolor),
        mask_texture_mul: flag_f(store, block_id, ids.mask_texture_mul),
        mask_texture_clip: flag_f(store, block_id, ids.mask_texture_clip),
    };
    let main_tex = texture_handle(store, block_id, ids.main_tex);
    let mask_tex = texture_handle(store, block_id, ids.mask_tex);
    let u = UiUnlitMaterialUniform {
        tint,
        overlay_tint,
        main_tex_st,
        mask_tex_st,
        rect,
        cutoff,
        flags: flags.to_bits(),
        pad_tail: [0; 2],
    };
    (u, main_tex, mask_tex)
}

fn texture_handle(store: &super::MaterialPropertyStore, block: i32, pid: i32) -> i32 {
    if pid < 0 {
        return 0;
    }
    match store.get(block, pid) {
        Some(super::MaterialPropertyValue::Texture(h)) => *h,
        _ => 0,
    }
}

/// Builds uniform and font atlas handle for `UI_TextUnlit`.
pub fn ui_text_unlit_material_uniform(
    store: &super::MaterialPropertyStore,
    block_id: i32,
    ids: &UiTextUnlitPropertyIds,
) -> (UiTextUnlitMaterialUniform, i32) {
    let tint_color = float4(store, block_id, ids.tint_color, [1.0, 1.0, 1.0, 1.0]);
    let overlay_tint = float4(store, block_id, ids.overlay_tint, [1.0, 1.0, 1.0, 0.73]);
    let outline_color = float4(store, block_id, ids.outline_color, [1.0, 1.0, 1.0, 0.0]);
    let background_color = float4(store, block_id, ids.background_color, [0.0, 0.0, 0.0, 0.0]);
    let range_v = float4(store, block_id, ids.range, [0.001, 0.001, 0.0, 0.0]);
    let face_dilate = float1(store, block_id, ids.face_dilate, 0.0);
    let face_softness = float1(store, block_id, ids.face_softness, 0.0);
    let outline_size = float1(store, block_id, ids.outline_size, 0.0);
    let rect = float4(store, block_id, ids.rect, [0.0, 0.0, 1.0, 1.0]);
    let mut mode: u32 = 0;
    if flag_f(store, block_id, ids.sdf) {
        mode = 1;
    }
    if flag_f(store, block_id, ids.msdf) {
        mode = 2;
    }
    if flag_f(store, block_id, ids.raster) {
        mode = 0;
    }
    let mut flags = mode & 3;
    if flag_f(store, block_id, ids.outline) {
        flags |= 1 << 8;
    }
    if flag_f(store, block_id, ids.rectclip) {
        flags |= 1 << 9;
    }
    if flag_f(store, block_id, ids.overlay) {
        flags |= 1 << 10;
    }
    let font_atlas = texture_handle(store, block_id, ids.font_atlas);
    let u = UiTextUnlitMaterialUniform {
        tint_color,
        overlay_tint,
        outline_color,
        background_color,
        range_xy: [range_v[0], range_v[1], range_v[2], range_v[3]],
        face_dilate,
        face_softness,
        outline_size,
        pad_scalar: 0.0,
        rect,
        flags,
        pad_flags: 0,
        pad_tail: [0; 2],
    };
    (u, font_atlas)
}
