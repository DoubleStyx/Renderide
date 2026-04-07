#define_import_path renderide_ui_common

fn rect_contains(local_xy: vec2f, rect: vec4f) -> bool {
    let min_v = rect.xy;
    let max_v = rect.xy + rect.zw;
    return local_xy.x >= min_v.x && local_xy.x <= max_v.x && local_xy.y >= min_v.y && local_xy.y <= max_v.y;
}

fn apply_st(uv: vec2f, st: vec4f) -> vec2f {
    return uv * st.xy + st.zw;
}

fn overlay_ndc_xy(frag_xy: vec2f, dims: vec2u) -> vec2f {
    let w = max(f32(dims.x), 1.0);
    let h = max(f32(dims.y), 1.0);
    let x = (frag_xy.x + 0.5) / w * 2.0 - 1.0;
    let y = 1.0 - (frag_xy.y + 0.5) / h * 2.0;
    return vec2f(x, y);
}

fn overlay_view_z_from_depth(d: f32, ndc_xy: vec2f, inv_proj: mat4x4f) -> f32 {
    let ndc_z = d * 2.0 - 1.0;
    let h = inv_proj * vec4f(ndc_xy.x, ndc_xy.y, ndc_z, 1.0);
    let v = h.xyz / max(h.w, 1e-8);
    return v.z;
}
