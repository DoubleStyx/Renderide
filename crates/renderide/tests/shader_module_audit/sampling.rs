//! Raw texture-sampling audits for material roots and shared shader modules.

use super::*;

#[derive(Clone, Copy)]
struct RawSampleException {
    label: &'static str,
    builtin: &'static str,
    texture: &'static str,
    reason: &'static str,
}

struct RawSampleCall {
    label: String,
    line: usize,
    builtin: &'static str,
    texture: String,
}

const RAW_SAMPLE_EXCEPTIONS: &[RawSampleException] = &[
    RawSampleException {
        label: "shaders/materials/billboardunlit.wgsl",
        builtin: "textureSampleGrad",
        texture: "_Tex",
        reason: "polar UV remapping supplies repaired gradients",
    },
    RawSampleException {
        label: "shaders/materials/depthprojection.wgsl",
        builtin: "textureSampleLevel",
        texture: "_DepthTex",
        reason: "vertex displacement depth samples intentionally use base LOD",
    },
    RawSampleException {
        label: "shaders/materials/fresnellerp.wgsl",
        builtin: "textureSampleGrad",
        texture: "_LerpTex",
        reason: "polar UV remapping supplies repaired gradients",
    },
    RawSampleException {
        label: "shaders/materials/grayscale.wgsl",
        builtin: "textureSampleLevel",
        texture: "_Gradient",
        reason: "gradient lookup intentionally samples the authored ramp base LOD",
    },
    RawSampleException {
        label: "shaders/materials/overlayfresnel.wgsl",
        builtin: "textureSampleGrad",
        texture: "tex",
        reason: "polar UV helper samples caller-provided texture with repaired gradients",
    },
    RawSampleException {
        label: "shaders/materials/overlayunlit.wgsl",
        builtin: "textureSampleGrad",
        texture: "tex",
        reason: "polar UV helper samples caller-provided texture with repaired gradients",
    },
    RawSampleException {
        label: "shaders/materials/toonwater.wgsl",
        builtin: "textureSampleLevel",
        texture: "_VoronoiTex",
        reason: "animated Voronoi control field intentionally samples base LOD",
    },
    RawSampleException {
        label: "shaders/materials/unlit.wgsl",
        builtin: "textureSampleGrad",
        texture: "_Tex",
        reason: "polar UV remapping supplies repaired gradients",
    },
    RawSampleException {
        label: "shaders/materials/unlitpolarmapping.wgsl",
        builtin: "textureSampleGrad",
        texture: "_MainTex",
        reason: "polar UV remapping supplies repaired gradients",
    },
    RawSampleException {
        label: "shaders/modules/core/texture_sampling.wgsl",
        builtin: "textureSampleLevel",
        texture: "tex",
        reason: "shared explicit-LOD wrapper",
    },
    RawSampleException {
        label: "shaders/modules/frame/globals.wgsl",
        builtin: "textureSampleLevel",
        texture: "light_cookie_2d_atlas",
        reason: "retains frame-global light-cookie atlas bindings",
    },
    RawSampleException {
        label: "shaders/modules/frame/globals.wgsl",
        builtin: "textureSampleLevel",
        texture: "light_cookie_point_atlas",
        reason: "retains frame-global light-cookie atlas bindings",
    },
    RawSampleException {
        label: "shaders/modules/frame/grab_pass.wgsl",
        builtin: "textureSample",
        texture: "rg::scene_color_array",
        reason: "scene-color graph resource, not a host material texture",
    },
    RawSampleException {
        label: "shaders/modules/frame/grab_pass.wgsl",
        builtin: "textureSample",
        texture: "rg::scene_color",
        reason: "scene-color graph resource, not a host material texture",
    },
    RawSampleException {
        label: "shaders/modules/lighting/light_cookies.wgsl",
        builtin: "textureSample",
        texture: "rg::light_cookie_2d_atlas",
        reason: "lighting atlas resource, not a host material texture",
    },
    RawSampleException {
        label: "shaders/modules/lighting/light_cookies.wgsl",
        builtin: "textureSample",
        texture: "rg::light_cookie_point_atlas",
        reason: "lighting atlas resource, not a host material texture",
    },
    RawSampleException {
        label: "shaders/modules/material/sample.wgsl",
        builtin: "textureSampleGrad",
        texture: "tex",
        reason: "shared polar UV helper samples caller-provided texture with repaired gradients",
    },
    RawSampleException {
        label: "shaders/modules/pbs/displace.wgsl",
        builtin: "textureSampleLevel",
        texture: "position_offset_map",
        reason: "vertex-stage displacement offset intentionally samples base LOD",
    },
    RawSampleException {
        label: "shaders/modules/pbs/displace.wgsl",
        builtin: "textureSampleLevel",
        texture: "vertex_offset_map",
        reason: "vertex-stage displacement height intentionally samples base LOD",
    },
    RawSampleException {
        label: "shaders/modules/skybox/projection360_material.wgsl",
        builtin: "textureSampleLevel",
        texture: "offset_tex",
        reason: "Projection360 offset field intentionally samples base LOD",
    },
    RawSampleException {
        label: "shaders/modules/skybox/projection360_material.wgsl",
        builtin: "textureSampleLevel",
        texture: "offset_mask",
        reason: "Projection360 offset mask intentionally samples base LOD",
    },
    RawSampleException {
        label: "shaders/modules/skybox/projection360_material.wgsl",
        builtin: "textureSampleLevel",
        texture: "main_tex",
        reason: "Projection360 equirectangular source intentionally samples base LOD",
    },
    RawSampleException {
        label: "shaders/modules/skybox/projection360_material.wgsl",
        builtin: "textureSampleLevel",
        texture: "second_tex",
        reason: "Projection360 second equirectangular source intentionally samples base LOD",
    },
    RawSampleException {
        label: "shaders/modules/skybox/projection360_material.wgsl",
        builtin: "textureSampleLevel",
        texture: "tint_tex",
        reason: "Projection360 tint ramp intentionally samples base LOD",
    },
    RawSampleException {
        label: "shaders/modules/skybox/projection360_material.wgsl",
        builtin: "textureSampleLevel",
        texture: "main_cube",
        reason: "Projection360 explicit cubemap LOD keyword",
    },
    RawSampleException {
        label: "shaders/modules/skybox/projection360_material.wgsl",
        builtin: "textureSampleLevel",
        texture: "second_cube",
        reason: "Projection360 explicit second cubemap LOD keyword",
    },
    RawSampleException {
        label: "shaders/modules/xiexe/toon2/lighting.wgsl",
        builtin: "textureSampleLevel",
        texture: "xb::_Matcap",
        reason: "matcap LOD derives from smoothness instead of sampler mip bias",
    },
    RawSampleException {
        label: "shaders/modules/xiexe/toon2/outline.wgsl",
        builtin: "textureSampleLevel",
        texture: "xb::_OutlineMask",
        reason: "outline vertex extrusion intentionally samples base LOD",
    },
];

#[test]
fn raw_texture_samples_are_documented_exceptions() -> io::Result<()> {
    let mut calls = Vec::new();
    for root in ["shaders/materials", "shaders/modules"] {
        for path in wgsl_files_recursive(root)? {
            collect_raw_sample_calls(&path, &mut calls)?;
        }
    }

    let mut used_exceptions = vec![false; RAW_SAMPLE_EXCEPTIONS.len()];
    let mut offenders = Vec::new();
    for call in &calls {
        if let Some(index) = RAW_SAMPLE_EXCEPTIONS.iter().position(|exception| {
            exception.label == call.label.as_str()
                && exception.builtin == call.builtin
                && exception.texture == call.texture.as_str()
                && !exception.reason.is_empty()
        }) {
            used_exceptions[index] = true;
        } else {
            offenders.push(format!(
                "{}:{} uses {}({}) without a raw-sampling exception",
                call.label, call.line, call.builtin, call.texture
            ));
        }
    }

    for (index, exception) in RAW_SAMPLE_EXCEPTIONS.iter().enumerate() {
        if !used_exceptions[index] {
            offenders.push(format!(
                "{} no longer uses {}({}); remove stale exception `{}`",
                exception.label, exception.builtin, exception.texture, exception.reason
            ));
        }
    }

    assert!(
        offenders.is_empty(),
        "host material texture samples must use biased helpers unless a raw sample is intentional:\n  {}",
        offenders.join("\n  ")
    );
    Ok(())
}

fn collect_raw_sample_calls(path: &Path, calls: &mut Vec<RawSampleCall>) -> io::Result<()> {
    let src = source_file(path)?;
    let label = file_label(path);
    for builtin in ["textureSampleLevel", "textureSampleGrad", "textureSample"] {
        collect_builtin_calls(&src, &label, builtin, calls);
    }
    Ok(())
}

fn collect_builtin_calls(
    src: &str,
    label: &str,
    builtin: &'static str,
    calls: &mut Vec<RawSampleCall>,
) {
    let needle = format!("{builtin}(");
    let mut search_start = 0;
    while let Some(relative_index) = src[search_start..].find(&needle) {
        let index = search_start + relative_index;
        search_start = index + needle.len();
        if source_line_is_comment(src, index)
            || builtin_shadowed_by_longer_name(src, index, builtin)
        {
            continue;
        }
        let call_start = index + builtin.len();
        let Some(call_end) = find_call_end(src, call_start) else {
            continue;
        };
        let args = split_call_args(&src[call_start + 1..call_end]);
        let Some(texture) = args.first() else {
            continue;
        };
        calls.push(RawSampleCall {
            label: label.to_owned(),
            line: line_number(src, index),
            builtin,
            texture: (*texture).to_owned(),
        });
    }
}

fn builtin_shadowed_by_longer_name(src: &str, index: usize, builtin: &str) -> bool {
    builtin == "textureSample"
        && (src[index..].starts_with("textureSampleLevel")
            || src[index..].starts_with("textureSampleGrad"))
}

fn source_line_is_comment(src: &str, index: usize) -> bool {
    let line_start = src[..index].rfind('\n').map_or(0, |pos| pos + 1);
    src[line_start..index].trim_start().starts_with("//")
}

fn line_number(src: &str, index: usize) -> usize {
    src[..index].bytes().filter(|b| *b == b'\n').count() + 1
}

fn find_call_end(src: &str, call_start: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (offset, c) in src[call_start..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(call_start + offset);
                }
            }
            _ => {}
        }
    }
    None
}

fn split_call_args(args: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut depth = 0usize;
    for (index, c) in args.char_indices() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                out.push(args[start..index].trim());
                start = index + c.len_utf8();
            }
            _ => {}
        }
    }
    out.push(args[start..].trim());
    out
}
