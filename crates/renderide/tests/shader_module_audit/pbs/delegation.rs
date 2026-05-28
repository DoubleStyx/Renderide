//! PBS shared-helper and clustered-lighting delegation audits.

use super::super::*;

#[test]
fn pbs_material_roots_use_shared_sampling_and_mask_helpers() -> io::Result<()> {
    for material in ["pbscolorsplat.wgsl", "pbscolorsplatspecular.wgsl"] {
        let src = material_source(material)?;
        assert!(
            src.contains("psamp::unpack_packed_normal_xy("),
            "{material} must use the shared packed-normal unpack helper"
        );
        assert!(
            !src.contains("fn unpack_normal_xy"),
            "{material} must not redeclare packed-normal unpacking"
        );
    }

    for material in ["pbscolormask.wgsl", "pbscolormaskspecular.wgsl"] {
        let src = material_source(material)?;
        assert!(
            src.contains("splat::color_mask_weights(mask)") && src.contains("mcolor::blend4_vec4("),
            "{material} must use shared color-mask weighting and blend helpers"
        );
    }

    for material in [
        "pbsdualsided.wgsl",
        "pbsdualsidedspecular.wgsl",
        "pbsdualsidedtransparent.wgsl",
        "pbsdualsidedtransparentspecular.wgsl",
    ] {
        let src = material_source(material)?;
        assert!(
            src.contains("psamp::sample_optional_two_sided_world_normal("),
            "{material} must use the shared two-sided normal sampling helper"
        );
        for forbidden in [
            "nd::decode_ts_normal_with_placeholder_sample",
            "pnorm::orthonormal_tbn",
        ] {
            assert!(
                !src.contains(forbidden),
                "{material} must delegate `{forbidden}` through pbs::sampling"
            );
        }
    }

    for material in [
        "pbsdistancelerp.wgsl",
        "pbsdistancelerpspecular.wgsl",
        "pbsdistancelerptransparent.wgsl",
        "pbsdistancelerpspeculartransparent.wgsl",
    ] {
        let src = material_source(material)?;
        assert!(
            src.contains("psamp::sample_optional_two_sided_world_normal("),
            "{material} must use the shared two-sided normal sampling helper"
        );
        for forbidden in [
            "nd::decode_ts_normal_with_placeholder_sample",
            "pnorm::orthonormal_tbn",
        ] {
            assert!(
                !src.contains(forbidden),
                "{material} must delegate `{forbidden}` through pbs::sampling"
            );
        }
    }

    let splat = module_source("pbs/splat.wgsl")?;
    assert!(
        splat.contains("fn color_mask_weights(mask: vec4<f32>) -> vec4<f32>")
            && splat.contains("mask * clamp(1.0 / sum, 0.0, 1.0)"),
        "pbs::splat must expose the Unity color-mask blend-weight policy"
    );

    Ok(())
}

#[test]
fn shared_pbs_surface_roots_pass_geometric_normals() -> io::Result<()> {
    let sampling = module_source("pbs/sampling.wgsl")?;
    assert!(
        sampling.contains(
            "fn two_sided_geometric_normal(world_n: vec3<f32>, front_facing: bool) -> vec3<f32>"
        ),
        "pbs/sampling.wgsl must expose a two-sided geometric normal helper for horizon occlusion"
    );

    let mut offenders = Vec::new();
    for root in ["shaders/materials", "shaders/modules/fur"] {
        for path in wgsl_files_recursive(root)? {
            let src = source_file(&path)?;
            if !src.contains("renderide::pbs::surface as psurf") {
                continue;
            }
            if src.contains("psurf::metallic(") || src.contains("psurf::specular(") {
                offenders.push(file_label(&path));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "shared PBS surface roots must pass a separate geometric normal for reflection-probe horizon occlusion:\n  {}",
        offenders.join("\n  ")
    );
    Ok(())
}

/// Material roots using the shared PBS lighting module should not also carry their own clustered loop.
#[test]
fn shared_pbs_lighting_roots_do_not_duplicate_clustered_lighting() -> io::Result<()> {
    let mut offenders = Vec::new();
    for path in wgsl_files_recursive("shaders/materials")? {
        let src = source_file(&path)?;
        if !src.contains("renderide::pbs::lighting") {
            continue;
        }

        for forbidden in [
            "#import renderide::ibl::sh2_ambient",
            "#import renderide::pbs::brdf",
            "#import renderide::pbs::cluster",
            "fn clustered_direct_lighting",
            "pcls::cluster_id_from_frag",
        ] {
            if src.contains(forbidden) {
                offenders.push(format!(
                    "{} still contains `{forbidden}`",
                    file_label(&path)
                ));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "materials importing renderide::pbs::lighting must delegate clustered PBS lighting:\n  {}",
        offenders.join("\n  ")
    );
    Ok(())
}

/// Standard PBS-like roots should delegate clustered Standard lighting to the shared PBS module.
#[test]
fn standard_material_roots_do_not_duplicate_clustered_pbs_lighting() -> io::Result<()> {
    let mut offenders = Vec::new();
    for path in wgsl_files_recursive("shaders/materials")? {
        let label = file_label(&path);
        let src = source_file(&path)?;
        if label == "shaders/materials/toonstandard.wgsl"
            || label == "shaders/materials/toonwater.wgsl"
        {
            continue;
        }

        for forbidden in [
            "#import renderide::ibl::sh2_ambient",
            "#import renderide::pbs::cluster",
            "cluster_id_from_frag",
            "direct_radiance_metallic",
            "direct_radiance_specular",
            "indirect_diffuse_metallic",
            "indirect_diffuse_specular",
            "indirect_specular",
        ] {
            if src.contains(forbidden) {
                offenders.push(format!("{label} still contains `{forbidden}`"));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "standard PBS-like material roots must delegate clustered PBS lighting:\n  {}",
        offenders.join("\n  ")
    );
    Ok(())
}
