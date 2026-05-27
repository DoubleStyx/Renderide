//! PBS material pass metadata audits.

use super::super::*;
use super::helpers::pass_directives;

#[test]
fn pbs_transparent_roots_keep_authored_pass_directives() -> io::Result<()> {
    for material in [
        "pbsdisplacetransparent.wgsl",
        "pbsdisplacespeculartransparent.wgsl",
        "pbsdistancelerptransparent.wgsl",
        "pbsdistancelerpspeculartransparent.wgsl",
        "pbsintersect.wgsl",
        "pbsintersectspecular.wgsl",
        "pbsslicetransparent.wgsl",
        "pbsslicetransparentspecular.wgsl",
        "pbstriplanartransparent.wgsl",
        "pbstriplanartransparentspecular.wgsl",
    ] {
        let src = material_source(material)?;
        assert_eq!(pass_directives(&src), ["forward_transparent"], "{material}");
    }

    for material in ["pbsrimtransparent.wgsl", "pbsrimtransparentspecular.wgsl"] {
        let src = material_source(material)?;
        assert_eq!(
            pass_directives(&src),
            ["forward_transparent_cull_back"],
            "{material}"
        );
    }

    for material in [
        "pbsrimtransparentzwrite.wgsl",
        "pbsrimtransparentzwritespecular.wgsl",
    ] {
        let src = material_source(material)?;
        assert_eq!(
            pass_directives(&src),
            ["depth_prepass", "forward_transparent_cull_back"],
            "{material}"
        );
    }

    for material in [
        "pbsvertexcolortransparent.wgsl",
        "pbsvertexcolortransparentspecular.wgsl",
    ] {
        let src = material_source(material)?;
        assert_eq!(
            pass_directives(&src),
            ["forward_transparent_cull_back"],
            "{material}"
        );
    }

    for material in [
        "pbsdualsidedtransparent.wgsl",
        "pbsdualsidedtransparentspecular.wgsl",
    ] {
        let src = material_source(material)?;
        assert_eq!(
            pass_directives(&src),
            [
                "forward_transparent_cull_front",
                "forward_transparent_cull_back"
            ],
            "{material}"
        );
    }

    Ok(())
}

#[test]
fn selected_pbs_material_roots_keep_material_offset_pass_metadata() -> io::Result<()> {
    for material in [
        "pbsdistancelerp.wgsl",
        "pbsdistancelerpspecular.wgsl",
        "pbsdistancelerpspeculartransparent.wgsl",
        "pbsdistancelerptransparent.wgsl",
        "pbsdualsided.wgsl",
        "pbsdualsidedspecular.wgsl",
        "pbsdualsidedtransparent.wgsl",
        "pbsdualsidedtransparentspecular.wgsl",
        "pbsintersect.wgsl",
        "pbsintersectspecular.wgsl",
    ] {
        let src = material_source(material)?;
        let mut pass_count = 0usize;
        for line in src
            .lines()
            .filter(|line| line.trim_start().starts_with("//#pass "))
        {
            pass_count += 1;
            assert!(
                line.contains("offset=material(0,0)"),
                "{material} pass directive must preserve material-driven Unity Offset state: {line}"
            );
        }
        assert!(pass_count > 0, "{material} must declare at least one pass");
    }
    Ok(())
}
