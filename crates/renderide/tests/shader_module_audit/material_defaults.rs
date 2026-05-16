//! Audits for source-declared material and texture defaults.

use super::*;

const EXPECTED_SHADER_DEFAULT_DIRECTIVES: &[(&str, &[&str])] = &[
    (
        "billboardunlit.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _OffsetMagnitude vec4 0.1 0.1 0.0 0.0",
            "//#mat_default _PointSize vec4 0.1 0.1 0.0 0.0",
            "//#mat_default _PolarPow float 1.0",
        ],
    ),
    (
        "blur.wgsl",
        &[
            "//#mat_default _DepthDivisor float 1.0",
            "//#mat_default _Iterations float 4.0",
            "//#mat_default _RefractionStrength float 0.01",
            "//#mat_default _Spread vec4 0.1 0.1 0.0 0.0",
        ],
    ),
    (
        "cadshader.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _OutlineColor vec4 0.0 0.0 0.0 1.0",
            "//#mat_default _OutlineWidth float 0.1",
        ],
    ),
    (
        "channelmatrix.wgsl",
        &[
            "//#mat_default _ClampMax vec4 2.0 2.0 2.0 0.0",
            "//#mat_default _LevelsB vec4 1.0 0.0 0.0 0.0",
            "//#mat_default _LevelsG vec4 0.0 0.0 1.0 0.0",
            "//#mat_default _LevelsR vec4 0.0 1.0 0.0 0.0",
        ],
    ),
    (
        "circle.wgsl",
        &["//#mat_default _Color vec4 1.0 1.0 1.0 1.0"],
    ),
    (
        "depthprojection.wgsl",
        &[
            "//#mat_default _Angle vec4 90.0 60.0 0.0 0.0",
            "//#mat_default _DepthTo float 1.0",
            "//#mat_default _DiscardOffset float 0.01",
            "//#mat_default _DiscardThreshold float 0.01",
            "//#mat_default _FarClip float 1.0",
        ],
    ),
    (
        "faceexplodeshader.wgsl",
        &["//#mat_default _Color vec4 1.0 1.0 1.0 1.0"],
    ),
    (
        "fogboxvolume.wgsl",
        &[
            "//#mat_default _AccumulationColor vec4 0.1 0.1 0.1 0.1",
            "//#mat_default _AccumulationColorBottom vec4 0.1 0.1 0.0 0.1",
            "//#mat_default _AccumulationColorTop vec4 0.1 0.1 0.1 0.1",
            "//#mat_default _AccumulationRate float 0.1",
            "//#mat_default _FogEnd float 1e+07",
            "//#mat_default _GammaCurve float 2.2",
        ],
    ),
    (
        "fresnel.wgsl",
        &[
            "//#mat_default _Exp float 1.0",
            "//#mat_default _FarColor vec4 0.0 0.0 0.0 1.0",
            "//#mat_default _GammaCurve float 1.0",
            "//#mat_default _NearColor vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale float 1.0",
            "//#mat_default _PolarPow float 1.0",
        ],
    ),
    (
        "fresnellerp.wgsl",
        &[
            "//#mat_default _Exp0 float 1.0",
            "//#mat_default _Exp1 float 1.0",
            "//#mat_default _FarColor0 vec4 0.0 0.0 0.0 1.0",
            "//#mat_default _FarColor1 vec4 0.2 0.2 0.2 1.0",
            "//#mat_default _GammaCurve float 2.2",
            "//#mat_default _LerpPolarPow float 1.0",
            "//#mat_default _NearColor0 vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NearColor1 vec4 0.8 0.8 0.8 0.8",
        ],
    ),
    (
        "furfx-2.0-10layer.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _SpecColor vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "furfx-2.0-20layer.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _SpecColor vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "furfx-3.0-10layer.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _SpecColor vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "furfx-3.0-20layer.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _SpecColor vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "furfx-3.0-shell-10layer.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _SpecColor vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "furfx-3.0-shell-20layer.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _SpecColor vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "furfx-advanced-10layer.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _SpecColor vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "furfx-advanced-20layer.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _SpecColor vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "furfx-advanced-40layer.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _SpecColor vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "furfx-advanced-5layer.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _SpecColor vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "furfx-advanced-shell-10layer.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _SpecColor vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "furfx-advanced-shell-20layer.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _SpecColor vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "furfx-advanced-shell-40layer.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _SpecColor vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "furfx-advanced-shell-5layer.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _SpecColor vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "furfx-basic-10layer.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _SpecColor vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "furfx-basic-20layer.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _SpecColor vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "furfx-basic-40layer.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _SpecColor vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "furfx-basic-5layer.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _SpecColor vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "furfx-selfshadow-blend-10layer.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _SpecColor vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "furfx-selfshadow-blend-20layer.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _SpecColor vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "furfx-selfshadow-noblend-10layer.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _SpecColor vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "furfx-selfshadow-noblend-20layer.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _SpecColor vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    ("gamma.wgsl", &["//#mat_default _Gamma float 2.2"]),
    (
        "getdepth.wgsl",
        &[
            "//#mat_default _ClipMax float 1.0",
            "//#mat_default _Multiply float 1.0",
        ],
    ),
    (
        "gradientskybox.wgsl",
        &["//#mat_default _BaseColor vec4 1.0 1.0 1.0 1.0"],
    ),
    (
        "grayscale.wgsl",
        &[
            "//#mat_default _Lerp float 1.0",
            "//#mat_default _RatioB float 0.11",
            "//#mat_default _RatioG float 0.59",
            "//#mat_default _RatioR float 0.3",
        ],
    ),
    (
        "hsv.wgsl",
        &[
            "//#mat_default _HSVMul vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _HSVOffset vec4 0.2 0.2 0.2 0.0",
        ],
    ),
    ("invert.wgsl", &["//#mat_default _Lerp float 1.0"]),
    (
        "newunlitshader.wgsl",
        &["//#mat_default _node_2829 vec4 0.5 0.5 0.5 1.0"],
    ),
    (
        "nosamplers.wgsl",
        &["//#mat_default _Color vec4 1.0 1.0 1.0 1.0"],
    ),
    (
        "overlay.wgsl",
        &["//#mat_default _Blend vec4 1.0 1.0 1.0 1.0"],
    ),
    (
        "overlayfresnel.wgsl",
        &[
            "//#mat_default _BehindFarColor vec4 0.0 0.0 0.0 1.0",
            "//#mat_default _BehindNearColor vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _Exp float 1.0",
            "//#mat_default _FrontFarColor vec4 0.0 0.0 0.0 1.0",
            "//#mat_default _FrontNearColor vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _GammaCurve float 2.2",
            "//#mat_default _PolarPow float 1.0",
        ],
    ),
    (
        "overlayunlit.wgsl",
        &[
            "//#mat_default _BehindColor vec4 0.5 0.5 0.5 0.5",
            "//#mat_default _FrontColor vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _PolarPow float 1.0",
        ],
    ),
    (
        "paintpbs.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _OutputScale float 10.0",
            "//#mat_default _PaintGain float 1.0",
            "//#mat_default _PaintTexOffsets vec4 0.0 0.333 0.5 0.777",
            "//#mat_default _PaintTexScales vec4 1.0 0.95 0.89 1.13",
            "//#mat_default _PaintTexShifts vec4 -0.7 0.2 -0.4 1.0",
            "//#mat_default _Pow float 1.0",
        ],
    ),
    (
        "pbscolormask.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 0.0 0.0 1.0",
            "//#mat_default _Color1 vec4 0.0 1.0 0.0 1.0",
            "//#mat_default _Color2 vec4 0.0 0.0 1.0 1.0",
            "//#mat_default _Color3 vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale float 1.0",
        ],
    ),
    (
        "pbscolormaskspecular.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 0.0 0.0 1.0",
            "//#mat_default _Color1 vec4 0.0 1.0 0.0 1.0",
            "//#mat_default _Color2 vec4 0.0 0.0 1.0 1.0",
            "//#mat_default _Color3 vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale float 1.0",
            "//#mat_default _SpecularColor vec4 1.0 1.0 1.0 0.5",
        ],
    ),
    (
        "pbscolorsplat.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _Color1 vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _Color2 vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _Color3 vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale float 1.0",
            "//#mat_default _NormalScale1 float 1.0",
            "//#mat_default _NormalScale2 float 1.0",
            "//#mat_default _NormalScale3 float 1.0",
        ],
    ),
    (
        "pbscolorsplatspecular.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _Color1 vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _Color2 vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _Color3 vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale0 float 1.0",
            "//#mat_default _NormalScale1 float 1.0",
            "//#mat_default _NormalScale2 float 1.0",
            "//#mat_default _NormalScale3 float 1.0",
            "//#mat_default _SpecularColor vec4 0.5 0.5 0.5 0.5",
            "//#mat_default _SpecularColor1 vec4 0.5 0.5 0.5 0.5",
            "//#mat_default _SpecularColor2 vec4 0.5 0.5 0.5 0.5",
            "//#mat_default _SpecularColor3 vec4 0.5 0.5 0.5 0.5",
        ],
    ),
    (
        "pbsdisplace.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale float 1.0",
            "//#mat_default _PositionOffsetMagnitude vec4 1.0 1.0 0.0 0.0",
            "//#mat_default _UVOffsetMagnitude float 0.1",
            "//#mat_default _VertexOffsetMagnitude float 0.1",
        ],
    ),
    (
        "pbsdisplaceshadow.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _VertexOffsetMagnitude float 0.1",
        ],
    ),
    (
        "pbsdisplacespecular.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale float 1.0",
            "//#mat_default _PositionOffsetMagnitude vec4 1.0 1.0 0.0 0.0",
            "//#mat_default _SpecularColor vec4 1.0 1.0 1.0 0.5",
            "//#mat_default _UVOffsetMagnitude float 0.1",
            "//#mat_default _VertexOffsetMagnitude float 0.1",
        ],
    ),
    (
        "pbsdisplacespeculartransparent.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale float 1.0",
            "//#mat_default _PositionOffsetMagnitude vec4 1.0 1.0 0.0 0.0",
            "//#mat_default _SpecularColor vec4 1.0 1.0 1.0 0.5",
            "//#mat_default _UVOffsetMagnitude float 0.1",
            "//#mat_default _VertexOffsetMagnitude float 0.1",
        ],
    ),
    (
        "pbsdisplacetransparent.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale float 1.0",
            "//#mat_default _PositionOffsetMagnitude vec4 1.0 1.0 0.0 0.0",
            "//#mat_default _UVOffsetMagnitude float 0.1",
            "//#mat_default _VertexOffsetMagnitude float 0.1",
        ],
    ),
    (
        "pbsdistancelerp.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _DisplaceDistanceFrom float 1.0",
            "//#mat_default _DisplaceMagnitudeTo float 0.1",
            "//#mat_default _DisplacementDirection vec4 0.0 1.0 0.0 0.0",
            "//#mat_default _EmissionColorTo vec4 1.5 1.5 1.5 0.0",
            "//#mat_default _EmissionDistanceFrom float 1.0",
            "//#mat_default _NormalScale float 1.0",
        ],
    ),
    (
        "pbsdistancelerpspecular.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _DisplaceDistanceFrom float 1.0",
            "//#mat_default _DisplaceMagnitudeTo float 0.1",
            "//#mat_default _DisplacementDirection vec4 0.0 1.0 0.0 0.0",
            "//#mat_default _EmissionColorTo vec4 1.5 1.5 1.5 0.0",
            "//#mat_default _EmissionDistanceFrom float 1.0",
            "//#mat_default _NormalScale float 1.0",
        ],
    ),
    (
        "pbsdistancelerpspeculartransparent.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _DisplaceDistanceFrom float 1.0",
            "//#mat_default _DisplaceMagnitudeTo float 0.1",
            "//#mat_default _DisplacementDirection vec4 0.0 1.0 0.0 0.0",
            "//#mat_default _EmissionColorTo vec4 1.5 1.5 1.5 0.0",
            "//#mat_default _EmissionDistanceFrom float 1.0",
            "//#mat_default _NormalScale float 1.0",
        ],
    ),
    (
        "pbsdistancelerptransparent.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _DisplaceDistanceFrom float 1.0",
            "//#mat_default _DisplaceMagnitudeTo float 0.1",
            "//#mat_default _DisplacementDirection vec4 0.0 1.0 0.0 0.0",
            "//#mat_default _EmissionColorTo vec4 1.5 1.5 1.5 0.0",
            "//#mat_default _EmissionDistanceFrom float 1.0",
            "//#mat_default _NormalScale float 1.0",
        ],
    ),
    (
        "pbsdualsided.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale float 1.0",
        ],
    ),
    (
        "pbsdualsidedspecular.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale float 1.0",
            "//#mat_default _SpecularColor vec4 1.0 1.0 1.0 0.5",
        ],
    ),
    (
        "pbsdualsidedtransparent.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale float 1.0",
        ],
    ),
    (
        "pbsdualsidedtransparentspecular.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale float 1.0",
            "//#mat_default _SpecularColor vec4 1.0 1.0 1.0 0.5",
        ],
    ),
    (
        "pbsintersect.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _EndTransitionEnd float 0.1",
            "//#mat_default _EndTransitionStart float 0.1",
            "//#mat_default _IntersectColor vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _IntersectEmissionColor vec4 1.0 0.0 0.0 1.0",
            "//#mat_default _NormalScale float 1.0",
        ],
    ),
    (
        "pbsintersectspecular.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _EndTransitionEnd float 0.1",
            "//#mat_default _EndTransitionStart float 0.1",
            "//#mat_default _IntersectColor vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _IntersectEmissionColor vec4 1.0 0.0 0.0 1.0",
            "//#mat_default _NormalScale float 1.0",
            "//#mat_default _SpecularColor vec4 1.0 1.0 1.0 0.5",
        ],
    ),
    (
        "pbslerp.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _Color1 vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale float 1.0",
            "//#mat_default _NormalScale1 float 1.0",
        ],
    ),
    (
        "pbslerpspecular.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _Color1 vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale float 1.0",
            "//#mat_default _NormalScale1 float 1.0",
            "//#mat_default _SpecularColor vec4 1.0 1.0 1.0 0.5",
            "//#mat_default _SpecularColor1 vec4 1.0 1.0 1.0 0.5",
        ],
    ),
    (
        "pbsmetallic.wgsl",
        &[
            "//#mat_default _BumpScale float 1.0",
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _DetailNormalMapScale float 1.0",
            "//#mat_default _EmissionColor vec4 0.0 0.0 0.0 1.0",
        ],
    ),
    (
        "pbsmultiuv.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale float 1.0",
        ],
    ),
    (
        "pbsmultiuvspecular.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale float 1.0",
            "//#mat_default _SpecularColor vec4 1.0 1.0 1.0 0.5",
        ],
    ),
    (
        "pbsrim.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale float 1.0",
            "//#mat_default _RimColor vec4 1.0 0.0 0.0 1.0",
        ],
    ),
    (
        "pbsrimspecular.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale float 1.0",
            "//#mat_default _RimColor vec4 1.0 0.0 0.0 1.0",
            "//#mat_default _SpecularColor vec4 1.0 1.0 1.0 0.5",
        ],
    ),
    (
        "pbsrimtransparent.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale float 1.0",
            "//#mat_default _RimColor vec4 1.0 0.0 0.0 1.0",
        ],
    ),
    (
        "pbsrimtransparentspecular.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale float 1.0",
            "//#mat_default _RimColor vec4 1.0 0.0 0.0 1.0",
            "//#mat_default _SpecularColor vec4 1.0 1.0 1.0 0.5",
        ],
    ),
    (
        "pbsrimtransparentzwrite.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale float 1.0",
            "//#mat_default _RimColor vec4 1.0 0.0 0.0 1.0",
        ],
    ),
    (
        "pbsrimtransparentzwritespecular.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale float 1.0",
            "//#mat_default _RimColor vec4 1.0 0.0 0.0 1.0",
            "//#mat_default _SpecularColor vec4 1.0 1.0 1.0 0.5",
        ],
    ),
    (
        "pbsslice.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _DetailNormalMapScale float 1.0",
            "//#mat_default _EdgeColor vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _EdgeEmissionColor vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _EdgeTransitionEnd float 0.1",
            "//#mat_default _NormalScale float 1.0",
        ],
    ),
    (
        "pbsslicespecular.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _DetailNormalMapScale float 1.0",
            "//#mat_default _EdgeColor vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _EdgeEmissionColor vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _EdgeTransitionEnd float 0.1",
            "//#mat_default _NormalScale float 1.0",
            "//#mat_default _SpecularColor vec4 1.0 1.0 1.0 0.5",
        ],
    ),
    (
        "pbsslicetransparent.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _DetailNormalMapScale float 1.0",
            "//#mat_default _EdgeColor vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _EdgeEmissionColor vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _EdgeTransitionEnd float 0.1",
            "//#mat_default _NormalScale float 1.0",
        ],
    ),
    (
        "pbsslicetransparentspecular.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _DetailNormalMapScale float 1.0",
            "//#mat_default _EdgeColor vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _EdgeEmissionColor vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _EdgeTransitionEnd float 0.1",
            "//#mat_default _NormalScale float 1.0",
            "//#mat_default _SpecularColor vec4 1.0 1.0 1.0 0.5",
        ],
    ),
    (
        "pbsspecular.wgsl",
        &[
            "//#mat_default _BumpScale float 1.0",
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _DetailNormalMapScale float 1.0",
            "//#mat_default _EmissionColor vec4 0.0 0.0 0.0 1.0",
            "//#mat_default _SpecColor vec4 0.2 0.2 0.2 1.0",
        ],
    ),
    (
        "pbsstencil.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale float 1.0",
        ],
    ),
    (
        "pbsstencilspecular.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale float 1.0",
            "//#mat_default _SpecularColor vec4 1.0 1.0 1.0 0.5",
        ],
    ),
    (
        "pbstriplanar.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale float 1.0",
            "//#mat_default _TriBlendPower float 4.0",
        ],
    ),
    (
        "pbstriplanarspecular.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale float 1.0",
            "//#mat_default _SpecularColor vec4 1.0 1.0 1.0 0.5",
            "//#mat_default _TriBlendPower float 4.0",
        ],
    ),
    (
        "pbstriplanartransparent.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale float 1.0",
            "//#mat_default _TriBlendPower float 4.0",
        ],
    ),
    (
        "pbstriplanartransparentspecular.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale float 1.0",
            "//#mat_default _SpecularColor vec4 1.0 1.0 1.0 0.5",
            "//#mat_default _TriBlendPower float 4.0",
        ],
    ),
    (
        "pbsvertexcolortransparent.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale float 1.0",
        ],
    ),
    (
        "pbsvertexcolortransparentspecular.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NormalScale float 1.0",
            "//#mat_default _SpecularColor vec4 1.0 1.0 1.0 0.5",
        ],
    ),
    (
        "pbsvoronoicrystal.wgsl",
        &[
            "//#mat_default _ColorTint vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _EdgeColor vec4 0.0 0.0 0.0 1.0",
            "//#mat_default _EdgeNormalStrength float 0.5",
            "//#mat_default _NormalStrength float 1.0",
            "//#mat_default _Scale vec4 1.0 1.0 0.0 0.0",
        ],
    ),
    (
        "pixelate.wgsl",
        &["//#mat_default _Resolution vec4 100.0 100.0 0.0 0.0"],
    ),
    ("posterize.wgsl", &["//#mat_default _Levels float 10.0"]),
    (
        "proceduralskybox.wgsl",
        &[
            "//#mat_default _GroundColor vec4 0.369 0.349 0.341 1.0",
            "//#mat_default _SkyTint vec4 0.5 0.5 0.5 1.0",
            "//#mat_default _SunColor vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _SunDirection vec4 0.577 0.577 0.577 0.0",
        ],
    ),
    (
        "projection360.wgsl",
        &[
            "//#mat_default _Exposure float 1.0",
            "//#mat_default _FOV vec4 6.283185 3.141593 0.0 0.0",
            "//#mat_default _Gamma float 1.0",
            "//#mat_default _MaxIntensity float 4.0",
            "//#mat_default _OffsetMagnitude vec4 0.1 0.1 0.0 0.0",
            "//#mat_default _PerspectiveFOV vec4 0.785398 0.785398 0.0 0.0",
            "//#mat_default _Tint vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _Tint0 vec4 1.0 0.0 0.0 1.0",
            "//#mat_default _Tint1 vec4 0.0 1.0 0.0 1.0",
        ],
    ),
    (
        "reflection.wgsl",
        &["//#mat_default _Color vec4 1.0 1.0 1.0 1.0"],
    ),
    (
        "refract.wgsl",
        &[
            "//#mat_default _DepthBias float 0.01",
            "//#mat_default _RefractionStrength float 0.01",
        ],
    ),
    (
        "testblend.wgsl",
        &["//#mat_default _Color vec4 1.0 1.0 1.0 1.0"],
    ),
    (
        "testshader.wgsl",
        &["//#mat_default _Color vec4 0.5 0.5 0.5 1.0"],
    ),
    (
        "textunlit.wgsl",
        &[
            "//#mat_default _OutlineColor vec4 1.0 1.0 1.0 0.0",
            "//#mat_default _Range vec4 0.001 0.001 0.0 0.0",
            "//#mat_default _TintColor vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "threshold.wgsl",
        &[
            "//#mat_default _Threshold float 0.5",
            "//#mat_default _Transition float 0.01",
        ],
    ),
    (
        "toonstandard.wgsl",
        &[
            "//#mat_default _BumpScale float 1.0",
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _Fresnel float 1.0",
            "//#mat_default _FresnelTint vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "toonwater.wgsl",
        &[
            "//#mat_default _BumpScale float 1.0",
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _Fresnel float 1.0",
            "//#mat_default _FresnelTint vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _PlanarReflection float 1.0",
        ],
    ),
    (
        "ui_circlesegment.wgsl",
        &[
            "//#mat_default _FillTint vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _OutlineTint vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _OverlayTint vec4 1.0 1.0 1.0 0.5",
            "//#mat_default _Rect vec4 0.0 0.0 1.0 1.0",
        ],
    ),
    (
        "ui_textunlit.wgsl",
        &[
            "//#mat_default _OutlineColor vec4 1.0 1.0 1.0 0.0",
            "//#mat_default _OverlayTint vec4 1.0 1.0 1.0 0.5",
            "//#mat_default _Range vec4 0.001 0.001 0.0 0.0",
            "//#mat_default _Rect vec4 0.0 0.0 1.0 1.0",
            "//#mat_default _TintColor vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "ui_unlit.wgsl",
        &[
            "//#mat_default _OverlayTint vec4 1.0 1.0 1.0 0.5",
            "//#mat_default _Rect vec4 0.0 0.0 1.0 1.0",
            "//#mat_default _Tint vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "unlit.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _OffsetMagnitude vec4 0.1 0.1 0.0 0.0",
            "//#mat_default _PolarPow float 1.0",
        ],
    ),
    (
        "unlitdistancelerp.wgsl",
        &[
            "//#mat_default _Distance float 1.0",
            "//#mat_default _FarColor vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _NearColor vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _Transition float 0.1",
        ],
    ),
    ("unlitpolarmapping.wgsl", &["//#mat_default _Pow float 1.0"]),
    (
        "uvrect.wgsl",
        &[
            "//#mat_default _ClipRect vec4 0.0 0.0 1.0 1.0",
            "//#mat_default _InnerColor vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _OuterColor vec4 0.0 0.0 0.0 1.0",
            "//#mat_default _Rect vec4 0.25 0.25 0.75 0.75",
        ],
    ),
    (
        "volumeunlit.wgsl",
        &[
            "//#mat_default _AccumulationCutoff float 100.0",
            "//#mat_default _Exp float 1.0",
            "//#mat_default _Gain float 0.1",
            "//#mat_default _HighClip float 1.0",
            "//#mat_default _HitThreshold float 0.5",
            "//#mat_default _StepSize float 0.1",
        ],
    ),
    (
        "xstoon2.0-cutout.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _Cutoff float 0.5",
            "//#mat_default _MatcapTint vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _OutlineColor vec4 0.0 0.0 0.0 1.0",
            "//#mat_default _RimColor vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _ShadowRim vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "xstoon2.0-cutouta2c-outlined.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _Cutoff float 0.5",
            "//#mat_default _MatcapTint vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _OutlineColor vec4 0.0 0.0 0.0 1.0",
            "//#mat_default _RimColor vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _ShadowRim vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "xstoon2.0-cutouta2c.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _Cutoff float 0.5",
            "//#mat_default _MatcapTint vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _OutlineColor vec4 0.0 0.0 0.0 1.0",
            "//#mat_default _RimColor vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _ShadowRim vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "xstoon2.0-cutouta2cmasked.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _Cutoff float 0.5",
            "//#mat_default _MatcapTint vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _OutlineColor vec4 0.0 0.0 0.0 1.0",
            "//#mat_default _RimColor vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _ShadowRim vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "xstoon2.0-dithered-outlined.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _Cutoff float 0.5",
            "//#mat_default _MatcapTint vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _OutlineColor vec4 0.0 0.0 0.0 1.0",
            "//#mat_default _RimColor vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _ShadowRim vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "xstoon2.0-dithered.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _Cutoff float 0.5",
            "//#mat_default _MatcapTint vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _OutlineColor vec4 0.0 0.0 0.0 1.0",
            "//#mat_default _RimColor vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _ShadowRim vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "xstoon2.0-fade.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _Cutoff float 0.5",
            "//#mat_default _MatcapTint vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _OutlineColor vec4 0.0 0.0 0.0 1.0",
            "//#mat_default _RimColor vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _ShadowRim vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "xstoon2.0-outlined.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _Cutoff float 0.5",
            "//#mat_default _MatcapTint vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _OutlineColor vec4 0.0 0.0 0.0 1.0",
            "//#mat_default _RimColor vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _ShadowRim vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "xstoon2.0-transparent.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _Cutoff float 0.5",
            "//#mat_default _MatcapTint vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _OutlineColor vec4 0.0 0.0 0.0 1.0",
            "//#mat_default _RimColor vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _ShadowRim vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "xstoon2.0.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _Cutoff float 0.5",
            "//#mat_default _MatcapTint vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _OutlineColor vec4 0.0 0.0 0.0 1.0",
            "//#mat_default _RimColor vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _ShadowRim vec4 1.0 1.0 1.0 1.0",
        ],
    ),
    (
        "xstoon2.0_outlined.wgsl",
        &[
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _Cutoff float 0.5",
            "//#mat_default _MatcapTint vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _OutlineColor vec4 0.0 0.0 0.0 1.0",
            "//#mat_default _RimColor vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _ShadowRim vec4 1.0 1.0 1.0 1.0",
        ],
    ),
];

#[test]
fn material_sources_declare_unity_property_defaults() -> io::Result<()> {
    let mut directive_count = 0usize;
    for (file_name, directives) in EXPECTED_SHADER_DEFAULT_DIRECTIVES {
        let src = material_source(file_name)?;
        for directive in *directives {
            directive_count += 1;
            assert!(
                src.contains(directive),
                "{file_name} must declare `{directive}`"
            );
        }
    }
    assert_eq!(directive_count, 453);
    Ok(())
}
