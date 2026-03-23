using System.Text;
using UnityShaderConverter.Analysis;
using UnityShaderParser.ShaderLab;

namespace UnityShaderConverter.Emission;

/// <summary>Emits Rust <c>wgpu::PrimitiveState</c>, blend, depth/stencil, and color mask snippets from <see cref="PassFixedFunctionState"/>.</summary>
public static class FixedFunctionRustEmitter
{
    /// <summary>Rust expression for <c>wgpu::PrimitiveState { ... }</c>.</summary>
    public static string EmitPrimitiveState(PassFixedFunctionState s)
    {
        var cull = ResolveCull(s);
        string cullFace = cull switch
        {
            CullMode.Off => "None",
            CullMode.Front => "Some(wgpu::Face::Front)",
            CullMode.Back => "Some(wgpu::Face::Back)",
            _ => "Some(wgpu::Face::Back)",
        };
        var sb = new StringBuilder();
        if (s.CullReferencesProperty)
            sb.AppendLine("    // Cull references a material property; using Unity default Back.");
        if (s.DepthBiasReferencesProperty)
            sb.AppendLine("    // Offset references a material property; depth bias left at default.");
        sb.Append("    wgpu::PrimitiveState {\n        topology: wgpu::PrimitiveTopology::TriangleList,\n        strip_index_format: None,\n        front_face: wgpu::FrontFace::Ccw,\n        cull_mode: ");
        sb.Append(cullFace);
        sb.Append(",\n        unclipped_depth: false,\n        polygon_mode: wgpu::PolygonMode::Fill,\n        conservative: false,\n    }");
        return sb.ToString();
    }

    private static CullMode ResolveCull(PassFixedFunctionState s)
    {
        if (s.CullReferencesProperty)
            return CullMode.Back;
        return s.CullMode ?? CullMode.Back;
    }

    /// <summary>Rust <c>wgpu::DepthStencilState { ... }</c> using <paramref name="depthFormatIdent"/> for <c>format</c>.</summary>
    public static string EmitDepthStencilState(PassFixedFunctionState s, string depthFormatIdent)
    {
        ComparisonMode depthCmp = ResolveDepthTest(s);
        bool depthWrite = ResolveDepthWrite(s);
        string cmpRust = ComparisonToCompareFunction(depthCmp);
        if (depthCmp == ComparisonMode.Off)
        {
            depthWrite = false;
            cmpRust = "wgpu::CompareFunction::Always";
        }

        var sb = new StringBuilder();
        if (s.DepthTestReferencesProperty)
            sb.AppendLine("    // ZTest references a material property; using LessEqual.");
        if (s.DepthWriteReferencesProperty)
            sb.AppendLine("    // ZWrite references a material property; using On.");
        sb.Append("    wgpu::DepthStencilState {\n        format: ");
        sb.Append(depthFormatIdent);
        sb.Append(",\n        depth_write_enabled: ");
        sb.Append(depthWrite ? "true" : "false");
        sb.Append(",\n        depth_compare: ");
        sb.Append(cmpRust);
        sb.Append(",\n        stencil: ");
        sb.Append(EmitStencilState(s));
        sb.Append(",\n        bias: wgpu::DepthBiasState {\n            constant: ");
        sb.Append(EmitDepthBiasConstant(s));
        sb.Append(",\n            slope_scale: ");
        sb.Append(EmitDepthBiasSlope(s));
        sb.Append(",\n            clamp: 0.0,\n        },\n    }");
        return sb.ToString();
    }

    private static ComparisonMode ResolveDepthTest(PassFixedFunctionState s)
    {
        if (s.DepthTestReferencesProperty)
            return ComparisonMode.LEqual;
        return s.DepthTest ?? ComparisonMode.LEqual;
    }

    private static bool ResolveDepthWrite(PassFixedFunctionState s)
    {
        if (s.DepthWriteReferencesProperty)
            return true;
        return s.DepthWrite ?? true;
    }

    private static string EmitDepthBiasConstant(PassFixedFunctionState s)
    {
        if (s.DepthBiasReferencesProperty || s.DepthBias is null)
            return "0";
        int c = (int)Math.Round(s.DepthBias.Value.Units, MidpointRounding.AwayFromZero);
        return c.ToString(System.Globalization.CultureInfo.InvariantCulture);
    }

    private static string EmitDepthBiasSlope(PassFixedFunctionState s)
    {
        if (s.DepthBiasReferencesProperty || s.DepthBias is null)
            return "0.0";
        return s.DepthBias.Value.Factor.ToString(System.Globalization.CultureInfo.InvariantCulture) + "f32";
    }

    private static string EmitStencilState(PassFixedFunctionState s)
    {
        if (s.StencilReferencesProperty || s.Stencil is null)
        {
            return @"wgpu::StencilState {
        front: wgpu::StencilFaceState::IGNORE,
        back: wgpu::StencilFaceState::IGNORE,
        read_mask: 0,
        write_mask: 0,
    }";
        }

        PassStencilConcrete t = s.Stencil;
        return $@"wgpu::StencilState {{
        front: {StencilFace(t.CompFront, t.PassFront, t.FailFront, t.ZFailFront)},
        back: {StencilFace(t.CompBack, t.PassBack, t.FailBack, t.ZFailBack)},
        read_mask: {t.ReadMask}u32,
        write_mask: {t.WriteMask}u32,
    }}";
    }

    private static string StencilFace(
        ComparisonMode comp,
        StencilOp pass,
        StencilOp fail,
        StencilOp zfail)
    {
        return $@"wgpu::StencilFaceState {{
            compare: {ComparisonToCompareFunction(comp)},
            fail_op: {StencilOpToWgpu(fail)},
            depth_fail_op: {StencilOpToWgpu(zfail)},
            pass_op: {StencilOpToWgpu(pass)},
        }}";
    }

    private static string StencilOpToWgpu(StencilOp op) =>
        op switch
        {
            StencilOp.Keep => "wgpu::StencilOperation::Keep",
            StencilOp.Zero => "wgpu::StencilOperation::Zero",
            StencilOp.Replace => "wgpu::StencilOperation::Replace",
            StencilOp.IncrSat => "wgpu::StencilOperation::IncrementClamp",
            StencilOp.DecrSat => "wgpu::StencilOperation::DecrementClamp",
            StencilOp.Invert => "wgpu::StencilOperation::Invert",
            StencilOp.IncrWrap => "wgpu::StencilOperation::IncrementWrap",
            StencilOp.DecrWrap => "wgpu::StencilOperation::DecrementWrap",
            _ => "wgpu::StencilOperation::Keep",
        };

    private static string ComparisonToCompareFunction(ComparisonMode m) =>
        m switch
        {
            ComparisonMode.Never => "wgpu::CompareFunction::Never",
            ComparisonMode.Less => "wgpu::CompareFunction::Less",
            ComparisonMode.Equal => "wgpu::CompareFunction::Equal",
            ComparisonMode.LEqual => "wgpu::CompareFunction::LessEqual",
            ComparisonMode.Greater => "wgpu::CompareFunction::Greater",
            ComparisonMode.NotEqual => "wgpu::CompareFunction::NotEqual",
            ComparisonMode.GEqual => "wgpu::CompareFunction::GreaterEqual",
            ComparisonMode.Always => "wgpu::CompareFunction::Always",
            ComparisonMode.Off => "wgpu::CompareFunction::Always",
            _ => "wgpu::CompareFunction::LessEqual",
        };

    /// <summary>Blend + write mask for one color attachment.</summary>
    public static string EmitColorTargetState(PassFixedFunctionState s, string surfaceFormatIdent)
    {
        string blend = EmitBlendComponent(s);
        string mask = EmitColorWrites(s);
        return $@"Some(wgpu::ColorTargetState {{
                format: {surfaceFormatIdent},
                blend: {blend},
                write_mask: {mask},
            }})";
    }

    private static string EmitBlendComponent(PassFixedFunctionState s)
    {
        PassBlendStateRt0? b = s.BlendRt0;
        if (b is null)
            return "None";
        if (b.BlendDisabled)
            return "None";
        if (b.HasPropertyReference || b.SourceRgb is null || b.DestRgb is null || b.SourceAlpha is null || b.DestAlpha is null)
            return @"Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                })";

        return $@"Some(wgpu::BlendState {{
                    color: wgpu::BlendComponent {{
                        src_factor: {BlendFactorToWgpu(b.SourceRgb.Value)},
                        dst_factor: {BlendFactorToWgpu(b.DestRgb.Value)},
                        operation: wgpu::BlendOperation::Add,
                    }},
                    alpha: wgpu::BlendComponent {{
                        src_factor: {BlendFactorToWgpu(b.SourceAlpha.Value)},
                        dst_factor: {BlendFactorToWgpu(b.DestAlpha.Value)},
                        operation: wgpu::BlendOperation::Add,
                    }},
                }})";
    }

    private static string BlendFactorToWgpu(BlendFactor f) =>
        f switch
        {
            BlendFactor.Zero => "wgpu::BlendFactor::Zero",
            BlendFactor.One => "wgpu::BlendFactor::One",
            BlendFactor.SrcColor => "wgpu::BlendFactor::Src",
            BlendFactor.OneMinusSrcColor => "wgpu::BlendFactor::OneMinusSrc",
            BlendFactor.DstColor => "wgpu::BlendFactor::Dst",
            BlendFactor.OneMinusDstColor => "wgpu::BlendFactor::OneMinusDst",
            BlendFactor.SrcAlpha => "wgpu::BlendFactor::SrcAlpha",
            BlendFactor.OneMinusSrcAlpha => "wgpu::BlendFactor::OneMinusSrcAlpha",
            BlendFactor.DstAlpha => "wgpu::BlendFactor::DstAlpha",
            BlendFactor.OneMinusDstAlpha => "wgpu::BlendFactor::OneMinusDstAlpha",
            BlendFactor.SrcAlphaSaturate => "wgpu::BlendFactor::SrcAlphaSaturated",
            _ => "wgpu::BlendFactor::One",
        };

    private static string EmitColorWrites(PassFixedFunctionState s)
    {
        if (s.ColorMaskReferencesProperty)
            return "wgpu::ColorWrites::ALL";
        string? m = s.ColorMask;
        if (string.IsNullOrEmpty(m) || m == "RGBA")
            return "wgpu::ColorWrites::ALL";
        if (m == "0" || m == "____")
            return "wgpu::ColorWrites::from_bits_truncate(0)";
        m = m.ToUpperInvariant();
        if (m.Contains('_'))
            return "wgpu::ColorWrites::ALL";
        var parts = new List<string>();
        if (m.Contains('R'))
            parts.Add("wgpu::ColorWrites::RED");
        if (m.Contains('G'))
            parts.Add("wgpu::ColorWrites::GREEN");
        if (m.Contains('B'))
            parts.Add("wgpu::ColorWrites::BLUE");
        if (m.Contains('A'))
            parts.Add("wgpu::ColorWrites::ALPHA");
        return parts.Count == 0 ? "wgpu::ColorWrites::from_bits_truncate(0)" : string.Join(" | ", parts);
    }
}
