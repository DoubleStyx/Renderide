using System.Globalization;
using System.Text;
using UnityShaderConverter.Analysis;
using UnityShaderConverter.Variants;
using UnityShaderParser.ShaderLab;

namespace UnityShaderConverter.Emission;

/// <summary>Dynamic fixed-function Rust emission when ShaderLab references material properties.</summary>
public static partial class RustEmitter
{
    private static bool ShaderUsesDynamicRenderState(ShaderFileDocument document) =>
        document.Passes.Any(static p => p.FixedFunctionState.HasDynamicRenderState());

    private static void EmitMaterialDynamicRenderStateHelpers(StringBuilder sb, ShaderFileDocument document)
    {
        if (!ShaderUsesDynamicRenderState(document))
            return;

        sb.AppendLine("/// Unity material floats mapped to WGPU fixed-function enums (property-driven ShaderLab).");
        sb.AppendLine("impl Material {");
        sb.AppendLine("    #[inline]");
        sb.AppendLine("    fn unity_cull_to_face(v: f32) -> Option<wgpu::Face> {");
        sb.AppendLine("        match v.round() as i32 {");
        sb.AppendLine("            0 => None,");
        sb.AppendLine("            1 => Some(wgpu::Face::Front),");
        sb.AppendLine("            _ => Some(wgpu::Face::Back),");
        sb.AppendLine("        }");
        sb.AppendLine("    }");
        sb.AppendLine();
        sb.AppendLine("    #[inline]");
        sb.AppendLine("    fn unity_compare_to_wgpu(v: f32) -> wgpu::CompareFunction {");
        sb.AppendLine("        match v.round() as i32 {");
        sb.AppendLine("            1 => wgpu::CompareFunction::Never,");
        sb.AppendLine("            2 => wgpu::CompareFunction::Less,");
        sb.AppendLine("            3 => wgpu::CompareFunction::Equal,");
        sb.AppendLine("            4 => wgpu::CompareFunction::LessEqual,");
        sb.AppendLine("            5 => wgpu::CompareFunction::Greater,");
        sb.AppendLine("            6 => wgpu::CompareFunction::NotEqual,");
        sb.AppendLine("            7 => wgpu::CompareFunction::GreaterEqual,");
        sb.AppendLine("            8 => wgpu::CompareFunction::Always,");
        sb.AppendLine("            _ => wgpu::CompareFunction::LessEqual,");
        sb.AppendLine("        }");
        sb.AppendLine("    }");
        sb.AppendLine();
        sb.AppendLine("    #[inline]");
        sb.AppendLine("    fn unity_blend_to_wgpu(v: f32) -> wgpu::BlendFactor {");
        sb.AppendLine("        match v.round() as i32 {");
        sb.AppendLine("            0 => wgpu::BlendFactor::Zero,");
        sb.AppendLine("            1 => wgpu::BlendFactor::One,");
        sb.AppendLine("            2 => wgpu::BlendFactor::Dst,");
        sb.AppendLine("            3 => wgpu::BlendFactor::Src,");
        sb.AppendLine("            4 => wgpu::BlendFactor::OneMinusDst,");
        sb.AppendLine("            5 => wgpu::BlendFactor::SrcAlpha,");
        sb.AppendLine("            6 => wgpu::BlendFactor::OneMinusSrcAlpha,");
        sb.AppendLine("            7 => wgpu::BlendFactor::DstAlpha,");
        sb.AppendLine("            8 => wgpu::BlendFactor::OneMinusDstAlpha,");
        sb.AppendLine("            9 => wgpu::BlendFactor::SrcAlphaSaturated,");
        sb.AppendLine("            10 => wgpu::BlendFactor::OneMinusSrc,");
        sb.AppendLine("            _ => wgpu::BlendFactor::One,");
        sb.AppendLine("        }");
        sb.AppendLine("    }");
        sb.AppendLine();
        sb.AppendLine("    #[inline]");
        sb.AppendLine("    fn unity_stencil_op_to_wgpu(v: f32) -> wgpu::StencilOperation {");
        sb.AppendLine("        match v.round() as i32 {");
        sb.AppendLine("            1 => wgpu::StencilOperation::Zero,");
        sb.AppendLine("            2 => wgpu::StencilOperation::Replace,");
        sb.AppendLine("            3 => wgpu::StencilOperation::IncrementClamp,");
        sb.AppendLine("            4 => wgpu::StencilOperation::DecrementClamp,");
        sb.AppendLine("            5 => wgpu::StencilOperation::Invert,");
        sb.AppendLine("            6 => wgpu::StencilOperation::IncrementWrap,");
        sb.AppendLine("            7 => wgpu::StencilOperation::DecrementWrap,");
        sb.AppendLine("            _ => wgpu::StencilOperation::Keep,");
        sb.AppendLine("        }");
        sb.AppendLine("    }");
        sb.AppendLine();
        sb.AppendLine("    #[inline]");
        sb.AppendLine("    fn unity_color_mask_bits(v: f32) -> wgpu::ColorWrites {");
        sb.AppendLine("        let bits = (v.round() as u8).min(15);");
        sb.AppendLine("        let mut m = wgpu::ColorWrites::empty();");
        sb.AppendLine("        if bits & 1 != 0 { m |= wgpu::ColorWrites::RED; }");
        sb.AppendLine("        if bits & 2 != 0 { m |= wgpu::ColorWrites::GREEN; }");
        sb.AppendLine("        if bits & 4 != 0 { m |= wgpu::ColorWrites::BLUE; }");
        sb.AppendLine("        if bits & 8 != 0 { m |= wgpu::ColorWrites::ALPHA; }");
        sb.AppendLine("        if bits == 0 { wgpu::ColorWrites::empty() } else { m }");
        sb.AppendLine("    }");
        sb.AppendLine("}");
        sb.AppendLine();
    }

    private static void EmitPipelineHelpersDynamicAware(
        StringBuilder sb,
        ShaderFileDocument document,
        IReadOnlyList<SpecializationAxis> axes,
        IReadOnlyList<PassVertexLayout> vertexLayoutsPerPass)
    {
        int passCount = document.Passes.Count;
        bool anyDynamic = ShaderUsesDynamicRenderState(document);

        for (int pi = 0; pi < passCount; pi++)
        {
            PassVertexLayout vLayout = pi < vertexLayoutsPerPass.Count ? vertexLayoutsPerPass[pi] : PassVertexLayout.Empty;
            EmitPassVertexBufferLayout(sb, vLayout, pi);
        }

        EmitMaterialUniformBlock(sb, document.Properties);
        EmitSpecializationPipelineOptions(sb, axes, passCount);
        EmitMaterialBindGroupRust(sb, document.Properties);

        for (int pi = 0; pi < passCount; pi++)
        {
            PassFixedFunctionState ff = document.Passes[pi].FixedFunctionState;
            sb.Append("/// Primitive assembly for pass ").Append(pi).Append(" (from ShaderLab `Cull`)");
            if (ff.EffectiveTags.Count > 0)
            {
                sb.Append(". Tags: ");
                sb.Append(
                    string.Join(
                        "; ",
                        ff.EffectiveTags.OrderBy(static k => k.Key, StringComparer.Ordinal)
                            .Select(kv => $"`{EscapeComment(kv.Key)}` = `{EscapeComment(kv.Value)}`")));
            }

            sb.AppendLine(".");
            if (anyDynamic)
            {
                sb.Append("pub fn primitive_state_pass").Append(pi).AppendLine("(material: &Material) -> wgpu::PrimitiveState {");
                sb.AppendLine(EmitDynamicPrimitiveStateBody(ff));
                sb.AppendLine("}");
            }
            else
            {
                sb.Append("pub fn primitive_state_pass").Append(pi).AppendLine("() -> wgpu::PrimitiveState {");
                sb.AppendLine(FixedFunctionRustEmitter.EmitPrimitiveState(ff));
                sb.AppendLine("}");
            }

            sb.AppendLine();
            sb.Append("/// Depth/stencil for pass ").Append(pi).Append(" when a depth attachment is used.");
            sb.AppendLine();
            if (anyDynamic)
            {
                sb.Append("pub fn depth_stencil_state_pass").Append(pi).AppendLine("(material: &Material, depth_format: wgpu::TextureFormat) -> wgpu::DepthStencilState {");
                sb.AppendLine(EmitDynamicDepthStencilBody(ff));
                sb.AppendLine("}");
            }
            else
            {
                sb.Append("pub fn depth_stencil_state_pass").Append(pi).AppendLine("(depth_format: wgpu::TextureFormat) -> wgpu::DepthStencilState {");
                sb.AppendLine(FixedFunctionRustEmitter.EmitDepthStencilState(ff, "depth_format"));
                sb.AppendLine("}");
            }

            sb.AppendLine();
            sb.Append("/// First color target for pass ").Append(pi).Append(" (blend + write mask from ShaderLab).");
            sb.AppendLine();
            if (anyDynamic)
            {
                sb.Append("pub fn color_target_state_pass").Append(pi).AppendLine("(material: &Material, surface_format: wgpu::TextureFormat) -> Option<wgpu::ColorTargetState> {");
                sb.Append("    ");
                sb.AppendLine(EmitDynamicColorTargetExpr(ff, "surface_format"));
                sb.AppendLine("}");
            }
            else
            {
                sb.Append("pub fn color_target_state_pass").Append(pi).AppendLine("(surface_format: wgpu::TextureFormat) -> Option<wgpu::ColorTargetState> {");
                sb.Append("    ");
                sb.AppendLine(FixedFunctionRustEmitter.EmitColorTargetState(ff, "surface_format"));
                sb.AppendLine("}");
            }

            sb.AppendLine();
        }

        for (int pi = 0; pi < passCount; pi++)
        {
            string fnStem = ShaderNaming.WgslPassStem(document.Passes[pi].PassName, pi);
            sb.Append("/// Builds a render pipeline for pass ").Append(pi).Append(" using generated fixed-function state.");
            sb.AppendLine();
            sb.Append("pub fn create_render_pipeline_pass").Append(pi).AppendLine("(");
            sb.AppendLine("    device: &wgpu::Device,");
            sb.AppendLine("    label: &str,");
            sb.AppendLine("    layout: &wgpu::PipelineLayout,");
            sb.AppendLine("    surface_format: wgpu::TextureFormat,");
            sb.AppendLine("    depth_format: Option<wgpu::TextureFormat>,");
            sb.AppendLine("    variant: &VariantKey,");
            sb.AppendLine("    vertex_layout_override: Option<&[wgpu::VertexBufferLayout]>,");
            if (anyDynamic)
                sb.AppendLine("    material: &Material,");
            sb.AppendLine(") -> wgpu::RenderPipeline {");
            sb.Append("    let vertex_buffers: &[wgpu::VertexBufferLayout] = match vertex_layout_override {");
            sb.Append(" Some(b) => b, None => VERTEX_BUFFER_LAYOUTS_PASS").Append(pi).AppendLine(" };");
            sb.AppendLine("    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {");
            sb.Append("        label: Some(");
            sb.Append('"').Append(EscapeRustString(document.ShaderName)).AppendLine("\"),");
            sb.Append("        source: shader_source_").Append(fnStem).AppendLine("(),");
            sb.AppendLine("    });");
            sb.Append("    let vertex_opts = pipeline_compilation_options_pass").Append(pi).AppendLine("_vertex(variant);");
            sb.Append("    let fragment_opts = pipeline_compilation_options_pass").Append(pi).AppendLine("_fragment(variant);");
            string colorCall = anyDynamic
                ? $"color_target_state_pass{pi}(material, surface_format)"
                : $"color_target_state_pass{pi}(surface_format)";
            string primCall = anyDynamic ? $"primitive_state_pass{pi}(material)" : $"primitive_state_pass{pi}()";
            string depthCall = anyDynamic
                ? $"depth_format.map(|df| depth_stencil_state_pass{pi}(material, df))"
                : $"depth_format.map(depth_stencil_state_pass{pi})";
            sb.AppendLine("    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {");
            sb.AppendLine("        label: Some(label),");
            sb.AppendLine("        layout: Some(layout),");
            sb.AppendLine("        vertex: wgpu::VertexState {");
            sb.AppendLine("            module: &module,");
            sb.Append("            entry_point: Some(\"");
            sb.Append(EscapeRustString(document.Passes[pi].VertexEntry!));
            sb.AppendLine("\"),");
            sb.AppendLine("            buffers: vertex_buffers,");
            sb.AppendLine("            compilation_options: vertex_opts,");
            sb.AppendLine("        },");
            sb.AppendLine("        fragment: Some(wgpu::FragmentState {");
            sb.AppendLine("            module: &module,");
            sb.Append("            entry_point: Some(\"");
            sb.Append(EscapeRustString(document.Passes[pi].FragmentEntry!));
            sb.AppendLine("\"),");
            sb.Append("            targets: &[").Append(colorCall).AppendLine("],");
            sb.AppendLine("            compilation_options: fragment_opts,");
            sb.AppendLine("        }),");
            sb.Append("        primitive: ").Append(primCall).AppendLine(",");
            sb.Append("        depth_stencil: ").Append(depthCall).AppendLine(",");
            sb.AppendLine("        multisample: wgpu::MultisampleState::default(),");
            sb.AppendLine("        multiview_mask: None,");
            sb.AppendLine("        cache: None,");
            sb.AppendLine("    })");
            sb.AppendLine("}");
            sb.AppendLine();
        }

        EmitCreateRenderPipelineDispatcherDynamicAware(sb, passCount, anyDynamic);
    }

    private static void EmitCreateRenderPipelineDispatcherDynamicAware(StringBuilder sb, int passCount, bool anyDynamic)
    {
        sb.AppendLine("/// Dispatches to `create_render_pipeline_passN` by index. Panics if `pass_index` is out of range.");
        sb.AppendLine("#[allow(clippy::too_many_arguments)]");
        sb.AppendLine("pub fn create_render_pipeline(");
        sb.AppendLine("    device: &wgpu::Device,");
        sb.AppendLine("    label: &str,");
        sb.AppendLine("    layout: &wgpu::PipelineLayout,");
        sb.AppendLine("    surface_format: wgpu::TextureFormat,");
        sb.AppendLine("    depth_format: Option<wgpu::TextureFormat>,");
        sb.AppendLine("    variant: &VariantKey,");
        sb.AppendLine("    pass_index: usize,");
        sb.AppendLine("    vertex_layout_override: Option<&[wgpu::VertexBufferLayout]>,");
        if (anyDynamic)
            sb.AppendLine("    material: &Material,");
        sb.AppendLine(") -> wgpu::RenderPipeline {");
        sb.AppendLine("    match pass_index {");
        for (int pi = 0; pi < passCount; pi++)
        {
            if (anyDynamic)
            {
                sb.Append("        ").Append(pi).Append(" => create_render_pipeline_pass").Append(pi).AppendLine(
                    "(device, label, layout, surface_format, depth_format, variant, vertex_layout_override, material),");
            }
            else
            {
                sb.Append("        ").Append(pi).Append(" => create_render_pipeline_pass").Append(pi).AppendLine(
                    "(device, label, layout, surface_format, depth_format, variant, vertex_layout_override),");
            }
        }

        sb.AppendLine("        _ => panic!(\"invalid pass_index for this shader module\"),");
        sb.AppendLine("    }");
        sb.AppendLine("}");
    }

    private static string EmitDynamicPrimitiveStateBody(PassFixedFunctionState s)
    {
        string cullFace = s.CullReferencesProperty && !string.IsNullOrEmpty(s.CullPropertyUniformName)
            ? $"Material::unity_cull_to_face(material.{RustFieldName(s.CullPropertyUniformName)})"
            : (s.CullMode ?? CullMode.Back) switch
            {
                CullMode.Off => "None",
                CullMode.Front => "Some(wgpu::Face::Front)",
                _ => "Some(wgpu::Face::Back)",
            };

        var sb = new StringBuilder();
        sb.AppendLine("    wgpu::PrimitiveState {");
        sb.AppendLine("        topology: wgpu::PrimitiveTopology::TriangleList,");
        sb.AppendLine("        strip_index_format: None,");
        sb.AppendLine("        front_face: wgpu::FrontFace::Ccw,");
        sb.Append("        cull_mode: ");
        sb.Append(cullFace);
        sb.AppendLine(",");
        sb.AppendLine("        unclipped_depth: false,");
        sb.AppendLine("        polygon_mode: wgpu::PolygonMode::Fill,");
        sb.AppendLine("        conservative: false,");
        sb.AppendLine("    }");
        return sb.ToString();
    }

    private static string EmitDynamicDepthStencilBody(PassFixedFunctionState s)
    {
        string depthCmpExpr;
        if (s.DepthTestReferencesProperty && !string.IsNullOrEmpty(s.DepthTestPropertyUniformName))
            depthCmpExpr = $"Material::unity_compare_to_wgpu(material.{RustFieldName(s.DepthTestPropertyUniformName)})";
        else
        {
            ComparisonMode m = s.DepthTest ?? ComparisonMode.LEqual;
            if (m == ComparisonMode.Off)
                depthCmpExpr = "wgpu::CompareFunction::Always";
            else
                depthCmpExpr = FixedFunctionRustEmitter.CompareFunctionPath(m);
        }

        string depthWriteExpr;
        if (s.DepthWriteReferencesProperty && !string.IsNullOrEmpty(s.DepthWritePropertyUniformName))
            depthWriteExpr = $"material.{RustFieldName(s.DepthWritePropertyUniformName)} > 0.5";
        else if ((s.DepthTest ?? ComparisonMode.LEqual) == ComparisonMode.Off)
            depthWriteExpr = "false";
        else
            depthWriteExpr = (s.DepthWrite ?? true) ? "true" : "false";

        if (s.DepthBiasReferencesProperty)
        {
            string units = !string.IsNullOrEmpty(s.DepthBiasUnitsPropertyUniformName)
                ? $"material.{RustFieldName(s.DepthBiasUnitsPropertyUniformName)}.round() as i32"
                : "0";
            string slopeScale = !string.IsNullOrEmpty(s.DepthBiasFactorPropertyUniformName)
                ? $"material.{RustFieldName(s.DepthBiasFactorPropertyUniformName)}"
                : "0.0_f32";
            return $@"    let depth_compare = {depthCmpExpr};
    let depth_write = {depthWriteExpr};
    wgpu::DepthStencilState {{
        format: depth_format,
        depth_write_enabled: depth_write,
        depth_compare: depth_compare,
        stencil: {EmitDynamicStencilExpr(s)},
        bias: wgpu::DepthBiasState {{
            constant: {units},
            slope_scale: {slopeScale},
            clamp: 0.0,
        }},
    }}";
        }

        string biasConst = s.DepthBias is null ? "0" : ((int)Math.Round(s.DepthBias.Value.Units, MidpointRounding.AwayFromZero)).ToString(CultureInfo.InvariantCulture);
        string slope = s.DepthBias is null ? "0.0" : s.DepthBias.Value.Factor.ToString(CultureInfo.InvariantCulture) + "f32";
        return $@"    let depth_compare = {depthCmpExpr};
    let depth_write = {depthWriteExpr};
    wgpu::DepthStencilState {{
        format: depth_format,
        depth_write_enabled: depth_write,
        depth_compare: depth_compare,
        stencil: {EmitDynamicStencilExpr(s)},
        bias: wgpu::DepthBiasState {{
            constant: {biasConst},
            slope_scale: {slope},
            clamp: 0.0,
        }},
    }}";
    }

    private static string EmitDynamicStencilExpr(PassFixedFunctionState s)
    {
        if (!s.StencilReferencesProperty)
        {
            if (s.Stencil is null)
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
        front: {FixedFunctionRustEmitter.StencilFaceRust(t.CompFront, t.PassFront, t.FailFront, t.ZFailFront)},
        back: {FixedFunctionRustEmitter.StencilFaceRust(t.CompBack, t.PassBack, t.FailBack, t.ZFailBack)},
        read_mask: {t.ReadMask}u32,
        write_mask: {t.WriteMask}u32,
    }}";
        }

        string readRef = !string.IsNullOrEmpty(s.StencilReadMaskPropertyUniformName)
            ? $"material.{RustFieldName(s.StencilReadMaskPropertyUniformName)}.round() as u32"
            : "255u32";
        string writeRef = !string.IsNullOrEmpty(s.StencilWriteMaskPropertyUniformName)
            ? $"material.{RustFieldName(s.StencilWriteMaskPropertyUniformName)}.round() as u32"
            : "255u32";
        string comp = !string.IsNullOrEmpty(s.StencilCompPropertyUniformName)
            ? $"Material::unity_compare_to_wgpu(material.{RustFieldName(s.StencilCompPropertyUniformName)})"
            : "wgpu::CompareFunction::Always";
        string passOp = !string.IsNullOrEmpty(s.StencilPassPropertyUniformName)
            ? $"Material::unity_stencil_op_to_wgpu(material.{RustFieldName(s.StencilPassPropertyUniformName)})"
            : "wgpu::StencilOperation::Keep";

        return $@"wgpu::StencilState {{
        front: wgpu::StencilFaceState {{
            compare: {comp},
            fail_op: wgpu::StencilOperation::Keep,
            depth_fail_op: wgpu::StencilOperation::Keep,
            pass_op: {passOp},
        }},
        back: wgpu::StencilFaceState {{
            compare: {comp},
            fail_op: wgpu::StencilOperation::Keep,
            depth_fail_op: wgpu::StencilOperation::Keep,
            pass_op: {passOp},
        }},
        read_mask: {readRef},
        write_mask: {writeRef},
    }}";
    }

    private static string EmitDynamicColorTargetExpr(PassFixedFunctionState s, string surfaceFormatIdent)
    {
        PassBlendStateRt0? b = s.BlendRt0;
        string maskExpr;
        if (s.ColorMaskReferencesProperty && !string.IsNullOrEmpty(s.ColorMaskPropertyUniformName))
            maskExpr = $"Material::unity_color_mask_bits(material.{RustFieldName(s.ColorMaskPropertyUniformName)})";
        else
            maskExpr = EmitStaticColorWritesInner(s);

        if (b is null)
            return $"Some(wgpu::ColorTargetState {{ format: {surfaceFormatIdent}, blend: None, write_mask: {maskExpr} }})";
        if (b.BlendDisabled)
            return $"Some(wgpu::ColorTargetState {{ format: {surfaceFormatIdent}, blend: None, write_mask: {maskExpr} }})";

        if (b.HasPropertyReference)
        {
            string srcRgb = !string.IsNullOrEmpty(b.SrcRgbPropertyUniformName)
                ? $"Material::unity_blend_to_wgpu(material.{RustFieldName(b.SrcRgbPropertyUniformName)})"
                : "wgpu::BlendFactor::SrcAlpha";
            string dstRgb = !string.IsNullOrEmpty(b.DstRgbPropertyUniformName)
                ? $"Material::unity_blend_to_wgpu(material.{RustFieldName(b.DstRgbPropertyUniformName)})"
                : "wgpu::BlendFactor::OneMinusSrcAlpha";
            string srcA = !string.IsNullOrEmpty(b.SrcAlphaPropertyUniformName)
                ? $"Material::unity_blend_to_wgpu(material.{RustFieldName(b.SrcAlphaPropertyUniformName)})"
                : "wgpu::BlendFactor::One";
            string dstA = !string.IsNullOrEmpty(b.DstAlphaPropertyUniformName)
                ? $"Material::unity_blend_to_wgpu(material.{RustFieldName(b.DstAlphaPropertyUniformName)})"
                : "wgpu::BlendFactor::OneMinusSrcAlpha";
            return $@"Some(wgpu::ColorTargetState {{
                format: {surfaceFormatIdent},
                blend: Some(wgpu::BlendState {{
                    color: wgpu::BlendComponent {{
                        src_factor: {srcRgb},
                        dst_factor: {dstRgb},
                        operation: wgpu::BlendOperation::Add,
                    }},
                    alpha: wgpu::BlendComponent {{
                        src_factor: {srcA},
                        dst_factor: {dstA},
                        operation: wgpu::BlendOperation::Add,
                    }},
                }}),
                write_mask: {maskExpr},
            }})";
        }

        if (b.SourceRgb is null || b.DestRgb is null || b.SourceAlpha is null || b.DestAlpha is null)
            return $"Some(wgpu::ColorTargetState {{ format: {surfaceFormatIdent}, blend: None, write_mask: {maskExpr} }})";

        return $@"Some(wgpu::ColorTargetState {{
                format: {surfaceFormatIdent},
                blend: Some(wgpu::BlendState {{
                    color: wgpu::BlendComponent {{
                        src_factor: {FixedFunctionRustEmitter.BlendFactorPath(b.SourceRgb.Value)},
                        dst_factor: {FixedFunctionRustEmitter.BlendFactorPath(b.DestRgb.Value)},
                        operation: wgpu::BlendOperation::Add,
                    }},
                    alpha: wgpu::BlendComponent {{
                        src_factor: {FixedFunctionRustEmitter.BlendFactorPath(b.SourceAlpha.Value)},
                        dst_factor: {FixedFunctionRustEmitter.BlendFactorPath(b.DestAlpha.Value)},
                        operation: wgpu::BlendOperation::Add,
                    }},
                }}),
                write_mask: {maskExpr},
            }})";
    }

    private static string EmitStaticColorWritesInner(PassFixedFunctionState s)
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
