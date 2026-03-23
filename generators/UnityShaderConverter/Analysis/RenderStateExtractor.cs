using UnityShaderParser.ShaderLab;

namespace UnityShaderConverter.Analysis;

/// <summary>Builds <see cref="PassFixedFunctionState"/> from ShaderLab pass commands and subshader tags.</summary>
public static class RenderStateExtractor
{
    /// <summary>Walks pass commands (last wins) and merges tags with <paramref name="subShaderTags"/>.</summary>
    public static PassFixedFunctionState Extract(
        IReadOnlyList<ShaderLabCommandNode>? commands,
        IReadOnlyDictionary<string, string> subShaderTags)
    {
        var passTags = new Dictionary<string, string>(StringComparer.Ordinal);
        CullMode? cull = null;
        bool cullProp = false;
        bool? zwrite = null;
        bool zwriteProp = false;
        ComparisonMode? ztest = null;
        bool ztestProp = false;
        PassBlendStateRt0? blend = null;
        PassStencilConcrete? stencil = null;
        bool stencilProp = false;
        string? colorMask = null;
        bool colorMaskProp = false;
        (float, float)? offset = null;
        bool offsetProp = false;
        int? lod = null;

        foreach (ShaderLabCommandNode cmd in commands ?? Array.Empty<ShaderLabCommandNode>())
        {
            switch (cmd)
            {
                case ShaderLabCommandTagsNode tn when tn.Tags is not null:
                    foreach (KeyValuePair<string, string> kv in tn.Tags)
                        passTags[kv.Key] = kv.Value;
                    break;
                case ShaderLabCommandCullNode c:
                    if (c.Mode.IsPropertyReference)
                    {
                        cullProp = true;
                        cull = null;
                    }
                    else
                    {
                        cullProp = false;
                        cull = c.Mode.Value;
                    }

                    break;
                case ShaderLabCommandZWriteNode zw:
                    if (zw.Enabled.IsPropertyReference)
                    {
                        zwriteProp = true;
                        zwrite = null;
                    }
                    else
                    {
                        zwriteProp = false;
                        zwrite = zw.Enabled.Value;
                    }

                    break;
                case ShaderLabCommandZTestNode zt:
                    if (zt.Mode.IsPropertyReference)
                    {
                        ztestProp = true;
                        ztest = null;
                    }
                    else
                    {
                        ztestProp = false;
                        ztest = zt.Mode.Value;
                    }

                    break;
                case ShaderLabCommandBlendNode b when b.RenderTarget == 0:
                    blend = ExtractBlendRt0(b);
                    break;
                case ShaderLabCommandColorMaskNode cm:
                    if (cm.Mask.IsPropertyReference)
                    {
                        colorMaskProp = true;
                        colorMask = null;
                    }
                    else
                    {
                        colorMaskProp = false;
                        colorMask = cm.Mask.Value;
                    }

                    break;
                case ShaderLabCommandOffsetNode off:
                    if (off.Factor.IsPropertyReference || off.Units.IsPropertyReference)
                    {
                        offsetProp = true;
                        offset = null;
                    }
                    else
                    {
                        offsetProp = false;
                        offset = (off.Factor.Value, off.Units.Value);
                    }

                    break;
                case ShaderLabCommandStencilNode st:
                    if (StencilAnyPropertyReference(st))
                    {
                        stencilProp = true;
                        stencil = null;
                    }
                    else
                    {
                        stencilProp = false;
                        stencil = BuildStencilConcrete(st);
                    }

                    break;
                case ShaderLabCommandLodNode l:
                    lod = l.LodLevel;
                    break;
            }
        }

        var effective = new Dictionary<string, string>(subShaderTags, StringComparer.Ordinal);
        foreach (KeyValuePair<string, string> kv in passTags)
            effective[kv.Key] = kv.Value;

        return new PassFixedFunctionState
        {
            CullMode = cull,
            CullReferencesProperty = cullProp,
            DepthWrite = zwrite,
            DepthWriteReferencesProperty = zwriteProp,
            DepthTest = ztest,
            DepthTestReferencesProperty = ztestProp,
            BlendRt0 = blend,
            Stencil = stencil,
            StencilReferencesProperty = stencilProp,
            ColorMask = colorMask,
            ColorMaskReferencesProperty = colorMaskProp,
            DepthBias = offset,
            DepthBiasReferencesProperty = offsetProp,
            Lod = lod,
            EffectiveTags = effective,
        };
    }

    private static PassBlendStateRt0 ExtractBlendRt0(ShaderLabCommandBlendNode b)
    {
        if (!b.Enabled)
        {
            return new PassBlendStateRt0 { BlendDisabled = true };
        }

        bool prop = IsBlendProp(b.SourceFactorRGB) || IsBlendProp(b.DestinationFactorRGB) ||
                    IsBlendProp(b.SourceFactorAlpha) || IsBlendProp(b.DestinationFactorAlpha);
        return new PassBlendStateRt0
        {
            BlendDisabled = false,
            HasPropertyReference = prop,
            SourceRgb = FactorOrNull(b.SourceFactorRGB),
            DestRgb = FactorOrNull(b.DestinationFactorRGB),
            SourceAlpha = FactorOrNull(b.SourceFactorAlpha),
            DestAlpha = FactorOrNull(b.DestinationFactorAlpha),
        };
    }

    private static bool IsBlendProp(PropertyReferenceOr<BlendFactor>? f) =>
        f is { IsPropertyReference: true };

    private static BlendFactor? FactorOrNull(PropertyReferenceOr<BlendFactor>? f)
    {
        if (f is null || f.Value.IsPropertyReference)
            return null;
        return f.Value.Value;
    }

    private static bool StencilAnyPropertyReference(ShaderLabCommandStencilNode st) =>
        st.Ref.IsPropertyReference || st.ReadMask.IsPropertyReference || st.WriteMask.IsPropertyReference ||
        st.ComparisonOperationFront.IsPropertyReference || st.PassOperationFront.IsPropertyReference ||
        st.FailOperationFront.IsPropertyReference || st.ZFailOperationFront.IsPropertyReference ||
        st.ComparisonOperationBack.IsPropertyReference || st.PassOperationBack.IsPropertyReference ||
        st.FailOperationBack.IsPropertyReference || st.ZFailOperationBack.IsPropertyReference;

    private static PassStencilConcrete BuildStencilConcrete(ShaderLabCommandStencilNode st) =>
        new()
        {
            Ref = st.Ref.Value,
            ReadMask = st.ReadMask.Value,
            WriteMask = st.WriteMask.Value,
            CompFront = st.ComparisonOperationFront.Value,
            PassFront = st.PassOperationFront.Value,
            FailFront = st.FailOperationFront.Value,
            ZFailFront = st.ZFailOperationFront.Value,
            CompBack = st.ComparisonOperationBack.Value,
            PassBack = st.PassOperationBack.Value,
            FailBack = st.FailOperationBack.Value,
            ZFailBack = st.ZFailOperationBack.Value,
        };
}
