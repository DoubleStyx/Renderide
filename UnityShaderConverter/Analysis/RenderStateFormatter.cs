using System.Text;
using UnityShaderParser.ShaderLab;

namespace UnityShaderConverter.Analysis;

/// <summary>Summarizes ShaderLab fixed-function state for generated Rust comments.</summary>
public static class RenderStateFormatter
{
    /// <summary>Produces a short multi-line summary of pass commands.</summary>
    public static string Summarize(IReadOnlyList<ShaderLabCommandNode> commands)
    {
        var sb = new StringBuilder();
        foreach (ShaderLabCommandNode cmd in commands)
        {
            switch (cmd)
            {
                case ShaderLabCommandTagsNode t:
                    sb.Append("Tags ");
                    if (t.Tags is not null)
                    {
                        foreach (KeyValuePair<string, string> kv in t.Tags)
                            sb.Append(kv.Key).Append('=').Append(kv.Value).Append(' ');
                    }

                    sb.AppendLine();
                    break;
                case ShaderLabCommandBlendNode b:
                    sb.Append("Blend enabled=").Append(b.Enabled).Append(" rt=").Append(b.RenderTarget)
                        .Append(" srcRGB=").Append(FormatBlendRef(b.SourceFactorRGB))
                        .Append(" dstRGB=").Append(FormatBlendRef(b.DestinationFactorRGB))
                        .AppendLine();
                    break;
                case ShaderLabCommandBlendOpNode bo:
                    sb.AppendLine("BlendOp (see ShaderLab AST)");
                    break;
                case ShaderLabCommandCullNode c:
                    sb.Append("Cull ").Append(FormatCull(c.Mode)).AppendLine();
                    break;
                case ShaderLabCommandZWriteNode zw:
                    sb.Append("ZWrite ").Append(FormatToggle(zw.Enabled)).AppendLine();
                    break;
                case ShaderLabCommandZTestNode zt:
                    sb.Append("ZTest ").Append(FormatCompare(zt.Mode)).AppendLine();
                    break;
                case ShaderLabCommandColorMaskNode cm:
                    sb.Append("ColorMask ").Append(FormatStringRef(cm.Mask)).AppendLine();
                    break;
                case ShaderLabCommandOffsetNode off:
                    sb.Append("Offset factor=").Append(FormatFloatRef(off.Factor))
                        .Append(" units=").Append(FormatFloatRef(off.Units)).AppendLine();
                    break;
                case ShaderLabCommandStencilNode st:
                    sb.AppendLine("Stencil (see ShaderLab AST)");
                    break;
                case ShaderLabCommandNameNode nm:
                    sb.Append("Name \"").Append(nm.Name).AppendLine("\"");
                    break;
                case ShaderLabCommandLodNode lod:
                    sb.Append("LOD ").Append(lod.LodLevel).AppendLine();
                    break;
                default:
                    sb.Append(cmd.GetType().Name).AppendLine();
                    break;
            }
        }

        return sb.ToString().TrimEnd();
    }

    private static string FormatToggle(PropertyReferenceOr<bool> enabled)
    {
        if (enabled.IsPropertyReference)
            return $"[{enabled.Property}]";
        return enabled.Value ? "On" : "Off";
    }

    private static string FormatCull(PropertyReferenceOr<CullMode> mode)
    {
        if (mode.IsPropertyReference)
            return $"[{mode.Property}]";
        return mode.Value.ToString();
    }

    private static string FormatCompare(PropertyReferenceOr<ComparisonMode> mode)
    {
        if (mode.IsPropertyReference)
            return $"[{mode.Property}]";
        return mode.Value.ToString();
    }

    private static string FormatStringRef(PropertyReferenceOr<string> mask)
    {
        if (mask.IsPropertyReference)
            return $"[{mask.Property}]";
        return mask.Value ?? "";
    }

    private static string FormatFloatRef(PropertyReferenceOr<float> v)
    {
        if (v.IsPropertyReference)
            return $"[{v.Property}]";
        return v.Value.ToString(System.Globalization.CultureInfo.InvariantCulture);
    }

    private static string FormatBlendRef(PropertyReferenceOr<BlendFactor>? factor)
    {
        if (factor is null)
            return "?";
        if (factor.Value.IsPropertyReference)
            return $"[{factor.Value.Property}]";
        return factor.Value.Value.ToString();
    }
}
