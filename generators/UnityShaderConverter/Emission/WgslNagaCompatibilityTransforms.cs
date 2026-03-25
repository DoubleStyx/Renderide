using System.Text;
using System.Text.RegularExpressions;

namespace UnityShaderConverter.Emission;

/// <summary>
/// WGSL tweaks so output is accepted by Naga (used by wgpu): Slang sometimes emits
/// <c>array&lt;T, i32(N)&gt;</c> for uniform blocks, which is not treated as host-shareable.
/// </summary>
public static class WgslNagaCompatibilityTransforms
{
    private static readonly Regex CommaInt32Size = new(
        @",\s*i32\s*\(\s*([0-9]+)\s*\)",
        RegexOptions.CultureInvariant,
        TimeSpan.FromSeconds(1));

    private static readonly Regex CommaUInt32Size = new(
        @",\s*u32\s*\(\s*([0-9]+)\s*\)",
        RegexOptions.CultureInvariant,
        TimeSpan.FromSeconds(1));

    /// <summary>
    /// Rewrites <c>array&lt;…, i32(N)&gt;</c> / <c>u32(N)</c> size operands to plain <c>, N</c> inside each
    /// top-level <c>array&lt;…&gt;</c> generic (handles nested <c>&lt;</c> inside the element type).
    /// </summary>
    public static string NormalizeHostSharableArraySizes(string wgsl)
    {
        if (string.IsNullOrEmpty(wgsl) || wgsl.IndexOf("array<", StringComparison.Ordinal) < 0)
            return wgsl;

        var sb = new StringBuilder(wgsl.Length);
        int i = 0;
        while (i < wgsl.Length)
        {
            int idx = wgsl.IndexOf("array<", i, StringComparison.Ordinal);
            if (idx < 0)
            {
                sb.Append(wgsl.AsSpan(i));
                break;
            }

            sb.Append(wgsl.AsSpan(i, idx - i));
            int typeStart = idx + "array<".Length;
            int depth = 1;
            int j = typeStart;
            for (; j < wgsl.Length; j++)
            {
                char c = wgsl[j];
                if (c == '<')
                    depth++;
                else if (c == '>')
                {
                    depth--;
                    if (depth == 0)
                    {
                        j++;
                        break;
                    }
                }
            }

            if (j > wgsl.Length || depth != 0)
            {
                sb.Append(wgsl.AsSpan(idx));
                break;
            }

            int innerLen = j - 1 - typeStart;
            string inner = innerLen > 0 ? wgsl.Substring(typeStart, innerLen) : string.Empty;
            inner = CommaInt32Size.Replace(inner, ", $1");
            inner = CommaUInt32Size.Replace(inner, ", $1");
            sb.Append("array<").Append(inner).Append('>');
            i = j;
        }

        return sb.ToString();
    }
}
