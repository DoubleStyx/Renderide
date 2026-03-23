using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace UnityShaderConverter.Analysis;

/// <summary>One user vertex attribute (WGSL <c>@location</c>) for pipeline layout emission.</summary>
public sealed record VertexLayoutAttribute(
    int ShaderLocation,
    string WgslScalarKind,
    string RustVertexFormatPath);

/// <summary>Interleaved vertex buffer layout derived from merged Slang→WGSL output.</summary>
public sealed class PassVertexLayout
{
    /// <summary>Shared empty layout when extraction fails or no <c>@location</c> inputs exist.</summary>
    public static PassVertexLayout Empty { get; } = new(Array.Empty<VertexLayoutAttribute>(), Array.Empty<uint>(), 0);

    /// <summary>Creates a layout with attributes sorted by <see cref="VertexLayoutAttribute.ShaderLocation"/>.</summary>
    public PassVertexLayout(IReadOnlyList<VertexLayoutAttribute> attributes, IReadOnlyList<uint> byteOffsets, uint arrayStride)
    {
        if (attributes.Count != byteOffsets.Count)
            throw new ArgumentException("Offsets must align with attributes.", nameof(byteOffsets));
        Attributes = attributes;
        ByteOffsets = byteOffsets;
        ArrayStride = arrayStride;
    }

    /// <summary>Sorted vertex attributes.</summary>
    public IReadOnlyList<VertexLayoutAttribute> Attributes { get; }

    /// <summary>Interleaved byte offset for each <see cref="Attributes"/> entry.</summary>
    public IReadOnlyList<uint> ByteOffsets { get; }

    /// <summary>Byte stride for the single interleaved <c>VertexBufferLayout</c>.</summary>
    public uint ArrayStride { get; }

    /// <summary>True when <see cref="Attributes"/> is non-empty.</summary>
    public bool HasAttributes => Attributes.Count > 0;
}

/// <summary>
/// Extracts vertex input locations and types from WGSL produced by Slang (e.g. <c>vertexInput_0</c> + <c>@vertex fn vert</c>).
/// </summary>
public static class WgslVertexLayoutExtractor
{
    /// <summary>
    /// Parses <paramref name="wgslText"/> for the <c>@vertex</c> entry <paramref name="vertexEntryName"/> and builds
    /// one interleaved buffer layout. On failure, returns <c>false</c> and an error message.
    /// </summary>
    public static bool TryExtract(string wgslText, string vertexEntryName, out PassVertexLayout layout, out string? error)
    {
        layout = PassVertexLayout.Empty;
        error = null;
        if (string.IsNullOrWhiteSpace(vertexEntryName))
        {
            error = "vertex entry name is empty.";
            return false;
        }

        string stripped = StripWgslComments(wgslText);
        if (!TryFindVertexEntryParameterStruct(stripped, vertexEntryName, out string? structName, out error))
            return false;

        if (structName is null)
        {
            error = "Vertex entry struct name could not be resolved.";
            return false;
        }

        if (!TryParseStructLocationFields(stripped, structName, out List<VertexLayoutAttribute> attrs, out error))
            return false;

        if (attrs.Count == 0)
        {
            layout = PassVertexLayout.Empty;
            return true;
        }

        attrs.Sort(static (a, b) => a.ShaderLocation.CompareTo(b.ShaderLocation));
        if (!TryComputeInterleavedLayout(attrs, out uint stride, out error))
            return false;

        IReadOnlyList<uint> offsets = ComputeOffsets(attrs);
        layout = new PassVertexLayout(attrs, offsets, stride);
        return true;
    }

    private static string StripWgslComments(string source)
    {
        var sb = new StringBuilder(source.Length);
        int i = 0;
        while (i < source.Length)
        {
            char c = source[i];
            if (c == '/' && i + 1 < source.Length && source[i + 1] == '/')
            {
                i += 2;
                while (i < source.Length && source[i] != '\n')
                    i++;
                continue;
            }

            if (c == '/' && i + 1 < source.Length && source[i + 1] == '*')
            {
                i += 2;
                while (i + 1 < source.Length)
                {
                    if (source[i] == '*' && source[i + 1] == '/')
                    {
                        i += 2;
                        break;
                    }

                    i++;
                }

                continue;
            }

            sb.Append(c);
            i++;
        }

        return sb.ToString();
    }

    private static bool TryFindVertexEntryParameterStruct(string text, string entryName, out string? structName, out string? error)
    {
        structName = null;
        error = null;
        // fn <name> ( ... firstParam : TypeName ... )
        string pattern = @"@vertex\s+fn\s+" + Regex.Escape(entryName) + @"\s*\([^)]*\)";
        Match m = Regex.Match(text, pattern, RegexOptions.Singleline);
        if (!m.Success)
        {
            error = $"No @vertex fn `{entryName}(...)` found in WGSL.";
            return false;
        }

        string inside = m.Value;
        int open = inside.IndexOf('(');
        int close = inside.LastIndexOf(')');
        if (open < 0 || close <= open)
        {
            error = $"Malformed vertex entry `{entryName}` parameters.";
            return false;
        }

        string paramList = inside.Substring(open + 1, close - open - 1);
        string[] rawParams = SplitTopLevelCommas(paramList);
        if (rawParams.Length == 0)
        {
            error = $"Vertex entry `{entryName}` has no parameters.";
            return false;
        }

        string first = rawParams[0].Trim();
        int colon = first.IndexOf(':');
        if (colon < 0)
        {
            error = $"First parameter of `{entryName}` has no type (`:`).";
            return false;
        }

        string typePart = first.Substring(colon + 1).Trim();
        // Strip template generics for matching struct name: "vertexInput_0" only
        int angle = typePart.IndexOf('<');
        if (angle >= 0)
            typePart = typePart.Substring(0, angle).Trim();

        if (typePart.Length == 0)
        {
            error = $"Could not read struct type for `{entryName}` first parameter.";
            return false;
        }

        structName = typePart;
        return true;
    }

    private static string[] SplitTopLevelCommas(string paramList)
    {
        var parts = new List<string>();
        int depth = 0;
        int start = 0;
        for (int i = 0; i < paramList.Length; i++)
        {
            char c = paramList[i];
            if (c == '(' || c == '<')
                depth++;
            else if (c == ')' || c == '>')
                depth = Math.Max(0, depth - 1);
            else if (c == ',' && depth == 0)
            {
                parts.Add(paramList.Substring(start, i - start));
                start = i + 1;
            }
        }

        parts.Add(paramList.Substring(start));
        return parts.Select(p => p.Trim()).Where(p => p.Length > 0).ToArray();
    }

    private static bool TryParseStructLocationFields(
        string text,
        string structName,
        out List<VertexLayoutAttribute> attrs,
        out string? error)
    {
        attrs = new List<VertexLayoutAttribute>();
        error = null;
        string structPattern = @"struct\s+" + Regex.Escape(structName) + @"\s*\{";
        Match sm = Regex.Match(text, structPattern);
        if (!sm.Success)
        {
            error = $"Struct `{structName}` not found in WGSL.";
            return false;
        }

        int bodyStart = sm.Index + sm.Length;
        int depth = 1;
        int i = bodyStart;
        for (; i < text.Length && depth > 0; i++)
        {
            if (text[i] == '{')
                depth++;
            else if (text[i] == '}')
                depth--;
        }

        if (depth != 0)
        {
            error = $"Unclosed struct `{structName}`.";
            return false;
        }

        string body = text.Substring(bodyStart, i - 1 - bodyStart);
        var fieldRegex = new Regex(
            @"(?<attrs>(?:@\w+(?:\([^)]*\))?\s*)+)\s*(?<ident>\w+)\s*:\s*(?<type>[^,;}\n]+)\s*[,;]?",
            RegexOptions.Singleline);
        foreach (Match fm in fieldRegex.Matches(body))
        {
            string attrBlock = fm.Groups["attrs"].Value;
            string typeStr = fm.Groups["type"].Value.Trim();
            if (attrBlock.Contains("@builtin", StringComparison.Ordinal))
                continue;
            Match loc = Regex.Match(attrBlock, @"@location\s*\(\s*(\d+)\s*\)");
            if (!loc.Success)
                continue;
            int location = int.Parse(loc.Groups[1].Value, CultureInfo.InvariantCulture);
            if (!TryMapWgslVertexType(typeStr, out string kind, out string formatPath, out error))
                return false;
            attrs.Add(new VertexLayoutAttribute(location, kind, formatPath));
        }

        return true;
    }

    private static bool TryMapWgslVertexType(string typeStr, out string kind, out string rustFormat, out string? error)
    {
        kind = "";
        rustFormat = "";
        error = null;
        string t = typeStr.Replace(" ", "").ToLowerInvariant();
        return t switch
        {
            "f32" => Set("f32", "wgpu::VertexFormat::Float32", out kind, out rustFormat),
            "u32" => Set("u32", "wgpu::VertexFormat::Uint32", out kind, out rustFormat),
            "i32" => Set("i32", "wgpu::VertexFormat::Sint32", out kind, out rustFormat),
            "vec2<f32>" => Set("vec2f", "wgpu::VertexFormat::Float32x2", out kind, out rustFormat),
            "vec3<f32>" => Set("vec3f", "wgpu::VertexFormat::Float32x3", out kind, out rustFormat),
            "vec4<f32>" => Set("vec4f", "wgpu::VertexFormat::Float32x4", out kind, out rustFormat),
            "vec2<u32>" => Set("vec2u", "wgpu::VertexFormat::Uint32x2", out kind, out rustFormat),
            "vec3<u32>" => Set("vec3u", "wgpu::VertexFormat::Uint32x3", out kind, out rustFormat),
            "vec4<u32>" => Set("vec4u", "wgpu::VertexFormat::Uint32x4", out kind, out rustFormat),
            "vec2<i32>" => Set("vec2i", "wgpu::VertexFormat::Sint32x2", out kind, out rustFormat),
            "vec3<i32>" => Set("vec3i", "wgpu::VertexFormat::Sint32x3", out kind, out rustFormat),
            "vec4<i32>" => Set("vec4i", "wgpu::VertexFormat::Sint32x4", out kind, out rustFormat),
            _ => Fail($"Unsupported WGSL vertex input type `{typeStr}`.", out error),
        };
    }

    private static bool Set(string k, string fmt, out string kind, out string rustFormat)
    {
        kind = k;
        rustFormat = fmt;
        return true;
    }

    private static bool Fail(string msg, out string? error)
    {
        error = msg;
        return false;
    }

    private static bool TryComputeInterleavedLayout(List<VertexLayoutAttribute> sorted, out uint stride, out string? error)
    {
        error = null;
        var seen = new HashSet<int>();
        foreach (VertexLayoutAttribute a in sorted)
        {
            if (!seen.Add(a.ShaderLocation))
            {
                error = $"Duplicate @location({a.ShaderLocation}) in vertex input.";
                stride = 0;
                return false;
            }
        }

        uint offset = 0;
        foreach (VertexLayoutAttribute a in sorted)
        {
            uint align = GetAttributeAlignment(a);
            offset = AlignUp(offset, align);
            offset += GetFormatSize(a);
        }

        stride = AlignUp(offset, 4);
        if (stride == 0)
            stride = 4;
        return true;
    }

    /// <summary>Byte offset for each attribute after computing interleaved packing (for Rust emission).</summary>
    public static IReadOnlyList<uint> ComputeOffsets(IReadOnlyList<VertexLayoutAttribute> sorted)
    {
        var list = new List<uint>(sorted.Count);
        uint offset = 0;
        foreach (VertexLayoutAttribute a in sorted)
        {
            uint align = GetAttributeAlignment(a);
            offset = AlignUp(offset, align);
            list.Add(offset);
            offset += GetFormatSize(a);
        }

        return list;
    }

    private static uint GetAttributeAlignment(VertexLayoutAttribute a) =>
        a.WgslScalarKind switch
        {
            "vec2f" or "vec2u" or "vec2i" => 8,
            "vec3f" or "vec3u" or "vec3i" => 4, // WGSL allows vec3 at 4-byte alignment in vertex buffers
            "vec4f" or "vec4u" or "vec4i" => 16,
            _ => 4,
        };

    private static uint GetFormatSize(VertexLayoutAttribute a) =>
        a.WgslScalarKind switch
        {
            "f32" or "u32" or "i32" => 4,
            "vec2f" or "vec2u" or "vec2i" => 8,
            "vec3f" or "vec3u" or "vec3i" => 12,
            "vec4f" or "vec4u" or "vec4i" => 16,
            _ => 4,
        };

    private static uint AlignUp(uint value, uint alignment)
    {
        uint m = value % alignment;
        return m == 0 ? value : value + (alignment - m);
    }
}
