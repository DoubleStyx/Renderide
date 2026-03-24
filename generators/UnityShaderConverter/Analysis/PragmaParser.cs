using System.Globalization;
using System.Text.RegularExpressions;

namespace UnityShaderConverter.Analysis;

/// <summary>Extracts <c>#pragma</c> entry points from HLSL program text.</summary>
public static partial class PragmaParser
{
    [GeneratedRegex(@"^\s*#\s*pragma\s+vertex\s+(\w+)", RegexOptions.IgnoreCase | RegexOptions.Multiline)]
    private static partial Regex VertexRegex();

    [GeneratedRegex(@"^\s*#\s*pragma\s+fragment\s+(\w+)", RegexOptions.IgnoreCase | RegexOptions.Multiline)]
    private static partial Regex FragmentRegex();

    /// <summary>Returns the vertex entry function name if found.</summary>
    public static bool TryGetVertexEntry(string program, out string name)
    {
        Match m = VertexRegex().Match(program);
        if (!m.Success)
        {
            name = "";
            return false;
        }

        name = m.Groups[1].Value;
        return true;
    }

    /// <summary>Returns the fragment entry function name if found.</summary>
    public static bool TryGetFragmentEntry(string program, out string name)
    {
        Match m = FragmentRegex().Match(program);
        if (!m.Success)
        {
            name = "";
            return false;
        }

        name = m.Groups[1].Value;
        return true;
    }

    /// <summary>True when a geometry shader entry is declared (not supported by this converter yet).</summary>
    public static bool HasGeometryStage(string program) =>
        GeometryStageRegex().IsMatch(program);

    [GeneratedRegex(@"^\s*#\s*pragma\s+geometry\s+\w+", RegexOptions.IgnoreCase | RegexOptions.Multiline)]
    private static partial Regex GeometryStageRegex();

    [GeneratedRegex(@"^\s*#\s*pragma\s+surface\s+(\w+)\s+(\w+)(?:\s+(.*))?\s*$", RegexOptions.IgnoreCase | RegexOptions.Multiline)]
    private static partial Regex SurfacePragmaRegex();

    [GeneratedRegex(@"^\s*#\s*pragma\s+target\s+([0-9]+(?:\.[0-9]+)?)", RegexOptions.IgnoreCase | RegexOptions.Multiline)]
    private static partial Regex ShaderTargetRegex();

    /// <summary>
    /// True when <paramref name="program"/> contains a line <c>#pragma surface …</c> (full shader source or a single CGPROGRAM body).
    /// Used to skip entire shaders from conversion before expensive ShaderLab parsing.
    /// </summary>
    public static bool HasSurfacePragma(string program) => SurfacePragmaRegex().IsMatch(program);

    /// <summary>
    /// Returns the first <c>#pragma target</c> value (e.g. <c>3.0</c>) when present.
    /// </summary>
    public static bool TryGetShaderTarget(string program, out float targetVersion)
    {
        targetVersion = 0f;
        Match m = ShaderTargetRegex().Match(program);
        if (!m.Success)
            return false;
        return float.TryParse(m.Groups[1].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out targetVersion);
    }
}
