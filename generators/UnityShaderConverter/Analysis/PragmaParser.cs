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
}
