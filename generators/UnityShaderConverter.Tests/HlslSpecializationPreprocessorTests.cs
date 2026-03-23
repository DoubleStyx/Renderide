using UnityShaderConverter.Emission;

namespace UnityShaderConverter.Tests;

/// <summary>Tests for <see cref="HlslSpecializationPreprocessor"/>.</summary>
public sealed class HlslSpecializationPreprocessorTests
{
    /// <summary><c>#ifdef</c> rewrites to the Slang <c>USC_*</c> name.</summary>
    [Fact]
    public void Rewrite_Ifdef_ReplacesKeywordWithSlangId()
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal) { ["NORMALMAP"] = "USC_NORMALMAP" };
        const string src = "#ifdef NORMALMAP\nfloat a = 1;\n#endif\n";
        string outS = HlslSpecializationPreprocessor.Rewrite(src, map);
        Assert.Contains("#ifdef USC_NORMALMAP", outS);
        Assert.DoesNotContain("#ifdef NORMALMAP", outS);
    }

    /// <summary><c>#ifndef</c> rewrites similarly.</summary>
    [Fact]
    public void Rewrite_Ifndef_ReplacesKeyword()
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal) { ["FOO"] = "USC_FOO" };
        const string src = "#ifndef FOO\n#endif\n";
        string outS = HlslSpecializationPreprocessor.Rewrite(src, map);
        Assert.Contains("#ifndef USC_FOO", outS);
    }

    /// <summary><c>defined(KEYWORD)</c> inside <c>#if</c> is rewritten.</summary>
    [Fact]
    public void Rewrite_Defined_ReplacesKeyword()
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal) { ["BAR"] = "USC_BAR" };
        const string src = "#if defined( BAR )\n#endif\n";
        string outS = HlslSpecializationPreprocessor.Rewrite(src, map);
        Assert.Contains("defined(USC_BAR)", outS);
        Assert.DoesNotContain("defined( BAR )", outS);
    }

    /// <summary>Non-axis identifiers are left unchanged.</summary>
    [Fact]
    public void Rewrite_LeavesOtherMacrosAlone()
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal) { ["A"] = "USC_A" };
        const string src = "#ifdef OTHER\n#endif\n";
        string outS = HlslSpecializationPreprocessor.Rewrite(src, map);
        Assert.Contains("#ifdef OTHER", outS);
    }
}
