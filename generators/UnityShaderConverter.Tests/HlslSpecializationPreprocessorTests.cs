using UnityShaderConverter.Emission;

namespace UnityShaderConverter.Tests;

/// <summary>Tests for specialization preprocessor rewriting via <see cref="SpecializationConditionalToRuntimeRewriter"/>.</summary>
public sealed class HlslSpecializationPreprocessorTests
{
    /// <summary>Axis <c>#ifdef</c> normalizes to <c>#if defined</c> then becomes runtime <c>if (USC_*)</c>.</summary>
    [Fact]
    public void Rewrite_Ifdef_BecomesRuntimeIfWithUscBool()
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal) { ["NORMALMAP"] = "USC_NORMALMAP" };
        const string src = "#ifdef NORMALMAP\nfloat a = 1;\n#endif\n";
        string outS = HlslSpecializationPreprocessor.Rewrite(src, map);
        Assert.Contains("if ((USC_NORMALMAP))", outS);
        Assert.Contains("float a = 1;", outS);
        Assert.DoesNotContain("#ifdef NORMALMAP", outS);
        Assert.DoesNotContain("defined(USC_NORMALMAP)", outS);
    }

    /// <summary><c>#ifndef</c> for an axis becomes <c>if (!(USC_*))</c> chain.</summary>
    [Fact]
    public void Rewrite_Ifndef_BecomesRuntimeIfOnNegatedBool()
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal) { ["FOO"] = "USC_FOO" };
        const string src = "#ifndef FOO\nfloat x = 1;\n#endif\n";
        string outS = HlslSpecializationPreprocessor.Rewrite(src, map);
        Assert.Contains("if (!(USC_FOO))", outS);
        Assert.Contains("float x = 1;", outS);
    }

    /// <summary><c>#if defined(KEYWORD)</c> becomes runtime <c>if ((USC_*))</c>, not <c>defined(USC_*)</c>.</summary>
    [Fact]
    public void Rewrite_DefinedChain_UsesRuntimeIfNotDefinedMacro()
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal) { ["BAR"] = "USC_BAR" };
        const string src = "#if defined( BAR )\nreturn 1;\n#endif\n";
        string outS = HlslSpecializationPreprocessor.Rewrite(src, map);
        Assert.Contains("if ((USC_BAR))", outS);
        Assert.Contains("return 1;", outS);
        Assert.DoesNotContain("defined(USC_BAR)", outS);
    }

    /// <summary><c>#elif</c> becomes <c>else if ((USC_*))</c>.</summary>
    [Fact]
    public void Rewrite_Elif_BecomesElseIf()
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["A"] = "USC_A",
            ["B"] = "USC_B",
        };
        const string src = """
            #if defined(A)
            x = 1;
            #elif defined(B)
            x = 2;
            #endif
            """;
        string outS = HlslSpecializationPreprocessor.Rewrite(src, map);
        Assert.Contains("if ((USC_A))", outS);
        Assert.Contains("else if ((USC_B))", outS);
        Assert.Contains("x = 1;", outS);
        Assert.Contains("x = 2;", outS);
    }

    /// <summary><c>#else</c> branch is preserved.</summary>
    [Fact]
    public void Rewrite_Else_Branch_Preserved()
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal) { ["A"] = "USC_A" };
        const string src = """
            #if defined(A)
            x = 1;
            #else
            x = 2;
            #endif
            """;
        string outS = HlslSpecializationPreprocessor.Rewrite(src, map);
        Assert.Contains("if ((USC_A))", outS);
        Assert.Contains("else {", outS);
        Assert.Contains("x = 2;", outS);
    }

    /// <summary>Non-axis identifiers are left as preprocessor <c>#if</c> (opaque block).</summary>
    [Fact]
    public void Rewrite_LeavesUnknownMacrosAsPreprocessorIf()
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal) { ["A"] = "USC_A" };
        const string src = "#ifdef OTHER\nfloat z = 1;\n#endif\n";
        string outS = HlslSpecializationPreprocessor.Rewrite(src, map);
        Assert.Contains("#ifdef OTHER", outS);
    }

    /// <summary>Groups containing <c>#define</c> are not converted (macros cannot sit inside runtime <c>if</c>).</summary>
    [Fact]
    public void Rewrite_SkipsIfGroupWithDefineInBody()
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal) { ["A"] = "USC_A" };
        const string src = """
            #if defined(A)
            #define X 1
            #endif
            """;
        string outS = HlslSpecializationPreprocessor.Rewrite(src, map);
        Assert.Contains("#if defined(A)", outS);
        Assert.Contains("#define X", outS);
    }

    /// <summary>Groups whose body declares global samplers are not converted.</summary>
    [Fact]
    public void Rewrite_SkipsIfGroupWithSampler2DInBody()
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal) { ["_TEXTURE"] = "USC__TEXTURE" };
        const string src = """
            #if defined(_TEXTURE)
            sampler2D _MainTex;
            #endif
            """;
        string outS = HlslSpecializationPreprocessor.Rewrite(src, map);
        Assert.Contains("#if defined(_TEXTURE)", outS);
        Assert.Contains("sampler2D", outS);
    }

    /// <summary>Nested spec-safe <c>#if</c> inside a converted body is rewritten recursively.</summary>
    [Fact]
    public void Rewrite_NestedSpecSafe_InnerConverted()
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["A"] = "USC_A",
            ["B"] = "USC_B",
        };
        const string src = """
            #if defined(A)
            #if defined(B)
            x = 1;
            #endif
            #endif
            """;
        string outS = HlslSpecializationPreprocessor.Rewrite(src, map);
        Assert.Contains("if ((USC_A))", outS);
        Assert.Contains("if ((USC_B))", outS);
    }

    /// <summary><see cref="SpecializationConditionalToRuntimeRewriter.IsSpecSafeCondition"/> accepts <c>||</c> / <c>&amp;&amp;</c> / <c>!</c>.</summary>
    [Fact]
    public void IsSpecSafe_OrAndNot_Ok()
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["A"] = "USC_A",
            ["B"] = "USC_B",
        };
        Assert.True(SpecializationConditionalToRuntimeRewriter.IsSpecSafeCondition("defined(A) || defined(B)", map));
        Assert.True(SpecializationConditionalToRuntimeRewriter.IsSpecSafeCondition("defined(A) && !defined(B)", map));
    }

    /// <summary>With entry names, spec-safe <c>#if</c> at file scope is not converted; only the fragment body is rewritten.</summary>
    [Fact]
    public void Rewrite_WithFragmentEntry_LeavesGlobalPreprocessorIf()
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal) { ["FOO"] = "USC_FOO" };
        const string src = """
            #if defined(FOO)
            static float g_unused = 1;
            #endif

            float4 frag(v2f i) : SV_Target {
            #if defined(FOO)
            float z = 1;
            #endif
            return float4(0,0,0,0);
            }
            """;
        string outS = HlslSpecializationPreprocessor.Rewrite(src, map, vertexEntry: null, fragmentEntry: "frag");
        Assert.Contains("#if defined(FOO)", outS);
        Assert.Contains("static float g_unused", outS);
        Assert.Contains("if ((USC_FOO))", outS);
        Assert.Contains("float z = 1;", outS);
    }

    /// <summary>Unknown <c>defined()</c> makes the condition not spec-safe.</summary>
    [Fact]
    public void IsSpecSafe_UnknownDefined_False()
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal) { ["A"] = "USC_A" };
        Assert.False(SpecializationConditionalToRuntimeRewriter.IsSpecSafeCondition("defined(A) || defined(UNKNOWN)", map));
    }
}
