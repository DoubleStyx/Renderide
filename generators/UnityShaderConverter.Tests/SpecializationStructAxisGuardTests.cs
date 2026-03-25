using UnityShaderConverter.Emission;

namespace UnityShaderConverter.Tests;

/// <summary>Tests for <see cref="SpecializationStructAxisGuard"/>.</summary>
public sealed class SpecializationStructAxisGuardTests
{
    [Fact]
    public void FindAxisKeywords_DetectsIfdefInsideStruct()
    {
        const string src = """
            struct v2f {
            #ifdef FOO
            float x;
            #endif
            };
            """;
        var axes = new[] { "FOO", "BAR" };
        HashSet<string> found = SpecializationStructAxisGuard.FindAxisKeywordsInStructConditionalBlocks(src, axes);
        Assert.Contains("FOO", found);
        Assert.DoesNotContain("BAR", found);
    }

    [Fact]
    public void FindAxisKeywords_DetectsIfDefinedInsideStruct()
    {
        const string src = """
            struct S
            {
            #if defined( BAR )
            float y;
            #endif
            };
            """;
        var axes = new[] { "BAR" };
        HashSet<string> found = SpecializationStructAxisGuard.FindAxisKeywordsInStructConditionalBlocks(src, axes);
        Assert.Contains("BAR", found);
    }

    [Fact]
    public void FindAxisKeywords_IgnoresIfdefOutsideStruct()
    {
        const string src = """
            #ifdef FOO
            float g;
            #endif
            struct v2f { float a; };
            """;
        var axes = new[] { "FOO" };
        HashSet<string> found = SpecializationStructAxisGuard.FindAxisKeywordsInStructConditionalBlocks(src, axes);
        Assert.Empty(found);
    }
}
