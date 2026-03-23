using UnityShaderConverter.Analysis;

namespace UnityShaderConverter.Tests;

/// <summary>Tests for <see cref="GlobMatcher"/>.</summary>
public sealed class GlobMatcherTests
{
    /// <summary>Sample shader path under Renderide matches default eligibility patterns.</summary>
    [Fact]
    public void MatchesAny_SampleShadersPath()
    {
        string[] patterns = { "UnityShaderConverter/SampleShaders/**/*.shader" };
        Assert.True(GlobMatcher.MatchesAny("UnityShaderConverter/SampleShaders/MinimalUnlit.shader", patterns));
        Assert.False(GlobMatcher.MatchesAny("third_party/Resonite.UnityShaders/Assets/Shaders/Common/Null.shader", patterns));
    }
}
