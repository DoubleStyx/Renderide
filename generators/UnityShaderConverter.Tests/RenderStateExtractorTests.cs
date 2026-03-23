using UnityShaderConverter.Analysis;
using UnityShaderParser.ShaderLab;

namespace UnityShaderConverter.Tests;

/// <summary>Tests for <see cref="RenderStateExtractor"/>.</summary>
public sealed class RenderStateExtractorTests
{
    /// <summary>With no pass commands, effective tags equal subshader tags.</summary>
    [Fact]
    public void Extract_NoCommands_CopiesSubShaderTags()
    {
        var sub = new Dictionary<string, string>(StringComparer.Ordinal) { ["Queue"] = "Geometry", ["RenderType"] = "Opaque" };
        PassFixedFunctionState s = RenderStateExtractor.Extract(Array.Empty<ShaderLabCommandNode>(), sub);
        Assert.Equal(2, s.EffectiveTags.Count);
        Assert.Equal("Geometry", s.EffectiveTags["Queue"]);
        Assert.Equal("Opaque", s.EffectiveTags["RenderType"]);
    }
}
