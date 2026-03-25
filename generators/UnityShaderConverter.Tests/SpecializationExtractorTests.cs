using UnityShaderConverter.Analysis;
using UnityShaderConverter.Config;
using UnityShaderConverter.Variants;

namespace UnityShaderConverter.Tests;

/// <summary>Tests for <see cref="SpecializationExtractor"/>.</summary>
public sealed class SpecializationExtractorTests
{
    [Fact]
    public void Extract_Disabled_returns_empty()
    {
        var doc = new ShaderFileDocument
        {
            SourcePath = "x.shader",
            ShaderName = "T/S",
            Properties = Array.Empty<ShaderPropertyRecord>(),
            SubShaderTags = new Dictionary<string, string>(),
            Passes = Array.Empty<ShaderPassDocument>(),
            MultiCompilePragmas = new[] { "#pragma multi_compile A B" },
            AnalyzerWarnings = Array.Empty<string>(),
            TotalSubShaderCount = 1,
        };
        var cfg = new CompilerConfigModel { EnableSlangSpecialization = false };
        IReadOnlyList<SpecializationAxis> axes = SpecializationExtractor.Extract(doc, cfg);
        Assert.Empty(axes);
    }

    [Fact]
    public void Extract_Picks_keywords_in_order()
    {
        var doc = new ShaderFileDocument
        {
            SourcePath = "x.shader",
            ShaderName = "T/S",
            Properties = Array.Empty<ShaderPropertyRecord>(),
            SubShaderTags = new Dictionary<string, string>(),
            Passes = Array.Empty<ShaderPassDocument>(),
            MultiCompilePragmas = new[] { "#pragma multi_compile FOO BAR" },
            AnalyzerWarnings = Array.Empty<string>(),
            TotalSubShaderCount = 1,
        };
        var cfg = new CompilerConfigModel { EnableSlangSpecialization = true, MaxSpecializationConstants = 8 };
        IReadOnlyList<SpecializationAxis> axes = SpecializationExtractor.Extract(doc, cfg);
        Assert.Equal(2, axes.Count);
        Assert.Equal("FOO", axes[0].Keyword);
        Assert.Equal(0, axes[0].ConstantId);
        Assert.True(axes[0].DefaultConstantValue);
        Assert.Equal("BAR", axes[1].Keyword);
        Assert.Equal(1, axes[1].ConstantId);
        Assert.False(axes[1].DefaultConstantValue);
    }

    /// <summary>Lines with a <c>_</c> off slot keep all specialization bools default false.</summary>
    [Fact]
    public void Extract_OptionalMultiCompile_AllDefaultsFalse()
    {
        var doc = new ShaderFileDocument
        {
            SourcePath = "x.shader",
            ShaderName = "T/S",
            Properties = Array.Empty<ShaderPropertyRecord>(),
            SubShaderTags = new Dictionary<string, string>(),
            Passes = Array.Empty<ShaderPassDocument>(),
            MultiCompilePragmas = new[] { "#pragma multi_compile _ OUTLINE" },
            AnalyzerWarnings = Array.Empty<string>(),
            TotalSubShaderCount = 1,
        };
        var cfg = new CompilerConfigModel { EnableSlangSpecialization = true, MaxSpecializationConstants = 8 };
        IReadOnlyList<SpecializationAxis> axes = SpecializationExtractor.Extract(doc, cfg);
        Assert.Single(axes);
        Assert.Equal("OUTLINE", axes[0].Keyword);
        Assert.False(axes[0].DefaultConstantValue);
    }
}
