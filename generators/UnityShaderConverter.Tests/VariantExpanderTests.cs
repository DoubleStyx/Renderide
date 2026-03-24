using UnityShaderConverter.Analysis;
using UnityShaderConverter.Config;
using UnityShaderConverter.Variants;

namespace UnityShaderConverter.Tests;

/// <summary>Tests for <see cref="VariantExpander"/>.</summary>
public sealed class VariantExpanderTests
{
    /// <summary>When there are no multi_compile pragmas, a single empty-variant list is returned.</summary>
    [Fact]
    public void Expand_NoMultiCompile_SingleEmptyVariant()
    {
        var doc = new ShaderFileDocument
        {
            SourcePath = "x.shader",
            ShaderName = "Test/Shader",
            Properties = Array.Empty<ShaderPropertyRecord>(),
            SubShaderTags = new Dictionary<string, string>(),
            Passes = Array.Empty<ShaderPassDocument>(),
            MultiCompilePragmas = Array.Empty<string>(),
        };
        var cfg = new CompilerConfigModel { MaxVariantCombinationsPerShader = 32, EnableSlangSpecialization = false };
        var result = VariantExpander.Expand(doc, cfg, null);
        Assert.Single(result);
        Assert.Empty(result[0]);
    }

    /// <summary>With specialization on, a huge Cartesian product does not throw; a single empty baseline variant is returned.</summary>
    [Fact]
    public void Expand_SpecializationOn_HugeProduct_ReturnsSingleEmptyVariant()
    {
        string many = string.Join(' ', Enumerable.Range(0, 30).Select(i => $"K{i}"));
        var doc = new ShaderFileDocument
        {
            SourcePath = "x.shader",
            ShaderName = "Test/BigVariants",
            Properties = Array.Empty<ShaderPropertyRecord>(),
            SubShaderTags = new Dictionary<string, string>(),
            Passes = Array.Empty<ShaderPassDocument>(),
            MultiCompilePragmas = new[] { "#pragma multi_compile " + many },
        };
        var cfg = new CompilerConfigModel
        {
            MaxVariantCombinationsPerShader = 4,
            EnableSlangSpecialization = true,
        };
        var result = VariantExpander.Expand(doc, cfg, null);
        Assert.Single(result);
        Assert.Empty(result[0]);
    }

    /// <summary>With specialization off, over-limit expansion still throws.</summary>
    [Fact]
    public void Expand_SpecializationOff_OverLimit_Throws()
    {
        string many = string.Join(' ', Enumerable.Range(0, 30).Select(i => $"K{i}"));
        var doc = new ShaderFileDocument
        {
            SourcePath = "x.shader",
            ShaderName = "Test/BigVariants",
            Properties = Array.Empty<ShaderPropertyRecord>(),
            SubShaderTags = new Dictionary<string, string>(),
            Passes = Array.Empty<ShaderPassDocument>(),
            MultiCompilePragmas = new[] { "#pragma multi_compile " + many },
        };
        var cfg = new CompilerConfigModel
        {
            MaxVariantCombinationsPerShader = 4,
            EnableSlangSpecialization = false,
        };
        Assert.Throws<InvalidOperationException>(() => VariantExpander.Expand(doc, cfg, null));
    }

    /// <summary>
    /// <see cref="VariantExpander.GetFirstCartesianVariantDefinesIgnoringProductLimit"/> returns the first keyword per group even when the Cartesian product exceeds the normal cap.
    /// </summary>
    [Fact]
    public void GetFirstCartesianVariantDefinesIgnoringProductLimit_ReturnsFirstKeywords()
    {
        string g1 = string.Join(' ', Enumerable.Range(0, 32).Select(i => $"A{i}"));
        string g2 = string.Join(' ', Enumerable.Range(0, 32).Select(i => $"B{i}"));
        var doc = new ShaderFileDocument
        {
            SourcePath = "x.shader",
            ShaderName = "Test/HugeFallback",
            Properties = Array.Empty<ShaderPropertyRecord>(),
            SubShaderTags = new Dictionary<string, string>(),
            Passes = Array.Empty<ShaderPassDocument>(),
            MultiCompilePragmas = new[]
            {
                "#pragma multi_compile " + g1,
                "#pragma multi_compile " + g2,
            },
        };
        IReadOnlyList<string> combo = VariantExpander.GetFirstCartesianVariantDefinesIgnoringProductLimit(doc, null);
        Assert.Equal(new[] { "A0", "B0" }, combo);
    }

    /// <summary><see cref="VariantExpander.AnalyzeMultiCompileGroups"/> reports product without expanding.</summary>
    [Fact]
    public void AnalyzeMultiCompileGroups_ReturnsProduct()
    {
        var doc = new ShaderFileDocument
        {
            SourcePath = "x.shader",
            ShaderName = "Test/Shader",
            Properties = Array.Empty<ShaderPropertyRecord>(),
            SubShaderTags = new Dictionary<string, string>(),
            Passes = Array.Empty<ShaderPassDocument>(),
            MultiCompilePragmas = new[] { "#pragma multi_compile A B", "#pragma multi_compile _ X" },
        };
        VariantExpander.MultiCompileAnalysis a = VariantExpander.AnalyzeMultiCompileGroups(doc);
        Assert.Equal(2, a.Groups.Count);
        Assert.Equal(4, a.Product);
    }

    /// <summary>JSON overrides replace automatic expansion.</summary>
    [Fact]
    public void Expand_ForcedVariantsFromJson()
    {
        var doc = new ShaderFileDocument
        {
            SourcePath = "x.shader",
            ShaderName = "Converter/MinimalUnlit",
            Properties = Array.Empty<ShaderPropertyRecord>(),
            SubShaderTags = new Dictionary<string, string>(),
            Passes = Array.Empty<ShaderPassDocument>(),
            MultiCompilePragmas = new[] { "#pragma multi_compile A B" },
        };
        var cfg = new CompilerConfigModel { MaxVariantCombinationsPerShader = 32, EnableSlangSpecialization = false };
        var vcfg = new VariantConfigModel
        {
            VariantsByShaderName = new Dictionary<string, List<VariantDefines>>
            {
                ["Converter/MinimalUnlit"] = new List<VariantDefines>
                {
                    new VariantDefines { Defines = new List<string> { "FOO" } },
                },
            },
        };
        var result = VariantExpander.Expand(doc, cfg, vcfg);
        Assert.Single(result);
        Assert.Single(result[0]);
        Assert.Equal("FOO", result[0][0]);
    }
}
