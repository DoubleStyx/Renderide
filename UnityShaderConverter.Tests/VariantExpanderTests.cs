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
        var cfg = new CompilerConfigModel { MaxVariantCombinationsPerShader = 32 };
        var result = VariantExpander.Expand(doc, cfg, null);
        Assert.Single(result);
        Assert.Empty(result[0]);
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
        var cfg = new CompilerConfigModel { MaxVariantCombinationsPerShader = 32 };
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
