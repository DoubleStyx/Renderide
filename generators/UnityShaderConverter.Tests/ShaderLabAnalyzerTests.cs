using UnityShaderConverter.Analysis;

namespace UnityShaderConverter.Tests;

/// <summary>Tests for <see cref="ShaderLabAnalyzer"/>.</summary>
public sealed class ShaderLabAnalyzerTests
{
    /// <summary>Ensures the sample MinimalUnlit shader parses and exposes one pass.</summary>
    [Fact]
    public void TryAnalyze_MinimalUnlit_Succeeds()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "MinimalUnlit.shader");
        Assert.True(File.Exists(path), $"Missing test file: {path}");
        bool ok = ShaderLabAnalyzer.TryAnalyze(path, out var doc, out var diags, out var errors);
        Assert.True(ok, string.Join("; ", errors) + string.Join("; ", diags));
        Assert.NotNull(doc);
        Assert.Equal("Converter/MinimalUnlit", doc!.ShaderName);
        Assert.Single(doc.Passes);
        Assert.Equal("vert", doc.Passes[0].VertexEntry);
        Assert.Equal("frag", doc.Passes[0].FragmentEntry);
        Assert.Single(doc.Properties);
        Assert.Equal("_Color", doc.Properties[0].Name);
    }

    /// <summary>
    /// Resonite shaders include <c>UnityCG.cginc</c>; the analyzer must resolve vendored CGIncludes from the UnityShaderParser test pack.
    /// </summary>
    [Fact]
    public void TryAnalyze_ResoniteUnlit_WithUnityCgIncludes_Succeeds()
    {
        string renderideRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".."));
        string path = Path.Combine(
            renderideRoot,
            "third_party",
            "Resonite.UnityShaders",
            "Assets",
            "Shaders",
            "Common",
            "Unlit.shader");
        Assert.True(File.Exists(path), $"Expected Resonite Unlit shader at {path}");

        bool ok = ShaderLabAnalyzer.TryAnalyze(path, out var doc, out var diags, out var errors);
        Assert.True(ok, string.Join("; ", errors) + string.Join("; ", diags));
        Assert.NotNull(doc);
        Assert.NotEmpty(doc!.Passes);
    }

    /// <summary>
    /// Surface shaders with <c>CGPROGRAM</c> directly under <c>SubShader</c> (no <c>Pass</c>) must be detected, not mis-reported as missing passes.
    /// </summary>
    [Fact]
    public void TryAnalyze_SurfaceImplicitSubshader_ExcludedWithSurfacePrefix()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "SurfaceImplicitSubshader.shader");
        Assert.True(File.Exists(path), $"Missing test file: {path}");
        bool ok = ShaderLabAnalyzer.TryAnalyze(path, out var doc, out var diags, out var errors);
        Assert.False(ok);
        Assert.Null(doc);
        Assert.Contains(
            errors,
            e => e.StartsWith(ShaderLabAnalyzer.SurfaceShaderNotSupportedPrefix, StringComparison.Ordinal) &&
                 e.Contains("shader source contains", StringComparison.Ordinal));
        Assert.True(ShaderLabAnalyzer.IsSurfaceShaderExclusion(errors));
    }

    /// <summary>Surface shaders are detected and excluded with a stable prefix and porting guidance.</summary>
    [Fact]
    public void TryAnalyze_SurfaceOnly_ExcludedWithSurfaceShaderPrefix()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "SurfaceOnly.shader");
        Assert.True(File.Exists(path), $"Missing test file: {path}");
        bool ok = ShaderLabAnalyzer.TryAnalyze(path, out var doc, out var diags, out var errors);
        Assert.False(ok);
        Assert.Null(doc);
        Assert.Contains(
            errors,
            e => e.StartsWith(ShaderLabAnalyzer.SurfaceShaderNotSupportedPrefix, StringComparison.Ordinal));
        Assert.True(ShaderLabAnalyzer.IsSurfaceShaderExclusion(errors));
        Assert.Contains(errors, e => e.Contains("#pragma vertex", StringComparison.Ordinal) && e.Contains("#pragma fragment", StringComparison.Ordinal));
    }

    /// <summary><see cref="ShaderLabAnalyzer.IsGeometryShaderExclusion"/> recognizes the stable geometry-stage message.</summary>
    [Fact]
    public void IsGeometryShaderExclusion_DetectsKnownMessage()
    {
        var errors = new List<string> { ShaderLabAnalyzer.GeometryShaderNotSupportedMessage };
        Assert.True(ShaderLabAnalyzer.IsGeometryShaderExclusion(errors));
        Assert.True(ShaderLabAnalyzer.IsExpectedUnsupportedShaderExclusion(errors));
        Assert.True(ShaderLabAnalyzer.IsParseFailureLoggedAtTraceOnly(ShaderLabAnalyzer.GeometryShaderNotSupportedMessage));
    }

    /// <summary>
    /// Resonite wireframe shaders use <c>#pragma geometry</c>; the analyzer must report the geometry exclusion, not unrelated parse noise.
    /// </summary>
    [Fact]
    public void TryAnalyze_ResoniteWireframe_GeometryPragma_ExcludedWithGeometryMessage()
    {
        string renderideRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".."));
        string path = Path.Combine(
            renderideRoot,
            "third_party",
            "Resonite.UnityShaders",
            "Assets",
            "Shaders",
            "Common",
            "Wireframe.shader");
        Assert.True(File.Exists(path), $"Expected Resonite Wireframe shader at {path}");

        bool ok = ShaderLabAnalyzer.TryAnalyze(path, out var doc, out var diags, out var errors);
        Assert.False(ok);
        Assert.Null(doc);
        Assert.Contains(errors, e => string.Equals(e, ShaderLabAnalyzer.GeometryShaderNotSupportedMessage, StringComparison.Ordinal));
        Assert.True(ShaderLabAnalyzer.IsGeometryShaderExclusion(errors));
    }
}
