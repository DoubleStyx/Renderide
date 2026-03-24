using UnityShaderConverter;

namespace UnityShaderConverter.Tests;

/// <summary>Tests for <see cref="UnityCgIncludesResolver"/> used by ShaderLab and <c>slangc</c> include paths.</summary>
public sealed class UnityCgIncludesResolverTests
{
    /// <summary>Bundled output should contain <c>UnityCG.cginc</c> when tests are built with the same layout as UnityShaderConverter.</summary>
    [Fact]
    public void TryBundledDirectory_PointsAtUnityCG()
    {
        string? bundled = UnityCgIncludesResolver.TryBundledDirectory();
        Assert.NotNull(bundled);
        Assert.True(File.Exists(Path.Combine(bundled, "UnityCG.cginc")));
        Assert.True(File.Exists(Path.Combine(bundled, "UnityStandardUtils.cginc")));
    }

    /// <summary><see cref="UnityCgIncludesResolver.ResolveForSlang"/> must match the first valid path in priority order.</summary>
    [Fact]
    public void ResolveForSlang_UsesBundledWhenNoOverride()
    {
        string shaderUnderOutput = Path.Combine(AppContext.BaseDirectory, "TestData", "MinimalUnlit.shader");
        Assert.True(File.Exists(shaderUnderOutput));

        string? resolved = UnityCgIncludesResolver.ResolveForSlang(cliOrEnvOverride: null, shaderUnderOutput);
        Assert.NotNull(resolved);
        Assert.True(File.Exists(Path.Combine(resolved, "UnityCG.cginc")));
    }

    /// <summary>Search list should be non-empty in the default test output layout and include no duplicate paths.</summary>
    [Fact]
    public void GetSearchDirectories_IsOrderedAndUnique()
    {
        string shaderUnderOutput = Path.Combine(AppContext.BaseDirectory, "TestData", "MinimalUnlit.shader");
        IReadOnlyList<string> dirs = UnityCgIncludesResolver.GetSearchDirectories(null, shaderUnderOutput);
        Assert.NotEmpty(dirs);
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (string d in dirs)
        {
            Assert.True(seen.Add(d), $"Duplicate include path: {d}");
            Assert.True(File.Exists(Path.Combine(d, "UnityCG.cginc")));
        }
    }
}
