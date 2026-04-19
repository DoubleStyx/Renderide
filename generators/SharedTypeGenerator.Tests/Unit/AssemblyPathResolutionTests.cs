using System.Globalization;
using SharedTypeGenerator.Options;
using SharedTypeGenerator.Tests.Unit.Support;
using Xunit;

namespace SharedTypeGenerator.Tests.Unit;

/// <summary>Unit tests for <see cref="AssemblyPathResolution"/>.</summary>
public sealed class AssemblyPathResolutionTests : IDisposable
{
    private readonly EnvVarScope _dllScope;
    private readonly EnvVarScope _dirScope;
    private readonly EnvVarScope _steamScope;

    /// <summary>Captures discovery env vars so empty-path resolution is deterministic.</summary>
    public AssemblyPathResolutionTests()
    {
        _dllScope = new EnvVarScope(ResoniteAssemblyDiscovery.RenderiteSharedDllEnvVar);
        _dirScope = new EnvVarScope(ResoniteAssemblyDiscovery.ResoniteDirEnvVar);
        _steamScope = new EnvVarScope(ResoniteAssemblyDiscovery.SteamPathEnvVar);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _dllScope.Dispose();
        _dirScope.Dispose();
        _steamScope.Dispose();
        GC.SuppressFinalize(this);
    }

    /// <summary>Empty assembly path with no discovery produces a friendly error.</summary>
    [SkippableFact]
    public void TryResolveOrValidate_empty_without_env_writes_error()
    {
        Environment.SetEnvironmentVariable(ResoniteAssemblyDiscovery.RenderiteSharedDllEnvVar, null);
        Environment.SetEnvironmentVariable(ResoniteAssemblyDiscovery.ResoniteDirEnvVar, null);
        Environment.SetEnvironmentVariable(ResoniteAssemblyDiscovery.SteamPathEnvVar, null);
        Skip.If(ResoniteAssemblyDiscovery.TryFindRenderiteSharedDll() != null,
            "Renderite.Shared.dll is discoverable via Steam/registry on this machine.");
        var options = new GeneratorOptions { AssemblyPath = null };
        var err = new StringWriter(CultureInfo.InvariantCulture);
        bool ok = AssemblyPathResolution.TryResolveOrValidate(options, err);
        Assert.False(ok);
        Assert.Contains("Could not find Renderite.Shared.dll", err.ToString(), StringComparison.Ordinal);
    }

    /// <summary>Existing file paths are normalized to full paths.</summary>
    [Fact]
    public void TryResolveOrValidate_existing_file_sets_full_path()
    {
        using var dllFile = new TempFile();
        File.WriteAllBytes(dllFile.FilePath, [0x4D, 0x5A]);
        var options = new GeneratorOptions { AssemblyPath = dllFile.FilePath };
        var err = new StringWriter(CultureInfo.InvariantCulture);
        bool ok = AssemblyPathResolution.TryResolveOrValidate(options, err);
        Assert.True(ok);
        Assert.Equal(Path.GetFullPath(dllFile.FilePath), options.AssemblyPath);
    }

    /// <summary>Missing explicit paths surface a clear error.</summary>
    [Fact]
    public void TryResolveOrValidate_missing_file_returns_false()
    {
        string missing = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N", CultureInfo.InvariantCulture) + ".dll");
        var options = new GeneratorOptions { AssemblyPath = missing };
        var err = new StringWriter(CultureInfo.InvariantCulture);
        bool ok = AssemblyPathResolution.TryResolveOrValidate(options, err);
        Assert.False(ok);
        Assert.Contains("does not exist", err.ToString(), StringComparison.Ordinal);
    }
}
