using System.Globalization;
using SharedTypeGenerator.Options;
using Xunit;

namespace SharedTypeGenerator.Tests.Unit;

/// <summary>Unit tests for <see cref="AssemblyPathResolution"/>.</summary>
public sealed class AssemblyPathResolutionTests : IDisposable
{
    private readonly string? _prevDll;
    private readonly string? _prevDir;
    private readonly string? _prevSteam;

    /// <summary>Captures discovery env vars so empty-path resolution is deterministic.</summary>
    public AssemblyPathResolutionTests()
    {
        _prevDll = Environment.GetEnvironmentVariable(ResoniteAssemblyDiscovery.RenderiteSharedDllEnvVar);
        _prevDir = Environment.GetEnvironmentVariable(ResoniteAssemblyDiscovery.ResoniteDirEnvVar);
        _prevSteam = Environment.GetEnvironmentVariable(ResoniteAssemblyDiscovery.SteamPathEnvVar);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        RestoreEnv(ResoniteAssemblyDiscovery.RenderiteSharedDllEnvVar, _prevDll);
        RestoreEnv(ResoniteAssemblyDiscovery.ResoniteDirEnvVar, _prevDir);
        RestoreEnv(ResoniteAssemblyDiscovery.SteamPathEnvVar, _prevSteam);
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

    private static void RestoreEnv(string name, string? value)
    {
        if (value is null)
            Environment.SetEnvironmentVariable(name, null);
        else
            Environment.SetEnvironmentVariable(name, value);
    }

    /// <summary>Existing file paths are normalized to full paths.</summary>
    [Fact]
    public void TryResolveOrValidate_existing_file_sets_full_path()
    {
        string temp = Path.Combine(Path.GetTempPath(), "fake-shared-" + Guid.NewGuid().ToString("N", CultureInfo.InvariantCulture) + ".dll");
        File.WriteAllBytes(temp, [0x4D, 0x5A]);
        try
        {
            var options = new GeneratorOptions { AssemblyPath = temp };
            var err = new StringWriter(CultureInfo.InvariantCulture);
            bool ok = AssemblyPathResolution.TryResolveOrValidate(options, err);
            Assert.True(ok);
            Assert.Equal(Path.GetFullPath(temp), options.AssemblyPath);
        }
        finally
        {
            try
            {
                File.Delete(temp);
            }
            catch
            {
                // ignored
            }
        }
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
