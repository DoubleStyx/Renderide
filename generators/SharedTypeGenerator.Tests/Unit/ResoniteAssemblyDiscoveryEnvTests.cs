using System.Globalization;
using SharedTypeGenerator.Options;
using Xunit;

namespace SharedTypeGenerator.Tests.Unit;

/// <summary>Environment variable precedence tests for <see cref="ResoniteAssemblyDiscovery"/>.</summary>
[Collection("ResoniteAssemblyDiscoveryEnv")]
public sealed class ResoniteAssemblyDiscoveryEnvTests : IDisposable
{
    private readonly string? _prevDll;
    private readonly string? _prevDir;
    private readonly string? _prevSteam;

    /// <summary>Captures and clears discovery env vars before each test.</summary>
    public ResoniteAssemblyDiscoveryEnvTests()
    {
        _prevDll = Environment.GetEnvironmentVariable(ResoniteAssemblyDiscovery.RenderiteSharedDllEnvVar);
        _prevDir = Environment.GetEnvironmentVariable(ResoniteAssemblyDiscovery.ResoniteDirEnvVar);
        _prevSteam = Environment.GetEnvironmentVariable(ResoniteAssemblyDiscovery.SteamPathEnvVar);
        ClearEnv();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Restore(ResoniteAssemblyDiscovery.RenderiteSharedDllEnvVar, _prevDll);
        Restore(ResoniteAssemblyDiscovery.ResoniteDirEnvVar, _prevDir);
        Restore(ResoniteAssemblyDiscovery.SteamPathEnvVar, _prevSteam);
        GC.SuppressFinalize(this);
    }

    /// <summary><c>RENDERITE_SHARED_DLL</c> wins when the file exists.</summary>
    [Fact]
    public void Renderite_shared_dll_env_wins()
    {
        string dll = Path.Combine(Path.GetTempPath(), "renderite-shared-" + Guid.NewGuid().ToString("N", CultureInfo.InvariantCulture) + ".dll");
        File.WriteAllBytes(dll, [0x4D, 0x5A]);
        try
        {
            Environment.SetEnvironmentVariable(ResoniteAssemblyDiscovery.RenderiteSharedDllEnvVar, dll);
            string? found = ResoniteAssemblyDiscovery.TryFindRenderiteSharedDll();
            Assert.Equal(Path.GetFullPath(dll), found);
        }
        finally
        {
            try
            {
                File.Delete(dll);
            }
            catch
            {
                // ignored
            }
        }
    }

    /// <summary><c>RESONITE_DIR</c> resolves the DLL when present.</summary>
    [Fact]
    public void Resonite_dir_env_finds_dll()
    {
        string dir = Path.Combine(Path.GetTempPath(), "resonite-dir-" + Guid.NewGuid().ToString("N", CultureInfo.InvariantCulture));
        string dll = Path.Combine(dir, ResoniteAssemblyDiscovery.SharedDllFileName);
        Directory.CreateDirectory(dir);
        File.WriteAllBytes(dll, [0x4D, 0x5A]);
        try
        {
            Environment.SetEnvironmentVariable(ResoniteAssemblyDiscovery.ResoniteDirEnvVar, dir);
            string? found = ResoniteAssemblyDiscovery.TryFindRenderiteSharedDll();
            Assert.Equal(Path.GetFullPath(dll), found);
        }
        finally
        {
            try
            {
                Directory.Delete(dir, recursive: true);
            }
            catch
            {
                // ignored
            }
        }
    }

    /// <summary><c>STEAM_PATH</c> resolves the DLL under the Resonite steamapps layout.</summary>
    [Fact]
    public void Steam_path_env_finds_dll()
    {
        string steam = Path.Combine(Path.GetTempPath(), "steam-root-" + Guid.NewGuid().ToString("N", CultureInfo.InvariantCulture));
        string dll = Path.Combine(steam, "steamapps", "common", ResoniteAssemblyDiscovery.ResoniteAppFolderName,
            ResoniteAssemblyDiscovery.SharedDllFileName);
        Directory.CreateDirectory(Path.GetDirectoryName(dll)!);
        File.WriteAllBytes(dll, [0x4D, 0x5A]);
        try
        {
            Environment.SetEnvironmentVariable(ResoniteAssemblyDiscovery.SteamPathEnvVar, steam);
            string? found = ResoniteAssemblyDiscovery.TryFindRenderiteSharedDll();
            Assert.Equal(Path.GetFullPath(dll), found);
        }
        finally
        {
            try
            {
                Directory.Delete(steam, recursive: true);
            }
            catch
            {
                // ignored
            }
        }
    }

    private static void ClearEnv()
    {
        Environment.SetEnvironmentVariable(ResoniteAssemblyDiscovery.RenderiteSharedDllEnvVar, null);
        Environment.SetEnvironmentVariable(ResoniteAssemblyDiscovery.ResoniteDirEnvVar, null);
        Environment.SetEnvironmentVariable(ResoniteAssemblyDiscovery.SteamPathEnvVar, null);
    }

    private static void Restore(string name, string? value)
    {
        if (value is null)
            Environment.SetEnvironmentVariable(name, null);
        else
            Environment.SetEnvironmentVariable(name, value);
    }
}
