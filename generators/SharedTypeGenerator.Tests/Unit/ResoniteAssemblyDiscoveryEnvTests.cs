using SharedTypeGenerator.Options;
using SharedTypeGenerator.Tests.Unit.Support;
using Xunit;

namespace SharedTypeGenerator.Tests.Unit;

/// <summary>Environment variable precedence tests for <see cref="ResoniteAssemblyDiscovery"/>.</summary>
[Collection("ResoniteAssemblyDiscoveryEnv")]
public sealed class ResoniteAssemblyDiscoveryEnvTests : IDisposable
{
    private readonly EnvVarScope _dllScope;
    private readonly EnvVarScope _dirScope;
    private readonly EnvVarScope _steamScope;

    /// <summary>Captures and clears discovery env vars before each test.</summary>
    public ResoniteAssemblyDiscoveryEnvTests()
    {
        _dllScope = new EnvVarScope(ResoniteAssemblyDiscovery.RenderiteSharedDllEnvVar);
        _dirScope = new EnvVarScope(ResoniteAssemblyDiscovery.ResoniteDirEnvVar);
        _steamScope = new EnvVarScope(ResoniteAssemblyDiscovery.SteamPathEnvVar);
        ClearEnv();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _steamScope.Dispose();
        _dirScope.Dispose();
        _dllScope.Dispose();
        GC.SuppressFinalize(this);
    }

    /// <summary><c>RENDERITE_SHARED_DLL</c> wins when the file exists.</summary>
    [Fact]
    public void Renderite_shared_dll_env_wins()
    {
        using var dllFile = new TempFile();
        File.WriteAllBytes(dllFile.FilePath, [0x4D, 0x5A]);
        Environment.SetEnvironmentVariable(ResoniteAssemblyDiscovery.RenderiteSharedDllEnvVar, dllFile.FilePath);
        string? found = ResoniteAssemblyDiscovery.TryFindRenderiteSharedDll();
        Assert.Equal(Path.GetFullPath(dllFile.FilePath), found);
    }

    /// <summary><c>RESONITE_DIR</c> resolves the DLL when present.</summary>
    [Fact]
    public void Resonite_dir_env_finds_dll()
    {
        using var dir = new TempDirectory();
        string dll = Path.Combine(dir.DirectoryPath, ResoniteAssemblyDiscovery.SharedDllFileName);
        File.WriteAllBytes(dll, [0x4D, 0x5A]);
        Environment.SetEnvironmentVariable(ResoniteAssemblyDiscovery.ResoniteDirEnvVar, dir.DirectoryPath);
        string? found = ResoniteAssemblyDiscovery.TryFindRenderiteSharedDll();
        Assert.Equal(Path.GetFullPath(dll), found);
    }

    /// <summary><c>STEAM_PATH</c> resolves the DLL under the Resonite steamapps layout.</summary>
    [Fact]
    public void Steam_path_env_finds_dll()
    {
        using var steam = new TempDirectory();
        string dll = Path.Combine(steam.DirectoryPath, "steamapps", "common", ResoniteAssemblyDiscovery.ResoniteAppFolderName,
            ResoniteAssemblyDiscovery.SharedDllFileName);
        Directory.CreateDirectory(Path.GetDirectoryName(dll)!);
        File.WriteAllBytes(dll, [0x4D, 0x5A]);
        Environment.SetEnvironmentVariable(ResoniteAssemblyDiscovery.SteamPathEnvVar, steam.DirectoryPath);
        string? found = ResoniteAssemblyDiscovery.TryFindRenderiteSharedDll();
        Assert.Equal(Path.GetFullPath(dll), found);
    }

    private static void ClearEnv()
    {
        Environment.SetEnvironmentVariable(ResoniteAssemblyDiscovery.RenderiteSharedDllEnvVar, null);
        Environment.SetEnvironmentVariable(ResoniteAssemblyDiscovery.ResoniteDirEnvVar, null);
        Environment.SetEnvironmentVariable(ResoniteAssemblyDiscovery.SteamPathEnvVar, null);
    }
}
