using SharedTypeGenerator.Options;
using Xunit;

namespace SharedTypeGenerator.Tests;

/// <summary>Serializes environment-mutating discovery tests so process-wide env vars do not race.</summary>
[CollectionDefinition(nameof(ResoniteAssemblyDiscoveryEnvCollection), DisableParallelization = true)]
public sealed class ResoniteAssemblyDiscoveryEnvCollection;

/// <summary>Unit tests for VDF path parsing in <see cref="ResoniteAssemblyDiscovery"/>.</summary>
public sealed class ResoniteAssemblyDiscoveryTests
{
    [Fact]
    public void ParseLibraryFolderPathsFromVdfLines_extracts_quoted_path_values()
    {
        string[] sample =
        {
            "\t\t\"1\"\t\t{",
            "\t\t\t\"path\"\t\t\"/data/SteamLibrary\"",
            "\t\t}",
            "\t\t\"2\"\t\t{",
            "\t\t\t\"path\"\t\t\"/home/user/.steam/steam\"",
        };

        IReadOnlyList<string> paths = ResoniteAssemblyDiscovery.ParseLibraryFolderPathsFromVdfLines(sample);

        Assert.Equal(2, paths.Count);
        Assert.Equal("/data/SteamLibrary", paths[0]);
        Assert.Equal("/home/user/.steam/steam", paths[1]);
    }

    [Fact]
    public void ParseLibraryFolderPathsFromVdfLines_returns_empty_when_no_paths()
    {
        IReadOnlyList<string> paths = ResoniteAssemblyDiscovery.ParseLibraryFolderPathsFromVdfLines(["foo", "bar"]);
        Assert.Empty(paths);
    }
}

/// <summary>End-to-end discovery tests that set process environment variables (serialized collection).</summary>
[Collection(nameof(ResoniteAssemblyDiscoveryEnvCollection))]
public sealed class ResoniteAssemblyDiscoveryEnvTests : IDisposable
{
    readonly string? _oldRenderiteSharedDll;
    readonly string? _oldResoniteDir;

    public ResoniteAssemblyDiscoveryEnvTests()
    {
        _oldRenderiteSharedDll = Environment.GetEnvironmentVariable(ResoniteAssemblyDiscovery.RenderiteSharedDllEnvVar);
        _oldResoniteDir = Environment.GetEnvironmentVariable(ResoniteAssemblyDiscovery.ResoniteDirEnvVar);
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable(ResoniteAssemblyDiscovery.RenderiteSharedDllEnvVar, _oldRenderiteSharedDll);
        Environment.SetEnvironmentVariable(ResoniteAssemblyDiscovery.ResoniteDirEnvVar, _oldResoniteDir);
    }

    [Fact]
    public void TryFindRenderiteSharedDll_prefers_RENDERITE_SHARED_DLL_when_file_exists()
    {
        string dir = Path.Combine(Path.GetTempPath(), "stg_dll_env_" + Guid.NewGuid());
        Directory.CreateDirectory(dir);
        string dll = Path.Combine(dir, ResoniteAssemblyDiscovery.SharedDllFileName);
        try
        {
            File.WriteAllText(dll, "");
            Environment.SetEnvironmentVariable(ResoniteAssemblyDiscovery.RenderiteSharedDllEnvVar, null);
            Environment.SetEnvironmentVariable(ResoniteAssemblyDiscovery.ResoniteDirEnvVar, null);
            Environment.SetEnvironmentVariable(ResoniteAssemblyDiscovery.RenderiteSharedDllEnvVar, dll);

            string? found = ResoniteAssemblyDiscovery.TryFindRenderiteSharedDll();

            Assert.Equal(dll, found);
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }

    [Fact]
    public void TryFindRenderiteSharedDll_uses_RESONITE_DIR_when_dll_present_and_no_shared_dll_env()
    {
        string dir = Path.Combine(Path.GetTempPath(), "stg_resonite_dir_" + Guid.NewGuid());
        Directory.CreateDirectory(dir);
        string dll = Path.Combine(dir, ResoniteAssemblyDiscovery.SharedDllFileName);
        try
        {
            File.WriteAllText(dll, "");
            Environment.SetEnvironmentVariable(ResoniteAssemblyDiscovery.RenderiteSharedDllEnvVar, null);
            Environment.SetEnvironmentVariable(ResoniteAssemblyDiscovery.ResoniteDirEnvVar, dir);

            string? found = ResoniteAssemblyDiscovery.TryFindRenderiteSharedDll();

            Assert.Equal(dll, found);
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }
}
