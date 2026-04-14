using SharedTypeGenerator.Options;
using Xunit;

namespace SharedTypeGenerator.Tests.Unit;

/// <summary>Unit tests for Steam <c>libraryfolders.vdf</c> path parsing used by <see cref="ResoniteAssemblyDiscovery"/>.</summary>
public sealed class ResoniteAssemblyDiscoveryVdfTests
{
    /// <summary><see cref="ResoniteAssemblyDiscovery.ParseLibraryFolderPathsFromVdfLines"/> extracts quoted paths from typical VDF-style lines.</summary>
    [Fact]
    public void ParseLibraryFolderPathsFromVdfLines_collects_multiple_path_entries()
    {
        string[] lines =
        [
            "\t\"path\"\t\t\"D:/SteamLibrary\"",
            "  \"path\"    \"/home/user/.local/share/Steam\"",
        ];

        IReadOnlyList<string> paths = ResoniteAssemblyDiscovery.ParseLibraryFolderPathsFromVdfLines(lines);

        Assert.Equal(2, paths.Count);
        Assert.Equal("D:/SteamLibrary", paths[0]);
        Assert.Equal("/home/user/.local/share/Steam", paths[1]);
    }

    /// <summary>Lines without a <c>path</c> entry are ignored.</summary>
    [Fact]
    public void ParseLibraryFolderPathsFromVdfLines_ignores_non_path_lines()
    {
        string[] lines =
        [
            "\"other\" \"value\"",
            "",
        ];

        IReadOnlyList<string> paths = ResoniteAssemblyDiscovery.ParseLibraryFolderPathsFromVdfLines(lines);

        Assert.Empty(paths);
    }

    /// <summary>Malformed or incomplete quoted values must not throw; they are skipped.</summary>
    [Fact]
    public void ParseLibraryFolderPathsFromVdfLines_skips_incomplete_quotes()
    {
        string[] lines =
        [
            "\"path\" no-closing-quote",
        ];

        IReadOnlyList<string> paths = ResoniteAssemblyDiscovery.ParseLibraryFolderPathsFromVdfLines(lines);

        Assert.Empty(paths);
    }
}
