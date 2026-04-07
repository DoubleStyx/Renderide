using System.Runtime.Versioning;
using Microsoft.Win32;

namespace SharedTypeGenerator.Options;

/// <summary>
/// Locates <c>Renderite.Shared.dll</c> using the same Steam and environment heuristics as the Rust bootstrapper
/// (<c>RESONITE_DIR</c>, <c>STEAM_PATH</c>, default Steam roots, <c>libraryfolders.vdf</c>, the Steam registry key on Windows,
/// and <c>~/Library/Application Support/Steam</c> on macOS).
/// </summary>
public static class ResoniteAssemblyDiscovery
{
    /// <summary>Managed assembly to resolve (sits next to other game managed DLLs).</summary>
    public const string SharedDllFileName = "Renderite.Shared.dll";

    /// <summary>Steam <c>steamapps/common</c> folder name for Resonite.</summary>
    public const string ResoniteAppFolderName = "Resonite";

    /// <summary>
    /// Environment variable set to the full path of <see cref="SharedDllFileName"/> (highest-priority override after an explicit CLI path).
    /// </summary>
    public const string RenderiteSharedDllEnvVar = "RENDERITE_SHARED_DLL";

    /// <summary>Environment variable for the Resonite game directory (same as the bootstrapper).</summary>
    public const string ResoniteDirEnvVar = "RESONITE_DIR";

    /// <summary>Environment variable for the Steam installation root (same as the bootstrapper).</summary>
    public const string SteamPathEnvVar = "STEAM_PATH";

    /// <summary>Returns an absolute path to <see cref="SharedDllFileName"/>, or <c>null</c> if discovery failed.</summary>
    public static string? TryFindRenderiteSharedDll()
    {
        if (TryPathFromEnv(RenderiteSharedDllEnvVar, requireFile: true) is { } fromExplicitDll)
            return fromExplicitDll;

        if (TryPathUnderResoniteDirEnv() is { } fromResoniteDir)
            return fromResoniteDir;

        if (TryPathFromSteamPathEnv() is { } fromSteamPath)
            return fromSteamPath;

        foreach (string steamBase in EnumerateSteamBasePaths())
        {
            if (TrySharedDllUnderResoniteInSteamRoot(steamBase) is { } p)
                return p;
        }

        foreach (string steamBase in EnumerateSteamBasePaths())
        {
            foreach (string libRoot in ParseLibraryFolderPathsFromVdf(steamBase))
            {
                if (TrySharedDllUnderResoniteInSteamRoot(libRoot) is { } p)
                    return p;
            }
        }

        return null;
    }

    /// <summary>Parses <c>libraryfolders.vdf</c> line-by-line for <c>"path"</c> entries (same approach as the bootstrapper).</summary>
    internal static IReadOnlyList<string> ParseLibraryFolderPathsFromVdfLines(IEnumerable<string> lines)
    {
        var paths = new List<string>();
        foreach (string line in lines)
        {
            int idx = line.IndexOf("\"path\"", StringComparison.Ordinal);
            if (idx < 0)
                continue;
            string rest = line[(idx + 6)..].TrimStart(' ', '\t');
            int start = rest.IndexOf('"');
            if (start < 0)
                continue;
            rest = rest[(start + 1)..];
            int end = rest.IndexOf('"');
            if (end < 0)
                continue;
            paths.Add(rest[..end]);
        }

        return paths;
    }

    /// <summary>Resolves an environment variable to a full path, optionally requiring that a file exists.</summary>
    static string? TryPathFromEnv(string variable, bool requireFile)
    {
        string? v = Environment.GetEnvironmentVariable(variable);
        if (string.IsNullOrWhiteSpace(v))
            return null;
        string full = Path.GetFullPath(v.Trim());
        if (requireFile && !File.Exists(full))
            return null;
        return full;
    }

    /// <summary>Returns <see cref="SharedDllFileName"/> under <see cref="ResoniteDirEnvVar"/> when present.</summary>
    static string? TryPathUnderResoniteDirEnv()
    {
        string? dir = Environment.GetEnvironmentVariable(ResoniteDirEnvVar);
        if (string.IsNullOrWhiteSpace(dir))
            return null;
        return TrySharedDllInDirectory(Path.GetFullPath(dir.Trim()));
    }

    /// <summary>Returns the shared DLL under <c>steamapps/common/Resonite</c> relative to <see cref="SteamPathEnvVar"/>.</summary>
    static string? TryPathFromSteamPathEnv()
    {
        string? steam = Environment.GetEnvironmentVariable(SteamPathEnvVar);
        if (string.IsNullOrWhiteSpace(steam))
            return null;
        return TrySharedDllUnderResoniteInSteamRoot(Path.GetFullPath(steam.Trim()));
    }

    /// <summary>Looks for the DLL under <c>{root}/steamapps/common/Resonite/</c>.</summary>
    static string? TrySharedDllUnderResoniteInSteamRoot(string steamOrLibraryRoot)
    {
        string resonite = Path.Combine(steamOrLibraryRoot, "steamapps", "common", ResoniteAppFolderName);
        return TrySharedDllInDirectory(resonite);
    }

    /// <summary>Returns the full path to <see cref="SharedDllFileName"/> in <paramref name="resoniteDirectory"/> if it exists.</summary>
    static string? TrySharedDllInDirectory(string resoniteDirectory)
    {
        string dll = Path.Combine(resoniteDirectory, SharedDllFileName);
        return File.Exists(dll) ? dll : null;
    }

    /// <summary>
    /// Steam roots to probe: <see cref="SteamPathEnvVar"/>, registry and common folders on Windows, standard Steam Client paths on macOS,
    /// and typical Linux locations (<c>~/.local/share/Steam</c>, <c>~/.steam/steam</c>).
    /// </summary>
    static IEnumerable<string> EnumerateSteamBasePaths()
    {
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var bases = new List<string>();

        void TryAdd(string? path)
        {
            if (string.IsNullOrWhiteSpace(path))
                return;
            string full = Path.GetFullPath(path.Trim());
            if (seen.Add(full))
                bases.Add(full);
        }

        TryAdd(Environment.GetEnvironmentVariable(SteamPathEnvVar));

        if (OperatingSystem.IsWindows())
        {
            TryAdd(TrySteamInstallPathFromRegistry());
            foreach (string envVar in new[] { "ProgramFiles(x86)", "ProgramFiles" })
            {
                string? pf = Environment.GetEnvironmentVariable(envVar);
                if (!string.IsNullOrEmpty(pf))
                    TryAdd(Path.Combine(pf, "Steam"));
            }

            string? local = Environment.GetEnvironmentVariable("LOCALAPPDATA");
            if (!string.IsNullOrEmpty(local))
                TryAdd(Path.Combine(local, "Steam"));
        }
        else if (OperatingSystem.IsMacOS())
        {
            string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            if (!string.IsNullOrEmpty(home))
            {
                // Default Steam Client layout on macOS (same as Valve’s documented install).
                TryAdd(Path.Combine(home, "Library", "Application Support", "Steam"));
                // Some setups use a compatibility symlink similar to Linux.
                TryAdd(Path.Combine(home, ".steam", "steam"));
            }
        }
        else
        {
            string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            if (!string.IsNullOrEmpty(home))
            {
                TryAdd(Path.Combine(home, ".local", "share", "Steam"));
                TryAdd(Path.Combine(home, ".steam", "steam"));
            }
        }

        return bases;
    }

    /// <summary>Reads the Steam install path from the Windows registry (WOW6432Node and 32-bit fallback keys).</summary>
    [SupportedOSPlatform("windows")]
    static string? TrySteamInstallPathFromRegistry()
    {
        try
        {
            foreach (string keyPath in new[] { @"SOFTWARE\WOW6432Node\Valve\Steam", @"SOFTWARE\Valve\Steam" })
            {
                using RegistryKey? steamKey = Registry.LocalMachine.OpenSubKey(keyPath);
                if (steamKey?.GetValue("InstallPath") is string installPath && !string.IsNullOrWhiteSpace(installPath))
                    return installPath;
            }
        }
        catch (SystemException)
        {
        }

        return null;
    }

    /// <summary>Loads <c>libraryfolders.vdf</c> under a Steam root and returns extra library folder paths.</summary>
    static IReadOnlyList<string> ParseLibraryFolderPathsFromVdf(string steamBase)
    {
        string vdfPath = Path.Combine(steamBase, "steamapps", "libraryfolders.vdf");
        if (!File.Exists(vdfPath))
            return Array.Empty<string>();

        try
        {
            string[] lines = File.ReadAllLines(vdfPath);
            return ParseLibraryFolderPathsFromVdfLines(lines);
        }
        catch (IOException)
        {
            return Array.Empty<string>();
        }
        catch (UnauthorizedAccessException)
        {
            return Array.Empty<string>();
        }
    }
}
