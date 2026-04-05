using System.Diagnostics;

namespace Renderide.Generators.Logging;

/// <summary>Resolves the Renderide tree root (directory containing <c>crates/renderide</c>) for generators and logging paths.</summary>
public static class RenderidePathResolver
{
    /// <summary>Runs <c>git rev-parse --show-toplevel</c>; returns null if git is missing or the command fails.</summary>
    public static string? TryGetGitRepositoryRoot()
    {
        var psi = new ProcessStartInfo
        {
            FileName = "git",
            Arguments = "rev-parse --show-toplevel",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };

        using var process = new Process { StartInfo = psi };
        try
        {
            process.Start();
        }
        catch
        {
            return null;
        }

        if (!process.WaitForExit(10_000))
        {
            try
            {
                process.Kill(entireProcessTree: true);
            }
            catch
            {
                // ignored
            }

            return null;
        }

        if (process.ExitCode != 0)
            return null;

        string line = process.StandardOutput.ReadToEnd().Split(
            new[] { '\r', '\n' },
            StringSplitOptions.RemoveEmptyEntries).FirstOrDefault() ?? "";
        return string.IsNullOrWhiteSpace(line) ? null : line.Trim();
    }

    /// <summary>Walks from the current working directory looking for <c>crates/renderide</c>.</summary>
    public static string FallbackRenderideRootFromCwd()
    {
        var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
        while (dir is not null)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, "crates", "renderide")))
                return dir.FullName;
            if (Directory.Exists(Path.Combine(dir.FullName, "Renderide", "crates", "renderide")))
                return Path.Combine(dir.FullName, "Renderide");
            dir = dir.Parent;
        }

        return Path.Combine(Directory.GetCurrentDirectory(), "Renderide");
    }

    /// <summary>Maps a git repository root (or null) to the directory that contains <c>crates/renderide</c>.</summary>
    public static string ResolveRenderideRoot(string? gitTopLevel)
    {
        if (gitTopLevel is not null)
        {
            if (Directory.Exists(Path.Combine(gitTopLevel, "crates", "renderide")))
                return Path.GetFullPath(gitTopLevel);
            string nested = Path.Combine(gitTopLevel, "Renderide");
            if (Directory.Exists(Path.Combine(nested, "crates", "renderide")))
                return Path.GetFullPath(nested);
        }

        return FallbackRenderideRootFromCwd();
    }
}
