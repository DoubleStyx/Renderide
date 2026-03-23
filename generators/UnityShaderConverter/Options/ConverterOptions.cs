using System.Diagnostics;
using CommandLine;

namespace UnityShaderConverter.Options;

/// <summary>Command-line options for <see cref="ConverterRunner.Run"/>.</summary>
public sealed class ConverterOptions
{
    /// <summary>When true, lowers the log threshold to trace.</summary>
    [Option('v', "verbose", Required = false, HelpText = "Verbose logging.")]
    public bool Verbose { get; set; }

    /// <summary>One or more roots to scan recursively for *.shader files.</summary>
    [Option('i', "input", Required = false, HelpText = "Directory containing .shader files (repeatable). Defaults to SampleShaders + Resonite Unity shaders under the repo.")]
    public IEnumerable<string> InputDirectories { get; set; } = Array.Empty<string>();

    /// <summary>Generated output root (<c>wgsl/</c>, bundled <c>wgsl_sources.rs</c> / <c>materials.rs</c>, <c>mod.rs</c>).</summary>
    [Option('o', "output", Required = false, HelpText = "Output directory (default: crates/renderide/src/shaders/generated).")]
    public string? OutputDirectory { get; set; }

    /// <summary>Path to <c>slangc</c>; overrides <c>SLANGC</c> environment variable.</summary>
    [Option("slangc", Required = false, HelpText = "Path to slangc executable.")]
    public string? SlangcPath { get; set; }

    /// <summary>When set, skips invoking <c>slangc</c> even for eligible shaders.</summary>
    [Option("skip-slang", Required = false, HelpText = "Do not run slangc; WGSL must already exist on disk for a shader to enter the Rust bundle.")]
    public bool SkipSlang { get; set; }

    /// <summary>Optional JSON file with slang eligibility globs, variant caps, and forced defines.</summary>
    [Option("compiler-config", Required = false, HelpText = "Path to compiler JSON config (merges over built-in defaults).")]
    public string? CompilerConfigPath { get; set; }

    /// <summary>Optional JSON file mapping shader names to explicit variant define lists.</summary>
    [Option("variant-config", Required = false, HelpText = "Path to variant override JSON.")]
    public string? VariantConfigPath { get; set; }

    /// <summary>Resolves default input and output paths relative to the git repository root (Renderide parent).</summary>
    public void DetermineDefaultPaths()
    {
        string renderideRoot = ResolveRenderideRoot(TryGetGitRepositoryRoot());

        if (OutputDirectory is null)
            OutputDirectory = Path.Combine(renderideRoot, "crates", "renderide", "src", "shaders", "generated");

        if (!InputDirectories.Any())
        {
            InputDirectories = new[]
            {
                Path.Combine(renderideRoot, "generators", "UnityShaderConverter", "SampleShaders"),
                Path.Combine(renderideRoot, "third_party", "Resonite.UnityShaders", "Assets", "Shaders"),
            };
        }
    }

    private static string ResolveRenderideRoot(string? gitTopLevel)
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

    private static string FallbackRenderideRootFromCwd()
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

    private static string? TryGetGitRepositoryRoot()
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
}
