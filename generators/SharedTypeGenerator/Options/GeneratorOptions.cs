using System.Diagnostics;
using CommandLine;

namespace SharedTypeGenerator.Options;

/// <summary>Command-line options for <see cref="SharedTypeGenerator.GeneratorRunner"/>: input assembly path, output path, and verbosity flags.</summary>
public class GeneratorOptions
{
    /// <summary>When true, lowers the log threshold to trace (includes per-type analysis logs).</summary>
    [Option('v', "verbose", Required = false, HelpText = "Output verbose messaging as Rust types are generated.")]
    public bool Verbose { get; set; }

    /// <summary>When true, the emitter writes a comment before each type's Rust definition (C# name and shape).</summary>
    [Option("il-verbose", Required = false, HelpText = "Emit a leading comment for each type in the Rust file (C# name and TypeShape).")]
    public bool IlVerbose { get; set; }

    /// <summary>Path to Renderite.Shared.dll (and sibling managed DLLs) to analyze.</summary>
    [Option('i', "assembly-path", Required = false, HelpText = "Path to Renderite.Shared.dll. When omitted, the generator searches RENDERITE_SHARED_DLL, RESONITE_DIR, STEAM_PATH, default Steam locations, and libraryfolders.vdf (same idea as the Rust bootstrapper).")]
    public string? AssemblyPath { get; set; }

    /// <summary>Optional path for generated <c>shared.rs</c>; when null, <see cref="DetermineDefaultOutputPath"/> is used.</summary>
    [Option('o', "output-rust-file", Required = false, Default = null, HelpText = "The destination .rs file to generate.")]
    public string? OutputRustFile { get; set; }

    /// <summary>Sets <see cref="OutputRustFile"/> to <c>crates/renderide/src/shared/shared.rs</c> under the git repository root.</summary>
    /// <exception cref="InvalidOperationException">Thrown when git is missing, times out, or the path cannot be resolved.</exception>
    public void DetermineDefaultOutputPath()
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
        catch (Exception e)
        {
            throw new InvalidOperationException(
                "Could not start `git` to resolve the default output path. Ensure git is installed and on PATH.", e);
        }

        const int timeoutMs = 10_000;
        if (!process.WaitForExit(timeoutMs))
        {
            try
            {
                process.Kill(entireProcessTree: true);
            }
            catch (InvalidOperationException)
            {
            }
            catch (System.ComponentModel.Win32Exception)
            {
            }

            throw new InvalidOperationException(
                $"Timed out after {timeoutMs}ms waiting for `git rev-parse --show-toplevel`.");
        }

        string stderr = process.StandardError.ReadToEnd();
        string stdout = process.StandardOutput.ReadToEnd();

        if (process.ExitCode != 0)
        {
            string detail = string.IsNullOrWhiteSpace(stderr) ? "(no stderr)" : stderr.Trim();
            throw new InvalidOperationException(
                $"Could not resolve git repository root (git exit code {process.ExitCode}): {detail}");
        }

        string? line = stdout
            .Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries)
            .FirstOrDefault();
        if (string.IsNullOrWhiteSpace(line))
            throw new InvalidOperationException("git rev-parse --show-toplevel produced no output; is this a git repository?");

        string gitRoot = line.Trim();
        string renderideRoot = Renderide.Generators.Logging.RenderidePathResolver.ResolveRenderideRoot(gitRoot);
        OutputRustFile = Path.Combine(renderideRoot, "crates", "renderide", "src", "shared", "shared.rs");
    }
}
