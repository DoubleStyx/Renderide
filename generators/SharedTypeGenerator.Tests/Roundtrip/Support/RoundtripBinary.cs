using System.Diagnostics;
using SharedTypeGenerator.Logging;
using Xunit;

namespace SharedTypeGenerator.Tests.Roundtrip.Support;

/// <summary>Locates the Rust <c>roundtrip</c> test binary and runs it with safe stream draining.</summary>
public static class RoundtripBinary
{
    /// <summary>Executable name (no extension) under <c>target/*/roundtrip</c>.</summary>
    public const string DefaultBinaryName = "roundtrip";

    /// <summary>Default wall-clock budget for the child process.</summary>
    public static readonly TimeSpan DefaultTimeout = TimeSpan.FromSeconds(30);

    /// <summary>Tries to resolve the repository root and find <c>roundtrip</c> under <c>target/debug</c>, <c>target/dev-fast</c>, or <c>target/release</c>.</summary>
    /// <returns>Full path to the binary, or <see langword="null"/> if not found.</returns>
    public static string? TryFind()
    {
        string? repoRoot = FindRepoRoot();
        if (repoRoot == null)
            return null;

        string debugPath = System.IO.Path.Combine(repoRoot, "target", "debug", DefaultBinaryName);
        string devFastPath = System.IO.Path.Combine(repoRoot, "target", "dev-fast", DefaultBinaryName);
        string releasePath = System.IO.Path.Combine(repoRoot, "target", "release", DefaultBinaryName);

        if (File.Exists(debugPath))
            return debugPath;
        if (File.Exists(devFastPath))
            return devFastPath;
        if (File.Exists(releasePath))
            return releasePath;
        return null;
    }

    /// <summary>Resolves the Renderide repository root from git or by walking from <see cref="AppContext.BaseDirectory"/>.</summary>
    /// <returns>Canonical repo root, or <see langword="null"/> if not found.</returns>
    private static string? FindRepoRoot()
    {
        string? gitTop = RenderidePathResolver.TryGetGitRepositoryRoot();
        if (gitTop is not null)
        {
            string resolved = RenderidePathResolver.ResolveRenderideRoot(gitTop);
            if (IsCanonicalRenderideRepoRoot(resolved))
                return resolved;
        }

        var dir = AppContext.BaseDirectory;
        while (!string.IsNullOrEmpty(dir))
        {
            if (IsCanonicalRenderideRepoRoot(dir))
                return dir;
            dir = System.IO.Path.GetDirectoryName(dir);
        }

        return null;
    }

    private static bool IsCanonicalRenderideRepoRoot(string dir) =>
        Directory.Exists(System.IO.Path.Combine(dir, "generators", "SharedTypeGenerator"))
        && Directory.Exists(System.IO.Path.Combine(dir, "crates", "renderide"));

    /// <summary>Returns <see langword="true"/> when <c>CI</c> is <c>true</c> or <c>1</c>.</summary>
    public static bool IsCiEnvironment()
    {
        string? ci = Environment.GetEnvironmentVariable("CI");
        return string.Equals(ci, "true", StringComparison.OrdinalIgnoreCase)
            || string.Equals(ci, "1", StringComparison.Ordinal);
    }

    /// <summary>Skips the test locally when <paramref name="binaryPath"/> is null; throws when CI requires the binary.</summary>
    /// <param name="binaryPath">Resolved path from <see cref="TryFind"/>.</param>
    public static void RequireOrSkip(string? binaryPath)
    {
        if (binaryPath != null)
            return;

        if (IsCiEnvironment())
        {
            throw new InvalidOperationException(
                "Roundtrip binary is required when CI=true. From the repository root run: cargo build -p renderide --bin roundtrip");
        }

        Skip.If(true,
            "Roundtrip binary not found under target/debug, target/dev-fast, or target/release. From the repository root run: cargo build -p renderide --bin roundtrip");
    }

    /// <summary>Runs <c>roundtrip &lt;typeName&gt; &lt;input&gt; &lt;output&gt;</c> with redirected stdout/stderr and a timeout.</summary>
    /// <param name="binaryPath">Path to the roundtrip executable.</param>
    /// <param name="typeName">C# type name passed to Rust dispatch.</param>
    /// <param name="inputPath">Packed bytes input file.</param>
    /// <param name="outputPath">Output file written by the binary.</param>
    /// <param name="timeout">Optional timeout; defaults to <see cref="DefaultTimeout"/>.</param>
    public static void Run(string binaryPath, string typeName, string inputPath, string outputPath, TimeSpan? timeout = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(binaryPath);
        TimeSpan limit = timeout ?? DefaultTimeout;
        int timeoutMs = (int)Math.Min(int.MaxValue, Math.Max(1, limit.TotalMilliseconds));

        var psi = new ProcessStartInfo
        {
            FileName = binaryPath,
            ArgumentList = { typeName, inputPath, outputPath },
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };

        using var proc = new Process { StartInfo = psi };
        proc.Start();

        Task<string> stdoutTask = proc.StandardOutput.ReadToEndAsync();
        Task<string> stderrTask = proc.StandardError.ReadToEndAsync();

        if (!proc.WaitForExit(timeoutMs))
        {
            TryKillProcessTree(proc);
            try
            {
                proc.WaitForExit(5000);
            }
            catch (InvalidOperationException)
            {
            }
            catch (System.ComponentModel.Win32Exception)
            {
            }

            Task.WaitAll([stdoutTask, stderrTask], millisecondsTimeout: 5000);
            string stderrTail = stderrTask.Status == TaskStatus.RanToCompletion ? stderrTask.Result.Trim() : string.Empty;
            string stdoutTail = stdoutTask.Status == TaskStatus.RanToCompletion ? stdoutTask.Result.Trim() : string.Empty;
            string msg = FormattableString.Invariant($"Roundtrip binary timed out after {limit.TotalSeconds}s (type={typeName}).");
            if (!string.IsNullOrWhiteSpace(stderrTail))
                msg += FormattableString.Invariant($" stderr: {stderrTail}");
            if (!string.IsNullOrWhiteSpace(stdoutTail))
                msg += FormattableString.Invariant($" stdout: {stdoutTail}");
            throw new TimeoutException(msg);
        }

        // Process has exited; drain both streams (started before WaitForExit to avoid pipe deadlocks).
        Task.WaitAll(stdoutTask, stderrTask);
        string stderr = stderrTask.Result;
        string stdout = stdoutTask.Result;

        if (proc.ExitCode != 0)
        {
            string msg = FormattableString.Invariant($"Roundtrip binary failed (exit {proc.ExitCode}).");
            if (!string.IsNullOrWhiteSpace(stderr))
                msg += FormattableString.Invariant($" stderr: {stderr.Trim()}");
            if (!string.IsNullOrWhiteSpace(stdout))
                msg += FormattableString.Invariant($" stdout: {stdout.Trim()}");
            throw new InvalidOperationException(msg);
        }
    }

    /// <summary>Best-effort termination of <paramref name="proc"/> and its children after a timeout.</summary>
    private static void TryKillProcessTree(Process proc)
    {
        try
        {
            proc.Kill(entireProcessTree: true);
        }
        catch (InvalidOperationException)
        {
        }
        catch (System.ComponentModel.Win32Exception)
        {
        }
    }
}
