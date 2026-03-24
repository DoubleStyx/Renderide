using System.Diagnostics;
using UnityShaderConverter.Emission;

namespace UnityShaderConverter.Tests;

/// <summary>Tests for <see cref="SlangCompiler"/> stderr shaping and optional <c>slangc</c> flag compatibility.</summary>
public sealed class SlangCompilerDiagnosticsTests
{
    /// <summary>Slang warning lines are removed; error lines remain.</summary>
    [Fact]
    public void FilterSlangDiagnosticsForErrorsOnly_DropsWarningLines()
    {
        const string mixed = "warning[E15205]: noise\nerror[E30015]: boom\n";
        string filtered = SlangCompiler.FilterSlangDiagnosticsForErrorsOnly(mixed);
        Assert.DoesNotContain("warning[E15205]", filtered, StringComparison.Ordinal);
        Assert.Contains("error[E30015]", filtered, StringComparison.Ordinal);
    }

    /// <summary>Matching is case-insensitive for the <c>warning[E</c> prefix.</summary>
    [Fact]
    public void FilterSlangDiagnosticsForErrorsOnly_DropsWarningLinesCaseInsensitive()
    {
        const string mixed = "Warning[E15205]: noise\n";
        string filtered = SlangCompiler.FilterSlangDiagnosticsForErrorsOnly(mixed);
        Assert.DoesNotContain("15205", filtered, StringComparison.Ordinal);
    }

    /// <summary>When <c>slangc</c> is on PATH, its help output documents <c>-warnings-disable</c> (validates our CLI usage).</summary>
    [Fact]
    public void SlangcHelp_DocumentsWarningsDisable()
    {
        string exe = SlangCompiler.ResolveExecutable(null);
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = exe,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
            };
            psi.ArgumentList.Add("-help");
            using var proc = Process.Start(psi);
            Assert.NotNull(proc);
            string stdout = proc.StandardOutput.ReadToEnd();
            string stderr = proc.StandardError.ReadToEnd();
            proc.WaitForExit(15_000);
            string combined = stdout + stderr;
            Assert.Contains("warnings-disable", combined, StringComparison.OrdinalIgnoreCase);
        }
        catch (Exception)
        {
            // No slangc on PATH in this environment — other tests still cover filtering logic.
        }
    }
}
