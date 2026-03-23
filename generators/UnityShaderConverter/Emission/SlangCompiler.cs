using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using NotEnoughLogs;
using UnityShaderConverter.Logging;

namespace UnityShaderConverter.Emission;

/// <summary>Invokes the external <c>slangc</c> tool to produce WGSL.</summary>
public sealed class SlangCompiler
{
    private readonly string _slangcExecutable;
    private readonly Logger _logger;

    /// <summary>Creates a compiler facade.</summary>
    public SlangCompiler(string slangcExecutable, Logger logger)
    {
        _slangcExecutable = slangcExecutable;
        _logger = logger;
    }

    /// <summary>
    /// Resolves <c>slangc</c> from <c>--slangc</c>, <c>SLANGC</c>, a <c>Slang.Sdk</c> layout under the app directory
    /// (<c>runtimes/&lt;rid&gt;/native/slangc</c> when published), then <c>PATH</c>.
    /// </summary>
    public static string ResolveExecutable(string? optionPath)
    {
        if (!string.IsNullOrWhiteSpace(optionPath))
            return optionPath;
        string? env = Environment.GetEnvironmentVariable("SLANGC");
        if (!string.IsNullOrWhiteSpace(env))
            return env;
        string? bundled = TryResolveBundledSlangc();
        if (!string.IsNullOrWhiteSpace(bundled))
            return bundled;
        return "slangc";
    }

    private static string? TryResolveBundledSlangc()
    {
        string rid = GetRuntimeRid();
        string exe = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "slangc.exe" : "slangc";
        string baseDir = AppContext.BaseDirectory;
        string candidate = Path.Combine(baseDir, "runtimes", rid, "native", exe);
        return File.Exists(candidate) ? candidate : null;
    }

    private static string GetRuntimeRid()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return RuntimeInformation.OSArchitecture == Architecture.Arm64 ? "win-arm64" : "win-x64";
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            return RuntimeInformation.OSArchitecture == Architecture.Arm64 ? "osx-arm64" : "osx-x64";
        if (RuntimeInformation.OSArchitecture == Architecture.Arm64)
            return "linux-arm64";
        return "linux-x64";
    }

    /// <summary>Compiles Slang to WGSL (single module when <c>slangc</c> supports it; otherwise merged stages).</summary>
    public bool TryCompileToWgsl(
        string slangPath,
        string wgslOutPath,
        string runtimeSlangIncludeDir,
        string shaderSourceIncludeDir,
        string vertexEntry,
        string fragmentEntry,
        IReadOnlyList<string> variantDefines,
        out string? stderr)
    {
        if (TryCompileToWgslCore(
                slangPath,
                wgslOutPath,
                runtimeSlangIncludeDir,
                shaderSourceIncludeDir,
                vertexEntry,
                fragmentEntry,
                variantDefines,
                useMatrixLayout: true,
                out stderr))
            return true;

        if (stderr is not null &&
            stderr.Contains("matrix-layout", StringComparison.OrdinalIgnoreCase))
        {
            TryDelete(wgslOutPath);
            if (TryCompileToWgslCore(
                    slangPath,
                    wgslOutPath,
                    runtimeSlangIncludeDir,
                    shaderSourceIncludeDir,
                    vertexEntry,
                    fragmentEntry,
                    variantDefines,
                    useMatrixLayout: false,
                    out stderr))
            {
                _logger.LogDebug(LogCategory.SlangCompile, "slangc succeeded without -matrix-layout (toolchain lacks that flag).");
                return true;
            }
        }

        return false;
    }

    private bool TryCompileToWgslCore(
        string slangPath,
        string wgslOutPath,
        string runtimeSlangIncludeDir,
        string shaderSourceIncludeDir,
        string vertexEntry,
        string fragmentEntry,
        IReadOnlyList<string> variantDefines,
        bool useMatrixLayout,
        out string? stderr)
    {
        stderr = null;
        Directory.CreateDirectory(Path.GetDirectoryName(wgslOutPath)!);

        var singleArgs = new List<string> { slangPath, "-target", "wgsl" };
        if (useMatrixLayout)
        {
            singleArgs.Add("-matrix-layout");
            singleArgs.Add("column-major");
        }

        AddDefines(singleArgs, variantDefines);
        singleArgs.AddRange(new[] { "-I", runtimeSlangIncludeDir, "-I", shaderSourceIncludeDir, "-o", wgslOutPath });

        if (RunProcess(singleArgs, out string errSingle) && File.Exists(wgslOutPath) && new FileInfo(wgslOutPath).Length > 0)
        {
            _logger.LogDebug(LogCategory.SlangCompile, $"Wrote combined WGSL for {Path.GetFileName(slangPath)}");
            return true;
        }

        _logger.LogDebug(LogCategory.SlangCompile, $"Combined WGSL compile failed; trying per-stage merge. {errSingle}");

        var vertArgs = new List<string> { slangPath, "-target", "wgsl" };
        if (useMatrixLayout)
        {
            vertArgs.Add("-matrix-layout");
            vertArgs.Add("column-major");
        }

        vertArgs.AddRange(new[] { "-entry", vertexEntry, "-stage", "vertex" });
        AddDefines(vertArgs, variantDefines);
        vertArgs.AddRange(new[] { "-I", runtimeSlangIncludeDir, "-I", shaderSourceIncludeDir, "-o", wgslOutPath + ".vert.tmp" });

        if (!RunProcess(vertArgs, out string errV))
        {
            stderr = string.IsNullOrEmpty(errV) ? errSingle : errV;
            return false;
        }

        var fragArgs = new List<string> { slangPath, "-target", "wgsl" };
        if (useMatrixLayout)
        {
            fragArgs.Add("-matrix-layout");
            fragArgs.Add("column-major");
        }

        fragArgs.AddRange(new[] { "-entry", fragmentEntry, "-stage", "fragment" });
        AddDefines(fragArgs, variantDefines);
        fragArgs.AddRange(new[] { "-I", runtimeSlangIncludeDir, "-I", shaderSourceIncludeDir, "-o", wgslOutPath + ".frag.tmp" });

        if (!RunProcess(fragArgs, out string errF))
        {
            stderr = errF;
            TryDelete(wgslOutPath + ".vert.tmp");
            TryDelete(wgslOutPath + ".frag.tmp");
            return false;
        }

        try
        {
            string vert = File.ReadAllText(wgslOutPath + ".vert.tmp");
            string frag = File.ReadAllText(wgslOutPath + ".frag.tmp");
            File.WriteAllText(
                wgslOutPath,
                "// Generated by UnityShaderConverter — merged vertex + fragment stages.\n" +
                vert +
                "\n\n" +
                frag);
            return true;
        }
        catch (Exception ex)
        {
            stderr = ex.Message;
            return false;
        }
        finally
        {
            TryDelete(wgslOutPath + ".vert.tmp");
            TryDelete(wgslOutPath + ".frag.tmp");
        }
    }

    private static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path))
                File.Delete(path);
        }
        catch
        {
            // ignored
        }
    }

    private void AddDefines(List<string> args, IReadOnlyList<string> variantDefines)
    {
        foreach (string d in variantDefines)
        {
            if (d.Length > 0)
            {
                args.Add("-D");
                args.Add(d);
            }
        }
    }

    private bool RunProcess(List<string> args, out string stderrCombined)
    {
        stderrCombined = string.Empty;
        var psi = new ProcessStartInfo
        {
            FileName = _slangcExecutable,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };
        foreach (string a in args)
            psi.ArgumentList.Add(a);

        _logger.LogDebug(LogCategory.SlangCompile, $"slangc {string.Join(" ", args.Select(a => a.Contains(' ') ? $"\"{a}\"" : a))}");

        using var proc = new Process { StartInfo = psi };
        try
        {
            proc.Start();
        }
        catch (Exception ex)
        {
            stderrCombined = ex.Message;
            return false;
        }

        string stdout = proc.StandardOutput.ReadToEnd();
        string stderr = proc.StandardError.ReadToEnd();
        proc.WaitForExit();
        stderrCombined = string.Join(
            Environment.NewLine,
            new[] { stderr, stdout }.Where(s => !string.IsNullOrWhiteSpace(s)));
        if (proc.ExitCode != 0)
            return false;
        return true;
    }
}
