using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using NotEnoughLogs;
using UnityShaderConverter.Logging;

namespace UnityShaderConverter.Emission;

/// <summary>Invokes the external <c>slangc</c> tool to produce WGSL.</summary>
/// <remarks>
/// Global Unity platform and sampling shims live in <c>runtime_slang/UnityCompat.slang</c> and
/// <c>runtime_slang/UnityCompatPostUnity.slang</c> (included by <see cref="SlangEmitter"/>), not in duplicate <c>-D</c> flags here.
/// </remarks>
public sealed class SlangCompiler
{
    /// <summary>Slang warning IDs that are noisy during Unity header conversion but rarely indicate WGSL failure.</summary>
    /// <remarks>30056 covers deprecated non-short-circuiting <c>?:</c> in vendored Unity <c>.cginc</c> files.</remarks>
    private static readonly string[] DefaultDisabledSlangWarningIds =
    {
        "15205",
        "15401",
        "15400",
        "30081",
        "30056",
    };

    private readonly string _slangcExecutable;
    private readonly Logger _logger;
    private readonly bool _suppressSlangWarnings;

    /// <summary>Creates a compiler facade.</summary>
    /// <param name="suppressSlangWarnings">When true, passes <c>-warnings-disable</c> and strips warning lines from logged stderr.</param>
    public SlangCompiler(string slangcExecutable, Logger logger, bool suppressSlangWarnings = false)
    {
        _slangcExecutable = slangcExecutable;
        _logger = logger;
        _suppressSlangWarnings = suppressSlangWarnings;
    }

    /// <summary>
    /// Removes lines containing Slang <c>warning[E…]</c> markers from combined stderr/stdout. Error lines are preserved.
    /// </summary>
    public static string FilterSlangDiagnosticsForErrorsOnly(string combinedStderrAndStdout)
    {
        if (string.IsNullOrWhiteSpace(combinedStderrAndStdout))
            return string.Empty;

        string[] lines = combinedStderrAndStdout.Split(new[] { '\r', '\n' }, StringSplitOptions.None);
        var sb = new StringBuilder();
        foreach (string line in lines)
        {
            if (line.Contains("warning[E", StringComparison.OrdinalIgnoreCase))
                continue;
            sb.AppendLine(line);
        }

        return sb.ToString().TrimEnd();
    }

    /// <summary>
    /// Resolves <c>slangc</c> from <c>--slangc</c>, the <c>Slang.Sdk</c> copy under the app directory
    /// (<c>runtimes/&lt;rid&gt;/native/slangc</c>), then <c>SLANGC</c>, then <c>PATH</c>.
    /// Prefer the SDK binary when present so flags such as <c>-matrix-layout</c> match the referenced <c>Slang.Sdk</c> package; override with <c>--slangc</c> or <c>SLANGC</c> when needed.
    /// </summary>
    public static string ResolveExecutable(string? optionPath)
    {
        if (!string.IsNullOrWhiteSpace(optionPath))
            return optionPath;
        string? bundled = TryResolveBundledSlangc();
        if (!string.IsNullOrWhiteSpace(bundled))
            return bundled;
        string? env = Environment.GetEnvironmentVariable("SLANGC");
        if (!string.IsNullOrWhiteSpace(env))
            return env;
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
    /// <param name="unityCgIncludesDir">Optional directory with <c>UnityCG.cginc</c> (same as ShaderLab resolution); when <c>null</c>, Unity <c>#include</c> may fail.</param>
    public bool TryCompileToWgsl(
        string slangPath,
        string wgslOutPath,
        string runtimeSlangIncludeDir,
        string? unityCgIncludesDir,
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
                unityCgIncludesDir,
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
                    unityCgIncludesDir,
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

    private string ApplySlangStderrPolicy(string raw) =>
        _suppressSlangWarnings ? FilterSlangDiagnosticsForErrorsOnly(raw) : raw;

    private void AppendSlangWarningPolicyArgs(List<string> args)
    {
        if (!_suppressSlangWarnings)
            return;
        args.Add("-warnings-disable");
        args.Add(string.Join(",", DefaultDisabledSlangWarningIds));
    }

    private bool TryCompileToWgslCore(
        string slangPath,
        string wgslOutPath,
        string runtimeSlangIncludeDir,
        string? unityCgIncludesDir,
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
        AppendSlangWarningPolicyArgs(singleArgs);
        if (useMatrixLayout)
        {
            singleArgs.Add("-matrix-layout");
            singleArgs.Add("column-major");
        }

        AddDefines(singleArgs, variantDefines);
        AppendIncludeDirectories(singleArgs, runtimeSlangIncludeDir, unityCgIncludesDir, shaderSourceIncludeDir);
        singleArgs.AddRange(new[] { "-o", wgslOutPath });

        if (RunProcess(singleArgs, out string errSingleRaw) && File.Exists(wgslOutPath) && new FileInfo(wgslOutPath).Length > 0)
        {
            _logger.LogDebug(LogCategory.SlangCompile, $"Wrote combined WGSL for {Path.GetFileName(slangPath)}");
            return true;
        }

        if (useMatrixLayout &&
            errSingleRaw.Contains("matrix-layout", StringComparison.OrdinalIgnoreCase))
        {
            stderr = ApplySlangStderrPolicy(errSingleRaw);
            return false;
        }

        _logger.LogDebug(
            LogCategory.SlangCompile,
            $"Combined WGSL compile failed; trying per-stage merge. {ApplySlangStderrPolicy(errSingleRaw)}");

        var vertArgs = new List<string> { slangPath, "-target", "wgsl" };
        AppendSlangWarningPolicyArgs(vertArgs);
        if (useMatrixLayout)
        {
            vertArgs.Add("-matrix-layout");
            vertArgs.Add("column-major");
        }

        vertArgs.AddRange(new[] { "-entry", vertexEntry, "-stage", "vertex" });
        AddDefines(vertArgs, variantDefines);
        AppendIncludeDirectories(vertArgs, runtimeSlangIncludeDir, unityCgIncludesDir, shaderSourceIncludeDir);
        vertArgs.AddRange(new[] { "-o", wgslOutPath + ".vert.tmp" });

        if (!RunProcess(vertArgs, out string errVRaw))
        {
            stderr = ApplySlangStderrPolicy(string.IsNullOrEmpty(errVRaw) ? errSingleRaw : errVRaw);
            return false;
        }

        var fragArgs = new List<string> { slangPath, "-target", "wgsl" };
        AppendSlangWarningPolicyArgs(fragArgs);
        if (useMatrixLayout)
        {
            fragArgs.Add("-matrix-layout");
            fragArgs.Add("column-major");
        }

        fragArgs.AddRange(new[] { "-entry", fragmentEntry, "-stage", "fragment" });
        AddDefines(fragArgs, variantDefines);
        AppendIncludeDirectories(fragArgs, runtimeSlangIncludeDir, unityCgIncludesDir, shaderSourceIncludeDir);
        fragArgs.AddRange(new[] { "-o", wgslOutPath + ".frag.tmp" });

        if (!RunProcess(fragArgs, out string errFRaw))
        {
            stderr = ApplySlangStderrPolicy(errFRaw);
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

    /// <summary>Appends <c>-I</c> in order: <c>runtime_slang</c>, Unity CGIncludes (if any), shader source directory.</summary>
    private static void AppendIncludeDirectories(
        List<string> args,
        string runtimeSlangIncludeDir,
        string? unityCgIncludesDir,
        string shaderSourceIncludeDir)
    {
        args.AddRange(new[] { "-I", runtimeSlangIncludeDir });
        if (!string.IsNullOrWhiteSpace(unityCgIncludesDir))
            args.AddRange(new[] { "-I", unityCgIncludesDir });
        args.AddRange(new[] { "-I", shaderSourceIncludeDir });
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
        var sw = Stopwatch.StartNew();
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
        sw.Stop();
        if (sw.ElapsedMilliseconds >= 3000)
        {
            string inputName = args.Count > 0 ? Path.GetFileName(args[0]) : "?";
            _logger.LogDebug(LogCategory.SlangCompile, $"slangc slow: {sw.ElapsedMilliseconds}ms ({inputName})");
        }

        stderrCombined = string.Join(
            Environment.NewLine,
            new[] { stderr, stdout }.Where(s => !string.IsNullOrWhiteSpace(s)));
        if (proc.ExitCode != 0)
            return false;
        return true;
    }
}
