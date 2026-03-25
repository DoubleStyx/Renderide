using System.Collections.Generic;
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
/// Slang <c>error[E00004]: cannot write output file</c> often follows unresolved diagnostics (e.g. warning <c>39019</c> when not
/// listed in <c>-warnings-disable</c>); it can also indicate a truly unwritable output path, read-only tree, or concurrent writers.
/// When the combined WGSL compile step fails but per-stage merge succeeds, intermediate diagnostics are logged at
/// <see cref="LogLevel.Trace"/> only. Final <c>slangc</c> failures are logged at <see cref="LogLevel.Error"/> with full stderr
/// by the shader converter runner regardless of verbose mode.
/// When <see cref="TryCompileToWgsl"/> is called with <c>preserveWgslPipelineOverridableConstants</c>, passes <c>-preserve-params</c>
/// so <c>[vk::constant_id]</c> globals are not dropped from WGSL output (pipeline <c>override</c> / <c>@id</c>).
/// Combined whole-module <c>slangc</c> can still return success while emitting WGSL that has no <c>@vertex</c>/<c>@fragment</c> entry points
/// (only module-scope declarations); that output is rejected and the per-stage merge path is used instead.
/// </remarks>
public sealed class SlangCompiler
{
    /// <summary>Slang warning IDs that are noisy during Unity header conversion but rarely indicate WGSL failure.</summary>
    /// <remarks>
    /// 30056 covers deprecated non-short-circuiting <c>?:</c> in vendored Unity <c>.cginc</c> files.
    /// 39019 is <c>implicit global shader parameter</c> for legacy <c>half4 _Foo;</c> uniforms; when enabled as a warning, Slang
    /// fails WGSL emission with <c>error[E00004]: cannot write output file</c> even though the message suggests adding <c>uniform</c>.
    /// 41000 is <c>unreachable-code</c> in <c>UnityShadowLibrary.cginc</c> (deferred + combined WGSL) where Unity’s preprocessor layout
    /// leaves a final <c>return</c> Slang marks unreachable for some keyword combinations.
    /// </remarks>
    private static readonly string[] DefaultDisabledSlangWarningIds =
    {
        "15205",
        "15401",
        "15400",
        "30081",
        "30056",
        "39019",
        "41000",
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

    /// <summary>Returns the first non-empty line of Slang stderr (or stdout), truncated for short digests (not for primary error reporting).</summary>
    public static string FormatSlangStderrSummary(string? stderr, int maxLength = 400)
    {
        if (string.IsNullOrWhiteSpace(stderr))
            return "(no stderr)";

        ReadOnlySpan<char> span = stderr.AsSpan().Trim();
        int newline = span.IndexOfAny('\r', '\n');
        ReadOnlySpan<char> first = newline >= 0 ? span[..newline] : span;
        string s = first.Trim().ToString();
        if (s.Length > maxLength)
            return string.Concat(s.AsSpan(0, maxLength), "…");
        return s;
    }

    /// <summary>
    /// Produces a single line for <see cref="LogLevel.Trace"/> when combined WGSL compile fails: a Slang
    /// <c>error[E…]</c> line after <see cref="FilterSlangDiagnosticsForErrorsOnly"/>, preferring a non-<c>E00004</c>
    /// error when present (Slang often reports <c>error[E00004]: cannot write output file</c> after a root type error).
    /// </summary>
    public static string FormatSlangStderrErrorDigest(string? stderr, int maxLength = 320)
    {
        if (string.IsNullOrWhiteSpace(stderr))
            return "Slang: (no stderr)";

        string withoutWarnings = FilterSlangDiagnosticsForErrorsOnly(stderr);
        string source = string.IsNullOrWhiteSpace(withoutWarnings) ? stderr.Trim() : withoutWarnings;

        var errorLines = new List<string>();
        foreach (string line in source.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries))
        {
            string t = line.Trim();
            if (t.Length == 0)
                continue;
            if (t.Contains("error[", StringComparison.OrdinalIgnoreCase))
                errorLines.Add(t);
        }

        static bool IsCannotWriteOutputFileError(string line) =>
            line.Contains("error[E00004]", StringComparison.OrdinalIgnoreCase) &&
            line.Contains("cannot write output file", StringComparison.OrdinalIgnoreCase);

        foreach (string t in errorLines)
        {
            if (IsCannotWriteOutputFileError(t))
                continue;
            if (t.Length > maxLength)
                return string.Concat(t.AsSpan(0, maxLength), "…");
            return t;
        }

        foreach (string t in errorLines)
        {
            if (t.Length > maxLength)
                return string.Concat(t.AsSpan(0, maxLength), "…");
            return t;
        }

        return $"Slang: {FormatSlangStderrSummary(source, maxLength)}";
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
    /// <param name="unityCgIncludesDirs">Ordered <c>-I</c> directories with Unity <c>.cginc</c> trees (may be empty).</param>
    /// <param name="preserveWgslPipelineOverridableConstants">When true, passes <c>-preserve-params</c> so specialization constants survive WGSL emission.</param>
    public bool TryCompileToWgsl(
        string slangPath,
        string wgslOutPath,
        string runtimeSlangIncludeDir,
        IReadOnlyList<string> unityCgIncludesDirs,
        string shaderSourceIncludeDir,
        string vertexEntry,
        string fragmentEntry,
        IReadOnlyList<string> variantDefines,
        bool preserveWgslPipelineOverridableConstants,
        out string? stderr)
    {
        if (TryCompileToWgslCore(
                slangPath,
                wgslOutPath,
                runtimeSlangIncludeDir,
                unityCgIncludesDirs,
                shaderSourceIncludeDir,
                vertexEntry,
                fragmentEntry,
                variantDefines,
                useMatrixLayout: true,
                preserveWgslPipelineOverridableConstants,
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
                    unityCgIncludesDirs,
                    shaderSourceIncludeDir,
                    vertexEntry,
                    fragmentEntry,
                    variantDefines,
                    useMatrixLayout: false,
                    preserveWgslPipelineOverridableConstants,
                    out stderr))
            {
                _logger.LogTrace(LogCategory.SlangCompile, "slangc succeeded without -matrix-layout (toolchain lacks that flag).");
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

    /// <summary>Slang may drop unused pipeline parameters; <c>-preserve-params</c> keeps <c>[vk::constant_id]</c> as WGSL <c>override</c>.</summary>
    private static void AppendPreserveParamsForWgslOverrides(List<string> args, bool preserve)
    {
        if (preserve)
            args.Add("-preserve-params");
    }

    /// <summary>
    /// Returns whether <paramref name="wgsl"/> appears to define both a vertex and a fragment stage (WGSL <c>@vertex</c> and <c>@fragment</c>).
    /// </summary>
    /// <remarks>
    /// Used after a &quot;successful&quot; combined <c>slangc</c> emit: some Slang versions with <c>-preserve-params</c> write a non-empty WGSL file
    /// that omits stage entry points; <see cref="TryCompileToWgslCore"/> then falls back to separate vertex and fragment compiles merged into one file.
    /// </remarks>
    public static bool WgslContainsVertexAndFragmentStageMarkers(string wgsl)
    {
        if (string.IsNullOrEmpty(wgsl))
            return false;
        return wgsl.Contains("@vertex", StringComparison.Ordinal) &&
               wgsl.Contains("@fragment", StringComparison.Ordinal);
    }

    private bool TryCompileToWgslCore(
        string slangPath,
        string wgslOutPath,
        string runtimeSlangIncludeDir,
        IReadOnlyList<string> unityCgIncludesDirs,
        string shaderSourceIncludeDir,
        string vertexEntry,
        string fragmentEntry,
        IReadOnlyList<string> variantDefines,
        bool useMatrixLayout,
        bool preserveWgslPipelineOverridableConstants,
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

        AppendPreserveParamsForWgslOverrides(singleArgs, preserveWgslPipelineOverridableConstants);

        AddDefines(singleArgs, variantDefines);
        AppendIncludeDirectories(singleArgs, runtimeSlangIncludeDir, unityCgIncludesDirs, shaderSourceIncludeDir);
        singleArgs.AddRange(new[] { "-o", wgslOutPath });

        bool ranSingle = RunProcess(singleArgs, out string errSingleRaw);
        bool singleWroteNonEmpty =
            ranSingle &&
            File.Exists(wgslOutPath) &&
            new FileInfo(wgslOutPath).Length > 0;

        if (singleWroteNonEmpty)
        {
            string combinedText = File.ReadAllText(wgslOutPath);
            if (WgslContainsVertexAndFragmentStageMarkers(combinedText))
            {
                _logger.LogTrace(LogCategory.SlangCompile, $"Wrote combined WGSL for {Path.GetFileName(slangPath)}");
                return true;
            }

            _logger.LogTrace(
                LogCategory.SlangCompile,
                $"Combined WGSL for {Path.GetFileName(slangPath)} is missing @vertex/@fragment entry points; using per-stage merge.");
        }

        if (!singleWroteNonEmpty &&
            useMatrixLayout &&
            errSingleRaw.Contains("matrix-layout", StringComparison.OrdinalIgnoreCase))
        {
            stderr = ApplySlangStderrPolicy(errSingleRaw);
            return false;
        }

        string policyStderr = ApplySlangStderrPolicy(errSingleRaw);
        string errorDigest = FormatSlangStderrErrorDigest(policyStderr);
        if (!singleWroteNonEmpty)
        {
            _logger.LogTrace(
                LogCategory.SlangCompile,
                $"Combined WGSL compile failed; trying per-stage merge. {errorDigest} (full stderr below).");
        }

        _logger.LogTrace(
            LogCategory.SlangCompile,
            $"Combined WGSL compile stderr: {policyStderr}");

        var vertArgs = new List<string> { slangPath, "-target", "wgsl" };
        AppendSlangWarningPolicyArgs(vertArgs);
        if (useMatrixLayout)
        {
            vertArgs.Add("-matrix-layout");
            vertArgs.Add("column-major");
        }

        AppendPreserveParamsForWgslOverrides(vertArgs, preserveWgslPipelineOverridableConstants);

        vertArgs.AddRange(new[] { "-entry", vertexEntry, "-stage", "vertex" });
        AddDefines(vertArgs, variantDefines);
        AppendIncludeDirectories(vertArgs, runtimeSlangIncludeDir, unityCgIncludesDirs, shaderSourceIncludeDir);
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

        AppendPreserveParamsForWgslOverrides(fragArgs, preserveWgslPipelineOverridableConstants);

        fragArgs.AddRange(new[] { "-entry", fragmentEntry, "-stage", "fragment" });
        AddDefines(fragArgs, variantDefines);
        AppendIncludeDirectories(fragArgs, runtimeSlangIncludeDir, unityCgIncludesDirs, shaderSourceIncludeDir);
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

    /// <summary>Appends <c>-I</c> in order: <c>runtime_slang</c>, each Unity/extra include root, shader source directory.</summary>
    private static void AppendIncludeDirectories(
        List<string> args,
        string runtimeSlangIncludeDir,
        IReadOnlyList<string> unityCgIncludesDirs,
        string shaderSourceIncludeDir)
    {
        args.AddRange(new[] { "-I", runtimeSlangIncludeDir });
        foreach (string d in unityCgIncludesDirs)
        {
            if (!string.IsNullOrWhiteSpace(d))
                args.AddRange(new[] { "-I", d });
        }

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

        _logger.LogTrace(LogCategory.SlangCompile, $"slangc {string.Join(" ", args.Select(a => a.Contains(' ') ? $"\"{a}\"" : a))}");

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
            _logger.LogTrace(LogCategory.SlangCompile, $"slangc slow: {sw.ElapsedMilliseconds}ms ({inputName})");
        }

        stderrCombined = string.Join(
            Environment.NewLine,
            new[] { stderr, stdout }.Where(s => !string.IsNullOrWhiteSpace(s)));
        if (proc.ExitCode != 0)
            return false;
        return true;
    }
}
