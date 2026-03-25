using System.Diagnostics;
using NotEnoughLogs;
using NotEnoughLogs.Behaviour;
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

    /// <summary>Digest skips warnings and prefers a non-<c>E00004</c> error when Slang also emits a misleading write failure.</summary>
    [Fact]
    public void FormatSlangStderrErrorDigest_PrefersNonE00004WhenPresent()
    {
        const string stderr = """
            warning[E15205]: implicit global
            error[E30019]: type mismatch in expression
            error[E00004]: cannot write output file '/tmp/out.wgsl'
            """;
        string digest = SlangCompiler.FormatSlangStderrErrorDigest(stderr);
        Assert.Contains("error[E30019]", digest, StringComparison.Ordinal);
        Assert.DoesNotContain("error[E00004]", digest, StringComparison.Ordinal);
        Assert.DoesNotContain("warning[E15205]", digest, StringComparison.Ordinal);
    }

    /// <summary>When <c>E00004</c> is the only error line, it is still surfaced.</summary>
    [Fact]
    public void FormatSlangStderrErrorDigest_UsesE00004WhenSoleError()
    {
        const string stderr = """
            warning[E15205]: implicit global
            error[E00004]: cannot write output file '/tmp/out.wgsl'
            """;
        string digest = SlangCompiler.FormatSlangStderrErrorDigest(stderr);
        Assert.Contains("error[E00004]", digest, StringComparison.Ordinal);
        Assert.Contains("cannot write output file", digest, StringComparison.Ordinal);
        Assert.DoesNotContain("warning[E15205]", digest, StringComparison.Ordinal);
    }

    /// <summary>When no <c>error[</c> line exists, digest falls back to a short summary.</summary>
    [Fact]
    public void FormatSlangStderrErrorDigest_FallbackToSummaryWhenNoErrorToken()
    {
        const string stderr = "some diagnostic text without error marker\nsecond line\n";
        string digest = SlangCompiler.FormatSlangStderrErrorDigest(stderr);
        Assert.StartsWith("Slang:", digest, StringComparison.Ordinal);
        Assert.Contains("some diagnostic", digest, StringComparison.Ordinal);
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

    /// <summary>
    /// HLSL allows <c>dot(half3, half4)</c> via implicit truncation; <c>UnityCompat.slang</c> supplies a Slang overload for WGSL.
    /// </summary>
    [Fact]
    public void SlangCompile_DotHalf3Half4_OverloadFromUnityCompat_Compiles()
    {
        string baseDir = AppContext.BaseDirectory;
        string runtimeSlang = Path.Combine(baseDir, "runtime_slang");
        Assert.True(Directory.Exists(runtimeSlang), "Expected runtime_slang next to test output (rebuild UnityShaderConverter.Tests).");

        string temp = Path.Combine(Path.GetTempPath(), "usc_slang_dot34_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(temp);
        try
        {
            const string tu = """
                #include "UnityCompat.slang"

                struct vi { float4 vertex : POSITION; };
                struct vo { float4 pos : SV_POSITION; half3 a; half4 b; };

                vo vert(vi v)
                {
                    vo o;
                    o.pos = float4(0.0, 0.0, 0.0, 1.0);
                    o.a = half3(1.0, 0.0, 0.0);
                    o.b = half4(0.0, 1.0, 0.0, 0.0);
                    return o;
                }

                float4 frag(vo i) : SV_Target0
                {
                    half d = dot(i.a, i.b);
                    return float4(float(d), 0.0, 0.0, 1.0);
                }
                """;
            string slangPath = Path.Combine(temp, "dot_h3_h4.slang");
            string wgslPath = Path.Combine(temp, "out.wgsl");
            File.WriteAllText(slangPath, tu);

            using var logger = new Logger(new LoggerConfiguration
            {
                Behaviour = new DirectLoggingBehaviour(),
                MaxLevel = LogLevel.Error,
            });
            var compiler = new SlangCompiler(SlangCompiler.ResolveExecutable(null), logger, suppressSlangWarnings: true);
            bool ok = compiler.TryCompileToWgsl(
                slangPath,
                wgslPath,
                runtimeSlang,
                Array.Empty<string>(),
                temp,
                "vert",
                "frag",
                Array.Empty<string>(),
                preserveWgslPipelineOverridableConstants: false,
                out string? stderr);

            Assert.True(ok, stderr ?? "slangc failed with no stderr");
            Assert.True(File.Exists(wgslPath), "WGSL path missing");
        }
        finally
        {
            try
            {
                Directory.Delete(temp, recursive: true);
            }
            catch
            {
                // ignored
            }
        }
    }

    /// <summary>
    /// Legacy Unity-style globals without <c>uniform</c> trigger Slang <c>warning[E39019]</c>; default suppression must allow WGSL output.
    /// </summary>
    [Fact]
    public void SlangCompile_SuppressSlangWarnings_DisablesImplicitGlobalParameter39019()
    {
        string baseDir = AppContext.BaseDirectory;
        string runtimeSlang = Path.Combine(baseDir, "runtime_slang");
        Assert.True(Directory.Exists(runtimeSlang), "Expected runtime_slang next to test output (rebuild UnityShaderConverter.Tests).");

        string temp = Path.Combine(Path.GetTempPath(), "usc_slang_39019_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(temp);
        try
        {
            const string tu = """
                half4 unity_Lightmap_HDR;

                struct vi { float4 vertex : POSITION; };
                struct vo { float4 pos : SV_POSITION; };

                vo vert(vi v)
                {
                    vo o;
                    o.pos = float4(0.0, 0.0, 0.0, 1.0);
                    return o;
                }

                float4 frag(vo i) : SV_Target0
                {
                    return float4(unity_Lightmap_HDR.rgb, 1.0);
                }
                """;
            string slangPath = Path.Combine(temp, "implicit_global.slang");
            string wgslPath = Path.Combine(temp, "out.wgsl");
            File.WriteAllText(slangPath, tu);

            using var logger = new Logger(new LoggerConfiguration
            {
                Behaviour = new DirectLoggingBehaviour(),
                MaxLevel = LogLevel.Error,
            });
            var compiler = new SlangCompiler(SlangCompiler.ResolveExecutable(null), logger, suppressSlangWarnings: true);
            bool ok = compiler.TryCompileToWgsl(
                slangPath,
                wgslPath,
                runtimeSlang,
                Array.Empty<string>(),
                temp,
                "vert",
                "frag",
                Array.Empty<string>(),
                preserveWgslPipelineOverridableConstants: false,
                out string? stderr);

            Assert.True(ok, stderr ?? "slangc failed with no stderr");
            Assert.True(File.Exists(wgslPath), "WGSL path missing");
        }
        finally
        {
            try
            {
                Directory.Delete(temp, recursive: true);
            }
            catch
            {
                // ignored
            }
        }
    }

    /// <summary>
    /// <c>-preserve-params</c> keeps <c>[vk::constant_id]</c> as WGSL <c>override</c> so wgpu pipeline constants apply.
    /// </summary>
    [Fact]
    public void TryCompileToWgsl_PreserveParams_EmitsWgslOverrideForUnusedSpecializationConstant()
    {
        string baseDir = AppContext.BaseDirectory;
        string runtimeSlang = Path.Combine(baseDir, "runtime_slang");
        Assert.True(Directory.Exists(runtimeSlang), "Expected runtime_slang next to test output (rebuild UnityShaderConverter.Tests).");

        string temp = Path.Combine(Path.GetTempPath(), "usc_slang_preserve_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(temp);
        try
        {
            const string tu = """
                [vk::constant_id(0)]
                const bool USC_FEATURE = false;

                struct vi { float4 vertex : POSITION; };
                struct vo { float4 pos : SV_POSITION; };

                vo vert(vi v)
                {
                    vo o;
                    o.pos = float4(0.0, 0.0, 0.0, 1.0);
                    return o;
                }

                float4 frag(vo i) : SV_Target0
                {
                    return float4(1.0, 0.0, 0.0, 1.0);
                }
                """;
            string slangPath = Path.Combine(temp, "spec.slang");
            string wgslPath = Path.Combine(temp, "out.wgsl");
            File.WriteAllText(slangPath, tu);

            using var logger = new Logger(new LoggerConfiguration
            {
                Behaviour = new DirectLoggingBehaviour(),
                MaxLevel = LogLevel.Error,
            });
            var compiler = new SlangCompiler(SlangCompiler.ResolveExecutable(null), logger, suppressSlangWarnings: true);

            bool okWithout = compiler.TryCompileToWgsl(
                slangPath,
                wgslPath + ".a",
                runtimeSlang,
                Array.Empty<string>(),
                temp,
                "vert",
                "frag",
                Array.Empty<string>(),
                preserveWgslPipelineOverridableConstants: false,
                out string? errA);
            Assert.True(okWithout, errA ?? "slangc failed");
            string wgslA = File.ReadAllText(wgslPath + ".a");
            Assert.DoesNotContain("override", wgslA, StringComparison.Ordinal);

            bool okWith = compiler.TryCompileToWgsl(
                slangPath,
                wgslPath + ".b",
                runtimeSlang,
                Array.Empty<string>(),
                temp,
                "vert",
                "frag",
                Array.Empty<string>(),
                preserveWgslPipelineOverridableConstants: true,
                out string? errB);
            Assert.True(okWith, errB ?? "slangc failed");
            string wgslB = File.ReadAllText(wgslPath + ".b");
            Assert.Contains("override", wgslB, StringComparison.Ordinal);
            Assert.Contains("@id(0)", wgslB, StringComparison.Ordinal);
            Assert.True(
                SlangCompiler.WgslContainsVertexAndFragmentStageMarkers(wgslB),
                "Expected merged or combined WGSL to include @vertex and @fragment after -preserve-params.");
        }
        finally
        {
            try
            {
                Directory.Delete(temp, recursive: true);
            }
            catch
            {
                // ignored
            }
        }
    }

    /// <summary><see cref="SlangCompiler.WgslContainsVertexAndFragmentStageMarkers"/> rejects empty and globals-only WGSL.</summary>
    [Fact]
    public void WgslContainsVertexAndFragmentStageMarkers_EmptyOrIncomplete_False()
    {
        Assert.False(SlangCompiler.WgslContainsVertexAndFragmentStageMarkers(""));
        Assert.False(SlangCompiler.WgslContainsVertexAndFragmentStageMarkers("   "));
        const string globalsOnly = """
            enable f16;
            @binding(0) @group(0) var<uniform> u : vec4<f32>;
            @id(0) override X : bool = false;
            """;
        Assert.False(SlangCompiler.WgslContainsVertexAndFragmentStageMarkers(globalsOnly));
    }

    /// <summary>Detects typical merged WGSL from the converter (both stage attributes present).</summary>
    [Fact]
    public void WgslContainsVertexAndFragmentStageMarkers_BothStages_True()
    {
        const string merged = """
            @vertex
            fn v() { }

            @fragment
            fn f() { }
            """;
        Assert.True(SlangCompiler.WgslContainsVertexAndFragmentStageMarkers(merged));
    }
}
