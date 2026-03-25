using NotEnoughLogs;
using NotEnoughLogs.Behaviour;
using UnityShaderConverter.Emission;

namespace UnityShaderConverter.Tests;

/// <summary>
/// Ensures <c>UnityCompat.slang</c> does not duplicate symbols defined by Unity’s <c>UnityCG.cginc</c> chain (regression for slangc ambiguity / redefinition).
/// </summary>
public sealed class UnityCompatUnityCgCompileTests
{
    /// <summary>
    /// When <c>slangc</c> is available, a minimal TU that includes compat then <c>UnityCG.cginc</c> then PostUnity must compile without duplicate <c>unity_ObjectToWorld</c> or <c>UnityObjectToClipPos</c> errors.
    /// </summary>
    [Fact]
    public void SlangCompile_UnityCompatThenUnityCg_NoDuplicateBuiltinErrors()
    {
        string baseDir = AppContext.BaseDirectory;
        string runtimeSlang = Path.Combine(baseDir, "runtime_slang");
        string unityCg = Path.Combine(baseDir, "UnityBuiltinCGIncludes");
        Assert.True(
            Directory.Exists(runtimeSlang) && Directory.Exists(unityCg),
            "Expected runtime_slang and UnityBuiltinCGIncludes next to test output (rebuild UnityShaderConverter.Tests).");

        string temp = Path.Combine(Path.GetTempPath(), "usc_compat_unitycg_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(temp);
        try
        {
            const string tu = """
                #include "UnityCompat.slang"
                #include "UnityCG.cginc"
                #include "UnityCompatPostUnity.slang"

                struct appdata { float4 vertex : POSITION; };
                struct v2f { float4 vertex : SV_POSITION; };

                v2f vert(appdata v)
                {
                    v2f o;
                    o.vertex = UnityObjectToClipPos(v.vertex);
                    return o;
                }

                float4 frag(v2f i) : SV_Target0
                {
                    return float4(unity_ObjectToWorld[0][0], 0.0, 0.0, 1.0);
                }
                """;
            string slangPath = Path.Combine(temp, "compat_unitycg_min.slang");
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
                new[] { unityCg },
                temp,
                "vert",
                "frag",
                Array.Empty<string>(),
                preserveWgslPipelineOverridableConstants: false,
                out string? stderr);

            Assert.True(ok, stderr ?? "slangc failed with no stderr");
            Assert.True(File.Exists(wgslPath) && new FileInfo(wgslPath).Length > 0, "empty WGSL output");
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
    /// Resonite <c>Common.cginc</c> EVR_* macros use a formal parameter named <c>tex</c> and call <c>tex2D(tex, …)</c> /
    /// <c>UNITY_SAMPLE_TEX2D(tex, …)</c>. Slang must not produce <c>samplertex</c> from <c>sampler##tex</c> during nested expansion.
    /// </summary>
    [Fact]
    public void SlangCompile_NestedMacroFormalTex_Tex2DAndUnitySample_Compiles()
    {
        string baseDir = AppContext.BaseDirectory;
        string runtimeSlang = Path.Combine(baseDir, "runtime_slang");
        string unityCg = Path.Combine(baseDir, "UnityBuiltinCGIncludes");
        Assert.True(
            Directory.Exists(runtimeSlang) && Directory.Exists(unityCg),
            "Expected runtime_slang and UnityBuiltinCGIncludes next to test output (rebuild UnityShaderConverter.Tests).");

        string temp = Path.Combine(Path.GetTempPath(), "usc_evr_tex_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(temp);
        try
        {
            const string tu = """
                #include "UnityCompat.slang"
                #include "UnityCG.cginc"
                #include "UnityCompatPostUnity.slang"

                UNITY_DECLARE_TEX2D(_EvrTestTex);

                struct appdata { float4 vertex : POSITION; };
                struct v2f { float4 vertex : SV_POSITION; float2 tc0 : TEXCOORD0; };

                #define EVR_LIKE_TEX2D(i, t, tex) float4 evr_c0 = tex2D(tex, i.t);
                #define EVR_LIKE_SAMPLE(i, t, tex) float4 evr_c1 = UNITY_SAMPLE_TEX2D(tex, i.t);

                v2f vert(appdata v)
                {
                    v2f o;
                    o.vertex = UnityObjectToClipPos(v.vertex);
                    o.tc0 = float2(0.0, 0.0);
                    return o;
                }

                float4 frag(v2f i) : SV_Target0
                {
                    EVR_LIKE_TEX2D(i, tc0, _EvrTestTex);
                    EVR_LIKE_SAMPLE(i, tc0, _EvrTestTex);
                    return evr_c0 + evr_c1;
                }
                """;
            string slangPath = Path.Combine(temp, "evr_tex_formal.slang");
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
                new[] { unityCg },
                temp,
                "vert",
                "frag",
                Array.Empty<string>(),
                preserveWgslPipelineOverridableConstants: false,
                out string? stderr);

            Assert.True(ok, stderr ?? "slangc failed with no stderr");
            Assert.True(File.Exists(wgslPath) && new FileInfo(wgslPath).Length > 0, "empty WGSL output");
            string wgslText = File.ReadAllText(wgslPath);
            Assert.DoesNotContain("samplertex", wgslText, StringComparison.Ordinal);
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
}
