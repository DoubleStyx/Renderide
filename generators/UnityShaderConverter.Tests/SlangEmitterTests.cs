using UnityShaderConverter.Analysis;
using UnityShaderConverter.Emission;
using UnityShaderConverter.Variants;
using UnityShaderParser.ShaderLab;

namespace UnityShaderConverter.Tests;

/// <summary>Snapshot-style checks for emitted Slang.</summary>
public sealed class SlangEmitterTests
{
    /// <summary>Emitted Slang includes compat header and strips <c>#pragma vertex</c>.</summary>
    [Fact]
    public void EmitPassSlang_IncludesUnityCompatAndStripsPragmas()
    {
        var pass = new ShaderPassDocument
        {
            PassName = null,
            PassIndex = 0,
            ProgramSource = "#pragma vertex vert\n#pragma fragment frag\nfloat4 main() { return 0; }\n",
            Pragmas = Array.Empty<string>(),
            VertexEntry = "vert",
            FragmentEntry = "frag",
            RenderStateSummary = "",
            FixedFunctionState = RenderStateExtractor.Extract(Array.Empty<ShaderLabCommandNode>(), new Dictionary<string, string>()),
        };
        string slang = SlangEmitter.EmitPassSlang(pass, Array.Empty<string>(), Array.Empty<SpecializationAxis>());
        Assert.Contains("#include \"UnityCompat.slang\"", slang);
        Assert.Contains("#include \"UnityCompatPostUnity.slang\"", slang);
        Assert.DoesNotContain("#pragma vertex", slang);
        Assert.DoesNotContain("#pragma fragment", slang);
    }

    /// <summary>UnityCG and Post-Unity compat are prepended so overrides apply before Resonite <c>Common.cginc</c>-first passes.</summary>
    [Fact]
    public void InsertUnityCompatPostInclude_PrependsUnityCgAndPostUnity()
    {
        const string body = "#include \"../Common.cginc\"\nfloat x;\n";
        string result = SlangEmitter.InsertUnityCompatPostIncludeAfterInitialIncludes(body);
        int unityCg = result.IndexOf("#include \"UnityCG.cginc\"", StringComparison.Ordinal);
        int post = result.IndexOf("UnityCompatPostUnity.slang", StringComparison.Ordinal);
        int common = result.IndexOf("#include \"../Common.cginc\"", StringComparison.Ordinal);
        int x = result.IndexOf("float x", StringComparison.Ordinal);
        Assert.True(unityCg >= 0);
        Assert.True(unityCg < post);
        Assert.True(post < common);
        Assert.True(common < x);
    }

    /// <summary>Specialization axes become <c>[vk::constant_id(n)]</c> bools before UnityCompat.</summary>
    [Fact]
    public void EmitPassSlang_WithAxes_InjectsVkConstantIdBools()
    {
        var pass = new ShaderPassDocument
        {
            PassName = null,
            PassIndex = 0,
            ProgramSource = "#pragma vertex vert\n#pragma fragment frag\nfloat4 main() { return 0; }\n",
            Pragmas = Array.Empty<string>(),
            VertexEntry = "vert",
            FragmentEntry = "frag",
            RenderStateSummary = "",
            FixedFunctionState = RenderStateExtractor.Extract(Array.Empty<ShaderLabCommandNode>(), new Dictionary<string, string>()),
        };
        var axes = new[]
        {
            new SpecializationAxis(0, "FOO", "USC_FOO", "foo"),
            new SpecializationAxis(1, "BAR", "USC_BAR", "bar"),
        };
        string slang = SlangEmitter.EmitPassSlang(pass, Array.Empty<string>(), axes);
        Assert.Contains("[vk::constant_id(0)]", slang);
        Assert.Contains("[vk::constant_id(1)]", slang);
        Assert.Contains("const bool USC_FOO", slang);
        Assert.Contains("const bool USC_BAR", slang);
    }

    /// <summary>Preprocessor maps <c>#ifdef</c> on axis keywords to <c>USC_*</c>.</summary>
    [Fact]
    public void EmitPassSlang_RewritesIfdefToSlangAxisName()
    {
        var pass = new ShaderPassDocument
        {
            PassName = null,
            PassIndex = 0,
            ProgramSource = "#pragma vertex vert\n#pragma fragment frag\n#ifdef FOO\nfloat z = 1;\n#endif\n",
            Pragmas = Array.Empty<string>(),
            VertexEntry = "vert",
            FragmentEntry = "frag",
            RenderStateSummary = "",
            FixedFunctionState = RenderStateExtractor.Extract(Array.Empty<ShaderLabCommandNode>(), new Dictionary<string, string>()),
        };
        var axes = new[] { new SpecializationAxis(0, "FOO", "USC_FOO", "foo") };
        string slang = SlangEmitter.EmitPassSlang(pass, Array.Empty<string>(), axes);
        Assert.Contains("#ifdef USC_FOO", slang);
        Assert.DoesNotContain("#ifdef FOO", slang);
    }

    /// <summary>Legacy <c>sampler2D _Tex;</c> becomes <c>UNITY_DECLARE_TEX2D</c> after PostUnity include resolution.</summary>
    [Fact]
    public void RewriteLegacySampler2DDeclarations_ReplacesUnityStyleTextures()
    {
        const string body = "  sampler2D _MainTex;\n  sampler2D _DepthTex;\n";
        string rewritten = SlangEmitter.RewriteLegacySampler2DDeclarations(body);
        Assert.Contains("UNITY_DECLARE_TEX2D(_MainTex)", rewritten, StringComparison.Ordinal);
        Assert.Contains("UNITY_DECLARE_TEX2D(_DepthTex)", rewritten, StringComparison.Ordinal);
        Assert.DoesNotContain("sampler2D _MainTex", rewritten, StringComparison.Ordinal);
    }

    /// <summary>Non-underscore <c>sampler2D LUT;</c> style declarations are rewritten for split samplers.</summary>
    [Fact]
    public void RewriteLegacySampler2DDeclarations_ReplacesCamelCaseTextureNames()
    {
        const string body = "sampler2D LUT;\n";
        string rewritten = SlangEmitter.RewriteLegacySampler2DDeclarations(body);
        Assert.Contains("UNITY_DECLARE_TEX2D(LUT)", rewritten, StringComparison.Ordinal);
    }

    /// <summary><c>sampler3D</c> volume textures need split declarations for <c>tex3D</c> under Slang.</summary>
    [Fact]
    public void RewriteLegacySampler3DDeclarations_ReplacesVolumeTextures()
    {
        const string body = "\t\tsampler3D _LUT;\n";
        string rewritten = SlangEmitter.RewriteLegacySampler3DDeclarations(body);
        Assert.Contains("UNITY_DECLARE_TEX3D(_LUT)", rewritten, StringComparison.Ordinal);
    }

    /// <summary><c>float4</c> appdata texcoord to <c>float2</c> UV needs a <c>.xy</c> swizzle for Slang.</summary>
    [Fact]
    public void RewriteFloat4TexcoordUvAssignments_AppendsXySwizzle()
    {
        const string body = "\t\t\to.uv = v.texcoord;\n";
        string rewritten = SlangEmitter.RewriteFloat4TexcoordUvAssignments(body);
        Assert.Contains("o.uv = v.texcoord.xy;", rewritten, StringComparison.Ordinal);
    }

    /// <summary>Already-swizzled RHS must not become <c>.xy.xy</c>.</summary>
    [Fact]
    public void RewriteFloat4TexcoordUvAssignments_SkipsWhenAlreadyXy()
    {
        const string body = "o.uv = v.texcoord.xy;\n";
        string rewritten = SlangEmitter.RewriteFloat4TexcoordUvAssignments(body);
        Assert.Equal(body.TrimEnd(), rewritten.TrimEnd(), StringComparer.Ordinal);
    }
}
