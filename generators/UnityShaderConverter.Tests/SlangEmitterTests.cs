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
        Assert.DoesNotContain("#pragma vertex", slang);
        Assert.DoesNotContain("#pragma fragment", slang);
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
}
