using UnityShaderConverter.Analysis;
using UnityShaderConverter.Emission;

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
        };
        string slang = SlangEmitter.EmitPassSlang(pass, Array.Empty<string>());
        Assert.Contains("#include \"UnityCompat.slang\"", slang);
        Assert.DoesNotContain("#pragma vertex", slang);
        Assert.DoesNotContain("#pragma fragment", slang);
    }
}
