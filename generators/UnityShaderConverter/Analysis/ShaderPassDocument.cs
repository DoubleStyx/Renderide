namespace UnityShaderConverter.Analysis;

/// <summary>One ShaderLab code pass worth of extracted data for codegen.</summary>
public sealed class ShaderPassDocument
{
    /// <summary>Pass display name from <c>Name "..."</c> when present.</summary>
    public string? PassName { get; init; }

    /// <summary>Zero-based index among code passes in the first subshader.</summary>
    public int PassIndex { get; init; }

    /// <summary>Raw HLSL/Cg program text (no implicit Unity preamble when parsing with preamble disabled).</summary>
    public required string ProgramSource { get; init; }

    /// <summary>Pragma lines extracted by UnityShaderParser.</summary>
    public required IReadOnlyList<string> Pragmas { get; init; }

    /// <summary>Vertex entry from <c>#pragma vertex</c>.</summary>
    public string? VertexEntry { get; init; }

    /// <summary>Fragment entry from <c>#pragma fragment</c>.</summary>
    public string? FragmentEntry { get; init; }

    /// <summary>Serialized render-state snapshot for Rust comments / future pipeline mapping.</summary>
    public required string RenderStateSummary { get; init; }

    /// <summary>Structured fixed-function state and merged tags for Rust <c>wgpu</c> emission.</summary>
    public required PassFixedFunctionState FixedFunctionState { get; init; }
}
