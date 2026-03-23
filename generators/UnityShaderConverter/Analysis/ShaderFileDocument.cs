namespace UnityShaderConverter.Analysis;

/// <summary>Parsed Unity <c>.shader</c> file ready for emission stages.</summary>
public sealed class ShaderFileDocument
{
    /// <summary>Absolute path to the source <c>.shader</c> file.</summary>
    public required string SourcePath { get; init; }

    /// <summary>Unity shader name from <c>Shader "..."</c>.</summary>
    public required string ShaderName { get; init; }

    /// <summary>Properties block entries.</summary>
    public required IReadOnlyList<ShaderPropertyRecord> Properties { get; init; }

    /// <summary>Tags from the first subshader.</summary>
    public required IReadOnlyDictionary<string, string> SubShaderTags { get; init; }

    /// <summary>Code passes (skips grab/use passes).</summary>
    public required IReadOnlyList<ShaderPassDocument> Passes { get; init; }

    /// <summary>All <c>#pragma multi_compile</c> / <c>shader_feature</c> lines across passes.</summary>
    public required IReadOnlyList<string> MultiCompilePragmas { get; init; }
}
