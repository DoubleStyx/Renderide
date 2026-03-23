using System.Text.Json.Serialization;

namespace UnityShaderConverter.Config;

/// <summary>Optional JSON overrides for per-shader variant define lists.</summary>
public sealed class VariantConfigModel
{
    /// <summary>Maps Unity shader names (from <c>Shader "Name"</c>) to explicit define lists per variant.</summary>
    [JsonPropertyName("variantsByShaderName")]
    public Dictionary<string, List<VariantDefines>> VariantsByShaderName { get; set; } = new();
}

/// <summary>One build-time variant: preprocessor macro names to pass to <c>slangc -D</c>.</summary>
public sealed class VariantDefines
{
    /// <summary>Macro names (without leading underscore unless part of the Unity keyword).</summary>
    [JsonPropertyName("defines")]
    public List<string> Defines { get; set; } = new();
}
