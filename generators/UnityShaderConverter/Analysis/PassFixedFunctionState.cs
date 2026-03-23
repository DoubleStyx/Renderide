using UnityShaderParser.ShaderLab;

namespace UnityShaderConverter.Analysis;

/// <summary>Concrete ShaderLab fixed-function state for one pass when values are not property-driven.</summary>
public sealed class PassFixedFunctionState
{
    /// <summary>Default state when no pass commands were parsed.</summary>
    public static PassFixedFunctionState Empty { get; } = new();

    /// <summary>Cull mode when set; <c>null</c> means Unity default (Back) unless <see cref="CullReferencesProperty"/>.</summary>
    public CullMode? CullMode { get; init; }

    /// <summary>True when <c>Cull</c> references a material property.</summary>
    public bool CullReferencesProperty { get; init; }

    /// <summary>Depth write when set; <c>null</c> means default On.</summary>
    public bool? DepthWrite { get; init; }

    /// <summary>True when <c>ZWrite</c> uses a property reference.</summary>
    public bool DepthWriteReferencesProperty { get; init; }

    /// <summary>Depth test when set; <c>null</c> means default LEqual.</summary>
    public ComparisonMode? DepthTest { get; init; }

    /// <summary>True when <c>ZTest</c> uses a property reference.</summary>
    public bool DepthTestReferencesProperty { get; init; }

    /// <summary>RT0 blend when parsed; <c>null</c> if never set or not representable.</summary>
    public PassBlendStateRt0? BlendRt0 { get; init; }

    /// <summary>Stencil block when fully concrete; <c>null</c> if absent or any field references a property.</summary>
    public PassStencilConcrete? Stencil { get; init; }

    /// <summary>True if a stencil command referenced a property.</summary>
    public bool StencilReferencesProperty { get; init; }

    /// <summary><c>ColorMask</c> string when concrete (e.g. RGBA, 0).</summary>
    public string? ColorMask { get; init; }

    /// <summary>True when color mask uses a property.</summary>
    public bool ColorMaskReferencesProperty { get; init; }

    /// <summary><c>Offset</c> factor and units when concrete.</summary>
    public (float Factor, float Units)? DepthBias { get; init; }

    /// <summary>True if offset references a property.</summary>
    public bool DepthBiasReferencesProperty { get; init; }

    /// <summary><c>LOD</c> level when set.</summary>
    public int? Lod { get; init; }

    /// <summary>Merged <c>Queue</c>, <c>RenderType</c>, etc.: pass <c>Tags</c> override subshader tags.</summary>
    public IReadOnlyDictionary<string, string> EffectiveTags { get; init; } =
        new Dictionary<string, string>(StringComparer.Ordinal);
}

/// <summary>Blend state for render target 0.</summary>
public sealed class PassBlendStateRt0
{
    /// <summary><c>Blend Off</c> or disabled.</summary>
    public bool BlendDisabled { get; init; }

    /// <summary>True if any factor is a property reference.</summary>
    public bool HasPropertyReference { get; init; }

    public BlendFactor? SourceRgb { get; init; }
    public BlendFactor? DestRgb { get; init; }
    public BlendFactor? SourceAlpha { get; init; }
    public BlendFactor? DestAlpha { get; init; }
}

/// <summary>Stencil state when all fields are literal.</summary>
public sealed class PassStencilConcrete
{
    public byte Ref { get; init; }
    public byte ReadMask { get; init; }
    public byte WriteMask { get; init; }
    public ComparisonMode CompFront { get; init; }
    public StencilOp PassFront { get; init; }
    public StencilOp FailFront { get; init; }
    public StencilOp ZFailFront { get; init; }
    public ComparisonMode CompBack { get; init; }
    public StencilOp PassBack { get; init; }
    public StencilOp FailBack { get; init; }
    public StencilOp ZFailBack { get; init; }
}
