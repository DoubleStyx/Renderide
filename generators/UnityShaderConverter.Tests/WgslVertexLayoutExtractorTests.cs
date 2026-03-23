using UnityShaderConverter.Analysis;

namespace UnityShaderConverter.Tests;

/// <summary>Tests for <see cref="WgslVertexLayoutExtractor"/>.</summary>
public sealed class WgslVertexLayoutExtractorTests
{
    /// <summary>Ensures merged minimal-style WGSL yields one vec4 at location 0.</summary>
    [Fact]
    public void TryExtract_MinimalVertexInput_SingleVec4()
    {
        const string wgsl = """
            struct v2f_0
            {
                @builtin(position) vertex_0 : vec4<f32>,
            };

            struct vertexInput_0
            {
                @location(0) vertex_1 : vec4<f32>,
            };

            @vertex
            fn vert( _S1 : vertexInput_0) -> v2f_0
            {
                var o_0 : v2f_0;
                o_0.vertex_0 = _S1.vertex_1;
                return o_0;
            }
            """;

        bool ok = WgslVertexLayoutExtractor.TryExtract(wgsl, "vert", out PassVertexLayout layout, out string? err);
        Assert.True(ok, err);
        Assert.Single(layout.Attributes);
        Assert.Equal(0, layout.Attributes[0].ShaderLocation);
        Assert.Equal("wgpu::VertexFormat::Float32x4", layout.Attributes[0].RustVertexFormatPath);
        Assert.Equal(16u, layout.ArrayStride);
        Assert.Equal(0u, layout.ByteOffsets[0]);
    }

    /// <summary>Ensures multiple locations produce interleaved offsets and stride.</summary>
    [Fact]
    public void TryExtract_MultiAttribute_InterleavedOffsets()
    {
        const string wgsl = """
            struct Vin {
                @location(1) n: vec3<f32>,
                @location(0) p: vec4<f32>,
            }

            @vertex
            fn main(x: Vin) -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
            """;

        bool ok = WgslVertexLayoutExtractor.TryExtract(wgsl, "main", out PassVertexLayout layout, out string? err);
        Assert.True(ok, err);
        Assert.Equal(2, layout.Attributes.Count);
        Assert.Equal(0, layout.Attributes[0].ShaderLocation);
        Assert.Equal(1, layout.Attributes[1].ShaderLocation);
        Assert.Equal(0u, layout.ByteOffsets[0]);
        Assert.Equal(16u, layout.ByteOffsets[1]);
        Assert.True(layout.ArrayStride >= 28);
    }

    /// <summary>Skips @builtin-only fields and still succeeds.</summary>
    [Fact]
    public void TryExtract_SkipsBuiltinFields()
    {
        const string wgsl = """
            struct In {
                @builtin(vertex_index) vi: u32,
                @location(0) pos: vec2<f32>,
            }
            @vertex
            fn v(i: In) -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0);
            }
            """;

        bool ok = WgslVertexLayoutExtractor.TryExtract(wgsl, "v", out PassVertexLayout layout, out string? err);
        Assert.True(ok, err);
        Assert.Single(layout.Attributes);
        Assert.Equal(0, layout.Attributes[0].ShaderLocation);
        Assert.Equal("wgpu::VertexFormat::Float32x2", layout.Attributes[0].RustVertexFormatPath);
    }

    /// <summary>Unknown entry name fails cleanly.</summary>
    [Fact]
    public void TryExtract_WrongEntry_Fails()
    {
        const string wgsl = "@vertex fn vert() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";
        bool ok = WgslVertexLayoutExtractor.TryExtract(wgsl, "missing", out _, out string? err);
        Assert.False(ok);
        Assert.NotNull(err);
    }
}
