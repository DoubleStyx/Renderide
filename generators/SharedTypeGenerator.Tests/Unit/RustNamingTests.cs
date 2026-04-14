using SharedTypeGenerator.Analysis;
using Xunit;

namespace SharedTypeGenerator.Tests.Unit;

/// <summary>Unit tests for <see cref="RustNaming"/> name mapping helpers (no assemblies or IO).</summary>
public sealed class RustNamingTests
{
    /// <summary>PascalCase field names become Rust snake_case with <c>2D</c>/<c>3D</c> handling.</summary>
    [Theory]
    [InlineData("MyField", "my_field")]
    [InlineData("UVScale", "uv_scale")]
    [InlineData("Texture2D", "texture_2d")]
    public void HumanizeField_maps_csharp_fields_to_snake_case(string csharp, string expectedRust)
    {
        Assert.Equal(expectedRust, csharp.HumanizeField());
    }

    /// <summary>Rust keywords used as identifiers must be prefixed with <c>r#</c>.</summary>
    [Fact]
    public void HumanizeField_escapes_rust_keywords()
    {
        Assert.Equal("r#type", "Type".HumanizeField());
        Assert.Equal("r#self", "Self".HumanizeField());
    }

    /// <summary>Enum / union variant names become PascalCase in Rust, with keyword escaping.</summary>
    [Theory]
    [InlineData("SomeVariant", "SomeVariant")]
    [InlineData("Texture2DMode", "Texture2DMode")]
    public void HumanizeVariant_preserves_pascal_shape(string csharp, string expectedRust)
    {
        Assert.Equal(expectedRust, csharp.HumanizeVariant());
    }

    /// <summary>Primitive names and numeric suffix forms stay unchanged where the helper is designed to preserve them.</summary>
    [Theory]
    [InlineData("i32", "i32")]
    [InlineData("f64", "f64")]
    [InlineData("bool", "bool")]
    public void HumanizeType_preserves_primitives(string name, string expected)
    {
        Assert.Equal(expected, name.HumanizeType());
    }

    /// <summary>Generics and namespace-qualified names are left to downstream handling; angle brackets block full rewriting.</summary>
    [Fact]
    public void HumanizeType_preserves_generic_syntax()
    {
        Assert.Equal("Foo<T>", "Foo<T>".HumanizeType());
    }

    /// <summary><see cref="RustNaming.ToScreamingSnakeTypeName"/> produces const-style names from Rust PascalCase.</summary>
    [Theory]
    [InlineData("LightData", "LIGHT_DATA")]
    [InlineData("RendererCommand", "RENDERER_COMMAND")]
    public void ToScreamingSnakeTypeName_maps_pascal_to_screaming_snake(string pascal, string screaming)
    {
        Assert.Equal(screaming, pascal.ToScreamingSnakeTypeName());
    }
}
