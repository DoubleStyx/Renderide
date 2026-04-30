using SharedTypeGenerator.Analysis;
using Xunit;

namespace SharedTypeGenerator.Tests.Unit;

/// <summary>Unit tests for <see cref="RustNaming"/> name mapping helpers (no assemblies or IO).</summary>
public sealed class RustNamingTests
{
    /// <summary>PascalCase field names become Rust snake_case with <c>2D</c>/<c>3D</c> handling and ID-suffix collapsing.</summary>
    [Theory]
    [InlineData("MyField", "my_field")]
    [InlineData("UVScale", "uv_scale")]
    [InlineData("Texture2D", "texture_2d")]
    [InlineData("Texture3D", "texture_3d")]
    [InlineData("IDs", "ids")]
    [InlineData("myIDs", "my_ids")]
    [InlineData("OBJECTIDs", "objectids")]
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

    /// <summary>Sequences of underscores collapse so the result has no doubled separators.</summary>
    [Fact]
    public void HumanizeField_collapses_double_underscores()
    {
        Assert.Equal("a_b", "__a__b__".HumanizeField());
    }

    /// <summary>Enum / union variant names become PascalCase in Rust, with keyword escaping.</summary>
    [Theory]
    [InlineData("SomeVariant", "SomeVariant")]
    [InlineData("Texture2DMode", "Texture2DMode")]
    public void HumanizeVariant_preserves_pascal_shape(string csharp, string expectedRust)
    {
        Assert.Equal(expectedRust, csharp.HumanizeVariant());
    }

    /// <summary><see cref="RustNaming.HumanizeVariant"/> handles snake input and digit-leading variants.</summary>
    [Fact]
    public void HumanizeVariant_pascalizes_with_keyword_escape()
    {
        Assert.Equal("Withunderscore", "with_underscore".HumanizeVariant());
        Assert.NotEmpty("1st".HumanizeVariant());
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

    /// <summary><see cref="RustNaming.ToScreamingSnakeTypeName"/> produces const-style names from PascalCase or snake input.</summary>
    [Theory]
    [InlineData("LightData", "LIGHT_DATA")]
    [InlineData("RendererCommand", "RENDERER_COMMAND")]
    [InlineData("my_type_name", "MY_TYPE_NAME")]
    public void ToScreamingSnakeTypeName_maps_to_screaming_snake(string input, string screaming)
    {
        Assert.Equal(screaming, input.ToScreamingSnakeTypeName());
    }
}
