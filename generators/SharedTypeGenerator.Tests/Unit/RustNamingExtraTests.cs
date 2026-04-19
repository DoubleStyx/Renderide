using SharedTypeGenerator.Analysis;
using Xunit;

namespace SharedTypeGenerator.Tests.Unit;

/// <summary>Extra coverage for <see cref="RustNaming"/> beyond basic humanization.</summary>
public sealed class RustNamingExtraTests
{
    /// <summary>IDs-style suffixes collapse to readable snake_case.</summary>
    [Theory]
    [InlineData("IDs", "ids")]
    [InlineData("myIDs", "my_ids")]
    [InlineData("Texture3D", "texture_3d")]
    [InlineData("OBJECTIDs", "objectids")]
    public void HumanizeField_special_token_rewrites(string input, string expected)
    {
        string actual = input.HumanizeField();
        Assert.Equal(expected, actual);
    }

    /// <summary>Multiple underscores collapse to a single underscore.</summary>
    [Fact]
    public void HumanizeField_collapses_double_underscores()
    {
        Assert.Equal("a_b", "__a__b__".HumanizeField());
    }

    /// <summary><see cref="RustNaming.HumanizeVariant"/> handles snake input and digits.</summary>
    [Fact]
    public void HumanizeVariant_pascalizes_with_keyword_escape()
    {
        string withUnderscore = "with_underscore".HumanizeVariant();
        Assert.Equal("Withunderscore", withUnderscore);

        string leadingDigit = "1st".HumanizeVariant();
        Assert.NotEmpty(leadingDigit);
    }

    /// <summary><see cref="RustNaming.ToScreamingSnakeTypeName"/> works on snake and mixed input.</summary>
    [Fact]
    public void ToScreamingSnakeTypeName_accepts_snake_and_mixed()
    {
        Assert.Equal("LIGHT_DATA", "LightData".ToScreamingSnakeTypeName());
        Assert.Equal("MY_TYPE_NAME", "my_type_name".ToScreamingSnakeTypeName());
    }
}
