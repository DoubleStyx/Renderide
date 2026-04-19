using System.Reflection;
using SharedTypeGenerator.Analysis;
using Xunit;

namespace SharedTypeGenerator.Tests.Unit;

/// <summary>Unit tests for <see cref="TypeAnalyzer"/> static name helpers.</summary>
public sealed class TypeAnalyzerNamingTests
{
    /// <summary>Nested CLR types flatten to a single Rust PascalCase identifier.</summary>
    [Fact]
    public void MapRustName_nested_type_flattens()
    {
        const string source = @"
namespace Naming {
  public class Outer {
    public class Inner { }
  }
}";
        (Assembly asm, _) = TestCompilation.Compile(source);
        Type? inner = asm.GetType("Naming.Outer+Inner");
        Assert.NotNull(inner);
        string rust = TypeAnalyzer.MapRustName(inner);
        Assert.Equal("OuterInner", rust);
    }

    /// <summary><see cref="TypeAnalyzer.IsPlainRustScalarLayoutType"/> matches scalar keywords only.</summary>
    [Theory]
    [InlineData("i32", true)]
    [InlineData("f32", true)]
    [InlineData("u8", true)]
    [InlineData("Vec3", false)]
    [InlineData("Guid", false)]
    public void IsPlainRustScalarLayoutType_table(string rustType, bool expected) =>
        Assert.Equal(expected, TypeAnalyzer.IsPlainRustScalarLayoutType(rustType));
}
