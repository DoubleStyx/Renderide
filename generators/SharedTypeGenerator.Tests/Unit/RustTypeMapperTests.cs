using System.Collections;
using System.Numerics;
using System.Reflection;
using SharedTypeGenerator.Analysis;
using Xunit;

namespace SharedTypeGenerator.Tests.Unit;

/// <summary>Unit tests for <see cref="RustTypeMapper"/> primitive and composite mappings.</summary>
public sealed class RustTypeMapperTests
{
    private readonly Assembly _assembly = typeof(RustTypeMapperTests).Assembly;

    /// <summary>Every supported primitive maps to the expected Rust keyword.</summary>
    [Fact]
    public void Map_primitives_match_expected_rust_types()
    {
        Assert.Equal("u8", RustTypeMapper.MapType(typeof(byte), _assembly));
        Assert.Equal("i8", RustTypeMapper.MapType(typeof(sbyte), _assembly));
        Assert.Equal("i16", RustTypeMapper.MapType(typeof(short), _assembly));
        Assert.Equal("u16", RustTypeMapper.MapType(typeof(ushort), _assembly));
        Assert.Equal("i32", RustTypeMapper.MapType(typeof(int), _assembly));
        Assert.Equal("u32", RustTypeMapper.MapType(typeof(uint), _assembly));
        Assert.Equal("i64", RustTypeMapper.MapType(typeof(long), _assembly));
        Assert.Equal("u64", RustTypeMapper.MapType(typeof(ulong), _assembly));
        Assert.Equal("i128", RustTypeMapper.MapType(typeof(Int128), _assembly));
        Assert.Equal("u128", RustTypeMapper.MapType(typeof(UInt128), _assembly));
        Assert.Equal("f32", RustTypeMapper.MapType(typeof(float), _assembly));
        Assert.Equal("f64", RustTypeMapper.MapType(typeof(double), _assembly));
        Assert.Equal("bool", RustTypeMapper.MapType(typeof(bool), _assembly));
        Assert.Equal("i128", RustTypeMapper.MapType(typeof(DateTime), _assembly));
        Assert.Equal("Option<String>", RustTypeMapper.MapType(typeof(string), _assembly));
    }

    /// <summary>Glam host types map by <see cref="Type.Name"/>.</summary>
    [Fact]
    public void Map_glam_vectors_and_matrix_by_name()
    {
        Assert.Equal("Vec2", RustTypeMapper.MapType(typeof(RenderVector2), _assembly));
        Assert.Equal("IVec2", RustTypeMapper.MapType(typeof(RenderVector2i), _assembly));
        Assert.Equal("Vec3", RustTypeMapper.MapType(typeof(RenderVector3), _assembly));
        Assert.Equal("IVec3", RustTypeMapper.MapType(typeof(RenderVector3i), _assembly));
        Assert.Equal("Vec4", RustTypeMapper.MapType(typeof(RenderVector4), _assembly));
        Assert.Equal("IVec4", RustTypeMapper.MapType(typeof(RenderVector4i), _assembly));
        Assert.Equal("Quat", RustTypeMapper.MapType(typeof(RenderQuaternion), _assembly));
        Assert.Equal("Mat4", RustTypeMapper.MapType(typeof(RenderMatrix4x4), _assembly));
    }

    /// <summary>Nested lists map to nested <c>Vec</c> types.</summary>
    [Fact]
    public void Map_list_and_nested_list()
    {
        Assert.Equal("Vec<i32>", RustTypeMapper.MapType(typeof(List<int>), _assembly));
        Assert.Equal("Vec<Vec<i32>>", RustTypeMapper.MapType(typeof(List<List<int>>), _assembly));
    }

    /// <summary>Nullable value types map to <c>Option&lt;T&gt;</c>.</summary>
    [Fact]
    public void Map_nullable_wraps_option()
    {
        Assert.Equal("Option<i32>", RustTypeMapper.MapType(typeof(int?), _assembly));
    }

    /// <summary>Reference types in the target assembly are collected as referenced generation roots.</summary>
    [Fact]
    public void MapInternal_collects_referenced_types_from_shared_assembly()
    {
        RustTypeMapper.MappingResult r = RustTypeMapper.Map(typeof(TestReferenced), _assembly);
        Assert.Equal("Option<TestReferenced>", r.RustType);
        Assert.Contains(typeof(TestReferenced), r.ReferencedTypes);
    }

    /// <summary><see cref="RustTypeMapper.BitsToRustUintType"/> picks the smallest uint that fits.</summary>
    [Theory]
    [InlineData(8, "u8")]
    [InlineData(16, "u16")]
    [InlineData(32, "u32")]
    [InlineData(64, "u64")]
    [InlineData(65, "u128")]
    public void BitsToRustUintType_boundaries(int bits, string expected)
    {
        Assert.Equal(expected, RustTypeMapper.BitsToRustUintType(bits));
    }

    /// <summary><see cref="RustTypeMapper.NormalizeRustTypeName"/> strips one <c>Option&lt;…&gt;</c> wrapper.</summary>
    [Fact]
    public void NormalizeRustTypeName_strips_single_option_only()
    {
        Assert.Equal("Foo", RustTypeMapper.NormalizeRustTypeName("Option<Foo>"));
        Assert.Equal("Option<Foo>", RustTypeMapper.NormalizeRustTypeName("Option<Option<Foo>>"));
    }

    /// <summary><see cref="RustTypeMapper.IsGlamRustType"/> matches glam primitives.</summary>
    [Fact]
    public void IsGlamRustType_truth_table()
    {
        Assert.True(RustTypeMapper.IsGlamRustType("Vec2"));
        Assert.True(RustTypeMapper.IsGlamRustType("Option<Vec3>"));
        Assert.False(RustTypeMapper.IsGlamRustType("i32"));
    }

    /// <summary><see cref="RustTypeMapper.IsGlamRustTypeRequiringCompositeNonPod"/> flags SIMD-heavy types.</summary>
    [Fact]
    public void IsGlamRustTypeRequiringCompositeNonPod_truth_table()
    {
        Assert.True(RustTypeMapper.IsGlamRustTypeRequiringCompositeNonPod("Quat"));
        Assert.True(RustTypeMapper.IsGlamRustTypeRequiringCompositeNonPod("Mat4"));
        Assert.True(RustTypeMapper.IsGlamRustTypeRequiringCompositeNonPod("Vec4"));
        Assert.False(RustTypeMapper.IsGlamRustTypeRequiringCompositeNonPod("Vec2"));
    }

    /// <summary><see cref="RustTypeMapper.StripVecElementType"/> removes one <c>Vec&lt;T&gt;</c> after option normalization.</summary>
    [Fact]
    public void StripVecElementType_removes_outer_vec()
    {
        Assert.Equal("Thing", RustTypeMapper.StripVecElementType("Vec<Thing>"));
        Assert.Equal("Thing", RustTypeMapper.StripVecElementType("Option<Vec<Thing>>"));
    }

    private sealed class TestReferenced
    {
    }

    private struct RenderVector2
    {
    }

    private struct RenderVector2i
    {
    }

    private struct RenderVector3
    {
    }

    private struct RenderVector3i
    {
    }

    private struct RenderVector4
    {
    }

    private struct RenderVector4i
    {
    }

    private struct RenderQuaternion
    {
    }

    private struct RenderMatrix4x4
    {
    }
}
