using System.Reflection;
using System.Runtime.InteropServices;
using SharedTypeGenerator.Analysis;
using Xunit;

namespace SharedTypeGenerator.Tests.Unit;

/// <summary>Unit tests for <see cref="ManagedLayoutSizing"/>.</summary>
public sealed class ManagedLayoutSizingTests
{
    /// <summary>Test enum with a one-byte underlying type.</summary>
    public enum TinyEnum : byte
    {
        /// <summary>Sole variant.</summary>
        A,
    }

    /// <summary>Sample explicit-layout struct exercising bool, enum, and primitive fields.</summary>
    [StructLayout(LayoutKind.Explicit, Size = 16)]
    internal struct Sample
    {
        /// <summary>Bool field at offset 0; managed size is 1 byte.</summary>
        [FieldOffset(0)] internal bool Flag;

        /// <summary>Byte-backed enum field at offset 1; managed size is 1 byte.</summary>
        [FieldOffset(1)] internal TinyEnum Tag;

        /// <summary>Int32 field at offset 4; managed size is 4 bytes.</summary>
        [FieldOffset(4)] internal int Value;
    }

    /// <summary>Sequential-layout struct without explicit field offsets.</summary>
    internal struct Sequential
    {
        /// <summary>Bool field; managed size is 1 byte.</summary>
        internal bool Flag = false;

        /// <summary>Int32 field; managed size is 4 bytes.</summary>
        internal int Value = 0;

        /// <summary>Default constructor wires the field initializers.</summary>
        public Sequential() { }
    }

    private static FieldInfo Field<T>(string name) =>
        typeof(T).GetField(name, BindingFlags.NonPublic | BindingFlags.Instance)
            ?? throw new InvalidOperationException($"Field {name} not found on {typeof(T).Name}");

    /// <summary><c>bool</c> sizes to one byte under managed layout (not the four-byte P/Invoke marshal size).</summary>
    [Fact]
    public void TryGetManagedFieldSize_bool_is_one_byte()
    {
        Assert.True(ManagedLayoutSizing.TryGetManagedFieldSize(Field<Sample>(nameof(Sample.Flag)), out int size));
        Assert.Equal(1, size);
    }

    /// <summary>Enums collapse to their underlying type's size.</summary>
    [Fact]
    public void TryGetManagedFieldSize_enum_uses_underlying_type()
    {
        Assert.True(ManagedLayoutSizing.TryGetManagedFieldSize(Field<Sample>(nameof(Sample.Tag)), out int size));
        Assert.Equal(1, size);
    }

    /// <summary>Plain primitives fall back to <see cref="Marshal.SizeOf(System.Type)"/>.</summary>
    [Fact]
    public void TryGetManagedFieldSize_int_is_four_bytes()
    {
        Assert.True(ManagedLayoutSizing.TryGetManagedFieldSize(Field<Sample>(nameof(Sample.Value)), out int size));
        Assert.Equal(4, size);
    }

    /// <summary>Summing managed sizes counts <c>bool</c> as 1 and respects enum underlying types.</summary>
    [Fact]
    public void SumManagedFieldSizes_combines_bool_enum_and_primitive()
    {
        FieldInfo[] fields = typeof(Sample).GetFields(BindingFlags.NonPublic | BindingFlags.Instance);
        Assert.Equal(1 + 1 + 4, ManagedLayoutSizing.SumManagedFieldSizes(fields));
    }

    /// <summary>Max field-end byte uses <c>FieldOffset + managed size</c>.</summary>
    [Fact]
    public void MaxFieldEndBytes_uses_explicit_offsets()
    {
        FieldInfo[] fields = typeof(Sample).GetFields(BindingFlags.NonPublic | BindingFlags.Instance);
        Assert.Equal(8, ManagedLayoutSizing.MaxFieldEndBytes(fields));
    }

    /// <summary>Sequential-layout structs without <see cref="FieldOffsetAttribute"/> report 0 max field end.</summary>
    [Fact]
    public void MaxFieldEndBytes_returns_zero_when_no_explicit_offsets()
    {
        FieldInfo[] fields = typeof(Sequential).GetFields(BindingFlags.NonPublic | BindingFlags.Instance);
        Assert.Equal(0, ManagedLayoutSizing.MaxFieldEndBytes(fields));
    }
}
