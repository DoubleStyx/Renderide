using System.Collections;
using System.Reflection;
using Renderite.Shared;

namespace SharedTypeGenerator.Tests;

/// <summary>Populates IMemoryPackable instances with random data to reduce the chance that
/// incorrect serialization mappings pass tests by producing identical default-value output.</summary>
public static class RandomInstancePopulator
{
    /// <summary>Populates the given instance with random data. Uses a seeded Random for determinism.</summary>
    /// <param name="instance">The object to populate (must implement IMemoryPackable).</param>
    /// <param name="type">The runtime type of the instance.</param>
    /// <param name="seed">Seed for Random to ensure deterministic, reproducible tests.</param>
    /// <param name="assembly">Assembly containing types for creating nested instances.</param>
    public static void Populate(object instance, Type type, int seed, Assembly assembly)
    {
        var rng = new Random(seed);
        var seen = new HashSet<object>(ReferenceEqualityComparer.Instance);
        var useMinimalValues = TypeHasListOfEnum(type);
        PopulateInternal(instance, type, rng, assembly, seen, useMinimalValues);
    }

    /// <summary>Returns true if the type has any List&lt;T&gt; field where T is an enum.
    /// Such types use minimal values (null strings, first enum, empty lists) to avoid
    /// C#/Rust layout or enum discriminant mismatches.</summary>
    private static bool TypeHasListOfEnum(Type type)
    {
        foreach (var f in type.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance))
        {
            if (IsListOfT(f.FieldType, out var elem) && elem != null && elem.IsEnum)
                return true;
        }
        return false;
    }

    /// <summary>Recursively populates fields. When useMinimalValues is true (for types with List&lt;Enum&gt;),
    /// uses null strings, first enum value, and empty enum lists to avoid C#/Rust layout mismatches.</summary>
    private static void PopulateInternal(object instance, Type type, Random rng, Assembly assembly, HashSet<object> seen, bool useMinimalValues = false)
    {
        if (instance == null) return;
        if (!seen.Add(instance)) return; // avoid cycles

        foreach (var field in type.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance))
        {
            // Populate list fields even when init-only so they are never null (avoids NullRef in Pack).
            var isInitOnlyList = field.IsInitOnly && IsListOfT(field.FieldType, out _);
            if (field.IsInitOnly && !isInitOnlyList) continue;

            var fieldType = field.FieldType;
            object? value;

            if (useMinimalValues && fieldType == typeof(string))
                value = null;
            else if (useMinimalValues && fieldType.IsEnum)
                value = Enum.GetValues(fieldType).GetValue(0)!;
            else if (useMinimalValues && IsListOfT(fieldType, out var enumElem) && enumElem != null && enumElem.IsEnum)
                value = Activator.CreateInstance(fieldType)!;
            else if (fieldType == typeof(int) || fieldType == typeof(long) || fieldType == typeof(short) || fieldType == typeof(byte))
                value = rng.Next();
            else if (fieldType == typeof(uint) || fieldType == typeof(ulong) || fieldType == typeof(ushort))
                value = (object)(uint)rng.Next();
            else if (fieldType == typeof(float) || fieldType == typeof(double))
                value = rng.NextDouble();
            else if (fieldType == typeof(bool))
                value = rng.Next(2) == 1;
            else if (fieldType == typeof(Guid))
                value = Guid.NewGuid();
            else if (fieldType == typeof(string))
                value = rng.Next(3) == 0 ? null : $"r{rng.Next(1000)}";
            else if (fieldType.IsEnum)
                value = GetRandomEnumValue(fieldType, rng);
            else if (fieldType.IsValueType && !fieldType.IsPrimitive && fieldType != typeof(Guid))
                value = PopulateValueType(fieldType, rng, assembly, seen, useMinimalValues);
            else if (IsMemoryPackableClass(fieldType))
                value = CreateAndPopulate(fieldType, rng, assembly, seen, useMinimalValues);
            else if (IsListOfT(fieldType, out var elementType) && elementType != null)
                value = CreateRandomList(elementType, rng, assembly, seen);
            else
                continue;

            try
            {
                field.SetValue(instance, value);
            }
            catch
            {
                // skip fields we cannot set (e.g. readonly, wrong type)
            }
        }
    }

    /// <summary>Picks a random enum value: non-flags types use a uniformly chosen <b>named</b> member so the
    /// underlying discriminant is always one Rust defines (integers in sparse gaps are invalid on Rust unpack);
    /// <see cref="FlagsAttribute"/> enums use a uniform raw integer from <c>0</c>..<c>mask</c>.</summary>
    private static object GetRandomEnumValue(Type enumType, Random rng)
    {
        if (enumType.IsDefined(typeof(FlagsAttribute), inherit: false))
            return GetRandomFlagsEnumUnderlying(enumType, rng);

        return GetRandomNonFlagsEnumUnderlying(enumType, rng);
    }

    /// <summary>Selects a random flags combination by drawing a uniform raw underlying integer from 0 through the OR of all defined bits.</summary>
    private static object GetRandomFlagsEnumUnderlying(Type enumType, Random rng)
    {
        var values = Enum.GetValues(enumType);
        if (values.Length == 0)
            return Activator.CreateInstance(enumType)!;

        ulong mask = 0;
        foreach (var v in values)
            mask |= Convert.ToUInt64(v);

        var underlying = Enum.GetUnderlyingType(enumType);
        ulong raw = RandomUInt64Inclusive(rng, 0, mask);
        return Enum.ToObject(enumType, Convert.ChangeType(raw, underlying));
    }

    /// <summary>Selects a random non-flags enum by uniformly picking among <see cref="Enum.GetValues"/> members.
    /// Sparse discriminants (gaps between named values) are excluded so C# and Rust roundtrip stays consistent.</summary>
    private static object GetRandomNonFlagsEnumUnderlying(Type enumType, Random rng)
    {
        var values = Enum.GetValues(enumType);
        if (values.Length == 0)
            return Activator.CreateInstance(enumType)!;

        return values.GetValue(rng.Next(values.Length))!;
    }

    /// <summary>Uniform <see cref="ulong"/> in <c>[min, max]</c> using rejection sampling (no modulo bias).</summary>
    private static ulong RandomUInt64Inclusive(Random rng, ulong min, ulong max)
    {
        if (min > max)
            throw new ArgumentOutOfRangeException(nameof(min));
        if (min == max)
            return min;

        ulong span = max - min;
        if (span == ulong.MaxValue)
            return NextUInt64(rng);

        ulong count = span + 1;
        ulong limit = ulong.MaxValue - ulong.MaxValue % count;
        ulong r;
        do
        {
            r = NextUInt64(rng);
        } while (r > limit);

        return min + r % count;
    }

    /// <summary>Reads eight bytes from <paramref name="rng"/> into a <see cref="ulong"/>.</summary>
    private static ulong NextUInt64(Random rng)
    {
        Span<byte> buf = stackalloc byte[8];
        rng.NextBytes(buf);
        return BitConverter.ToUInt64(buf);
    }

    private static object PopulateValueType(Type valueType, Random rng, Assembly assembly, HashSet<object> seen, bool useMinimalValues = false)
    {
        var instance = Activator.CreateInstance(valueType)!;
        PopulateInternal(instance, valueType, rng, assembly, seen, useMinimalValues);
        return instance;
    }

    private static bool IsMemoryPackableClass(Type type)
    {
        return type is { IsClass: true, IsAbstract: false } && typeof(IMemoryPackable).IsAssignableFrom(type);
    }

    private static object CreateAndPopulate(Type type, Random rng, Assembly assembly, HashSet<object> seen, bool useMinimalValues = false)
    {
        var instance = Activator.CreateInstance(type) ?? throw new InvalidOperationException($"Cannot create {type.Name}");
        PopulateInternal(instance, type, rng, assembly, seen, useMinimalValues);
        return instance;
    }

    private static bool IsListOfT(Type type, out Type? elementType)
    {
        elementType = null;
        if (!type.IsGenericType) return false;
        var def = type.GetGenericTypeDefinition();
        if (def != typeof(List<>)) return false;
        elementType = type.GetGenericArguments()[0];
        return true;
    }

    /// <summary>Creates a list with 0–4 random elements. Enum elements use the same underlying-value
    /// distribution as <see cref="GetRandomEnumValue"/>.</summary>
    private static IList CreateRandomList(Type elementType, Random rng, Assembly assembly, HashSet<object> seen)
    {
        var listType = typeof(List<>).MakeGenericType(elementType);
        var list = (IList)Activator.CreateInstance(listType)!;
        var count = rng.Next(0, 4);
        for (int i = 0; i < count; i++)
        {
            object? elem;
            if (elementType.IsValueType && !elementType.IsPrimitive && elementType != typeof(Guid))
                elem = PopulateValueType(elementType, rng, assembly, seen);
            else if (IsMemoryPackableClass(elementType))
                elem = CreateAndPopulate(elementType, rng, assembly, seen);
            else if (elementType == typeof(int) || elementType == typeof(long))
                elem = rng.Next();
            else if (elementType == typeof(float) || elementType == typeof(double))
                elem = rng.NextDouble();
            else if (elementType == typeof(bool))
                elem = rng.Next(2) == 1;
            else if (elementType == typeof(Guid))
                elem = Guid.NewGuid();
            else if (elementType.IsEnum)
                elem = GetRandomEnumValue(elementType, rng);
            else
                continue;
            list.Add(elem!);
        }
        return list;
    }
}
