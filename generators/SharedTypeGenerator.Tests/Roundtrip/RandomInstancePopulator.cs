using System.Reflection;
using Renderite.Shared;

namespace SharedTypeGenerator.Tests;

/// <summary>Populates IMemoryPackable instances with random data to reduce the chance that
/// incorrect serialization mappings pass tests by producing identical default-value output.</summary>
public static partial class RandomInstancePopulator
{
    /// <summary>Populates the given instance with random data. Uses a seeded <see cref="Random"/> for determinism.</summary>
    /// <remarks>
    /// Types that contain <c>List&lt;T&gt;</c> where <c>T</c> is an enum use a minimal pattern (null strings, first enum value, empty lists)
    /// to avoid C#/Rust layout or enum discriminant mismatches during cross-language roundtrip.
    /// </remarks>
    /// <param name="instance">The object to populate (must implement <see cref="IMemoryPackable"/>).</param>
    /// <param name="type">The runtime type of the instance.</param>
    /// <param name="seed">Seed for <see cref="Random"/> to ensure deterministic, reproducible tests.</param>
    /// <param name="assembly">Assembly containing types for creating nested instances.</param>
    public static void Populate(object instance, Type type, int seed, Assembly assembly)
    {
        var rng = new Random(seed);
        var seen = new HashSet<object>(ReferenceEqualityComparer.Instance);
        var useMinimalValues = TypeHasListOfEnum(type);
        PopulateInternal(instance, type, rng, assembly, seen, useMinimalValues);
    }

    /// <summary>Returns true if the type has any <see cref="List{T}"/> field where the element type is an enum.</summary>
    private static bool TypeHasListOfEnum(Type type)
    {
        foreach (var f in type.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance))
        {
            if (TypeProbe.IsListOfT(f.FieldType, out Type? elem) && elem != null && elem.IsEnum)
                return true;
        }

        return false;
    }

    /// <summary>Recursively populates fields.</summary>
    private static void PopulateInternal(
        object instance,
        Type type,
        Random rng,
        Assembly assembly,
        HashSet<object> seen,
        bool useMinimalValues = false)
    {
        if (instance == null)
            return;
        if (!seen.Add(instance))
            return;

        foreach (var field in type.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance))
        {
            // Populate list fields even when init-only so they are never null (avoids NullRef in Pack).
            var isInitOnlyList = field.IsInitOnly && TypeProbe.IsListOfT(field.FieldType, out _);
            if (field.IsInitOnly && !isInitOnlyList)
                continue;

            var fieldType = field.FieldType;
            if (!TryComputeFieldValue(fieldType, rng, assembly, seen, useMinimalValues, out object? value))
                continue;

            _ = TrySetField(field, instance, value);
        }
    }

    /// <summary>Computes the next field value, or returns <see langword="false"/> to skip the field.</summary>
    private static bool TryComputeFieldValue(
        Type fieldType,
        Random rng,
        Assembly assembly,
        HashSet<object> seen,
        bool useMinimalValues,
        out object? value)
    {
        if (useMinimalValues && fieldType == typeof(string))
        {
            value = null;
            return true;
        }

        if (useMinimalValues && fieldType.IsEnum)
        {
            value = Enum.GetValues(fieldType).GetValue(0)!;
            return true;
        }

        if (useMinimalValues && TypeProbe.IsListOfT(fieldType, out Type? enumElem) && enumElem != null && enumElem.IsEnum)
        {
            value = Activator.CreateInstance(fieldType)!;
            return true;
        }

        if (fieldType == typeof(int) || fieldType == typeof(long) || fieldType == typeof(short) || fieldType == typeof(byte))
        {
            value = rng.Next();
            return true;
        }

        if (fieldType == typeof(uint) || fieldType == typeof(ulong) || fieldType == typeof(ushort))
        {
            value = (object)(uint)rng.Next();
            return true;
        }

        if (fieldType == typeof(float) || fieldType == typeof(double))
        {
            value = rng.NextDouble();
            return true;
        }

        if (fieldType == typeof(bool))
        {
            value = rng.Next(2) == 1;
            return true;
        }

        if (fieldType == typeof(Guid))
        {
            value = Guid.NewGuid();
            return true;
        }

        if (fieldType == typeof(string))
        {
            value = rng.Next(3) == 0 ? null : $"r{rng.Next(1000)}";
            return true;
        }

        if (fieldType.IsEnum)
        {
            value = GetRandomEnumValue(fieldType, rng);
            return true;
        }

        if (fieldType.IsValueType && !fieldType.IsPrimitive && fieldType != typeof(Guid))
        {
            value = PopulateValueType(fieldType, rng, assembly, seen, useMinimalValues);
            return true;
        }

        if (TypeProbe.IsMemoryPackableClass(fieldType))
        {
            value = CreateAndPopulate(fieldType, rng, assembly, seen, useMinimalValues);
            return true;
        }

        if (TypeProbe.IsListOfT(fieldType, out Type? elementType) && elementType != null)
        {
            value = CreateRandomList(elementType, rng, assembly, seen);
            return true;
        }

        value = null;
        return false;
    }

    /// <summary>Assigns <paramref name="value"/> to <paramref name="field"/>; returns <see langword="false"/> when reflection rejects the write.</summary>
    private static bool TrySetField(FieldInfo field, object? instance, object? value)
    {
        try
        {
            field.SetValue(instance, value);
            return true;
        }
        catch
        {
            // Match pre-refactor behavior: ignore fields we cannot set (readonly, wrong type, etc.).
            return false;
        }
    }
}
