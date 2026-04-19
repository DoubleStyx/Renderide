using System.Collections;
using System.Reflection;
using Renderite.Shared;

namespace SharedTypeGenerator.Tests;

public static partial class RandomInstancePopulator
{
    /// <summary>Shared type-shape probes for list detection and packable classes.</summary>
    private static class TypeProbe
    {
        /// <summary>Returns true when <paramref name="type"/> is <see cref="List{T}"/> and sets <paramref name="elementType"/>.</summary>
        public static bool IsListOfT(Type type, out Type? elementType)
        {
            elementType = null;
            if (!type.IsGenericType)
                return false;
            var def = type.GetGenericTypeDefinition();
            if (def != typeof(List<>))
                return false;
            elementType = type.GetGenericArguments()[0];
            return true;
        }

        /// <summary>Returns true for concrete classes implementing <see cref="IMemoryPackable"/>.</summary>
        public static bool IsMemoryPackableClass(Type type) =>
            type is { IsClass: true, IsAbstract: false } && typeof(IMemoryPackable).IsAssignableFrom(type);
    }

    /// <summary>Allocates a value type and recursively populates it.</summary>
    private static object PopulateValueType(Type valueType, Random rng, Assembly assembly, HashSet<object> seen, bool useMinimalValues = false)
    {
        var instance = Activator.CreateInstance(valueType)!;
        PopulateInternal(instance, valueType, rng, assembly, seen, useMinimalValues);
        return instance;
    }

    /// <summary>Allocates a packable class and recursively populates it.</summary>
    private static object CreateAndPopulate(Type type, Random rng, Assembly assembly, HashSet<object> seen, bool useMinimalValues = false)
    {
        var instance = Activator.CreateInstance(type) ?? throw new InvalidOperationException($"Cannot create {type.Name}");
        PopulateInternal(instance, type, rng, assembly, seen, useMinimalValues);
        return instance;
    }

    /// <summary>Creates a list with 0–4 elements using the same element distribution as the pre-refactor loop.</summary>
    private static IList CreateRandomList(Type elementType, Random rng, Assembly assembly, HashSet<object> seen)
    {
        var listType = typeof(List<>).MakeGenericType(elementType);
        var list = (IList)Activator.CreateInstance(listType)!;
        var count = rng.Next(0, 4);
        for (int i = 0; i < count; i++)
        {
            object? elem = TryGetListElementValue(elementType, rng, assembly, seen);
            if (elem is null)
                continue;
            list.Add(elem);
        }

        return list;
    }

    /// <summary>Produces one list element; returns <see langword="null"/> when the type is not handled (same as <c>continue</c> in the original loop).</summary>
    private static object? TryGetListElementValue(Type elementType, Random rng, Assembly assembly, HashSet<object> seen)
    {
        if (elementType.IsValueType && !elementType.IsPrimitive && elementType != typeof(Guid))
            return PopulateValueType(elementType, rng, assembly, seen);
        if (TypeProbe.IsMemoryPackableClass(elementType))
            return CreateAndPopulate(elementType, rng, assembly, seen);
        if (elementType == typeof(int) || elementType == typeof(long))
            return rng.Next();
        if (elementType == typeof(float) || elementType == typeof(double))
            return rng.NextDouble();
        if (elementType == typeof(bool))
            return rng.Next(2) == 1;
        if (elementType == typeof(Guid))
            return Guid.NewGuid();
        if (elementType.IsEnum)
            return GetRandomEnumValue(elementType, rng);
        return null;
    }
}
