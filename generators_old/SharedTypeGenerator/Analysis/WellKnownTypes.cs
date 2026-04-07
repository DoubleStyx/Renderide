using System.Reflection;

namespace SharedTypeGenerator.Analysis;

/// <summary>Resolves key Renderite.Shared types referenced during analysis and field classification.</summary>
public sealed class WellKnownTypes
{
    /// <summary>Interface implemented by memory-packed types.</summary>
    public Type IMemoryPackable { get; }

    /// <summary>Open generic definition of <c>PolymorphicMemoryPackableEntity&lt;T&gt;</c>.</summary>
    public Type PolymorphicMemoryPackableEntityDefinition { get; }

    /// <summary>Builds well-known type references from all types in the loaded shared assembly.</summary>
    /// <exception cref="InvalidOperationException">Thrown when required types are missing.</exception>
    public WellKnownTypes(Type[] assemblyTypes)
    {
        IMemoryPackable = Array.Find(assemblyTypes, t => t.Name == "IMemoryPackable")
            ?? throw new InvalidOperationException("Assembly does not contain type IMemoryPackable.");
        Type? poly = Array.Find(assemblyTypes, t => t.Name == "PolymorphicMemoryPackableEntity`1");
        if (poly is null)
            throw new InvalidOperationException("Assembly does not contain PolymorphicMemoryPackableEntity`1.");
        PolymorphicMemoryPackableEntityDefinition = poly.GetGenericTypeDefinition();
    }
}
