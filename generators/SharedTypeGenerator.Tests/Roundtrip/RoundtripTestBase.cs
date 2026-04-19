using System.Reflection;
using SharedTypeGenerator.IR;
using SharedTypeGenerator.Tests.Roundtrip.Support;

namespace SharedTypeGenerator.Tests;

/// <summary>Base for roundtrip tests. Loads Renderite.Shared, provides the analyzed type list and C# pack helper.</summary>
public abstract class RoundtripTestBase
{
    /// <summary>Lazily loaded shared assembly plus analyzer output (thread-safe).</summary>
    protected static (Assembly Assembly, List<TypeDescriptor> Types) Loaded => RenderiteSharedAssembly.Loaded.Value;

    /// <summary>Returns true when the descriptor is exercised by the Rust roundtrip binary.</summary>
    protected static bool CanRoundtrip(TypeDescriptor d)
    {
        return d.Shape is TypeShape.PackableStruct or TypeShape.PolymorphicBase
            && d.PackSteps.Count > 0;
    }

    /// <summary>Resolves a concrete CLR type for a non-polymorphic descriptor name.</summary>
    protected static Type? GetConcreteType(Assembly asm, TypeDescriptor d)
    {
        if (d.Shape == TypeShape.PolymorphicBase)
            return null;
        var name = d.CSharpName;
        if (name.Contains('`'))
            name = name[..name.IndexOf('`')];
        return asm.GetTypes().FirstOrDefault(t => t.Name == name);
    }

    /// <summary>Creates an instance via <see cref="Activator.CreateInstance(Type)"/>.</summary>
    protected static object CreateInstance(Assembly asm, Type type)
    {
        return Activator.CreateInstance(type) ?? throw new InvalidOperationException($"Could not create {type.Name}");
    }

    /// <summary>Packs <paramref name="obj"/> using the same buffer size as generated Rust dispatch.</summary>
    protected static (byte[] Buffer, int Length) PackToBuffer(object obj) => PackingHelpers.PackToBuffer(obj);
}
