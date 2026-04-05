using System.Reflection;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Mono.Cecil.Rocks;
using SharedTypeGenerator.IR;

namespace SharedTypeGenerator.Analysis;

/// <summary>Extracts the polymorphic type registry from a PolymorphicMemoryPackableEntity{T}
/// subclass's static constructor by scanning for Ldtoken instructions.</summary>
public class PolymorphicAnalyzer
{
    private readonly AssemblyDefinition _assemblyDef;
    private readonly Assembly _assembly;

    /// <summary>Creates an analyzer for the given Cecil + reflection assembly pair.</summary>
    public PolymorphicAnalyzer(AssemblyDefinition assemblyDef, Assembly assembly)
    {
        _assemblyDef = assemblyDef;
        _assembly = assembly;
    }

    /// <summary>Reads the static constructor of the given type to find all Ldtoken instructions
    /// that register subtypes via InitTypes. Returns the ordered variant list.</summary>
    public List<PolymorphicVariant> ExtractVariants(Type type)
    {
        var variants = new List<PolymorphicVariant>();

        TypeDefinition? typeDef = _assemblyDef.MainModule.GetType(type.FullName);
        if (typeDef == null) return variants;

        MethodDefinition? cctor = typeDef.GetStaticConstructor();
        if (cctor == null) return variants;

        foreach (Instruction instruction in cctor.Body.Instructions)
        {
            if (instruction.OpCode.Code != Code.Ldtoken) continue;

            if (instruction.Operand is not TypeDefinition tokenType) continue;

            Type? runtimeType = _assembly.GetType(tokenType.FullName);
            if (runtimeType == null) continue;

            variants.Add(new PolymorphicVariant
            {
                CSharpName = tokenType.Name,
                RustName = tokenType.Name.HumanizeType(),
                RuntimeType = runtimeType,
            });
        }

        return variants;
    }

    /// <summary>Collects all runtime Type objects referenced by the polymorphic registry,
    /// so the caller can queue them for generation.</summary>
    public List<Type> GetReferencedTypes(List<PolymorphicVariant> variants)
    {
        return variants.Select(v => v.RuntimeType).ToList();
    }
}
