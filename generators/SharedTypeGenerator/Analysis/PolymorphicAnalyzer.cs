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

        string? cecilName = type.FullName?.Replace("+", "/", StringComparison.Ordinal);
        TypeDefinition? typeDef = cecilName is not null ? _assemblyDef.MainModule.GetType(cecilName) : null;
        if (typeDef == null) return variants;

        MethodDefinition? cctor = typeDef.GetStaticConstructor();
        if (cctor == null) return variants;

        foreach (Instruction instruction in cctor.Body.Instructions)
        {
            if (instruction.OpCode.Code != Code.Ldtoken) continue;

            TypeDefinition? tokenDef = instruction.Operand switch
            {
                TypeDefinition td => td,
                TypeReference tr => tr.Resolve(),
                _ => null,
            };
            if (tokenDef == null) continue;

            string reflectionName = tokenDef.FullName.Replace("/", "+", StringComparison.Ordinal);
            Type? runtimeType = _assembly.GetType(reflectionName);
            if (runtimeType == null) continue;

            variants.Add(new PolymorphicVariant
            {
                CSharpName = tokenDef.Name,
                RustName = tokenDef.Name.HumanizeType(),
                RuntimeType = runtimeType,
            });
        }

        return variants;
    }

    /// <summary>Collects all runtime Type objects referenced by the polymorphic registry,
    /// so the caller can queue them for generation.</summary>
    public static List<Type> GetReferencedTypes(IReadOnlyList<PolymorphicVariant> variants) =>
        variants.Select(v => v.RuntimeType).ToList();
}
