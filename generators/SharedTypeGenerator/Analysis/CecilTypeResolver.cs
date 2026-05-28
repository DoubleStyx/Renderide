using Mono.Cecil;

namespace SharedTypeGenerator.Analysis;

/// <summary>Resolves reflection <see cref="Type"/> instances to their matching Cecil type definitions.</summary>
internal static class CecilTypeResolver
{
    /// <summary>Returns the Cecil type definition for <paramref name="type"/>, or <see langword="null"/> when it is not in the module.</summary>
    public static TypeDefinition? Resolve(AssemblyDefinition assemblyDef, Type type)
    {
        string? cecilName = GetCecilFullName(type);
        return cecilName is null ? null : assemblyDef.MainModule.GetType(cecilName);
    }

    /// <summary>Converts a reflection full name into Cecil's nested-type name format.</summary>
    public static string? GetCecilFullName(Type type)
    {
        if (type.IsGenericType && !type.IsGenericTypeDefinition)
            type = type.GetGenericTypeDefinition();

        string? fullName = type.FullName;
        return string.IsNullOrEmpty(fullName)
            ? null
            : fullName.Replace('+', '/');
    }
}
