using System.Reflection;
using Mono.Cecil;
using NotEnoughLogs;
using SharedTypeGenerator.Logging;

namespace SharedTypeGenerator.Analysis;

/// <summary>Mono.Cecil helpers for explicit-layout struct metadata when reflection attributes are missing or incomplete.</summary>
internal static class CecilLayoutInspector
{
    /// <summary>Returns whether the type is marked explicit layout in IL metadata.</summary>
    /// <param name="assemblyDef">Assembly containing <paramref name="type"/>.</param>
    /// <param name="type">CLR value type to inspect.</param>
    public static bool HasExplicitLayout(AssemblyDefinition assemblyDef, Type type)
    {
        if (!type.IsValueType || type.IsEnum)
            return false;
        string? fullName = type.FullName;
        if (string.IsNullOrEmpty(fullName))
            return false;
        TypeDefinition? typeDef = assemblyDef.MainModule.GetType(fullName);
        return typeDef != null && (typeDef.Attributes & Mono.Cecil.TypeAttributes.ExplicitLayout) != 0;
    }

    /// <summary>
    /// Reads declared struct size from the ClassLayout table or <c>StructLayoutAttribute</c> custom attributes.
    /// </summary>
    /// <param name="assemblyDef">Assembly containing <paramref name="type"/>.</param>
    /// <param name="type">CLR value type.</param>
    /// <param name="logger">Optional logger for non-fatal Cecil read failures.</param>
    public static int GetExplicitLayoutSizeOrZero(AssemblyDefinition assemblyDef, Type type, Logger? logger = null)
    {
        try
        {
            string? fullName = type.FullName;
            if (string.IsNullOrEmpty(fullName))
                return 0;
            TypeDefinition? typeDef = assemblyDef.MainModule.GetType(fullName);
            if (typeDef == null)
                return 0;
            if (typeDef.ClassSize > 0)
                return typeDef.ClassSize;
            CustomAttribute? attr = typeDef.CustomAttributes
                .FirstOrDefault(a => a.AttributeType.Name == "StructLayoutAttribute");
            if (attr == null)
                return 0;
            foreach (Mono.Cecil.CustomAttributeNamedArgument prop in attr.Properties)
            {
                if (prop.Name == "Size" && prop.Argument.Value is int size && size > 0)
                    return size;
            }

            if (attr.ConstructorArguments.Count >= 2 && attr.ConstructorArguments[1].Value is int sizeArg && sizeArg > 0)
                return sizeArg;
        }
        catch (Exception ex) when (ex is ArgumentException or InvalidOperationException or BadImageFormatException)
        {
            logger?.LogTrace(LogCategory.Analysis, $"Cecil explicit layout size read failed for {type.FullName}: {ex.Message}");
        }

        return 0;
    }
}
