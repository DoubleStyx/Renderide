using System.Linq;
using SharedTypeGenerator.IR;

namespace SharedTypeGenerator.Analysis;

/// <summary>
/// After analysis, IL-derived <see cref="WriteField"/> steps may still be <see cref="FieldKind.Pod"/> or
/// <see cref="FieldKind.ValueList"/> for C# <c>Write&lt;T&gt;</c> / <c>WriteValueList&lt;T&gt;</c> calls on
/// nested shared structs. When SIMD glam or restricted-variant enums make the Rust element type
/// non-<c>Pod</c>, upgrade those steps so emitted code uses the MemoryPackable path
/// (<c>write_object_required</c> / <c>write_object_list</c>) instead of the bytemuck-<c>Pod</c> path.
/// </summary>
internal static class PackStepNonPodPostProcessor
{
    /// <summary>Rewrites <see cref="TypeDescriptor.PackSteps"/> and <see cref="TypeDescriptor.UnpackOnlySteps"/> in place.</summary>
    public static void Apply(List<TypeDescriptor> types)
    {
        HashSet<string> nonPodRust = types
            .Where(t => !t.IsPod && t.Shape == TypeShape.PodStruct)
            .Select(t => t.RustName)
            .ToHashSet(StringComparer.Ordinal);

        foreach (TypeDescriptor td in types)
        {
            if (td.PackSteps.Count > 0)
                td.PackSteps = RewriteSteps(td.PackSteps, td.Fields, nonPodRust);
            if (td.UnpackOnlySteps.Count > 0)
                td.UnpackOnlySteps = RewriteSteps(td.UnpackOnlySteps, td.Fields, nonPodRust);
        }
    }

    private static List<SerializationStep> RewriteSteps(
        List<SerializationStep> steps,
        List<FieldDescriptor> fields,
        HashSet<string> nonPodRust)
    {
        Dictionary<string, FieldDescriptor> byRustName = fields.ToDictionary(f => f.RustName, f => f);

        List<SerializationStep> result = [];
        foreach (SerializationStep step in steps)
        {
            result.Add(step switch
            {
                WriteField wf => RewriteWriteField(wf, byRustName, nonPodRust),
                ConditionalBlock cb => new ConditionalBlock(
                    cb.ConditionField,
                    RewriteSteps(cb.Steps, fields, nonPodRust)),
                _ => step,
            });
        }

        return result;
    }

    private static WriteField RewriteWriteField(
        WriteField wf,
        Dictionary<string, FieldDescriptor> byRustName,
        HashSet<string> nonPodRust)
    {
        if (!byRustName.TryGetValue(wf.FieldName, out FieldDescriptor? fd))
            return wf;

        if (wf.Kind == FieldKind.Pod)
        {
            string normalized = RustTypeMapper.NormalizeRustTypeName(fd.RustType);
            if (nonPodRust.Contains(normalized))
                return wf with { Kind = FieldKind.ObjectRequired };
        }
        else if (wf.Kind == FieldKind.ValueList)
        {
            string element = RustTypeMapper.StripVecElementType(fd.RustType);
            if (nonPodRust.Contains(element))
                return wf with { Kind = FieldKind.ObjectList };
        }

        return wf;
    }
}
