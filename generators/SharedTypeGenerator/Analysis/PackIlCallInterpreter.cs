using System.Reflection;
using Mono.Cecil;
using SharedTypeGenerator.IR;

namespace SharedTypeGenerator.Analysis;

/// <summary>
/// Maps a single <see cref="MethodReference"/> call inside a <c>Pack</c> method body to zero or one <see cref="SerializationStep"/>.
/// Shared by the conditional-aware IL walker so call handling is not duplicated.
/// </summary>
internal static class PackIlCallInterpreter
{
    /// <summary>Appends a <see cref="SerializationStep"/> when <paramref name="callRef"/> is a known MemoryPacker API; no-op for read-side-only calls.</summary>
    public static void AppendStepForCall(
        MethodReference callRef,
        Stack<string> fieldNameStack,
        FieldInfo[] fields,
        FieldClassifier classifier,
        List<SerializationStep> steps)
    {
        MemoryPackOperation operation = MemoryPackCallClassifier.Classify(callRef);
        switch (operation)
        {
            case MemoryPackOperation.Write:
            case MemoryPackOperation.WriteObject:
            case MemoryPackOperation.WriteObjectRequired:
            case MemoryPackOperation.WriteValueList:
            case MemoryPackOperation.WriteObjectList:
            case MemoryPackOperation.WritePolymorphicList:
            case MemoryPackOperation.WriteStringList:
            case MemoryPackOperation.WriteNestedValueList:
                AppendWriteField(operation, fieldNameStack, fields, classifier, steps);
                break;

            case MemoryPackOperation.PackedBools:
                {
                    List<string> boolNames = fieldNameStack.Reverse().Select(n => n.HumanizeField()).ToList();
                    fieldNameStack.Clear();
                    steps.Add(new PackedBools(boolNames));
                    break;
                }

            case MemoryPackOperation.CallBase:
                {
                    steps.Add(new CallBase());
                    break;
                }

            case MemoryPackOperation.Ignore:
                break;
        }
    }

    private static void AppendWriteField(
        MemoryPackOperation operation,
        Stack<string> fieldNameStack,
        FieldInfo[] fields,
        FieldClassifier classifier,
        List<SerializationStep> steps)
    {
        string name = FieldNameStackHelpers.PopLastFieldAndClear(fieldNameStack);
        string rustName = name.HumanizeField();
        FieldInfo? field = FindField(fields, rustName);
        FieldKind kind = field != null
            ? classifier.Classify(field.FieldType, operation)
            : FallbackKind(operation);
        steps.Add(new WriteField(rustName, kind));
    }

    private static FieldKind FallbackKind(MemoryPackOperation operation) =>
        operation switch
        {
            MemoryPackOperation.WriteObject => FieldKind.Object,
            MemoryPackOperation.WriteObjectRequired => FieldKind.ObjectRequired,
            MemoryPackOperation.WriteValueList => FieldKind.ValueList,
            MemoryPackOperation.WriteObjectList => FieldKind.ObjectList,
            MemoryPackOperation.WritePolymorphicList => FieldKind.PolymorphicList,
            MemoryPackOperation.WriteStringList => FieldKind.StringList,
            MemoryPackOperation.WriteNestedValueList => FieldKind.NestedValueList,
            _ => FieldKind.Pod,
        };

    private static FieldInfo? FindField(FieldInfo[] fields, string rustName)
    {
        return fields.FirstOrDefault(f => f.Name.HumanizeField() == rustName);
    }
}
