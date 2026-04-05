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
        switch (callRef.Name)
        {
            case "Write" when callRef.Parameters.Count == 1:
                {
                    string name = PopLastField(fieldNameStack);
                    string rustName = name.HumanizeField();
                    FieldInfo? field = FindField(fields, rustName);
                    FieldKind kind = field != null
                        ? classifier.Classify(field.FieldType, "Write")
                        : FieldKind.Pod;
                    steps.Add(new WriteField(rustName, kind));
                    break;
                }

            case "Write" when callRef.Parameters.All(p => p.ParameterType.Name == "Boolean"):
                {
                    List<string> boolNames = fieldNameStack.Reverse().Select(n => n.HumanizeField()).ToList();
                    fieldNameStack.Clear();
                    steps.Add(new PackedBools(boolNames));
                    break;
                }

            case "WriteObject":
                {
                    string name = PopLastField(fieldNameStack);
                    steps.Add(new WriteField(name.HumanizeField(), FieldKind.Object));
                    break;
                }

            case "WriteObjectRequired":
                {
                    string name = PopLastField(fieldNameStack);
                    steps.Add(new WriteField(name.HumanizeField(), FieldKind.ObjectRequired));
                    break;
                }

            case "WriteValueList":
            case "WriteEnumValueList":
                {
                    string name = PopLastField(fieldNameStack);
                    string rustName = name.HumanizeField();
                    FieldInfo? field = FindField(fields, rustName);
                    FieldKind kind = field != null
                        ? classifier.Classify(field.FieldType, "WriteValueList")
                        : FieldKind.ValueList;
                    steps.Add(new WriteField(rustName, kind));
                    break;
                }

            case "WriteObjectList":
                {
                    string name = PopLastField(fieldNameStack);
                    steps.Add(new WriteField(name.HumanizeField(), FieldKind.ObjectList));
                    break;
                }

            case "WritePolymorphicList":
                {
                    string name = PopLastField(fieldNameStack);
                    steps.Add(new WriteField(name.HumanizeField(), FieldKind.PolymorphicList));
                    break;
                }

            case "WriteStringList":
                {
                    string name = PopLastField(fieldNameStack);
                    steps.Add(new WriteField(name.HumanizeField(), FieldKind.StringList));
                    break;
                }

            case "WriteNestedValueList":
                {
                    string name = PopLastField(fieldNameStack);
                    steps.Add(new WriteField(name.HumanizeField(), FieldKind.NestedValueList));
                    break;
                }

            case "Pack" or "Unpack":
                {
                    steps.Add(new CallBase());
                    break;
                }

            // Read-side methods -- we only parse Pack, so these shouldn't appear,
            // but handle gracefully by treating them identically to their Write counterparts where applicable.
            case "Read" when callRef.Parameters.Count == 1:
            case "ReadObject":
            case "ReadValueList":
            case "ReadEnumValueList":
            case "ReadObjectList":
            case "ReadPolymorphicList":
            case "ReadStringList":
            case "ReadNestedValueList":
                break;

            case "Read" when callRef.Parameters.All(p => p.ParameterType.Name == "Boolean&"):
                break;
        }
    }

    private static string PopLastField(Stack<string> stack)
    {
        if (stack.Count == 0)
            return "_unknown";

        string last = stack.Pop();
        stack.Clear();
        return last;
    }

    private static FieldInfo? FindField(FieldInfo[] fields, string rustName)
    {
        return fields.FirstOrDefault(f => f.Name.HumanizeField() == rustName);
    }
}
