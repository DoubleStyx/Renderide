using System.Reflection;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Mono.Cecil.Rocks;
using SharedTypeGenerator.IR;

namespace SharedTypeGenerator.Analysis;

/// <summary>Conditional-block-aware Pack IL parsing for <see cref="PackMethodParser"/>.</summary>
public partial class PackMethodParser
{
    /// <summary>Second pass: re-parse with proper conditional block nesting.
    /// The first ParseMethodBody is simple -- this one correctly handles if-blocks.</summary>
    public List<SerializationStep> ParseWithConditionals(Type type, FieldInfo[] fields)
    {
        TypeDefinition? typeDef = _assemblyDef.MainModule.GetType(type.Namespace + '.' + type.Name);
        if (typeDef == null) return [];

        MethodDefinition? methodDef = typeDef.GetMethods().FirstOrDefault(m => m.Name == "Pack");
        if (methodDef == null)
        {
            if (type.BaseType != null)
                return [new CallBase()];
            return [];
        }

        return ParseBodyWithConditionals(type, methodDef, fields);
    }

    private List<SerializationStep> ParseBodyWithConditionals(Type type, MethodDefinition methodDef, FieldInfo[] fields)
    {
        var rootSteps = new List<SerializationStep>();
        var contextStack = new Stack<(List<SerializationStep> Steps, Instruction? EndTarget)>();
        contextStack.Push((rootSteps, null));

        var fieldNameStack = new Stack<string>();
        bool skip = false;

        foreach (Instruction instruction in methodDef.Body.Instructions)
        {
            if (skip) { skip = false; continue; }

            while (contextStack.Count > 1 && contextStack.Peek().EndTarget == instruction)
                contextStack.Pop();

            var currentSteps = contextStack.Peek().Steps;

            if (instruction.OpCode.Code is Code.Ldfld or Code.Ldflda)
            {
                string name = ((FieldReference)instruction.Operand).Name;
                fieldNameStack.Push(name);
            }

            if (instruction.OpCode.Code == Code.Brfalse_S)
            {
                string conditionField = PopLastField(fieldNameStack).HumanizeField();
                var endTarget = (Instruction)instruction.Operand;
                var innerSteps = new List<SerializationStep>();
                var block = new ConditionalBlock(conditionField, innerSteps);
                currentSteps.Add(block);
                contextStack.Push((innerSteps, endTarget));
            }

            if (instruction.OpCode.Code is Code.Call && instruction.Operand is MethodReference callRef)
            {
                if (instruction.Next?.OpCode.Code is Code.Stfld)
                    fieldNameStack.Push(((FieldReference)instruction.Next.Operand).Name);

                EmitStep(type, callRef, fieldNameStack, fields, currentSteps);
            }
        }

        return rootSteps;
    }

    private void EmitStep(
        Type type,
        MethodReference callRef,
        Stack<string> fieldNameStack,
        FieldInfo[] fields,
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
                        ? _classifier.Classify(field.FieldType, "Write")
                        : FieldKind.Pod;
                    steps.Add(new WriteField(rustName, kind));
                    break;
                }

            case "Write" when callRef.Parameters.All(p => p.ParameterType.Name == "Boolean"):
                {
                    var boolNames = fieldNameStack.Reverse().Select(n => n.HumanizeField()).ToList();
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
                        ? _classifier.Classify(field.FieldType, "WriteValueList")
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
