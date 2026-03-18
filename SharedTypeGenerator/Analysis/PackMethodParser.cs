using System.Reflection;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Mono.Cecil.Rocks;
using SharedTypeGenerator.IR;

namespace SharedTypeGenerator.Analysis;

/// <summary>Parses the IL of a Pack method to produce an ordered list of SerializationSteps.
/// Only reads Pack (not Unpack) because they are symmetric -- the same step list drives both.
/// Produces pure IR with zero Rust emission.</summary>
public class PackMethodParser
{
    private readonly AssemblyDefinition _assemblyDef;
    private readonly Assembly _assembly;
    private readonly FieldClassifier _classifier;

    public PackMethodParser(AssemblyDefinition assemblyDef, Assembly assembly, FieldClassifier classifier)
    {
        _assemblyDef = assemblyDef;
        _assembly = assembly;
        _classifier = classifier;
    }

    /// <summary>Parses the Pack method of the given type, returning the serialization steps.
    /// Recursively follows base.Pack() calls to include inherited serialization.</summary>
    public List<SerializationStep> Parse(Type type, FieldInfo[] fields)
    {
        var steps = new List<SerializationStep>();
        ParseMethod(type, "Pack", fields, steps);
        return steps;
    }

    private void ParseMethod(Type type, string methodName, FieldInfo[] fields, List<SerializationStep> steps)
    {
        TypeDefinition? typeDef = _assemblyDef.MainModule.GetType(type.Namespace + '.' + type.Name);
        if (typeDef == null) return;

        MethodDefinition? methodDef = typeDef.GetMethods().FirstOrDefault(m => m.Name == methodName);
        if (methodDef == null)
        {
            if (type.BaseType != null)
                ParseMethod(type.BaseType, methodName, fields, steps);
            return;
        }

        ParseMethodBody(type, methodDef, fields, steps);
    }

    private void ParseMethodBody(Type type, MethodDefinition methodDef, FieldInfo[] fields, List<SerializationStep> steps)
    {
        var fieldNameStack = new Stack<string>();
        var instructions = methodDef.Body.Instructions;

        // Pre-scan to find conditional block boundaries (Brfalse_S targets)
        var conditionalTargets = new Dictionary<Instruction, string>();
        bool skip = false;

        foreach (Instruction instruction in instructions)
        {
            if (skip) { skip = false; continue; }

            if (instruction.OpCode.Code is Code.Ldfld or Code.Ldflda)
            {
                string name = ((FieldReference)instruction.Operand).Name;
                fieldNameStack.Push(name);
            }

            if (instruction.OpCode.Code == Code.Brfalse_S)
            {
                string conditionField = PopLastField(fieldNameStack).HumanizeField();
                conditionalTargets[instruction] = conditionField;
            }

            if (instruction.OpCode.Code is Code.Call && instruction.Operand is MethodReference callRef)
            {
                if (instruction.Next?.OpCode.Code is Code.Stfld)
                    fieldNameStack.Push(((FieldReference)instruction.Next.Operand).Name);

                HandleCall(type, callRef, instruction, fieldNameStack, fields, steps, conditionalTargets);
            }
        }
    }

    private void HandleCall(
        Type type,
        MethodReference callRef,
        Instruction instruction,
        Stack<string> fieldNameStack,
        FieldInfo[] fields,
        List<SerializationStep> steps,
        Dictionary<Instruction, string> conditionalTargets)
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

            // Read-side methods -- we only parse Pack, so these shouldn't appear,
            // but handle gracefully by treating them identically to their Write counterparts.
            case "Read" when callRef.Parameters.Count == 1:
            case "ReadObject":
            case "ReadValueList":
            case "ReadObjectList":
            case "ReadPolymorphicList":
            case "ReadStringList":
            case "ReadNestedValueList":
                break;

            case "Read" when callRef.Parameters.All(p => p.ParameterType.Name == "Boolean&"):
                break;
        }
    }

    /// <summary>Parses the Unpack method to find steps that run only during unpack,
    /// e.g. decodedTime = DateTime.UtcNow. These are emitted only in unpack, not pack.</summary>
    public List<SerializationStep> ParseUnpackOnlySteps(Type type)
    {
        TypeDefinition? typeDef = ResolveTypeDef(type);
        if (typeDef == null)
        {
            if (type.BaseType != null)
                return ParseUnpackOnlySteps(type.BaseType);
            return [];
        }

        MethodDefinition? methodDef = typeDef.GetMethods().FirstOrDefault(m => m.Name == "Unpack");
        if (methodDef == null)
        {
            if (type.BaseType != null)
                return ParseUnpackOnlySteps(type.BaseType);
            return [];
        }

        var steps = new List<SerializationStep>();
        var instructions = methodDef.Body.Instructions.ToList();

        for (int i = 0; i < instructions.Count; i++)
        {
            if (instructions[i].OpCode.Code != Code.Call || instructions[i].Operand is not MethodReference callRef)
                continue;
            if (callRef.Name != "get_UtcNow")
                continue;

            Instruction? next = i + 1 < instructions.Count ? instructions[i + 1] : null;
            if (next?.OpCode.Code != Code.Stfld || next.Operand is not FieldReference fieldRef)
                continue;

            steps.Add(new TimestampNow(fieldRef.Name.HumanizeField()));
        }

        return steps;
    }

    private TypeDefinition? ResolveTypeDef(Type type)
    {
        string fullName = string.IsNullOrEmpty(type.Namespace) ? type.Name : type.Namespace + '.' + type.Name;
        return _assemblyDef.MainModule.GetType(fullName);
    }

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

            // Close any conditional blocks whose end target is this instruction
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

        // Pop returns the most recently pushed item, which is the field we want
        string last = stack.Pop();
        stack.Clear();
        return last;
    }

    private static FieldInfo? FindField(FieldInfo[] fields, string rustName)
    {
        return fields.FirstOrDefault(f => f.Name.HumanizeField() == rustName);
    }
}
