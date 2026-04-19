using System.Reflection;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Mono.Cecil.Rocks;
using SharedTypeGenerator.IR;

namespace SharedTypeGenerator.Analysis;

/// <summary>Parses the IL of a Pack method to produce an ordered list of SerializationSteps.
/// Only reads Pack (not Unpack) for the step list because Pack and Unpack are symmetric for emission.
/// Produces pure IR with zero Rust emission.</summary>
public sealed class PackMethodParser
{
    private readonly AssemblyDefinition _assemblyDef;
    private readonly FieldClassifier _classifier;

    /// <summary>Creates a parser for the module described by <paramref name="assemblyDef"/>.</summary>
    public PackMethodParser(AssemblyDefinition assemblyDef, FieldClassifier classifier)
    {
        _assemblyDef = assemblyDef;
        _classifier = classifier;
    }

    /// <summary>Parses the Pack method with <c>if</c> blocks modeled as <see cref="ConditionalBlock"/>.</summary>
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

        return ParseBodyWithConditionals(methodDef, fields);
    }

    /// <summary>Walks Pack IL, pairing <c>brfalse</c> / <c>brfalse.s</c> with field loads to build <see cref="ConditionalBlock"/> scopes.</summary>
    private List<SerializationStep> ParseBodyWithConditionals(MethodDefinition methodDef, FieldInfo[] fields)
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

            List<SerializationStep> currentSteps = contextStack.Peek().Steps;

            if (instruction.OpCode.Code is Code.Ldfld or Code.Ldflda)
            {
                string name = ((FieldReference)instruction.Operand).Name;
                fieldNameStack.Push(name);
            }

            if (instruction.OpCode.Code is Code.Brfalse_S or Code.Brfalse)
            {
                string conditionField = FieldNameStackHelpers.PopLastFieldAndClear(fieldNameStack).HumanizeField();
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

                PackIlCallInterpreter.AppendStepForCall(callRef, fieldNameStack, fields, _classifier, currentSteps);
            }
        }

        return rootSteps;
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
        List<Instruction> instructions = methodDef.Body.Instructions.ToList();

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
}
