using SharedTypeGenerator.Analysis;
using SharedTypeGenerator.IR;
using Xunit;

namespace SharedTypeGenerator.Tests.Unit;

/// <summary>Unit tests for <see cref="PackStepNonPodPostProcessor"/>.</summary>
public sealed class PackStepNonPodPostProcessorTests
{
    /// <summary>Pod steps targeting a non-Pod Rust struct name are upgraded to <see cref="FieldKind.ObjectRequired"/>.</summary>
    [Fact]
    public void Upgrades_pod_write_field_when_target_type_is_non_pod_struct()
    {
        var types = new List<TypeDescriptor>
        {
            new()
            {
                CSharpName = "Nested",
                RustName = "Nested",
                Shape = TypeShape.PodStruct,
                Fields = [],
                IsPod = false,
            },
            new()
            {
                CSharpName = "Host",
                RustName = "Host",
                Shape = TypeShape.PodStruct,
                Fields =
                [
                    new FieldDescriptor
                    {
                        CSharpName = "nested",
                        RustName = "nested",
                        RustType = "Nested",
                        Kind = FieldKind.Pod,
                    },
                ],
                IsPod = true,
                PackSteps = [new WriteField("nested", FieldKind.Pod)],
            },
        };

        PackStepNonPodPostProcessor.Apply(types);
        WriteField? wf = types[1].PackSteps[0] as WriteField;
        Assert.NotNull(wf);
        Assert.Equal(FieldKind.ObjectRequired, wf.Kind);
    }

    /// <summary>Primitive Pod targets stay <see cref="FieldKind.Pod"/>.</summary>
    [Fact]
    public void Leaves_pod_primitive_unchanged()
    {
        var types = new List<TypeDescriptor>
        {
            new()
            {
                CSharpName = "Host",
                RustName = "Host",
                Shape = TypeShape.PodStruct,
                Fields =
                [
                    new FieldDescriptor
                    {
                        CSharpName = "x",
                        RustName = "x",
                        RustType = "i32",
                        Kind = FieldKind.Pod,
                    },
                ],
                IsPod = true,
                PackSteps = [new WriteField("x", FieldKind.Pod)],
            },
        };

        PackStepNonPodPostProcessor.Apply(types);
        WriteField? wf = types[0].PackSteps[0] as WriteField;
        Assert.NotNull(wf);
        Assert.Equal(FieldKind.Pod, wf.Kind);
    }

    /// <summary>Nested <see cref="ConditionalBlock"/> steps rewrite inner <see cref="WriteField"/> steps.</summary>
    [Fact]
    public void Rewrites_conditional_inner_steps()
    {
        var types = new List<TypeDescriptor>
        {
            new()
            {
                CSharpName = "Nested",
                RustName = "Nested",
                Shape = TypeShape.PodStruct,
                Fields = [],
                IsPod = false,
            },
            new()
            {
                CSharpName = "Host",
                RustName = "Host",
                Shape = TypeShape.PodStruct,
                Fields =
                [
                    new FieldDescriptor
                    {
                        CSharpName = "nested",
                        RustName = "nested",
                        RustType = "Nested",
                        Kind = FieldKind.Pod,
                    },
                ],
                IsPod = true,
                PackSteps =
                [
                    new ConditionalBlock("flag", [new WriteField("nested", FieldKind.Pod)]),
                ],
            },
        };

        PackStepNonPodPostProcessor.Apply(types);
        WriteField? wf = (types[1].PackSteps[0] as ConditionalBlock)?.Steps[0] as WriteField;
        Assert.NotNull(wf);
        Assert.Equal(FieldKind.ObjectRequired, wf.Kind);
    }
}
