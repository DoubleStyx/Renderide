using SharedTypeGenerator.Analysis;
using SharedTypeGenerator.IR;
using SharedTypeGenerator.Tests.Unit.Support;
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
            Ir.PodStruct("Nested", isPod: false),
            Ir.PodStruct(
                "Host",
                fields: [Ir.PodField("nested", "Nested")],
                packSteps: [new WriteField("nested", FieldKind.Pod)]),
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
            Ir.PodStruct(
                "Host",
                fields: [Ir.PodField("x", "i32")],
                packSteps: [new WriteField("x", FieldKind.Pod)]),
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
            Ir.PodStruct("Nested", isPod: false),
            Ir.PodStruct(
                "Host",
                fields: [Ir.PodField("nested", "Nested")],
                packSteps: [new ConditionalBlock("flag", [new WriteField("nested", FieldKind.Pod)])]),
        };

        PackStepNonPodPostProcessor.Apply(types);
        WriteField? wf = (types[1].PackSteps[0] as ConditionalBlock)?.Steps[0] as WriteField;
        Assert.NotNull(wf);
        Assert.Equal(FieldKind.ObjectRequired, wf.Kind);
    }
}
