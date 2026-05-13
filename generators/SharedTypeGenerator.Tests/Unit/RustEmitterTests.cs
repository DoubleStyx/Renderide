using System.Globalization;
using NotEnoughLogs;
using NotEnoughLogs.Behaviour;
using SharedTypeGenerator.Emission;
using SharedTypeGenerator.IR;
using SharedTypeGenerator.Tests.Unit.Support;
using Xunit;

namespace SharedTypeGenerator.Tests.Unit;

/// <summary>Unit tests for generated Rust emitter decisions.</summary>
public sealed class RustEmitterTests
{
    /// <summary>Single-member value enums emit <c>if</c> instead of a one-arm <c>match</c>.</summary>
    [Fact]
    public void ValueEnum_single_member_emits_if_decode()
    {
        string text = Emit([
            ValueEnum("SingleEnum", [
                new EnumMember { Name = "Only", Value = 0, IsDefault = true },
            ]),
        ]);

        Assert.Contains("*self = if raw == 0 {", text, StringComparison.Ordinal);
        Assert.Contains("if i == 0 {", text, StringComparison.Ordinal);
        Assert.DoesNotContain("*self = match raw", text, StringComparison.Ordinal);
        Assert.DoesNotContain("match i {", text, StringComparison.Ordinal);
    }

    /// <summary>Multi-member value enums keep the generated <c>match</c> arms.</summary>
    [Fact]
    public void ValueEnum_multi_member_keeps_match_decode()
    {
        string text = Emit([
            ValueEnum("PairEnum", [
                new EnumMember { Name = "First", Value = 0, IsDefault = true },
                new EnumMember { Name = "Second", Value = 1, IsDefault = false },
            ]),
        ]);

        Assert.Contains("*self = match raw {", text, StringComparison.Ordinal);
        Assert.Contains("match i {", text, StringComparison.Ordinal);
        Assert.Contains("0 => Self::First,", text, StringComparison.Ordinal);
        Assert.Contains("1 => Self::Second,", text, StringComparison.Ordinal);
    }

    /// <summary>Roundtrip dispatch uses inline format args in the unknown-type error.</summary>
    [Fact]
    public void Roundtrip_dispatch_emits_inline_format_args()
    {
        string text = Emit([
            new TypeDescriptor
            {
                CSharpName = "Packet",
                RustName = "Packet",
                Shape = TypeShape.PackableStruct,
                Fields = [],
            },
        ]);

        Assert.Contains("format!(\"Unknown type: {type_name}\")", text, StringComparison.Ordinal);
        Assert.DoesNotContain("format!(\"Unknown type: {}\", type_name)", text, StringComparison.Ordinal);
    }

    private static string Emit(List<TypeDescriptor> types)
    {
        using var sw = new StringWriter(CultureInfo.InvariantCulture);
        using (var writer = new RustWriter(sw))
        {
            var emitter = new RustEmitter(writer, CreateLogger(), "UnitTestEngine");
            emitter.Emit(types);
        }

        return sw.ToString();
    }

    private static TypeDescriptor ValueEnum(string name, List<EnumMember> members) =>
        new()
        {
            CSharpName = name,
            RustName = name,
            Shape = TypeShape.ValueEnum,
            Fields = [],
            EnumMembers = members,
            RustUnderlyingType = "i32",
        };

    private static Logger CreateLogger() =>
        new(
            [new CollectingSink()],
            new LoggerConfiguration
            {
                Behaviour = new DirectLoggingBehaviour(),
                MaxLevel = LogLevel.Trace,
            });
}
