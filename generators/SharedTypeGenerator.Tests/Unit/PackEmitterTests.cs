using System.Globalization;
using System.Reflection;
using NotEnoughLogs;
using NotEnoughLogs.Behaviour;
using NotEnoughLogs.Sinks;
using SharedTypeGenerator.Analysis;
using SharedTypeGenerator.Emission;
using SharedTypeGenerator.IR;
using SharedTypeGenerator.Logging;
using Xunit;

namespace SharedTypeGenerator.Tests.Unit;

/// <summary>Unit tests for <see cref="PackEmitter"/> line helpers and emission paths.</summary>
public sealed class PackEmitterTests
{
    private static readonly BindingFlags StaticNonPublic = BindingFlags.Static | BindingFlags.NonPublic;

    /// <summary>Snapshot <see cref="PackEmitter"/> pack lines for each <see cref="FieldKind"/>.</summary>
    [Fact]
    public void PackLine_covers_each_field_kind()
    {
        Logger logger = CreateLogger();
        foreach (FieldKind kind in Enum.GetValues<FieldKind>())
        {
            if (kind == FieldKind.StringList)
            {
                Assert.Throws<TargetInvocationException>(() => InvokePackLine(logger, kind));
                continue;
            }

            string line = InvokePackLine(logger, kind);
            Assert.False(string.IsNullOrWhiteSpace(line), kind.ToString());
        }
    }

    /// <summary>Unknown <see cref="FieldKind"/> values produce a FIXME line and a warning log.</summary>
    [Fact]
    public void PackLine_unknown_kind_emits_fixme_and_logs()
    {
        var sink = new CollectingSink();
        using var logger = new Logger(
            [sink],
            new LoggerConfiguration
            {
                Behaviour = new DirectLoggingBehaviour(),
                MaxLevel = LogLevel.Warning,
            });
        var bogus = (FieldKind)9999;
        string line = InvokePackLine(logger, bogus);
        Assert.Contains("FIXME", line, StringComparison.Ordinal);
        Assert.Contains(bogus.ToString(), line, StringComparison.Ordinal);
        Assert.True(sink.Messages.Exists(m => m.Contains("FIXME", StringComparison.Ordinal)), "expected warning log");
    }

    /// <summary>Empty step lists emit no-op bindings for pack and unpack.</summary>
    [Fact]
    public void EmitPack_and_EmitUnpack_empty_steps_are_no_ops()
    {
        using var sw = new StringWriter(CultureInfo.InvariantCulture);
        using (var w = new RustWriter(sw))
        {
            PackEmitter.EmitPack(w, CreateLogger(), "T", [], []);
            PackEmitter.EmitUnpack(w, CreateLogger(), "T", [], []);
        }

        string text = sw.ToString();
        Assert.Contains("let _ = self;", text, StringComparison.Ordinal);
        Assert.Contains("let _ = packer;", text, StringComparison.Ordinal);
        Assert.Contains("Ok(())", text, StringComparison.Ordinal);
    }

    /// <summary><see cref="PackedBools"/> pads to eight slots on pack and unpack.</summary>
    [Fact]
    public void PackedBools_pads_to_eight()
    {
        using var sw = new StringWriter(CultureInfo.InvariantCulture);
        var steps = new List<SerializationStep> { new PackedBools(["a", "b"]) };
        using (var w = new RustWriter(sw))
        {
            PackEmitter.EmitPack(w, CreateLogger(), "T", steps, []);
            PackEmitter.EmitUnpack(w, CreateLogger(), "T", steps, []);
        }

        string text = sw.ToString();
        Assert.Contains("write_packed_bools_array([self.a, self.b", text, StringComparison.Ordinal);
        Assert.Contains("read_packed_bools", text, StringComparison.Ordinal);
        Assert.Contains("__p.bit0", text, StringComparison.Ordinal);
    }

    /// <summary><see cref="ConditionalBlock"/> emits matching <c>if</c> for pack and unpack.</summary>
    [Fact]
    public void ConditionalBlock_emits_if_for_pack_and_unpack()
    {
        using var sw = new StringWriter(CultureInfo.InvariantCulture);
        var inner = new List<SerializationStep> { new WriteField("x", FieldKind.Pod) };
        var steps = new List<SerializationStep> { new ConditionalBlock("flag", inner) };
        using (var w = new RustWriter(sw))
        {
            PackEmitter.EmitPack(w, CreateLogger(), "T", steps, []);
            PackEmitter.EmitUnpack(w, CreateLogger(), "T", steps, []);
        }

        string text = sw.ToString();
        Assert.Contains("if self.flag {", text, StringComparison.Ordinal);
        Assert.Contains("packer.write(&self.x);", text, StringComparison.Ordinal);
        Assert.Contains("self.x = unpacker.read()?;", text, StringComparison.Ordinal);
    }

    /// <summary><see cref="PackEmitter.EmitExplicitPack"/> handles bool, enum, object-required, pod, and padding.</summary>
    [Fact]
    public void EmitExplicitPack_covers_field_kinds_and_padding()
    {
        using var sw = new StringWriter(CultureInfo.InvariantCulture);
        var fields = new List<FieldDescriptor>
        {
            new()
            {
                CSharpName = "B",
                RustName = "b",
                RustType = "u8",
                Kind = FieldKind.Bool,
            },
            new()
            {
                CSharpName = "E",
                RustName = "e",
                RustType = "MyEnum",
                Kind = FieldKind.Enum,
            },
            new()
            {
                CSharpName = "R",
                RustName = "r",
                RustType = "Thing",
                Kind = FieldKind.ObjectRequired,
            },
            new()
            {
                CSharpName = "P",
                RustName = "p",
                RustType = "u32",
                Kind = FieldKind.Pod,
            },
        };

        using (var w = new RustWriter(sw))
            PackEmitter.EmitExplicitPack(w, CreateLogger(), "Explicit", fields, paddingBytes: 4);

        string text = sw.ToString();
        Assert.Contains("write_bool(self.b != 0);", text, StringComparison.Ordinal);
        Assert.Contains("write_object_required(&mut self.e);", text, StringComparison.Ordinal);
        Assert.Contains("write_object_required(&mut self.r);", text, StringComparison.Ordinal);
        Assert.Contains("write(&self.p);", text, StringComparison.Ordinal);
        Assert.Contains("write(&self._padding);", text, StringComparison.Ordinal);
    }

    /// <summary><see cref="PackEmitter.EmitExplicitUnpack"/> handles representative kinds and padding copy.</summary>
    [Fact]
    public void EmitExplicitUnpack_covers_field_kinds_and_padding()
    {
        using var sw = new StringWriter(CultureInfo.InvariantCulture);
        var fields = new List<FieldDescriptor>
        {
            new()
            {
                CSharpName = "B",
                RustName = "b",
                RustType = "u8",
                Kind = FieldKind.Bool,
            },
            new()
            {
                CSharpName = "E",
                RustName = "e",
                RustType = "MyEnum",
                Kind = FieldKind.FlagsEnum,
            },
            new()
            {
                CSharpName = "P",
                RustName = "p",
                RustType = "u32",
                Kind = FieldKind.Pod,
            },
        };

        using (var w = new RustWriter(sw))
            PackEmitter.EmitExplicitUnpack(w, CreateLogger(), "Explicit", fields, paddingBytes: 2);

        string text = sw.ToString();
        Assert.Contains("read_bool()? as u8;", text, StringComparison.Ordinal);
        Assert.Contains("read_object_required", text, StringComparison.Ordinal);
        Assert.Contains("unpacker.read()?;", text, StringComparison.Ordinal);
        Assert.Contains("self._padding.copy_from_slice", text, StringComparison.Ordinal);
    }

    /// <summary><see cref="PackEmitter"/> object unpack strips a single <c>Option&lt;…&gt;</c> for generic inference.</summary>
    [Fact]
    public void UnpackObjectLine_strips_option_wrapper()
    {
        Logger logger = CreateLogger();
        var fields = new List<FieldDescriptor>
        {
            new()
            {
                CSharpName = "Obj",
                RustName = "obj",
                RustType = "Option<FooBar>",
                Kind = FieldKind.Object,
            },
        };

        string line = InvokeUnpackLine(logger, "T", "obj", FieldKind.Object, fields);
        Assert.Contains("read_object::<FooBar>()", line, StringComparison.Ordinal);
    }

    /// <summary>Polymorphic list unpack strips <c>Vec&lt;…&gt;</c> to the element decode name.</summary>
    [Fact]
    public void UnpackPolymorphicListLine_strips_vec_wrapper()
    {
        Logger logger = CreateLogger();
        var fields = new List<FieldDescriptor>
        {
            new()
            {
                CSharpName = "Items",
                RustName = "items",
                RustType = "Vec<Thing>",
                Kind = FieldKind.PolymorphicList,
            },
        };

        string line = InvokeUnpackLine(logger, "T", "items", FieldKind.PolymorphicList, fields);
        Assert.Contains("decode_thing", line, StringComparison.Ordinal);
    }

    private static string InvokePackLine(Logger logger, FieldKind kind)
    {
        MethodInfo? m = typeof(PackEmitter).GetMethod("PackLine", StaticNonPublic);
        Assert.NotNull(m);
        return (string)m.Invoke(null, [logger, "Type", "field", kind])!;
    }

    private static string InvokeUnpackLine(Logger logger, string typeName, string fieldName, FieldKind kind,
        List<FieldDescriptor> fields)
    {
        MethodInfo? m = typeof(PackEmitter).GetMethod("UnpackLine", StaticNonPublic);
        Assert.NotNull(m);
        return (string)m.Invoke(null, [logger, typeName, fieldName, kind, fields])!;
    }

    private static Logger CreateLogger()
    {
        return new Logger(
            [new CollectingSink()],
            new LoggerConfiguration
            {
                Behaviour = new DirectLoggingBehaviour(),
                MaxLevel = LogLevel.Trace,
            });
    }

    /// <summary>Captures log messages for assertions.</summary>
    private sealed class CollectingSink : ILoggerSink
    {
        /// <summary>Recorded formatted messages.</summary>
        public List<string> Messages { get; } = [];

        /// <inheritdoc />
        public void Log(LogLevel level, ReadOnlySpan<char> category, ReadOnlySpan<char> content) =>
            Messages.Add(content.ToString());

        /// <inheritdoc />
        public void Log(LogLevel level, ReadOnlySpan<char> category, ReadOnlySpan<char> format, params object[] args) =>
            Messages.Add(string.Format(CultureInfo.InvariantCulture, format.ToString(), args));
    }
}
