using System.Globalization;
using System.Text;
using SharedTypeGenerator.Emission;
using Xunit;

namespace SharedTypeGenerator.Tests.Unit;

/// <summary>Unit tests for <see cref="RustWriter"/> indentation and emission helpers.</summary>
public sealed class RustWriterTests
{
    /// <summary>Struct emission round-trips through a <see cref="MemoryStream"/>.</summary>
    [Fact]
    public void BeginStruct_writes_members_and_closes_block()
    {
        using var ms = new MemoryStream();
        using (var writer = new RustWriter(new StreamWriter(ms, Encoding.UTF8, leaveOpen: true) { NewLine = "\n" },
                   disposeWriter: true))
        {
            using (writer.BeginStruct("Demo", "Clone, Copy"))
            {
                writer.StructMember("alpha", "i32");
                writer.StructMember("_padding", "[u8; 4]");
            }
        }

        string text = Encoding.UTF8.GetString(ms.ToArray());
        Assert.Contains("pub struct Demo", text, StringComparison.Ordinal);
        Assert.Contains("pub alpha: i32,", text, StringComparison.Ordinal);
        Assert.Contains("pub _padding: [u8; 4],", text, StringComparison.Ordinal);
        Assert.Contains('}', text);
    }

    /// <summary>Synthetic <c>_padding</c> must not be passed through field humanization.</summary>
    [Fact]
    public void StructMember_synthetic_padding_not_humanized()
    {
        using var sw = new StringWriter(CultureInfo.InvariantCulture);
        using (var w = new RustWriter(sw))
        using (w.BeginStruct("S", "Clone"))
        {
            w.StructMember("_padding", "[u8; 4]");
        }

        Assert.Contains("pub _padding:", sw.ToString(), StringComparison.Ordinal);
    }

    /// <summary>Nested <see cref="RustWriter.BeginIf"/> blocks increase indentation by four spaces per level.</summary>
    [Fact]
    public void Nested_if_blocks_indent()
    {
        using var sw = new StringWriter(CultureInfo.InvariantCulture);
        using (var w = new RustWriter(sw))
        {
            using (w.BeginIf("self.a"))
            {
                using (w.BeginIf("self.b"))
                {
                    w.Line("inner();");
                }
            }
        }

        string s = sw.ToString();
        Assert.Contains("if self.a {", s, StringComparison.Ordinal);
        Assert.Contains("    if self.b {", s, StringComparison.Ordinal);
        Assert.Contains("        inner();", s, StringComparison.Ordinal);
    }

    /// <summary><see cref="RustWriter.BeginIf"/> nests correctly.</summary>
    [Fact]
    public void BeginIf_emits_condition_brace()
    {
        using var sw = new StringWriter(CultureInfo.InvariantCulture);
        using (var w = new RustWriter(sw))
        using (w.BeginIf("self.ok"))
        {
            w.Line("do_work();");
        }

        Assert.Contains("if self.ok {", sw.ToString(), StringComparison.Ordinal);
    }

    /// <summary><see cref="RustWriter.BeginExternStruct"/> emits <c>#[repr(C)]</c> after derives.</summary>
    [Fact]
    public void BeginExternStruct_repr_and_derive_order()
    {
        using var sw = new StringWriter(CultureInfo.InvariantCulture);
        using (var w = new RustWriter(sw))
        using (w.BeginExternStruct("PodThing", "Pod", "Zeroable"))
        {
            w.StructMember("x", "u32");
        }

        string text = sw.ToString();
        int derive = text.IndexOf("#[derive", StringComparison.Ordinal);
        int repr = text.IndexOf("#[repr(C)]", StringComparison.Ordinal);
        Assert.True(repr >= 0 && derive >= 0);
        Assert.True(derive < repr);
    }

    /// <summary>Comment, doc, and fixme prefixes are applied.</summary>
    [Fact]
    public void Comment_DocLine_Fixme_prefixes()
    {
        using var sw = new StringWriter(CultureInfo.InvariantCulture);
        using (var w = new RustWriter(sw))
        {
            w.Comment("note");
            w.DocLine("doc text");
            w.Fixme("later");
        }

        string text = sw.ToString();
        Assert.Contains("// note", text, StringComparison.Ordinal);
        Assert.Contains("/// doc text", text, StringComparison.Ordinal);
        Assert.Contains("// FIXME: later", text, StringComparison.Ordinal);
    }

    /// <summary><see cref="RustWriter.EnumMember"/> can mark the default variant.</summary>
    [Fact]
    public void EnumMember_default_attribute()
    {
        using var sw = new StringWriter(CultureInfo.InvariantCulture);
        using (var w = new RustWriter(sw))
        using (w.BeginEnum("E", "u8"))
        {
            w.EnumMember("A", isDefault: true);
            w.EnumMember("B");
        }

        string text = sw.ToString();
        Assert.Contains("#[default]", text, StringComparison.Ordinal);
    }
}
