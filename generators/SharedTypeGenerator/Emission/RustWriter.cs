using SharedTypeGenerator.Analysis;

namespace SharedTypeGenerator.Emission;

/// <summary>Low-level Rust code writer with indentation management.
/// Uses the Block/Indent RAII pattern via IDisposable for matching braces.</summary>
public class RustWriter : IDisposable
{
    private readonly StreamWriter _writer;
    private readonly Stream _stream;
    private int _indent;

    /// <summary>Opens <paramref name="path"/> for write (truncating if it exists).</summary>
    public RustWriter(string path)
    {
        _stream = File.OpenWrite(path);
        _writer = new StreamWriter(_stream);
    }

    private void WriteIndent()
    {
        for (int i = 0; i < _indent; i++)
            _writer.Write("    ");
    }

    /// <summary>Writes a full-line Rust comment at the current indent.</summary>
    public void Comment(string text)
    {
        WriteIndent();
        _writer.Write("// ");
        _writer.WriteLine(text);
    }

    public void Fixme(string text) => Comment($"FIXME: {text}");
    public void Note(string text) => Comment($"NOTE: {text}");
    public void Todo(string text) => Comment($"TODO: {text}");

    /// <summary>Writes an arbitrary line with current indentation.</summary>
    public void Line(string text)
    {
        WriteIndent();
        _writer.WriteLine(text);
    }

    /// <summary>Writes a blank line.</summary>
    public void BlankLine() => _writer.WriteLine();

    /// <summary>Opens <c>pub enum</c> with <c>#[repr(...)]</c> and optional extra derives; dispose the returned <see cref="Block"/> to close the body.</summary>
    public Block BeginEnum(string name, string reprType, params string[] extraDerives)
    {
        string derives = extraDerives.Length > 0
            ? $"Clone, Copy, Debug, Default, {string.Join(", ", extraDerives)}"
            : "Clone, Copy, Debug, Default";
        WriteIndent(); _writer.WriteLine($"#[derive({derives})]");
        WriteIndent(); _writer.WriteLine($"#[repr({reprType.HumanizeType()})]");
        WriteIndent(); _writer.Write("pub enum "); _writer.Write(name.HumanizeType()); _writer.WriteLine(" {");
        return new Block(this, false);
    }

    /// <summary>Opens a tagged <c>pub enum</c> (union-style variants with payloads), without a primitive repr.</summary>
    public Block BeginUnion(string name, params string[] extraDerives)
    {
        string derives = extraDerives.Length > 0
            ? $"Debug, {string.Join(", ", extraDerives)}"
            : "Debug";
        WriteIndent(); _writer.WriteLine($"#[derive({derives})]");
        WriteIndent(); _writer.Write("pub enum "); _writer.Write(name.HumanizeType()); _writer.WriteLine(" {");
        return new Block(this, false);
    }

    /// <summary>Opens a normal Rust <c>pub struct</c> with fields.</summary>
    public Block BeginStruct(string name, params string[] extraDerives)
    {
        string derives = extraDerives.Length > 0
            ? $"Debug, Default, {string.Join(", ", extraDerives)}"
            : "Debug, Default";
        WriteIndent(); _writer.WriteLine($"#[derive({derives})]");
        WriteIndent(); _writer.Write("pub struct "); _writer.Write(name.HumanizeType()); _writer.WriteLine(" {");
        return new Block(this, false);
    }

    /// <summary>Opens <c>#[repr(C)] pub struct</c> for layout-compatible POD types.</summary>
    public Block BeginExternStruct(string name, params string[] extraDerives)
    {
        string derives = extraDerives.Length > 0
            ? $"Debug, Default, {string.Join(", ", extraDerives)}"
            : "Debug, Default";
        WriteIndent(); _writer.WriteLine($"#[derive({derives})]");
        WriteIndent(); _writer.WriteLine("#[repr(C)]");
        WriteIndent(); _writer.Write("pub struct "); _writer.Write(name.HumanizeType()); _writer.WriteLine(" {");
        return new Block(this, false);
    }

    /// <summary>Emits a single-field <c>#[repr(transparent)]</c> newtype wrapper.</summary>
    public void TransparentStruct(string name, string innerType)
    {
        WriteIndent(); _writer.WriteLine("#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]");
        WriteIndent(); _writer.WriteLine("#[repr(transparent)]");
        WriteIndent(); _writer.Write("pub struct "); _writer.Write(name.HumanizeType());
        _writer.Write("(pub "); _writer.Write(innerType.HumanizeType()); _writer.WriteLine(");");
    }

    /// <summary>Opens <c>impl TypeName</c> for inherent methods.</summary>
    public Block BeginImpl(string name)
    {
        WriteIndent(); _writer.Write("impl "); _writer.Write(name.HumanizeType()); _writer.WriteLine(" {");
        return new Block(this, false);
    }

    /// <summary>Opens <c>impl Trait for Type</c>.</summary>
    public Block BeginTraitImpl(string traitName, string typeName)
    {
        WriteIndent(); _writer.Write("impl "); _writer.Write(traitName);
        _writer.Write(" for "); _writer.Write(typeName.HumanizeType()); _writer.WriteLine(" {");
        return new Block(this, false);
    }

    /// <summary>Opens a function inside the current impl; empty <paramref name="returnType"/> omits the return arrow.</summary>
    public Block BeginMethod(string name, string returnType, string[]? generics, string[] parameters, bool isPublic = true)
    {
        WriteIndent();
        _writer.Write(isPublic ? "pub fn " : "fn ");
        _writer.Write(name);
        if (generics is { Length: > 0 })
        {
            _writer.Write('<');
            _writer.Write(string.Join(", ", generics));
            _writer.Write('>');
        }
        _writer.Write('(');
        _writer.Write(string.Join(", ", parameters));
        _writer.Write(')');
        if (!string.IsNullOrWhiteSpace(returnType))
        {
            _writer.Write(" -> ");
            _writer.Write(returnType);
        }
        _writer.WriteLine(" {");
        return new Block(this, false);
    }

    /// <summary>Opens an <c>if</c> block with the given condition expression.</summary>
    public Block BeginIf(string condition)
    {
        WriteIndent(); _writer.Write("if "); _writer.Write(condition); _writer.WriteLine(" {");
        return new Block(this, false);
    }

    /// <summary>Emits a <c>pub field: type,</c> line; names starting with <c>_</c> are not humanized (e.g. padding).</summary>
    public void StructMember(string name, string type)
    {
        WriteIndent();
        string rustName = name.StartsWith("_") ? name : name.HumanizeField();
        _writer.Write("pub "); _writer.Write(rustName);
        _writer.Write(": "); _writer.Write(type.HumanizeType()); _writer.WriteLine(',');
    }

    /// <summary>Emits a unit enum variant, optionally marked <c>#[default]</c>.</summary>
    public void EnumMember(string name, bool isDefault = false)
    {
        if (isDefault)
        {
            WriteIndent(); _writer.WriteLine("#[default]");
        }
        WriteIndent(); _writer.Write(name.HumanizeField()); _writer.WriteLine(',');
    }

    /// <summary>Emits a C-style enum variant with an explicit discriminant.</summary>
    public void EnumMemberWithValue(string name, string value, bool isDefault = false)
    {
        if (isDefault)
        {
            WriteIndent(); _writer.WriteLine("#[default]");
        }
        WriteIndent(); _writer.Write(name.HumanizeField());
        _writer.Write(" = "); _writer.Write(value); _writer.WriteLine(',');
    }

    /// <summary>Emits a tuple-style enum variant <c>Variant(Type)</c> for tagged unions.</summary>
    public void EnumVariantWithPayload(string variantName, string payloadType)
    {
        WriteIndent();
        _writer.Write(variantName.HumanizeField()); _writer.Write('(');
        _writer.Write(payloadType.HumanizeType()); _writer.WriteLine("),");
    }

    internal void Indent() => _indent++;
    internal void Dedent() => _indent--;

    /// <summary>Closes the current block with <c>}</c> or <c>};</c> when <paramref name="semicolon"/> is true.</summary>
    public void CloseBlock(bool semicolon)
    {
        WriteIndent();
        _writer.WriteLine(semicolon ? "};" : "}");
    }

    public void Dispose()
    {
        _writer.Flush();
        _stream.Flush();
        _writer.Dispose();
        _stream.Dispose();
        GC.SuppressFinalize(this);
    }

    /// <summary>RAII scope that increments indent on creation and decrements + closes brace on dispose.</summary>
    public sealed class Block : IDisposable
    {
        private readonly RustWriter _writer;
        private readonly bool _semicolon;

        public Block(RustWriter writer, bool semicolon)
        {
            _writer = writer;
            _semicolon = semicolon;
            _writer.Indent();
        }

        public void Dispose()
        {
            _writer.Dedent();
            _writer.CloseBlock(_semicolon);
        }
    }
}
