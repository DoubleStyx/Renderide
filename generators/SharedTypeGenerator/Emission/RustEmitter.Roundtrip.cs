using System.Linq;
using SharedTypeGenerator.IR;

namespace SharedTypeGenerator.Emission;

public partial class RustEmitter
{
    private void EmitRoundtripDispatch(List<TypeDescriptor> types)
    {
        var roundtripable = types.Where(t =>
            (t.Shape == TypeShape.PackableStruct || t.Shape == TypeShape.GeneralStruct)
            && (t.PackSteps.Count > 0 || t.Shape == TypeShape.PackableStruct)
            && !t.RustName.Contains('<')).ToList();

        if (roundtripable.Count == 0) return;

        _w.BlankLine();
        _w.Comment("Roundtrip dispatch for C#–Rust serialization tests. Called by the roundtrip binary.");
        using (_w.BeginMethod("roundtrip_dispatch", "std::io::Result<Vec<u8>>", null, ["type_name: &str", "input: &[u8]"], isPublic: true))
        {
            _w.Line("use super::packing::default_entity_pool::DefaultEntityPool;");
            _w.Line("let mut pool = DefaultEntityPool;");
            _w.Line("let mut unpacker = MemoryUnpacker::new(input, &mut pool);");
            _w.Line($"let mut output = vec![0u8; {PackEmitter.RoundtripBufferBytes}];");
            _w.Line("let original_len = output.len();");
            _w.Line("let mut packer = MemoryPacker::new(&mut output[..]);");
            _w.BlankLine();
            _w.Line("match type_name {");
            foreach (var t in roundtripable)
            {
                string rustName = t.RustName;
                _w.Line($"    \"{t.CSharpName}\" => {{ let mut x = {rustName}::default(); x.unpack(&mut unpacker).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?; x.pack(&mut packer); }}");
            }
            _w.Line("    _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, format!(\"Unknown type: {}\", type_name))),");
            _w.Line("}");
            _w.BlankLine();
            _w.Line("let written = original_len - packer.remaining_len();");
            _w.Line("output.truncate(written);");
            _w.Line("Ok(output)");
        }
    }
}
