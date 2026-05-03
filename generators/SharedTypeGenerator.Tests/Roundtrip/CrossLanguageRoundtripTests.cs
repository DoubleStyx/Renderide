using System.Collections;
using System.Reflection;
using SharedTypeGenerator.IR;
using SharedTypeGenerator.Tests.Roundtrip.Support;
using SharedTypeGenerator.Tests.Unit.Support;
using Xunit;

namespace SharedTypeGenerator.Tests;

/// <summary>C# -> Rust -> C# roundtrip tests: C# packs, the Rust <c>roundtrip</c> binary unpacks and repacks, and the byte buffers must match.</summary>
/// <remarks>
/// These sources are only compiled when <c>Renderite.Shared.dll</c> is discoverable at build time (see <c>SharedTypeGenerator.Tests.csproj</c>).
/// When the DLL is missing, the roundtrip test files are excluded from the project so the suite still builds.
/// </remarks>
[Trait("Category", "Roundtrip")]
public sealed class CrossLanguageRoundtripTests : RoundtripTestBase
{
    /// <summary>Yields concrete types that support roundtrip for theory data.</summary>
    public static IEnumerable<object[]> RoundtripableTypes()
    {
        var (asm, types) = Loaded;
        foreach (var d in types)
        {
            if (!CanRoundtrip(d))
                continue;
            var type = GetConcreteType(asm, d);
            if (type == null)
                continue;
            yield return [d.CSharpName, type];
        }
    }

    /// <summary>Packs in C#, runs the Rust binary, and compares output bytes to the original pack.</summary>
    [SkippableTheory]
    [MemberData(nameof(RoundtripableTypes))]
    public void CSharpToRustToCSharp_ByteCompare(string typeName, Type type)
    {
        string? binary = RoundtripBinary.TryFind();
        RoundtripBinary.RequireOrSkip(binary);

        var (asm, _) = Loaded;
        var original = CreateInstance(asm, type);
        RandomInstancePopulator.Populate(original, type, typeName.GetHashCode(StringComparison.Ordinal), asm);
        var (buffer, length) = PackToBuffer(original);
        var bytesA = buffer.AsSpan(0, length).ToArray();

        using var input = new TempFile();
        using var output = new TempFile();
        File.WriteAllBytes(input.FilePath, bytesA);
        RoundtripBinary.Run(binary!, typeName, input.FilePath, output.FilePath);
        var bytesB = File.ReadAllBytes(output.FilePath);
        Assert.True(bytesA.SequenceEqual(bytesB), $"{typeName}: C# packed bytes != Rust roundtrip bytes");
    }
}
