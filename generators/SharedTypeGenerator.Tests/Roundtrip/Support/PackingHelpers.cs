using Renderite.Shared;
using SharedTypeGenerator.Emission;

namespace SharedTypeGenerator.Tests.Roundtrip.Support;

/// <summary>C# <see cref="MemoryPacker"/> helpers shared by roundtrip tests.</summary>
public static class PackingHelpers
{
    /// <summary>Packs <paramref name="obj"/> into a buffer sized <see cref="PackEmitter.RoundtripBufferBytes"/>.</summary>
    /// <param name="obj">Instance implementing <see cref="IMemoryPackable"/>.</param>
    /// <returns>Buffer and written byte length.</returns>
    public static (byte[] Buffer, int Length) PackToBuffer(object obj)
    {
        var buffer = new byte[PackEmitter.RoundtripBufferBytes];
        var span = buffer.AsSpan();
        var packer = new MemoryPacker(span);

        if (obj is not IMemoryPackable packable)
            throw new InvalidOperationException($"{obj.GetType().Name} does not implement IMemoryPackable");
        packable.Pack(ref packer);

        var written = packer.ComputeLength(buffer.AsSpan());
        return (buffer, written);
    }
}
