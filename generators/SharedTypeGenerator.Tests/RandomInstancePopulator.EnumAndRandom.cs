using System.Globalization;

namespace SharedTypeGenerator.Tests;

public static partial class RandomInstancePopulator
{
    /// <summary>Picks a random enum value: non-flags types use a uniformly chosen <b>named</b> member so the
    /// underlying discriminant is always one Rust defines (integers in sparse gaps are invalid on Rust unpack);
    /// <see cref="FlagsAttribute"/> enums use a uniform raw integer from <c>0</c>..<c>mask</c>.</summary>
    private static object GetRandomEnumValue(Type enumType, Random rng)
    {
        if (enumType.IsDefined(typeof(FlagsAttribute), inherit: false))
            return GetRandomFlagsEnumUnderlying(enumType, rng);

        return GetRandomNonFlagsEnumUnderlying(enumType, rng);
    }

    /// <summary>Selects a random flags combination by drawing a uniform raw underlying integer from 0 through the OR of all defined bits.</summary>
    private static object GetRandomFlagsEnumUnderlying(Type enumType, Random rng)
    {
        var values = Enum.GetValues(enumType);
        if (values.Length == 0)
            return Activator.CreateInstance(enumType)!;

        ulong mask = 0;
        foreach (var v in values)
            mask |= Convert.ToUInt64(v, CultureInfo.InvariantCulture);

        var underlying = Enum.GetUnderlyingType(enumType);
        ulong raw = RandomUInt64Inclusive(rng, 0, mask);
        return Enum.ToObject(enumType, Convert.ChangeType(raw, underlying, CultureInfo.InvariantCulture));
    }

    /// <summary>Selects a random non-flags enum by uniformly picking among <see cref="Enum.GetValues"/> members.
    /// Sparse discriminants (gaps between named values) are excluded so C# and Rust roundtrip stays consistent.</summary>
    private static object GetRandomNonFlagsEnumUnderlying(Type enumType, Random rng)
    {
        var values = Enum.GetValues(enumType);
        if (values.Length == 0)
            return Activator.CreateInstance(enumType)!;

        return values.GetValue(rng.Next(values.Length))!;
    }

    /// <summary>Uniform <see cref="ulong"/> in <c>[min, max]</c> using rejection sampling (no modulo bias).</summary>
    private static ulong RandomUInt64Inclusive(Random rng, ulong min, ulong max)
    {
        ArgumentOutOfRangeException.ThrowIfGreaterThan(min, max);
        if (min == max)
            return min;

        ulong span = max - min;
        if (span == ulong.MaxValue)
            return NextUInt64(rng);

        ulong count = span + 1;
        ulong limit = ulong.MaxValue - ulong.MaxValue % count;
        ulong r;
        do
        {
            r = NextUInt64(rng);
        } while (r > limit);

        return min + r % count;
    }

    /// <summary>Reads eight bytes from <paramref name="rng"/> into a <see cref="ulong"/>.</summary>
    private static ulong NextUInt64(Random rng)
    {
        Span<byte> buf = stackalloc byte[8];
        rng.NextBytes(buf);
        return BitConverter.ToUInt64(buf);
    }
}
