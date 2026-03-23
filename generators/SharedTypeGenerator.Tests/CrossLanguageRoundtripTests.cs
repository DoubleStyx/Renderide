using System.Collections;
using System.Diagnostics;
using System.Reflection;
using Renderite.Shared;
using SharedTypeGenerator.Analysis;
using SharedTypeGenerator.IR;
using Xunit;

namespace SharedTypeGenerator.Tests;

/// <summary>C# → Rust → C# roundtrip tests: C# packs, the Rust <c>roundtrip</c> binary unpacks and repacks, and the byte buffers must match.</summary>
public sealed class CrossLanguageRoundtripTests : RoundtripTestBase
{
    private static (Assembly asm, List<TypeDescriptor> types)? _cache;
    private static (Assembly asm, List<TypeDescriptor> types) Cached => _cache ??= LoadAssemblyAndTypes();

    private static string? GetRoundtripBinaryPath()
    {
        var repoRoot = FindRepoRoot();
        if (repoRoot == null) return null;

        var debugPath = Path.Combine(repoRoot, "target", "debug", "roundtrip");
        var releasePath = Path.Combine(repoRoot, "target", "release", "roundtrip");

        if (File.Exists(debugPath)) return debugPath;
        if (File.Exists(releasePath)) return releasePath;
        return null;
    }

    private static string? FindRepoRoot()
    {
        var dir = AppContext.BaseDirectory;
        while (!string.IsNullOrEmpty(dir))
        {
            if (Directory.Exists(Path.Combine(dir, "generators", "SharedTypeGenerator")) &&
                Directory.Exists(Path.Combine(dir, "crates", "renderide")))
                return dir;
            dir = Path.GetDirectoryName(dir);
        }
        return null;
    }

    private static bool IsCiEnvironment()
    {
        string? ci = Environment.GetEnvironmentVariable("CI");
        return string.Equals(ci, "true", StringComparison.OrdinalIgnoreCase)
            || string.Equals(ci, "1", StringComparison.Ordinal);
    }

    private static void RequireRoundtripBinaryOrSkip(string? binary)
    {
        if (binary != null)
            return;
        if (IsCiEnvironment())
        {
            throw new InvalidOperationException(
                "Roundtrip binary is required when CI=true. From the repository root run: cargo build -p renderide --bin roundtrip");
        }

        Skip.If(true,
            "Roundtrip binary not found under target/debug or target/release. From the repository root run: cargo build -p renderide --bin roundtrip");
    }

    public static IEnumerable<object[]> RoundtripableTypes()
    {
        var (asm, types) = Cached;
        foreach (var d in types)
        {
            if (!CanRoundtrip(d)) continue;
            var type = GetConcreteType(asm, d);
            if (type == null) continue;
            yield return [d.CSharpName, type];
        }
    }

    [SkippableTheory]
    [MemberData(nameof(RoundtripableTypes))]
    public void CSharpToRustToCSharp_ByteCompare(string typeName, Type type)
    {
        var binary = GetRoundtripBinaryPath();
        RequireRoundtripBinaryOrSkip(binary);

        var (asm, _) = Cached;
        var original = CreateInstance(asm, type);
        RandomInstancePopulator.Populate(original, type, typeName.GetHashCode(StringComparison.Ordinal), asm);
        var (buffer, length) = PackToBuffer(original);
        var bytesA = buffer.AsSpan(0, length).ToArray();

        var inputPath = Path.GetTempFileName();
        var outputPath = Path.GetTempFileName();
        try
        {
            File.WriteAllBytes(inputPath, bytesA);
            RunRoundtripBinary(binary!, typeName, inputPath, outputPath);
            var bytesB = File.ReadAllBytes(outputPath);
            Assert.True(bytesA.SequenceEqual(bytesB), $"{typeName}: C# packed bytes != Rust roundtrip bytes");
        }
        finally
        {
            if (File.Exists(inputPath)) File.Delete(inputPath);
            if (File.Exists(outputPath)) File.Delete(outputPath);
        }
    }

    private static void RunRoundtripBinary(string binaryPath, string typeName, string inputPath, string outputPath)
    {
        var psi = new ProcessStartInfo
        {
            FileName = binaryPath,
            ArgumentList = { typeName, inputPath, outputPath },
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        using var proc = Process.Start(psi)!;
        var stderr = proc.StandardError.ReadToEnd();
        proc.WaitForExit(5000);
        if (proc.ExitCode != 0)
            throw new InvalidOperationException($"Roundtrip binary failed (exit {proc.ExitCode}): {stderr}");
    }
}
