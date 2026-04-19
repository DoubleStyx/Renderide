using System.Reflection;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Mono.Cecil;

namespace SharedTypeGenerator.Tests.Unit;

/// <summary>Compiles in-memory C# sources for Cecil + reflection tests.</summary>
internal static class TestCompilation
{
    /// <summary>Compiles <paramref name="source"/> into a dynamic assembly and returns reflection + Cecil views.</summary>
    public static (Assembly Reflection, AssemblyDefinition Cecil) Compile(string source, string assemblyName = "TestAsm")
    {
        SyntaxTree tree = CSharpSyntaxTree.ParseText(source, CSharpParseOptions.Default.WithLanguageVersion(LanguageVersion.Latest));
        IEnumerable<MetadataReference> refs = GetMetadataReferences();
        CSharpCompilation compilation = CSharpCompilation.Create(
            assemblyName,
            [tree],
            refs,
            new CSharpCompilationOptions(
                OutputKind.DynamicallyLinkedLibrary,
                optimizationLevel: OptimizationLevel.Debug));
        using var ms = new MemoryStream();
        var emitResult = compilation.Emit(ms);
        if (!emitResult.Success)
        {
            string errors = string.Join(
                "\n",
                emitResult.Diagnostics.Where(d => d.Severity == DiagnosticSeverity.Error));
            throw new InvalidOperationException($"Compilation failed:\n{errors}");
        }

        byte[] bytes = ms.ToArray();
        Assembly reflection = Assembly.Load(bytes);
        // Cecil defers IL reads; keep a stable on-disk image so method bodies stay readable after this returns.
        string path = Path.Combine(Path.GetTempPath(), $"{assemblyName}-{Guid.NewGuid():N}.dll");
        File.WriteAllBytes(path, bytes);
        var readerParameters = new ReaderParameters { AssemblyResolver = new DefaultAssemblyResolver() };
        AssemblyDefinition cecil = AssemblyDefinition.ReadAssembly(path, readerParameters);
        return (reflection, cecil);
    }

    private static IEnumerable<MetadataReference> GetMetadataReferences()
    {
        var types = new[]
        {
            typeof(object),
            typeof(List<>),
            typeof(Enumerable),
            typeof(DateTime),
            typeof(System.Runtime.CompilerServices.RuntimeHelpers),
        };
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (Type t in types)
        {
            string loc = t.Assembly.Location;
            if (string.IsNullOrEmpty(loc) || !seen.Add(loc))
                continue;
            yield return MetadataReference.CreateFromFile(loc);
        }
    }
}
