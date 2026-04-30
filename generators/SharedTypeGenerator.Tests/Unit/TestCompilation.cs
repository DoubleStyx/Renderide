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
        (Assembly reflection, AssemblyDefinition cecil, _) = CompileImpl(source, assemblyName, useLoadFrom: false);
        return (reflection, cecil);
    }

    /// <summary>
    /// Like <see cref="Compile"/>, but also returns the on-disk path and loads the reflection view via
    /// <see cref="Assembly.LoadFrom(string)"/> so callers that build path-based components (e.g.,
    /// <see cref="SharedTypeGenerator.Analysis.TypeAnalyzer"/>) see the same Assembly instance the test does.
    /// </summary>
    public static (Assembly Reflection, AssemblyDefinition Cecil, string Path) CompileToFile(
        string source, string assemblyName = "TestAsm") =>
        CompileImpl(source, assemblyName, useLoadFrom: true);

    private static (Assembly Reflection, AssemblyDefinition Cecil, string Path) CompileImpl(
        string source, string assemblyName, bool useLoadFrom)
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
        // Cecil defers IL reads; keep a stable on-disk image so method bodies stay readable after this returns.
        string path = Path.Combine(Path.GetTempPath(), $"{assemblyName}-{Guid.NewGuid():N}.dll");
        File.WriteAllBytes(path, bytes);
        Assembly reflection = useLoadFrom ? Assembly.LoadFrom(path) : Assembly.Load(bytes);
        var readerParameters = new ReaderParameters { AssemblyResolver = new DefaultAssemblyResolver() };
        AssemblyDefinition cecil = AssemblyDefinition.ReadAssembly(path, readerParameters);
        return (reflection, cecil, path);
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
