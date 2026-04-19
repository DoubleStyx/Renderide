using System.Globalization;
using System.Reflection;
using NotEnoughLogs;
using SharedTypeGenerator.Emission;
using NotEnoughLogs.Behaviour;
using Renderite.Shared;
using SharedTypeGenerator.Analysis;
using SharedTypeGenerator.IR;
using SharedTypeGenerator.Logging;
using SharedTypeGenerator.Options;

namespace SharedTypeGenerator.Tests;

/// <summary>Base for roundtrip tests. Loads Renderite.Shared, provides the analyzed type list and C# pack helper.</summary>
public abstract class RoundtripTestBase
{
    private static string GetAssemblyPath()
    {
        var path = Environment.GetEnvironmentVariable(ResoniteAssemblyDiscovery.RenderiteSharedDllEnvVar);
        if (!string.IsNullOrEmpty(path) && File.Exists(path))
            return path;

        var discovered = ResoniteAssemblyDiscovery.TryFindRenderiteSharedDll();
        if (discovered != null)
            return discovered;

        var libPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "lib", "Renderite.Shared.dll");
        var fullLib = Path.GetFullPath(libPath);
        if (File.Exists(fullLib))
            return fullLib;

        throw new InvalidOperationException(
            "Renderite.Shared.dll not found. Set RENDERITE_SHARED_DLL, RESONITE_DIR, install Resonite via Steam, or copy the DLL to generators/SharedTypeGenerator.Tests/lib/");
    }

    protected static (Assembly Assembly, List<TypeDescriptor> Types) LoadAssemblyAndTypes()
    {
        var path = GetAssemblyPath();
        var dir = Path.GetDirectoryName(path)!;
        AppDomain.CurrentDomain.AssemblyResolve += (_, args) =>
        {
            var name = new System.Reflection.AssemblyName(args.Name).Name;
            var dll = Path.Combine(dir, name + ".dll");
            return File.Exists(dll) ? Assembly.LoadFrom(dll) : null;
        };
        var assembly = Assembly.LoadFrom(path);

        // Keep analyzer logs out of the repo: resolve under a temp tree (same naming as production helpers).
        string tempLogsRoot = Path.Combine(
            Path.GetTempPath(),
            "renderide-SharedTypeGenerator-tests",
            Guid.NewGuid().ToString("N", CultureInfo.InvariantCulture));
        string logFilePath = LogsLayout.EnsureUniqueTestSharedTypeGeneratorLogFilePathUnderLogsRoot(tempLogsRoot);
        try
        {
            using var logSink = SharedTypeGeneratorLogging.CreateMainSink(logFilePath);
            using var logger = new Logger(
                new[] { logSink },
                new LoggerConfiguration
                {
                    Behaviour = new DirectLoggingBehaviour(),
                    MaxLevel = LogLevel.Info,
                });
            var analyzer = new TypeAnalyzer(logger, path);
            var types = analyzer.Analyze();

            return (assembly, types);
        }
        finally
        {
            try
            {
                if (Directory.Exists(tempLogsRoot))
                    Directory.Delete(tempLogsRoot, recursive: true);
            }
            catch
            {
                // Best-effort cleanup (file locks / AV on some platforms).
            }
        }
    }

    protected static bool CanRoundtrip(TypeDescriptor d)
    {
        return d.Shape is TypeShape.PackableStruct or TypeShape.PolymorphicBase
            && d.PackSteps.Count > 0;
    }

    protected static Type? GetConcreteType(Assembly asm, TypeDescriptor d)
    {
        if (d.Shape == TypeShape.PolymorphicBase)
            return null;
        var name = d.CSharpName;
        if (name.Contains('`'))
            name = name[..name.IndexOf('`')];
        return asm.GetTypes().FirstOrDefault(t => t.Name == name);
    }

    protected static object CreateInstance(Assembly asm, Type type)
    {
        return Activator.CreateInstance(type) ?? throw new InvalidOperationException($"Could not create {type.Name}");
    }

    protected static (byte[] Buffer, int Length) PackToBuffer(object obj)
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
