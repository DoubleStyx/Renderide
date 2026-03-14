using System.Reflection;
using NotEnoughLogs;
using NotEnoughLogs.Behaviour;
using Renderite.Shared;
using SharedTypeGenerator.Analysis;
using SharedTypeGenerator.IR;
using SharedTypeGenerator.Logging;

namespace SharedTypeGenerator.Tests;

/// <summary>Base for roundtrip tests. Loads Renderite.Shared, provides type list and pack/unpack helpers.</summary>
public abstract class RoundtripTestBase
{
    private static string GetAssemblyPath()
    {
        var path = Environment.GetEnvironmentVariable("RENDERITE_SHARED_DLL");
        if (!string.IsNullOrEmpty(path) && File.Exists(path))
            return path;

        var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        var steamPath = Path.Combine(home, ".steam", "steam", "steamapps", "common", "Resonite", "Renderite.Shared.dll");
        if (File.Exists(steamPath))
            return steamPath;

        var libPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "SharedTypeGenerator.Tests", "lib", "Renderite.Shared.dll");
        var fullLib = Path.GetFullPath(libPath);
        if (File.Exists(fullLib))
            return fullLib;

        throw new InvalidOperationException(
            "Renderite.Shared.dll not found. Set RENDERITE_SHARED_DLL or copy the DLL to SharedTypeGenerator.Tests/lib/");
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

        using var logger = new Logger(new LoggerConfiguration
        {
            Behaviour = new DirectLoggingBehaviour(),
            MaxLevel = LogLevel.Warning,
        });
        var analyzer = new TypeAnalyzer(logger, path);
        var types = analyzer.Analyze();

        return (assembly, types);
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
        var buffer = new byte[1024 * 1024];
        var span = buffer.AsSpan();
        var packer = new MemoryPacker(span);

        if (obj is not IMemoryPackable packable)
            throw new InvalidOperationException($"{obj.GetType().Name} does not implement IMemoryPackable");
        packable.Pack(ref packer);

        var written = packer.ComputeLength(buffer.AsSpan());
        return (buffer, written);
    }

    protected static object UnpackFromBuffer(Type type, byte[] buffer, int length, IMemoryPackerEntityPool pool)
    {
        var instance = Activator.CreateInstance(type) ?? throw new InvalidOperationException($"Could not create {type.Name}");
        var span = buffer.AsSpan(0, length);
        var unpacker = new MemoryUnpacker(span, pool);

        if (instance is not IMemoryPackable packable)
            throw new InvalidOperationException($"{type.Name} does not implement IMemoryPackable");
        packable.Unpack(ref unpacker);

        return instance;
    }

}
