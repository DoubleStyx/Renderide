using System.Globalization;
using System.Reflection;
using NotEnoughLogs;
using NotEnoughLogs.Behaviour;
using SharedTypeGenerator.Analysis;
using SharedTypeGenerator.IR;
using SharedTypeGenerator.Logging;
using SharedTypeGenerator.Options;

namespace SharedTypeGenerator.Tests.Roundtrip.Support;

/// <summary>
/// Loads <c>Renderite.Shared.dll</c>, registers a one-shot <see cref="AppDomain.AssemblyResolve"/> handler for sibling DLLs,
/// and runs <see cref="TypeAnalyzer"/> under an isolated temp logs root.
/// </summary>
public static class RenderiteSharedAssembly
{
    /// <summary>Prefix for per-process temp log roots under the system temp directory.</summary>
    public const string RenderideTestsLogsTempPrefix = "renderide-SharedTypeGenerator-tests";

    private static int _resolveHookInstalled;

    private static int _processExitRegistered;

    private static string? _logsRootForExitCleanup;

    /// <summary>Thread-safe lazy load of the shared assembly and analyzed type list.</summary>
    public static Lazy<(Assembly Assembly, List<TypeDescriptor> Types)> Loaded { get; } = new(
        LoadAssemblyAndTypes,
        LazyThreadSafetyMode.ExecutionAndPublication);

    /// <summary>Resolves <c>Renderite.Shared.dll</c> using env, discovery, or the test <c>lib</c> folder.</summary>
    private static string GetAssemblyPath()
    {
        var path = Environment.GetEnvironmentVariable(ResoniteAssemblyDiscovery.RenderiteSharedDllEnvVar);
        if (!string.IsNullOrEmpty(path) && File.Exists(path))
            return path;

        var discovered = ResoniteAssemblyDiscovery.TryFindRenderiteSharedDll();
        if (discovered != null)
            return discovered;

        var libPath = System.IO.Path.Combine(
            AppContext.BaseDirectory,
            "..",
            "..",
            "..",
            "lib",
            "Renderite.Shared.dll");
        var fullLib = System.IO.Path.GetFullPath(libPath);
        if (File.Exists(fullLib))
            return fullLib;

        throw new InvalidOperationException(
            "Renderite.Shared.dll not found. Set RENDERITE_SHARED_DLL, RESONITE_DIR, install Resonite via Steam, or copy the DLL to generators/SharedTypeGenerator.Tests/lib/");
    }

    private static void EnsureAssemblyResolveHookOnce(string dependencyDirectory)
    {
        if (Interlocked.CompareExchange(ref _resolveHookInstalled, 1, 0) != 0)
            return;

        AppDomain.CurrentDomain.AssemblyResolve += (_, args) =>
        {
            var name = new AssemblyName(args.Name).Name;
            var dll = System.IO.Path.Combine(dependencyDirectory, name + ".dll");
            return File.Exists(dll) ? Assembly.LoadFrom(dll) : null;
        };
    }

    private static (Assembly Assembly, List<TypeDescriptor> Types) LoadAssemblyAndTypes()
    {
        var path = GetAssemblyPath();
        var dir = System.IO.Path.GetDirectoryName(path)!;
        EnsureAssemblyResolveHookOnce(dir);
        var assembly = Assembly.LoadFrom(path);

        string tempLogsRoot = System.IO.Path.Combine(
            Path.GetTempPath(),
            RenderideTestsLogsTempPrefix,
            Guid.NewGuid().ToString("N", CultureInfo.InvariantCulture));
        Directory.CreateDirectory(tempLogsRoot);
        _logsRootForExitCleanup = tempLogsRoot;
        RegisterProcessExitCleanupOnce();

        string logFilePath = LogsLayout.EnsureUniqueTestSharedTypeGeneratorLogFilePathUnderLogsRoot(tempLogsRoot);
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

    /// <summary>Deletes the temp logs tree on process exit so xUnit teardown order cannot race cleanup.</summary>
    private static void RegisterProcessExitCleanupOnce()
    {
        if (Interlocked.CompareExchange(ref _processExitRegistered, 1, 0) != 0)
            return;

        AppDomain.CurrentDomain.ProcessExit += (_, _) =>
        {
            string? root = _logsRootForExitCleanup;
            if (string.IsNullOrEmpty(root))
                return;
            try
            {
                var parent = Directory.GetParent(root)?.FullName;
                if (Directory.Exists(root))
                    Directory.Delete(root, recursive: true);
                if (!string.IsNullOrEmpty(parent)
                    && parent.Contains(RenderideTestsLogsTempPrefix, StringComparison.Ordinal)
                    && Directory.Exists(parent)
                    && Directory.GetFileSystemEntries(parent).Length == 0)
                    Directory.Delete(parent);
            }
            catch (IOException)
            {
            }
            catch (UnauthorizedAccessException)
            {
            }
        };
    }
}
