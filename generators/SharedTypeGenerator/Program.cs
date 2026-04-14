using System.Runtime;
using CommandLine;
using NotEnoughLogs;
using NotEnoughLogs.Behaviour;
using Renderide.Generators.Logging;
using SharedTypeGenerator;
using SharedTypeGenerator.Logging;
using SharedTypeGenerator.Options;

GCSettings.LatencyMode = GCLatencyMode.Batch;

Parser.Default.ParseArguments<GeneratorOptions>(args)
    .WithParsed(options =>
    {
        if (string.IsNullOrWhiteSpace(options.AssemblyPath))
        {
            string? discovered = ResoniteAssemblyDiscovery.TryFindRenderiteSharedDll();
            if (discovered == null)
            {
                Console.Error.WriteLine(
                    "Could not find Renderite.Shared.dll. Set RENDERITE_SHARED_DLL or RESONITE_DIR, install Resonite via Steam, or pass -i / --assembly-path.");
                Environment.Exit(1);
                return;
            }

            options.AssemblyPath = discovered;
        }
        else
        {
            options.AssemblyPath = Path.GetFullPath(options.AssemblyPath.Trim());
            if (!File.Exists(options.AssemblyPath))
            {
                Console.Error.WriteLine($"Assembly path does not exist: {options.AssemblyPath}");
                Environment.Exit(1);
                return;
            }
        }

        LogLevel maxLevel = ResolveMaxLogLevel(options.Verbose);
        int exitCode = 0;
        {
            string? gitRoot = RenderidePathResolver.TryGetGitRepositoryRoot();
            string logFilePath = LogsLayout.EnsureNewSharedTypeGeneratorLogFilePath(gitRoot);
            using var logSink = SharedTypeGeneratorLogging.CreateMainSink(logFilePath);
            using var logger = new Logger(
                new[] { logSink },
                new LoggerConfiguration
                {
                    Behaviour = new DirectLoggingBehaviour(),
                    MaxLevel = maxLevel,
                });
            logger.LogInfo(LogCategory.Startup, $"Log file (this run): {logFilePath}");
            try
            {
                GeneratorRunner.Run(options, logger);
            }
            catch (Exception ex)
            {
                logger.LogError(LogCategory.Bug, $"SharedTypeGenerator failed: {ex.Message}\n{ex}");
                exitCode = 1;
            }
        }

        if (exitCode != 0)
            Environment.Exit(exitCode);
    });

static LogLevel ResolveMaxLogLevel(bool verbose)
{
    if (verbose)
        return LogLevel.Trace;
    return LogLevel.Info;
}
