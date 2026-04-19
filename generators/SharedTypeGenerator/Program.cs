using System.Runtime;
using CommandLine;
using NotEnoughLogs;
using NotEnoughLogs.Behaviour;
using SharedTypeGenerator;
using SharedTypeGenerator.Logging;
using SharedTypeGenerator.Options;

GCSettings.LatencyMode = GCLatencyMode.Batch;

Parser.Default.ParseArguments<GeneratorOptions>(args)
    .WithParsed(options =>
    {
        if (!AssemblyPathResolution.TryResolveOrValidate(options, Console.Error))
        {
            Environment.Exit(1);
            return;
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

/// <summary>Maps the <c>--verbose</c> flag to the maximum <see cref="LogLevel"/> used for this process.</summary>
static LogLevel ResolveMaxLogLevel(bool verbose)
{
    if (verbose)
        return LogLevel.Trace;
    return LogLevel.Info;
}
