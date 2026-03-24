using System.Runtime;
using CommandLine;
using NotEnoughLogs;
using NotEnoughLogs.Behaviour;
using Renderide.Generators.Logging;
using UnityShaderConverter.Logging;
using UnityShaderConverter.Options;

namespace UnityShaderConverter;

internal static class Program
{
    private static int Main(string[] args)
    {
        GCSettings.LatencyMode = GCLatencyMode.Batch;
        return Parser.Default.ParseArguments<ConverterOptions>(args)
            .MapResult(
                options => Run(options),
                _ => 1);
    }

    private static int Run(ConverterOptions options)
    {
        options.DetermineDefaultPaths();
        LogLevel maxLevel = options.Verbose ? LogLevel.Trace : LogLevel.Info;
        string renderideRoot = RenderidePathResolver.ResolveRenderideRoot(RenderidePathResolver.TryGetGitRepositoryRoot());
        string logsDirectory = Path.Combine(renderideRoot, "logs");
        Directory.CreateDirectory(logsDirectory);
        string logFilePath = Path.Combine(logsDirectory, "UnityShaderConverter.log");
        using var logSink = new SuppressWarningsSink(new TruncatingFileSink(logFilePath));
        using var logger = new Logger(
            new[] { logSink },
            new LoggerConfiguration
            {
                Behaviour = new DirectLoggingBehaviour(),
                MaxLevel = maxLevel,
            });
        logger.LogInfo(LogCategory.Startup, $"Log file (truncated this run): {logFilePath}");
        return ConverterRunner.Run(options, logger);
    }
}
