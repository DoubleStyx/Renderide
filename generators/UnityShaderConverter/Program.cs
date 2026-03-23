using System.Runtime;
using CommandLine;
using NotEnoughLogs;
using NotEnoughLogs.Behaviour;
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
        LogLevel maxLevel = options.Verbose ? LogLevel.Trace :
#if DEBUG
            LogLevel.Trace;
#else
            LogLevel.Info;
#endif
        using var logger = new Logger(new LoggerConfiguration
        {
            Behaviour = new DirectLoggingBehaviour(),
            MaxLevel = maxLevel,
        });
        return ConverterRunner.Run(options, logger);
    }
}
