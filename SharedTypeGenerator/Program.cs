using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime;
using CommandLine;
using NotEnoughLogs;
using NotEnoughLogs.Behaviour;
using SharedTypeGenerator.Analysis;
using SharedTypeGenerator.Emission;
using SharedTypeGenerator.IR;
using SharedTypeGenerator.Logging;
using SharedTypeGenerator.Options;

GCSettings.LatencyMode = GCLatencyMode.Batch;

using Logger logger = new(new LoggerConfiguration
{
    Behaviour = new DirectLoggingBehaviour(),
    #if DEBUG
    MaxLevel = LogLevel.Trace,
    #else
    MaxLevel = LogLevel.Info,
    #endif
});

Parser.Default.ParseArguments<GeneratorOptions>(args)
    .WithParsed([SuppressMessage("ReSharper", "AccessToDisposedClosure")] (options) =>
    {
        logger.LogDebug(LogCategory.Startup, "Parsed generator options");
        if (options.OutputRustFile == null) options.DetermineDefaultOutputPath();

        Debug.Assert(options.OutputRustFile != null);

        string output = options.OutputRustFile;
        if (File.Exists(output))
        {
            logger.LogDebug(LogCategory.Output, "Deleting existing output file");
            File.Delete(output);
        }

        // ── Frontend: Analyze ──
        logger.LogInfo(LogCategory.Startup, "Loading assembly...");
        Stopwatch stopwatch = Stopwatch.StartNew();

        var analyzer = new TypeAnalyzer(logger, options.AssemblyPath);
        string engineVersion = analyzer.DetectEngineVersion(options.AssemblyPath);
        logger.LogInfo(LogCategory.Generator, $"Generating for engine version {engineVersion}");

        List<TypeDescriptor> types = analyzer.Analyze();
        logger.LogInfo(LogCategory.Analysis, $"Analyzed {types.Count} types in {stopwatch.ElapsedMilliseconds}ms");

        // ── Backend: Emit ──
        stopwatch.Restart();
        using (var writer = new RustWriter(output))
        {
            var emitter = new RustEmitter(writer, engineVersion);
            emitter.Emit(types);
        }

        logger.LogInfo(LogCategory.Emission, $"Emitted to {output} in {stopwatch.ElapsedMilliseconds}ms");
    });
