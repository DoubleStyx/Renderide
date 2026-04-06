using System.Diagnostics;
using NotEnoughLogs;
using SharedTypeGenerator.Analysis;
using SharedTypeGenerator.Emission;
using SharedTypeGenerator.IR;
using SharedTypeGenerator.Logging;
using SharedTypeGenerator.Options;

namespace SharedTypeGenerator;

/// <summary>Runs the analyze-then-emit pipeline for a single set of CLI options.</summary>
public static class GeneratorRunner
{
    /// <summary>
    /// Loads the target assembly, analyzes reachable types from <c>RendererCommand</c>, and writes <c>shared.rs</c>
    /// (default path under the Renderide repo: <c>crates/renderide/src/shared/shared.rs</c>).
    /// Cross-language roundtrip tests additionally require <c>cargo build -p renderide --bin roundtrip</c> so the
    /// Rust binary can exercise <c>roundtrip_dispatch</c> from the generated file.
    /// </summary>
    /// <param name="options">Input assembly path, output path, and verbosity flags.</param>
    /// <param name="logger">Sink for analysis and emission progress.</param>
    public static void Run(GeneratorOptions options, Logger logger)
    {
        logger.LogDebug(LogCategory.Startup, "Parsed generator options");
        if (string.IsNullOrWhiteSpace(options.AssemblyPath))
            throw new InvalidOperationException("AssemblyPath must be set (CLI resolution should run before Run).");
        if (options.OutputRustFile == null)
            options.DetermineDefaultOutputPath();

        Debug.Assert(options.OutputRustFile != null);

        string output = options.OutputRustFile;
        if (File.Exists(output))
        {
            logger.LogDebug(LogCategory.Output, "Deleting existing output file");
            File.Delete(output);
        }

        logger.LogInfo(LogCategory.Startup, "Loading assembly...");
        Stopwatch stopwatch = Stopwatch.StartNew();

        var analyzer = new TypeAnalyzer(logger, options.AssemblyPath);
        string engineVersion = analyzer.DetectEngineVersion(options.AssemblyPath);
        logger.LogInfo(LogCategory.Generator, $"Generating for engine version {engineVersion}");

        List<TypeDescriptor> types = analyzer.Analyze();
        PackStepNonPodPostProcessor.Apply(types);
        logger.LogInfo(LogCategory.Analysis, $"Analyzed {types.Count} types in {stopwatch.ElapsedMilliseconds}ms");

        stopwatch.Restart();
        using (var writer = new RustWriter(output))
        {
            var emitter = new RustEmitter(writer, logger, engineVersion, options.IlVerbose);
            emitter.Emit(types);
        }

        logger.LogInfo(LogCategory.Emission, $"Emitted to {output} in {stopwatch.ElapsedMilliseconds}ms");
    }
}
