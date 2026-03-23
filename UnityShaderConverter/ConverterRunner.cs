using System.Linq;
using NotEnoughLogs;
using UnityShaderConverter.Analysis;
using UnityShaderConverter.Config;
using UnityShaderConverter.Emission;
using UnityShaderConverter.Logging;
using UnityShaderConverter.Options;
using UnityShaderConverter.Variants;
using UnityShaderParser.Common;

namespace UnityShaderConverter;

/// <summary>Orchestrates discovery, parsing, Slang/WGSL emission, and Rust codegen into <c>crates/shaders/src/generated</c>.</summary>
public static class ConverterRunner
{
    /// <summary>Runs the full pipeline for the given options and returns a process exit code.</summary>
    /// <param name="options">Resolved paths and tool flags (call <see cref="ConverterOptions.DetermineDefaultPaths"/> first).</param>
    /// <param name="logger">Structured log sink (e.g. NotEnoughLogs).</param>
    public static int Run(ConverterOptions options, Logger logger)
    {
        options.DetermineDefaultPaths();
        string outputDir = Path.GetFullPath(options.OutputDirectory ?? ".");
        string slangOut = Path.Combine(outputDir, "slang");
        string wgslOut = Path.Combine(outputDir, "wgsl");
        Directory.CreateDirectory(slangOut);
        Directory.CreateDirectory(wgslOut);
        Directory.CreateDirectory(outputDir);

        string exeDir = AppContext.BaseDirectory;
        var defaults = DefaultCompilerConfig.LoadFromOutputDirectory(exeDir);
        var compilerConfig = ConfigLoader.MergeCompilerConfig(defaults, options.CompilerConfigPath);
        VariantConfigModel? variantConfig = ConfigLoader.LoadVariantConfig(options.VariantConfigPath);

        string renderideRoot = FindRenderideRoot(outputDir);
        logger.LogInfo(LogCategory.Startup, $"Renderide root: {renderideRoot}");
        logger.LogInfo(LogCategory.Startup, $"Output directory: {outputDir}");

        string runtimeSlangDir = Path.Combine(exeDir, "runtime_slang");
        if (!Directory.Exists(runtimeSlangDir))
        {
            logger.LogInfo(LogCategory.Startup, $"runtime_slang not next to executable ({runtimeSlangDir}); using source tree path.");
            runtimeSlangDir = Path.GetFullPath(Path.Combine(renderideRoot, "UnityShaderConverter", "runtime_slang"));
        }

        IReadOnlyList<string> inputs = ShaderDiscovery.Enumerate(options.InputDirectories ?? Array.Empty<string>());
        logger.LogInfo(LogCategory.Startup, $"Discovered {inputs.Count} .shader files.");

        string slangcExe = SlangCompiler.ResolveExecutable(options.SlangcPath);
        var slangCompiler = new SlangCompiler(slangcExe, logger);

        var rustModuleNames = new HashSet<string>(StringComparer.Ordinal);
        int variantFailures = 0;

        foreach (string shaderPath in inputs)
        {
            string relForGlob = Path.GetRelativePath(renderideRoot, shaderPath).Replace('\\', '/');
            if (!ShaderLabAnalyzer.TryAnalyze(shaderPath, out ShaderFileDocument? doc, out List<Diagnostic> diags, out List<string> errors))
            {
                foreach (string e in errors)
                    logger.LogWarning(LogCategory.Parse, $"{shaderPath}: {e}");
                foreach (Diagnostic d in diags.Where(d => (d.Kind & DiagnosticFlags.OnlyErrors) != 0))
                    logger.LogWarning(LogCategory.Parse, d.ToString());
                continue;
            }

            foreach (Diagnostic d in diags.Where(d => (d.Kind & DiagnosticFlags.Warning) != 0))
                logger.LogWarning(LogCategory.Parse, d.ToString());

            IReadOnlyList<IReadOnlyList<string>> variants;
            try
            {
                variants = VariantExpander.Expand(doc!, compilerConfig, variantConfig);
            }
            catch (InvalidOperationException ex)
            {
                logger.LogWarning(LogCategory.Variants, $"{shaderPath}: {ex.Message}");
                variantFailures++;
                continue;
            }

            bool slangEligible = GlobMatcher.MatchesAny(relForGlob, compilerConfig.SlangEligibleGlobPatterns);
            string shaderDir = Path.GetDirectoryName(shaderPath) ?? ".";

            string modName = RustEmitter.ModuleNameFromShaderName(doc!.ShaderName);
            var wgslArtifacts = new List<(int PassIndex, int VariantIndex, string RelativeIncludePath)>();
            int expectedWgslArtifacts = doc.Passes.Count * variants.Count;
            bool allWgslPresent = true;

            for (int vi = 0; vi < variants.Count; vi++)
            {
                IReadOnlyList<string> defines = variants[vi];
                for (int pi = 0; pi < doc.Passes.Count; pi++)
                {
                    ShaderPassDocument pass = doc.Passes[pi];
                    string slangName = RustEmitter.SlangFileName(doc.ShaderName, pi, vi);
                    string wgslName = RustEmitter.WgslFileName(doc.ShaderName, pi, vi);
                    string slangPath = Path.Combine(slangOut, slangName);
                    string wgslPath = Path.Combine(wgslOut, wgslName);

                    string slangSource = SlangEmitter.EmitPassSlang(pass, defines);
                    File.WriteAllText(slangPath, slangSource);
                    logger.LogDebug(LogCategory.Slang, $"Wrote {slangPath}");

                    if (!options.SkipSlang && slangEligible)
                    {
                        if (!slangCompiler.TryCompileToWgsl(
                                slangPath,
                                wgslPath,
                                runtimeSlangDir,
                                shaderDir,
                                pass.VertexEntry!,
                                pass.FragmentEntry!,
                                defines,
                                out string? err))
                        {
                            logger.LogWarning(LogCategory.SlangCompile, $"{shaderPath} pass {pi} variant {vi}: slangc failed: {err}");
                        }
                        else
                        {
                            logger.LogInfo(LogCategory.SlangCompile, $"WGSL {wgslPath}");
                        }
                    }

                    if (!File.Exists(wgslPath) || new FileInfo(wgslPath).Length == 0)
                    {
                        logger.LogWarning(
                            LogCategory.Output,
                            $"No WGSL at {wgslPath} for `{doc.ShaderName}` (pass {pi}, variant {vi}); skipping Rust for this shader until WGSL exists.");
                        allWgslPresent = false;
                    }
                    else
                    {
                        string relToRust = Path.Combine("wgsl", wgslName).Replace('\\', '/');
                        wgslArtifacts.Add((pi, vi, relToRust));
                    }
                }
            }

            if (!allWgslPresent || wgslArtifacts.Count != expectedWgslArtifacts)
            {
                logger.LogWarning(
                    LogCategory.Rust,
                    $"Skipping Rust module `{modName}` for `{doc.ShaderName}`: expected {expectedWgslArtifacts} WGSL file(s), have {wgslArtifacts.Count}.");
                continue;
            }

            string rustPath = Path.Combine(outputDir, modName + ".rs");
            string sourceComment = Path.GetRelativePath(renderideRoot, doc.SourcePath);
            string rustSource = RustEmitter.EmitShaderModule(doc, wgslArtifacts, sourceComment);
            File.WriteAllText(rustPath, rustSource);
            rustModuleNames.Add(modName);
            logger.LogInfo(LogCategory.Rust, $"Wrote {rustPath}");
        }

        WriteGeneratedModRust(outputDir, rustModuleNames.OrderBy(s => s, StringComparer.Ordinal).ToList(), logger);
        logger.LogInfo(LogCategory.Output, $"Done. Rust modules: {rustModuleNames.Count}, variant limit skips: {variantFailures}.");
        return 0;
    }

    private static string FindRenderideRoot(string outputDirectory)
    {
        var dir = new DirectoryInfo(Path.GetFullPath(outputDirectory));
        while (dir is not null)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, "crates", "renderide")))
                return dir.FullName;
            dir = dir.Parent;
        }

        return Path.GetFullPath(Path.Combine(outputDirectory, "..", "..", ".."));
    }

    private static void WriteGeneratedModRust(string generatedRoot, IReadOnlyList<string> moduleNames, Logger logger)
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("//! @generated by UnityShaderConverter — shader modules with WGSL and material stubs.");
        sb.AppendLine();
        foreach (string m in moduleNames)
        {
            sb.Append("pub mod ").Append(m).AppendLine(";");
        }

        string path = Path.Combine(generatedRoot, "mod.rs");
        File.WriteAllText(path, sb.ToString());
        logger.LogInfo(LogCategory.Rust, $"Wrote {path}");
    }
}
