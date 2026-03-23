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

/// <summary>Orchestrates discovery, parsing, Slang/WGSL emission, and Rust codegen into <c>crates/renderide/src/shaders/generated</c>.</summary>
public static class ConverterRunner
{
    /// <summary>Runs the full pipeline for the given options and returns a process exit code.</summary>
    /// <param name="options">Resolved paths and tool flags (call <see cref="ConverterOptions.DetermineDefaultPaths"/> first).</param>
    /// <param name="logger">Structured log sink (e.g. NotEnoughLogs).</param>
    public static int Run(ConverterOptions options, Logger logger)
    {
        options.DetermineDefaultPaths();
        string outputDir = Path.GetFullPath(options.OutputDirectory ?? ".");
        string wgslOut = Path.Combine(outputDir, "wgsl");
        Directory.CreateDirectory(wgslOut);
        Directory.CreateDirectory(outputDir);

        string legacySlangDir = Path.Combine(outputDir, "slang");
        TryDeleteDirectoryRecursive(legacySlangDir, logger);

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
            runtimeSlangDir = Path.GetFullPath(Path.Combine(renderideRoot, "generators", "UnityShaderConverter", "runtime_slang"));
        }

        IReadOnlyList<string> inputRootsFull = (options.InputDirectories ?? Array.Empty<string>())
            .Select(Path.GetFullPath)
            .Distinct(StringComparer.Ordinal)
            .ToList();

        IReadOnlyList<string> inputs = ShaderDiscovery.Enumerate(options.InputDirectories ?? Array.Empty<string>());
        logger.LogInfo(LogCategory.Startup, $"Discovered {inputs.Count} .shader files.");

        string slangcExe = SlangCompiler.ResolveExecutable(options.SlangcPath);
        var slangCompiler = new SlangCompiler(slangcExe, logger);

        string tempSlangDir = Path.Combine(Path.GetTempPath(), "UnityShaderConverter", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempSlangDir);

        var bundleEntries = new List<(string ModName, ShaderFileDocument Document, string SourceComment, List<(int PassIndex, int VariantIndex, string WgslRelativePath)> Artifacts)>();
        var modNameOwners = new Dictionary<string, string>(StringComparer.Ordinal);
        int variantFailures = 0;

        try
        {
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
                if (!modNameOwners.TryGetValue(modName, out string? ownerPath))
                    modNameOwners[modName] = shaderPath;
                else if (!string.Equals(Path.GetFullPath(ownerPath), Path.GetFullPath(shaderPath), StringComparison.Ordinal))
                {
                    logger.LogWarning(
                        LogCategory.Rust,
                        $"Skipping `{shaderPath}`: Rust module name `{modName}` is already used by `{ownerPath}` (duplicate Unity shader name).");
                    continue;
                }

                string? inputRoot = ShaderInputPaths.FindContainingInputRoot(shaderPath, inputRootsFull);
                string nestedWgslDirPosix = inputRoot is not null
                    ? ShaderInputPaths.GetWgslNestedDirectoryRelativeToWgslRoot(shaderPath, inputRoot)
                    : string.Empty;

                var wgslArtifacts = new List<(int PassIndex, int VariantIndex, string RelativeIncludePath)>();
                int expectedWgslArtifacts = doc.Passes.Count * variants.Count;
                bool allWgslPresent = true;

                for (int vi = 0; vi < variants.Count; vi++)
                {
                    IReadOnlyList<string> defines = variants[vi];
                    for (int pi = 0; pi < doc.Passes.Count; pi++)
                    {
                        ShaderPassDocument pass = doc.Passes[pi];
                        string wgslFileName = RustEmitter.WgslFileName(doc.ShaderName, pi, vi);
                        string wgslPath = CombineWgslOutputPath(wgslOut, nestedWgslDirPosix, wgslFileName);
                        Directory.CreateDirectory(Path.GetDirectoryName(wgslPath)!);

                        if (!options.SkipSlang && slangEligible)
                        {
                            string slangSource = SlangEmitter.EmitPassSlang(pass, defines);
                            string tempSlangPath = Path.Combine(tempSlangDir, "compile.slang");
                            try
                            {
                                File.WriteAllText(tempSlangPath, slangSource);
                                logger.LogDebug(LogCategory.Slang, $"Transient Slang → slangc ({tempSlangPath})");
                                if (!slangCompiler.TryCompileToWgsl(
                                        tempSlangPath,
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
                            finally
                            {
                                TryDeleteFile(tempSlangPath);
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
                            string relToGenerated = RelativePathFromGeneratedToWgsl(outputDir, wgslPath);
                            wgslArtifacts.Add((pi, vi, relToGenerated));
                        }
                    }
                }

                if (!allWgslPresent || wgslArtifacts.Count != expectedWgslArtifacts)
                {
                    logger.LogWarning(
                        LogCategory.Rust,
                        $"Skipping bundle entry `{modName}` for `{doc.ShaderName}`: expected {expectedWgslArtifacts} WGSL file(s), have {wgslArtifacts.Count}.");
                    continue;
                }

                string sourceComment = Path.GetRelativePath(renderideRoot, doc.SourcePath);
                bundleEntries.Add((modName, doc, sourceComment.Replace('\\', '/'), wgslArtifacts));
                logger.LogInfo(LogCategory.Rust, $"Queued bundle entry `{modName}` ({wgslArtifacts.Count} WGSL artifact(s)).");
            }

            bundleEntries.Sort((a, b) => string.CompareOrdinal(a.ModName, b.ModName));
            var wgslList = bundleEntries
                .Select(e => (e.ModName, e.Document, e.SourceComment, (IReadOnlyList<(int, int, string)>)e.Artifacts))
                .ToList();
            var materialsList = bundleEntries
                .Select(e => (e.ModName, e.Document, e.SourceComment))
                .ToList();

            string wgslSourcesPath = Path.Combine(outputDir, "wgsl_sources.rs");
            string materialsPath = Path.Combine(outputDir, "materials.rs");
            File.WriteAllText(wgslSourcesPath, RustEmitter.EmitWgslSourcesFile(wgslList));
            File.WriteAllText(materialsPath, RustEmitter.EmitMaterialsFile(materialsList));
            logger.LogInfo(LogCategory.Rust, $"Wrote {wgslSourcesPath}");
            logger.LogInfo(LogCategory.Rust, $"Wrote {materialsPath}");

            WriteGeneratedModRust(outputDir, logger);
            RemoveOrphanGeneratedRustModules(outputDir, logger);
        }
        finally
        {
            TryDeleteDirectoryRecursive(tempSlangDir, logger);
        }

        logger.LogInfo(LogCategory.Output, $"Done. Shaders in bundle: {bundleEntries.Count}, variant limit skips: {variantFailures}.");
        return 0;
    }

    /// <summary>Builds <c>wgsl/…</c> output path from POSIX-style nested directory segments.</summary>
    internal static string CombineWgslOutputPath(string wgslRoot, string nestedDirPosix, string wgslFileName)
    {
        var parts = new List<string> { wgslRoot };
        foreach (string seg in nestedDirPosix.Split('/', StringSplitOptions.RemoveEmptyEntries))
            parts.Add(seg);
        parts.Add(wgslFileName);
        return Path.Combine(parts.ToArray());
    }

    /// <summary>Path for <c>include_str!</c> relative to the generated Rust file directory.</summary>
    internal static string RelativePathFromGeneratedToWgsl(string generatedRoot, string wgslAbsolutePath)
    {
        string rel = Path.GetRelativePath(generatedRoot, wgslAbsolutePath);
        return rel.Replace('\\', '/');
    }

    private static void TryDeleteFile(string path)
    {
        try
        {
            if (File.Exists(path))
                File.Delete(path);
        }
        catch
        {
            // ignored
        }
    }

    private static void TryDeleteDirectoryRecursive(string path, Logger logger)
    {
        try
        {
            if (Directory.Exists(path))
            {
                Directory.Delete(path, recursive: true);
                logger.LogInfo(LogCategory.Output, $"Removed directory {path}");
            }
        }
        catch (Exception ex)
        {
            logger.LogWarning(LogCategory.Output, $"Could not remove {path}: {ex.Message}");
        }
    }

    private static void RemoveOrphanGeneratedRustModules(string generatedRoot, Logger logger)
    {
        foreach (string path in Directory.GetFiles(generatedRoot, "*.rs"))
        {
            string name = Path.GetFileName(path);
            if (name is "mod.rs" or "wgsl_sources.rs" or "materials.rs")
                continue;
            try
            {
                File.Delete(path);
                logger.LogInfo(LogCategory.Rust, $"Removed legacy generated Rust file {path}");
            }
            catch (Exception ex)
            {
                logger.LogWarning(LogCategory.Rust, $"Could not delete legacy {path}: {ex.Message}");
            }
        }
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

    private static void WriteGeneratedModRust(string generatedRoot, Logger logger)
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("//! @generated by UnityShaderConverter — bundled WGSL sources and material stubs.");
        sb.AppendLine();
        sb.AppendLine("pub mod wgsl_sources;");
        sb.AppendLine("pub mod materials;");
        string path = Path.Combine(generatedRoot, "mod.rs");
        File.WriteAllText(path, sb.ToString());
        logger.LogInfo(LogCategory.Rust, $"Wrote {path}");
    }
}
