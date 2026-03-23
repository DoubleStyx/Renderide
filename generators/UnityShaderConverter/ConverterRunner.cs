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

/// <summary>Orchestrates discovery, parsing, Slang/WGSL emission, and Rust codegen into the configured shader output directory (default <c>src/shaders/generated/&lt;mod&gt;/</c>).</summary>
public static class ConverterRunner
{
    /// <summary>Runs the full pipeline for the given options and returns a process exit code.</summary>
    public static int Run(ConverterOptions options, Logger logger)
    {
        options.DetermineDefaultPaths();
        string outputDir = Path.GetFullPath(options.OutputDirectory ?? ".");
        Directory.CreateDirectory(outputDir);

        string renderideRoot = FindRenderideRoot(outputDir);
        logger.LogInfo(LogCategory.Startup, $"Renderide root: {renderideRoot}");
        logger.LogInfo(LogCategory.Startup, $"Shader output directory: {outputDir}");

        string exeDir = AppContext.BaseDirectory;
        var defaults = DefaultCompilerConfig.LoadFromOutputDirectory(exeDir);
        var compilerConfig = ConfigLoader.MergeCompilerConfig(defaults, options.CompilerConfigPath);
        VariantConfigModel? variantConfig = ConfigLoader.LoadVariantConfig(options.VariantConfigPath);

        string runtimeSlangDir = Path.Combine(exeDir, "runtime_slang");
        if (!Directory.Exists(runtimeSlangDir))
        {
            logger.LogInfo(LogCategory.Startup, $"runtime_slang not next to executable ({runtimeSlangDir}); using source tree path.");
            runtimeSlangDir = Path.GetFullPath(Path.Combine(renderideRoot, "generators", "UnityShaderConverter", "runtime_slang"));
        }

        IReadOnlyList<string> inputs = ShaderDiscovery.Enumerate(options.InputDirectories ?? Array.Empty<string>());
        logger.LogInfo(LogCategory.Startup, $"Discovered {inputs.Count} .shader files.");

        string slangcExe = SlangCompiler.ResolveExecutable(options.SlangcPath);
        logger.LogDebug(LogCategory.Startup, $"slangc executable: {slangcExe}");
        var slangCompiler = new SlangCompiler(slangcExe, logger);

        string tempSlangDir = Path.Combine(Path.GetTempPath(), "UnityShaderConverter", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempSlangDir);

        var bundleEntries = new List<ShaderBundleEntry>();
        var modNameOwners = new Dictionary<string, string>(StringComparer.Ordinal);
        int variantFailures = 0;
        int specializationFallbackShaders = 0;

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

                if (doc!.Passes.Count == 0)
                {
                    logger.LogWarning(LogCategory.Parse, $"{shaderPath}: no code passes; skipping.");
                    continue;
                }

                VariantExpander.MultiCompileAnalysis multiAnalysis = VariantExpander.AnalyzeMultiCompileGroups(doc);
                if (multiAnalysis.Groups.Count > 0)
                {
                    logger.LogDebug(
                        LogCategory.Variants,
                        $"{shaderPath}: multi_compile groups={multiAnalysis.Groups.Count}, Cartesian product={multiAnalysis.Product}");
                }

                IReadOnlyList<IReadOnlyList<string>> variants;
                try
                {
                    variants = VariantExpander.Expand(doc, compilerConfig, variantConfig);
                }
                catch (InvalidOperationException ex)
                {
                    logger.LogWarning(LogCategory.Variants, $"{shaderPath}: {ex.Message}");
                    variantFailures++;
                    continue;
                }

                bool slangEligible = GlobMatcher.MatchesAny(relForGlob, compilerConfig.SlangEligibleGlobPatterns);
                string shaderDir = Path.GetDirectoryName(shaderPath) ?? ".";

                string modName = RustEmitter.ModuleNameFromShaderName(doc.ShaderName);
                if (!modNameOwners.TryGetValue(modName, out string? ownerPath))
                    modNameOwners[modName] = shaderPath;
                else if (!string.Equals(Path.GetFullPath(ownerPath), Path.GetFullPath(shaderPath), StringComparison.Ordinal))
                {
                    logger.LogWarning(
                        LogCategory.Rust,
                        $"Skipping `{shaderPath}`: Rust module name `{modName}` is already used by `{ownerPath}` (duplicate Unity shader name).");
                    continue;
                }

                IReadOnlyList<SpecializationAxis> axes = SpecializationExtractor.Extract(doc, compilerConfig);
                var axisKeywords = new HashSet<string>(axes.Select(a => a.Keyword), StringComparer.Ordinal);
                List<string> firstVariantDefines = variants[0].Where(s => s.Length > 0).ToList();
                List<string> baselineDefines = firstVariantDefines.Where(d => !axisKeywords.Contains(d)).ToList();

                List<string> fallbackFullVariantDefines;
                if (axes.Count > 0)
                {
                    try
                    {
                        fallbackFullVariantDefines = VariantExpander
                            .GetFirstCartesianVariantDefines(doc, compilerConfig, variantConfig)
                            .Where(s => s.Length > 0)
                            .ToList();
                    }
                    catch (InvalidOperationException ex)
                    {
                        logger.LogWarning(LogCategory.Variants, $"{shaderPath}: fallback variant list unavailable: {ex.Message}");
                        fallbackFullVariantDefines = new List<string>();
                    }
                }
                else
                {
                    fallbackFullVariantDefines = firstVariantDefines;
                }

                string modDir = Path.Combine(outputDir, modName);
                Directory.CreateDirectory(modDir);

                bool allPassesOk = true;
                bool anyPassDroppedSpecialization = false;
                for (int pi = 0; pi < doc.Passes.Count; pi++)
                {
                    ShaderPassDocument pass = doc.Passes[pi];
                    string wgslPath = Path.Combine(modDir, RustEmitter.WgslPassFileName(pass, pi));

                    if (!options.SkipSlang && slangEligible)
                    {
                        PassCompileOutcome outcome = TryCompilePassWithOptionalFallback(
                            doc,
                            pass,
                            baselineDefines,
                            axes,
                            fallbackFullVariantDefines,
                            tempSlangDir,
                            wgslPath,
                            runtimeSlangDir,
                            shaderDir,
                            slangCompiler,
                            shaderPath,
                            pi,
                            logger);
                        if (outcome == PassCompileOutcome.Failed)
                            allPassesOk = false;
                        else if (outcome == PassCompileOutcome.OkWithoutSpecialization)
                            anyPassDroppedSpecialization = true;
                    }
                    else if (!File.Exists(wgslPath) || new FileInfo(wgslPath).Length == 0)
                    {
                        logger.LogWarning(
                            LogCategory.Output,
                            $"No WGSL at {wgslPath} for `{doc.ShaderName}` pass {pi}; use --skip-slang only when files exist.");
                        allPassesOk = false;
                    }
                    else if (options.SkipSlang)
                    {
                        try
                        {
                            string wgsl = File.ReadAllText(wgslPath);
                            if (!wgsl.Contains("Material block (UnityShaderConverter)", StringComparison.Ordinal))
                            {
                                wgsl = WgslMaterialUniformInjector.PrependMaterialBlock(wgsl, doc.Properties);
                                File.WriteAllText(wgslPath, wgsl);
                            }
                        }
                        catch (Exception ex)
                        {
                            logger.LogWarning(LogCategory.Output, $"{shaderPath} pass {pi}: WGSL post-process failed: {ex.Message}");
                            allPassesOk = false;
                        }
                    }
                }

                if (!allPassesOk)
                {
                    logger.LogWarning(LogCategory.Rust, $"Skipping shader module `{modName}` for `{doc.ShaderName}` (missing WGSL).");
                    TryDeleteDirectoryRecursive(modDir, logger);
                    continue;
                }

                var vertexLayouts = new List<PassVertexLayout>();
                for (int pi = 0; pi < doc.Passes.Count; pi++)
                {
                    string wgslPath = Path.Combine(modDir, RustEmitter.WgslPassFileName(doc.Passes[pi], pi));
                    string wgslText = File.ReadAllText(wgslPath);
                    string vertEntry = doc.Passes[pi].VertexEntry ?? "";
                    if (!WgslVertexLayoutExtractor.TryExtract(wgslText, vertEntry, out PassVertexLayout vLayout, out string? vErr))
                    {
                        logger.LogWarning(
                            LogCategory.Rust,
                            $"{shaderPath} pass {pi}: vertex layout extraction failed ({vErr}); emitting empty `VERTEX_BUFFER_LAYOUTS_PASS{pi}`.");
                        vertexLayouts.Add(PassVertexLayout.Empty);
                    }
                    else
                        vertexLayouts.Add(vLayout);
                }

                IReadOnlyList<SpecializationAxis> rustAxes =
                    anyPassDroppedSpecialization ? Array.Empty<SpecializationAxis>() : axes;

                string sourceComment = Path.GetRelativePath(renderideRoot, doc.SourcePath);
                bundleEntries.Add(new ShaderBundleEntry(
                    modName,
                    doc,
                    sourceComment.Replace('\\', '/'),
                    doc.Passes.Count,
                    rustAxes,
                    vertexLayouts));
                if (anyPassDroppedSpecialization)
                    specializationFallbackShaders++;
                logger.LogDebug(
                    LogCategory.Rust,
                    $"Queued shader `{modName}` ({doc.Passes.Count} pass(es), {rustAxes.Count} specialization axis/axes in Rust output).");
            }

            bundleEntries.Sort((a, b) => string.CompareOrdinal(a.ModName, b.ModName));
            var currentMods = bundleEntries.Select(e => e.ModName).ToHashSet(StringComparer.Ordinal);

            string cleanShadersRoot = Path.GetFullPath(outputDir).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            if (string.Equals(Path.GetFileName(cleanShadersRoot), "generated", StringComparison.OrdinalIgnoreCase))
                cleanShadersRoot = Path.GetDirectoryName(cleanShadersRoot) ?? cleanShadersRoot;
            CleanLegacyBundleFiles(cleanShadersRoot, logger);
            RemoveStaleConverterShaderDirectories(outputDir, currentMods, logger);

            foreach (ShaderBundleEntry e in bundleEntries)
            {
                string modDir = Path.Combine(outputDir, e.ModName);
                Directory.CreateDirectory(modDir);
                File.WriteAllText(Path.Combine(modDir, "mod.rs"), RustEmitter.EmitShaderCrateModRs());
                File.WriteAllText(
                    Path.Combine(modDir, "material.rs"),
                    RustEmitter.EmitShaderMaterialRs(e.Document, e.SourceComment, e.PassCount, e.Axes, e.VertexLayouts));
                logger.LogDebug(LogCategory.Rust, $"Wrote {modDir}");
            }

            TryWriteGeneratedBundleModAndMergeRoot(renderideRoot, outputDir, bundleEntries, logger);
        }
        finally
        {
            TryDeleteDirectoryRecursive(tempSlangDir, logger);
        }

        int totalPasses = bundleEntries.Sum(static e => e.PassCount);
        int modulesWithSpec = bundleEntries.Count(static e => e.Axes.Count > 0);
        logger.LogInfo(
            LogCategory.Output,
            $"[UnityShaderConverter] modules_written={bundleEntries.Count} total_passes={totalPasses} " +
            $"specialization_active={modulesWithSpec} specialization_fallback_shaders={specializationFallbackShaders} " +
            $"variant_limit_skips={variantFailures} slangc={slangcExe}");
        return 0;
    }

    private sealed record ShaderBundleEntry(
        string ModName,
        ShaderFileDocument Document,
        string SourceComment,
        int PassCount,
        IReadOnlyList<SpecializationAxis> Axes,
        IReadOnlyList<PassVertexLayout> VertexLayouts);

    private enum PassCompileOutcome
    {
        Failed,
        Ok,
        OkWithoutSpecialization,
    }

    private static PassCompileOutcome TryCompilePassWithOptionalFallback(
        ShaderFileDocument shaderFile,
        ShaderPassDocument pass,
        List<string> baselineDefines,
        IReadOnlyList<SpecializationAxis> axes,
        List<string> fullFirstVariantDefines,
        string tempSlangDir,
        string wgslPath,
        string runtimeSlangDir,
        string shaderSourceIncludeDir,
        SlangCompiler slangCompiler,
        string shaderPath,
        int passIndex,
        Logger logger)
    {
        string tempSlangPath = Path.Combine(tempSlangDir, "compile.slang");
        try
        {
            if (TryCompileOnce(
                    shaderFile,
                    pass,
                    baselineDefines,
                    axes,
                    tempSlangPath,
                    wgslPath,
                    runtimeSlangDir,
                    shaderSourceIncludeDir,
                    slangCompiler,
                    shaderPath,
                    passIndex,
                    logger))
                return axes.Count > 0 ? PassCompileOutcome.Ok : PassCompileOutcome.OkWithoutSpecialization;

            if (axes.Count > 0)
            {
                logger.LogWarning(
                    LogCategory.SlangCompile,
                    $"{shaderPath} pass {passIndex}: retrying without specialization injection (full first-variant defines only).");
                if (TryCompileOnce(
                        shaderFile,
                        pass,
                        fullFirstVariantDefines,
                        Array.Empty<SpecializationAxis>(),
                        tempSlangPath,
                        wgslPath,
                        runtimeSlangDir,
                        shaderSourceIncludeDir,
                        slangCompiler,
                        shaderPath,
                        passIndex,
                        logger))
                    return PassCompileOutcome.OkWithoutSpecialization;
            }

            return PassCompileOutcome.Failed;
        }
        finally
        {
            TryDeleteFile(tempSlangPath);
        }
    }

    private static bool TryCompileOnce(
        ShaderFileDocument shaderFile,
        ShaderPassDocument pass,
        List<string> baselineDefines,
        IReadOnlyList<SpecializationAxis> axes,
        string tempSlangPath,
        string wgslPath,
        string runtimeSlangDir,
        string shaderSourceIncludeDir,
        SlangCompiler slangCompiler,
        string shaderPath,
        int passIndex,
        Logger logger)
    {
        string slangSource = SlangEmitter.EmitPassSlang(pass, baselineDefines, axes);
        File.WriteAllText(tempSlangPath, slangSource);
        logger.LogDebug(LogCategory.Slang, $"Transient Slang → slangc ({tempSlangPath})");
        if (!slangCompiler.TryCompileToWgsl(
                tempSlangPath,
                wgslPath,
                runtimeSlangDir,
                shaderSourceIncludeDir,
                pass.VertexEntry!,
                pass.FragmentEntry!,
                baselineDefines,
                out string? err))
        {
            logger.LogWarning(LogCategory.SlangCompile, $"{shaderPath} pass {passIndex}: slangc failed: {err}");
            return false;
        }

        try
        {
            string wgsl = File.ReadAllText(wgslPath);
            wgsl = WgslMaterialUniformInjector.PrependMaterialBlock(wgsl, shaderFile.Properties);
            File.WriteAllText(wgslPath, wgsl);
        }
        catch (Exception ex)
        {
            logger.LogWarning(LogCategory.Output, $"{shaderPath} pass {passIndex}: WGSL post-process failed: {ex.Message}");
            return false;
        }

        logger.LogDebug(LogCategory.SlangCompile, $"WGSL {wgslPath}");
        return true;
    }

    private static void TryWriteGeneratedBundleModAndMergeRoot(
        string renderideRoot,
        string outputDir,
        List<ShaderBundleEntry> bundleEntries,
        Logger logger)
    {
        string expectedGen = Path.GetFullPath(
            Path.Combine(renderideRoot, "crates", "renderide", "src", "shaders", "generated"));
        if (!string.Equals(Path.GetFullPath(outputDir), expectedGen, StringComparison.OrdinalIgnoreCase))
            return;

        string genModPath = Path.Combine(outputDir, "mod.rs");
        try
        {
            File.WriteAllText(
                genModPath,
                RustEmitter.EmitGeneratedShadersModRs(bundleEntries.Select(e => e.ModName).ToList()));
            logger.LogDebug(LogCategory.Rust, $"Wrote {genModPath}");
        }
        catch (Exception ex)
        {
            logger.LogWarning(LogCategory.Rust, $"Could not write {genModPath}: {ex.Message}");
            return;
        }

        string shadersRoot = Path.Combine(renderideRoot, "crates", "renderide", "src", "shaders");
        string rootModPath = Path.Combine(shadersRoot, "mod.rs");
        try
        {
            string? existing = File.Exists(rootModPath) ? File.ReadAllText(rootModPath) : null;
            File.WriteAllText(rootModPath, RustEmitter.MergeShadersRootModRs(existing));
            logger.LogDebug(LogCategory.Rust, $"Merged {rootModPath}");
        }
        catch (Exception ex)
        {
            logger.LogWarning(LogCategory.Rust, $"Could not merge {rootModPath}: {ex.Message}");
        }
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

    private static void CleanLegacyBundleFiles(string shadersRoot, Logger logger)
    {
        foreach (string legacy in new[] { "wgsl_sources.rs", "materials.rs" })
        {
            string p = Path.Combine(shadersRoot, legacy);
            if (!File.Exists(p))
                continue;
            try
            {
                File.Delete(p);
                logger.LogInfo(LogCategory.Rust, $"Removed legacy {p}");
            }
            catch (Exception ex)
            {
                logger.LogWarning(LogCategory.Rust, $"Could not delete legacy {p}: {ex.Message}");
            }
        }
    }

    private static void RemoveStaleConverterShaderDirectories(string shadersRoot, HashSet<string> currentMods, Logger logger)
    {
        if (!Directory.Exists(shadersRoot))
            return;
        foreach (string dir in Directory.GetDirectories(shadersRoot))
        {
            string name = Path.GetFileName(dir);
            if (name is null || currentMods.Contains(name))
                continue;
            string marker = Path.Combine(dir, "mod.rs");
            if (!File.Exists(marker))
                continue;
            string head = File.ReadAllText(marker);
            if (!head.Contains("UnityShaderConverter", StringComparison.Ordinal))
                continue;
            TryDeleteDirectoryRecursive(dir, logger);
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
}
