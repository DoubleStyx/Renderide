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

/// <summary>Orchestrates discovery, parsing, Slang/WGSL emission, and Rust codegen into the configured shader output directory (default <c>src/shaders/&lt;mod&gt;/</c>).</summary>
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

        string? unityCgIncludes = ResolveUnityCgIncludesOption(options, logger);
        if (unityCgIncludes is not null)
            logger.LogInfo(LogCategory.Startup, $"Unity CGIncludes (--cg-includes or env): {unityCgIncludes}");
        else
            logger.LogInfo(LogCategory.Startup, "Unity CGIncludes: auto (bundled UnityBuiltinCGIncludes if present, else walk from each shader path)");

        string slangcExe = SlangCompiler.ResolveExecutable(options.SlangcPath);
        logger.LogDebug(LogCategory.Startup, $"slangc executable: {slangcExe}");
        bool suppressSlangWarnings = options.ForceNoSuppressSlangWarnings
            ? false
            : options.ForceSuppressSlangWarnings || compilerConfig.SuppressSlangWarnings;
        var slangCompiler = new SlangCompiler(slangcExe, logger, suppressSlangWarnings);

        string tempSlangDir = Path.Combine(Path.GetTempPath(), "UnityShaderConverter", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempSlangDir);

        var bundleEntries = new List<ShaderBundleEntry>();
        var modNameOwners = new Dictionary<string, string>(StringComparer.Ordinal);
        var generationFailures = new List<ShaderGenerationFailure>();
        var preservedFailedModNames = new HashSet<string>(StringComparer.Ordinal);
        int variantFailures = 0;
        int specializationFallbackShaders = 0;
        bool loggedMissingSlangCgIncludes = false;

        try
        {
            foreach (string shaderPath in inputs)
            {
                string relForGlob = Path.GetRelativePath(renderideRoot, shaderPath).Replace('\\', '/');
                if (!ShaderLabAnalyzer.TryAnalyze(shaderPath, unityCgIncludes, out ShaderFileDocument? doc, out List<Diagnostic> diags, out List<string> errors))
                {
                    foreach (string e in errors)
                        logger.LogDebug(LogCategory.Parse, $"{shaderPath}: {e}");
                    foreach (Diagnostic d in diags.Where(d => (d.Kind & DiagnosticFlags.OnlyErrors) != 0))
                        logger.LogDebug(LogCategory.Parse, d.ToString());
                    var errParts = new List<string>();
                    errParts.AddRange(errors);
                    errParts.AddRange(diags.Where(d => (d.Kind & DiagnosticFlags.OnlyErrors) != 0).Select(d => d.ToString()));
                    generationFailures.Add(new ShaderGenerationFailure(
                        shaderPath,
                        null,
                        "Parse",
                        errParts.Count > 0 ? string.Join("; ", errParts) : "ShaderLab analysis failed"));
                    continue;
                }

                foreach (Diagnostic d in diags.Where(d => (d.Kind & DiagnosticFlags.Warning) != 0))
                    logger.LogDebug(LogCategory.Parse, d.ToString());

                if (doc!.Passes.Count == 0)
                {
                    generationFailures.Add(new ShaderGenerationFailure(
                        shaderPath,
                        RustEmitter.ModuleNameFromShaderName(doc.ShaderName),
                        "NoPasses",
                        "no code passes"));
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
                    logger.LogDebug(LogCategory.Variants, $"{shaderPath}: {ex.Message}");
                    variantFailures++;
                    generationFailures.Add(new ShaderGenerationFailure(
                        shaderPath,
                        RustEmitter.ModuleNameFromShaderName(doc.ShaderName),
                        "Variants",
                        ex.Message));
                    continue;
                }

                bool slangEligible = GlobMatcher.MatchesAny(relForGlob, compilerConfig.SlangEligibleGlobPatterns);
                if (slangEligible && compilerConfig.SlangExcludeGlobPatterns is { Count: > 0 } slangExcludes
                    && GlobMatcher.MatchesAny(relForGlob, slangExcludes))
                {
                    slangEligible = false;
                }
                string shaderDir = Path.GetDirectoryName(shaderPath) ?? ".";
                string? slangUnityCgIncludes = UnityCgIncludesResolver.ResolveForSlang(unityCgIncludes, shaderPath);
                if (slangEligible && !options.SkipSlang && slangUnityCgIncludes is null)
                {
                    if (!loggedMissingSlangCgIncludes)
                    {
                        logger.LogWarning(
                            LogCategory.SlangCompile,
                            "No Unity CGIncludes directory found for slangc (expected UnityBuiltinCGIncludes next to the converter, " +
                            "or --cg-includes / UNITY_SHADER_CONVERTER_CG_INCLUDES / UNITY_CG_INCLUDES, or repo walk from the shader). " +
                            "Unity #include lines will fail until this is fixed.");
                        loggedMissingSlangCgIncludes = true;
                    }
                }

                string modName = RustEmitter.ModuleNameFromShaderName(doc.ShaderName);
                if (!modNameOwners.TryGetValue(modName, out string? ownerPath))
                    modNameOwners[modName] = shaderPath;
                else if (!string.Equals(Path.GetFullPath(ownerPath), Path.GetFullPath(shaderPath), StringComparison.Ordinal))
                {
                    logger.LogDebug(
                        LogCategory.Rust,
                        $"Skipping `{shaderPath}`: Rust module name `{modName}` is already used by `{ownerPath}` (duplicate Unity shader name).");
                    generationFailures.Add(new ShaderGenerationFailure(
                        shaderPath,
                        modName,
                        "DuplicateMod",
                        $"Rust module `{modName}` already used by `{ownerPath}` (duplicate Unity shader name)"));
                    continue;
                }

                IReadOnlyList<SpecializationAxis> axes = SpecializationExtractor.Extract(doc, compilerConfig);
                var axisKeywords = new HashSet<string>(axes.Select(a => a.Keyword), StringComparer.Ordinal);
                List<string> firstVariantDefines = variants[0].Where(s => s.Length > 0).ToList();
                List<string> baselineDefines = firstVariantDefines.Where(d => !axisKeywords.Contains(d)).ToList();

                List<string> fallbackFullVariantDefines;
                if (axes.Count > 0)
                {
                    fallbackFullVariantDefines = VariantExpander
                        .GetFirstCartesianVariantDefinesIgnoringProductLimit(doc, variantConfig)
                        .Where(s => s.Length > 0)
                        .ToList();
                }
                else
                {
                    fallbackFullVariantDefines = firstVariantDefines;
                }

                string modDir = Path.Combine(outputDir, modName);
                Directory.CreateDirectory(modDir);

                bool allPassesOk = true;
                bool anyPassDroppedSpecialization = false;
                int firstFailedPassIndex = -1;
                string? firstPassFailureDetail = null;
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
                            slangUnityCgIncludes,
                            shaderDir,
                            slangCompiler,
                            shaderPath,
                            pi,
                            logger,
                            out string? passFailDetail);
                        if (outcome == PassCompileOutcome.Failed)
                        {
                            allPassesOk = false;
                            if (firstFailedPassIndex < 0)
                            {
                                firstFailedPassIndex = pi;
                                firstPassFailureDetail = passFailDetail;
                            }
                        }
                        else if (outcome == PassCompileOutcome.OkWithoutSpecialization)
                            anyPassDroppedSpecialization = true;
                    }
                    else if (!File.Exists(wgslPath) || new FileInfo(wgslPath).Length == 0)
                    {
                        logger.LogDebug(
                            LogCategory.Output,
                            $"No WGSL at {wgslPath} for `{doc.ShaderName}` pass {pi}; use --skip-slang only when files exist.");
                        allPassesOk = false;
                        if (firstFailedPassIndex < 0)
                        {
                            firstFailedPassIndex = pi;
                            firstPassFailureDetail = $"missing or empty WGSL at {wgslPath} (use --skip-slang only when files exist)";
                        }
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
                            logger.LogDebug(LogCategory.Output, $"{shaderPath} pass {pi}: WGSL post-process failed: {ex.Message}");
                            allPassesOk = false;
                            if (firstFailedPassIndex < 0)
                            {
                                firstFailedPassIndex = pi;
                                firstPassFailureDetail = $"WGSL post-process failed: {ex.Message}";
                            }
                        }
                    }
                }

                if (!allPassesOk)
                {
                    preservedFailedModNames.Add(modName);
                    string detail = firstFailedPassIndex >= 0
                        ? $"pass {firstFailedPassIndex}: {firstPassFailureDetail ?? "unknown error"}"
                        : "one or more passes failed";
                    generationFailures.Add(new ShaderGenerationFailure(shaderPath, modName, "CompileOrWgsl", detail));
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
                        logger.LogDebug(
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
            var preservedModNames = new HashSet<string>(StringComparer.Ordinal);
            foreach (ShaderBundleEntry e in bundleEntries)
                preservedModNames.Add(e.ModName);
            foreach (string m in preservedFailedModNames)
                preservedModNames.Add(m);

            string cleanShadersRoot = Path.GetFullPath(outputDir).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            if (string.Equals(Path.GetFileName(cleanShadersRoot), "generated", StringComparison.OrdinalIgnoreCase))
                cleanShadersRoot = Path.GetDirectoryName(cleanShadersRoot) ?? cleanShadersRoot;
            CleanLegacyBundleFiles(cleanShadersRoot, logger);
            RemoveStaleConverterShaderDirectories(outputDir, preservedModNames, logger);

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

            TryMergeShadersCrateRootMod(renderideRoot, outputDir, bundleEntries, preservedFailedModNames, logger);
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
        LogGenerationFailureReport(generationFailures, logger);
        return 0;
    }

    /// <summary>Resolves optional Unity CGIncludes directory from CLI or environment.</summary>
    private static string? ResolveUnityCgIncludesOption(ConverterOptions options, Logger logger)
    {
        string? raw = options.UnityCgIncludesDirectory;
        if (string.IsNullOrWhiteSpace(raw))
            raw = Environment.GetEnvironmentVariable("UNITY_SHADER_CONVERTER_CG_INCLUDES");
        if (string.IsNullOrWhiteSpace(raw))
            raw = Environment.GetEnvironmentVariable("UNITY_CG_INCLUDES");
        if (string.IsNullOrWhiteSpace(raw))
            return null;
        try
        {
            string full = Path.GetFullPath(raw.Trim());
            if (!File.Exists(Path.Combine(full, "UnityCG.cginc")))
            {
                logger.LogWarning(LogCategory.Startup, $"CGIncludes path missing UnityCG.cginc (ignored): {full}");
                return null;
            }

            return full;
        }
        catch (ArgumentException)
        {
            return null;
        }
    }

    /// <summary>Describes one shader that did not complete code generation; emitted in <see cref="LogGenerationFailureReport"/>.</summary>
    /// <param name="ShaderPath">Absolute or input path of the <c>.shader</c> file.</param>
    /// <param name="ModName">Rust module directory name when known; otherwise <c>null</c>.</param>
    /// <param name="Category">High-level failure bucket (Parse, Variants, etc.).</param>
    /// <param name="Detail">Human-readable explanation or tool stderr.</param>
    private sealed record ShaderGenerationFailure(string ShaderPath, string? ModName, string Category, string Detail);

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
        string? unityCgIncludesForSlang,
        string shaderSourceIncludeDir,
        SlangCompiler slangCompiler,
        string shaderPath,
        int passIndex,
        Logger logger,
        out string? failureDetail)
    {
        failureDetail = null;
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
                    unityCgIncludesForSlang,
                    shaderSourceIncludeDir,
                    slangCompiler,
                    shaderPath,
                    passIndex,
                    logger,
                    out string? err0))
                return axes.Count > 0 ? PassCompileOutcome.Ok : PassCompileOutcome.OkWithoutSpecialization;

            if (axes.Count > 0)
            {
                logger.LogDebug(
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
                        unityCgIncludesForSlang,
                        shaderSourceIncludeDir,
                        slangCompiler,
                        shaderPath,
                        passIndex,
                        logger,
                        out string? err1))
                    return PassCompileOutcome.OkWithoutSpecialization;
                failureDetail = err1 ?? err0;
            }
            else
                failureDetail = err0;

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
        string? unityCgIncludesForSlang,
        string shaderSourceIncludeDir,
        SlangCompiler slangCompiler,
        string shaderPath,
        int passIndex,
        Logger logger,
        out string? failureDetail)
    {
        failureDetail = null;
        string slangSource = SlangEmitter.EmitPassSlang(pass, baselineDefines, axes);
        File.WriteAllText(tempSlangPath, slangSource);
        logger.LogDebug(LogCategory.Slang, $"Transient Slang → slangc ({tempSlangPath})");
        if (!slangCompiler.TryCompileToWgsl(
                tempSlangPath,
                wgslPath,
                runtimeSlangDir,
                unityCgIncludesForSlang,
                shaderSourceIncludeDir,
                pass.VertexEntry!,
                pass.FragmentEntry!,
                baselineDefines,
                out string? err))
        {
            failureDetail = string.IsNullOrWhiteSpace(err) ? "slangc failed with no stderr" : err;
            logger.LogDebug(LogCategory.SlangCompile, $"{shaderPath} pass {passIndex}: slangc failed: {failureDetail}");
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
            failureDetail = $"WGSL post-process failed: {ex.Message}";
            logger.LogDebug(LogCategory.Output, $"{shaderPath} pass {passIndex}: {failureDetail}");
            return false;
        }

        logger.LogDebug(LogCategory.SlangCompile, $"WGSL {wgslPath}");
        return true;
    }

    /// <summary>
    /// When output is the crate <c>shaders/</c> directory, merges <c>pub mod &lt;shader&gt;;</c> lines into <c>shaders/mod.rs</c> between the UnityShaderConverter markers.
    /// </summary>
    private static void TryMergeShadersCrateRootMod(
        string renderideRoot,
        string outputDir,
        List<ShaderBundleEntry> bundleEntries,
        HashSet<string> preservedFailedModNames,
        Logger logger)
    {
        string expectedShadersRoot = Path.GetFullPath(
            Path.Combine(renderideRoot, "crates", "renderide", "src", "shaders"));
        if (!string.Equals(Path.GetFullPath(outputDir), expectedShadersRoot, StringComparison.OrdinalIgnoreCase))
            return;

        string rootModPath = Path.Combine(expectedShadersRoot, "mod.rs");
        try
        {
            string? existing = File.Exists(rootModPath) ? File.ReadAllText(rootModPath) : null;
            var successMods = bundleEntries.Select(e => e.ModName).ToList();
            var preservedList = preservedFailedModNames.ToList();
            File.WriteAllText(
                rootModPath,
                RustEmitter.MergeShadersRootModRs(existing, successMods, preservedList));
            logger.LogDebug(LogCategory.Rust, $"Merged {rootModPath}");
        }
        catch (Exception ex)
        {
            logger.LogWarning(LogCategory.Rust, $"Could not merge {rootModPath}: {ex.Message}");
        }
    }

    /// <summary>Writes a single conspicuous block listing every shader generation failure (after normal summary output).</summary>
    private static void LogGenerationFailureReport(IReadOnlyList<ShaderGenerationFailure> failures, Logger logger)
    {
        if (failures.Count == 0)
            return;

        logger.LogWarning(LogCategory.FailureReport, $"=== UnityShaderConverter: failed generations ({failures.Count}) ===");
        foreach (ShaderGenerationFailure f in failures)
        {
            string mod = f.ModName is { Length: > 0 } mn ? $" mod={mn}" : "";
            logger.LogWarning(
                LogCategory.FailureReport,
                $"  [{f.Category}]{mod} {f.ShaderPath}: {f.Detail}");
        }

        logger.LogWarning(
            LogCategory.FailureReport,
            "=== End failed generations — fix parse/variant/duplicate issues or slangc/WGSL errors above, then re-run ===");
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

    /// <summary>Removes converter-owned shader directories under <paramref name="shadersRoot"/> that are not in <paramref name="preservedModNames"/>.</summary>
    internal static void RemoveStaleConverterShaderDirectories(string shadersRoot, HashSet<string> preservedModNames, Logger logger)
    {
        if (!Directory.Exists(shadersRoot))
            return;
        foreach (string dir in Directory.GetDirectories(shadersRoot))
        {
            string name = Path.GetFileName(dir);
            if (name is null || preservedModNames.Contains(name))
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
