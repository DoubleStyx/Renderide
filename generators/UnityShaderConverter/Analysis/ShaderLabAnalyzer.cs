using System.Linq;
using UnityShaderParser.Common;
using UnityShaderParser.HLSL.PreProcessor;
using UnityShaderParser.ShaderLab;

namespace UnityShaderConverter.Analysis;

/// <summary>Parses ShaderLab via UnityShaderParser and builds <see cref="ShaderFileDocument"/>.</summary>
public static class ShaderLabAnalyzer
{
    /// <summary>
    /// Prefix for the single error returned when a shader is skipped because it uses <c>#pragma surface</c>.
    /// Used by <see cref="IsSurfaceShaderExclusion"/> for failure categorization.
    /// </summary>
    public const string SurfaceShaderNotSupportedPrefix = "Surface shader not supported:";

    /// <summary>
    /// Exact analyzer message when a pass uses <c>#pragma geometry</c>. Used by <see cref="IsGeometryShaderExclusion"/>.
    /// </summary>
    public const string GeometryShaderNotSupportedMessage =
        "Pass uses #pragma geometry which is not supported by UnityShaderConverter yet.";

    /// <summary>Shared porting guidance appended to every surface-shader exclusion message.</summary>
    private const string SurfaceShaderPortingGuidance =
        "UnityShaderConverter does not expand surface shaders. In Unity, use Compile and show code, copy the HLSL for the pass you need into a shader that uses only " +
        "`#pragma vertex` and `#pragma fragment` (remove `#pragma surface` from ShaderLab), or maintain a hand-written port.";

    /// <summary>Parses a single <c>.shader</c> file using auto-detected Unity <c>CGIncludes</c>.</summary>
    public static bool TryAnalyze(string shaderPath, out ShaderFileDocument? document, out List<Diagnostic> diagnostics, out List<string> errors) =>
        TryAnalyze(shaderPath, null, out document, out diagnostics, out errors);

    /// <summary>Parses a single <c>.shader</c> file.</summary>
    /// <param name="unityCgIncludesDirectory">
    /// Optional directory that contains <c>UnityCG.cginc</c>, prepended to the include search list (CLI <c>--cg-includes</c> or <c>UNITY_SHADER_CONVERTER_CG_INCLUDES</c>).
    /// </param>
    public static bool TryAnalyze(
        string shaderPath,
        string? unityCgIncludesDirectory,
        out ShaderFileDocument? document,
        out List<Diagnostic> diagnostics,
        out List<string> errors)
    {
        diagnostics = new List<Diagnostic>();
        errors = new List<string>();
        document = null;
        string source = File.ReadAllText(shaderPath);

        // Surface shaders use Unity-only syntax inside CGPROGRAM; parsing embedded HLSL produces long cascades of
        // HLSLParsing diagnostics that are irrelevant once we know conversion is unsupported.
        if (PragmaParser.HasSurfacePragma(source))
        {
            errors.Add(
                $"{SurfaceShaderNotSupportedPrefix} shader source contains `#pragma surface`. {SurfaceShaderPortingGuidance}");
            return false;
        }

        string basePath = Path.GetDirectoryName(shaderPath) ?? ".";
        string fileName = Path.GetFileName(shaderPath);
        IPreProcessorIncludeResolver includeResolver = CreateIncludeResolver(shaderPath, unityCgIncludesDirectory);
        var config = new ShaderLabParserConfig
        {
            BasePath = basePath,
            FileName = fileName,
            ParseEmbeddedHLSL = true,
            IncludeProgramBlockPreamble = false,
            ThrowExceptionOnError = false,
            IncludeResolver = includeResolver,
            Defines = new Dictionary<string, string>(StringComparer.Ordinal)
            {
                // Matches UnityShaderParser.Tests embedded-HLSL setup so built-in .cginc branches resolve.
                ["SHADER_API_D3D11"] = "1",
            },
        };

        ShaderNode root = ShaderParser.ParseUnityShader(source, config, out var lexerParserDiags);
        diagnostics.AddRange(lexerParserDiags);
        if (root.SubShaders is null || root.SubShaders.Count == 0)
        {
            errors.Add("Shader has no SubShader blocks.");
            return false;
        }

        int totalSubShaders = root.SubShaders.Count;
        var analyzerWarnings = new List<string>();
        if (totalSubShaders > 1)
        {
            analyzerWarnings.Add(
                $"Shader defines {totalSubShaders} SubShader blocks; UnityShaderConverter uses only the first SubShader.");
        }

        SubShaderNode sub0 = root.SubShaders[0];
        if (TryBuildSurfaceShaderExclusionError(sub0, out string? surfaceErr))
        {
            errors.Add(surfaceErr!);
            return false;
        }

        var properties = ExtractProperties(root.Properties ?? new List<ShaderPropertyNode>());
        var passes = new List<ShaderPassDocument>();
        var multiCompiles = new List<string>();
        var subShaderTags = ExtractSubShaderTags(sub0);
        int codePassIndex = 0;
        foreach (ShaderPassNode pass in sub0.Passes ?? new List<ShaderPassNode>())
        {
            if (pass is ShaderCodePassNode codePass)
            {
                if (!TryExtractPass(
                        codePass,
                        codePassIndex,
                        subShaderTags,
                        programSource => ExtractPragmaLines(programSource),
                        out ShaderPassDocument? passDoc,
                        out string? passErr))
                {
                    if (passErr is not null)
                        errors.Add(passErr);
                    return false;
                }

                passes.Add(passDoc!);
                foreach (string line in ExtractMultiCompileLines(passDoc!.ProgramSource))
                    multiCompiles.Add(line);
                codePassIndex++;
            }
        }

        if (passes.Count == 0)
        {
            if (!TryExtractImplicitSubShaderProgramPass(
                    sub0,
                    subShaderTags,
                    programSource => ExtractPragmaLines(programSource),
                    out ShaderPassDocument? implicitPass,
                    out string? implicitErr))
            {
                errors.Add(implicitErr ?? "No CGPROGRAM/HLSLPROGRAM passes found in first SubShader.");
                return false;
            }

            passes.Add(implicitPass!);
            foreach (string line in ExtractMultiCompileLines(implicitPass!.ProgramSource))
                multiCompiles.Add(line);
        }

        document = new ShaderFileDocument
        {
            SourcePath = Path.GetFullPath(shaderPath),
            ShaderName = root.Name ?? Path.GetFileNameWithoutExtension(shaderPath),
            Properties = properties,
            SubShaderTags = subShaderTags,
            Passes = passes,
            MultiCompilePragmas = DeduplicateSorted(multiCompiles),
            AnalyzerWarnings = analyzerWarnings,
            TotalSubShaderCount = totalSubShaders,
        };
        return true;
    }

    /// <summary>True when <paramref name="errors"/> includes a surface-shader exclusion (see <see cref="SurfaceShaderNotSupportedPrefix"/>).</summary>
    public static bool IsSurfaceShaderExclusion(IReadOnlyList<string> errors) =>
        errors.Any(static e => e.StartsWith(SurfaceShaderNotSupportedPrefix, StringComparison.Ordinal));

    /// <summary>True when <paramref name="errors"/> includes the geometry-stage exclusion message.</summary>
    public static bool IsGeometryShaderExclusion(IReadOnlyList<string> errors) =>
        errors.Any(static e => string.Equals(e, GeometryShaderNotSupportedMessage, StringComparison.Ordinal));

    /// <summary>
    /// Surface- or geometry-shader skips are expected for many vendored Unity shaders; logs treat them as trace-only.
    /// </summary>
    public static bool IsExpectedUnsupportedShaderExclusion(IReadOnlyList<string> errors) =>
        IsSurfaceShaderExclusion(errors) || IsGeometryShaderExclusion(errors);

    /// <summary>
    /// When true, the parse failure line is logged at trace instead of error (surface or geometry exclusion only).
    /// </summary>
    public static bool IsParseFailureLoggedAtTraceOnly(string errorLine) =>
        errorLine.StartsWith(SurfaceShaderNotSupportedPrefix, StringComparison.Ordinal) ||
        string.Equals(errorLine, GeometryShaderNotSupportedMessage, StringComparison.Ordinal);

    /// <summary>
    /// Builds an include resolver: optional override, then <c>UnityBuiltinCGIncludes</c> next to the app, then repo walk from <paramref name="shaderPath"/>.
    /// </summary>
    private static IPreProcessorIncludeResolver CreateIncludeResolver(string shaderPath, string? unityCgIncludesDirectory)
    {
        IReadOnlyList<string> paths = UnityCgIncludesResolver.GetSearchDirectories(unityCgIncludesDirectory, shaderPath);
        if (paths.Count == 0)
            return new DefaultPreProcessorIncludeResolver();
        return new DefaultPreProcessorIncludeResolver(paths.ToList());
    }

    /// <summary>
    /// When any code pass in the first subshader contains <c>#pragma surface</c>, conversion is skipped for the whole file.
    /// </summary>
    /// <remarks>
    /// Unity allows <c>CGPROGRAM</c> directly under <c>SubShader</c> (no <c>Pass</c> wrapper); those blocks live in
    /// <see cref="SubShaderNode.ProgramBlocks"/> and must be checked here too.
    /// </remarks>
    private static bool TryBuildSurfaceShaderExclusionError(SubShaderNode sub0, out string? error)
    {
        foreach (HLSLProgramBlock block in sub0.ProgramBlocks ?? new List<HLSLProgramBlock>())
        {
            if (PragmaParser.HasSurfacePragma(block.CodeWithoutIncludes))
            {
                error =
                    $"{SurfaceShaderNotSupportedPrefix} subshader-level CGPROGRAM contains `#pragma surface`. {SurfaceShaderPortingGuidance}";
                return true;
            }
        }

        int passIndex = 0;
        foreach (ShaderPassNode pass in sub0.Passes ?? new List<ShaderPassNode>())
        {
            if (pass is not ShaderCodePassNode codePass)
                continue;
            HLSLProgramBlock? blockNullable = codePass.ProgramBlock;
            if (blockNullable is null)
            {
                passIndex++;
                continue;
            }

            string program = blockNullable.Value.CodeWithoutIncludes;
            if (PragmaParser.HasSurfacePragma(program))
            {
                error =
                    $"{SurfaceShaderNotSupportedPrefix} pass {passIndex} contains `#pragma surface`. {SurfaceShaderPortingGuidance}";
                return true;
            }

            passIndex++;
        }

        error = null;
        return false;
    }

    /// <summary>
    /// Unity sometimes places <c>CGPROGRAM</c> directly under <c>SubShader</c>; the parser stores that in
    /// <see cref="SubShaderNode.ProgramBlocks"/> instead of a <see cref="ShaderCodePassNode"/>.
    /// </summary>
    private static bool TryExtractImplicitSubShaderProgramPass(
        SubShaderNode sub0,
        IReadOnlyDictionary<string, string> subShaderTags,
        Func<string, IReadOnlyList<string>> extractPragmas,
        out ShaderPassDocument? passDoc,
        out string? error)
    {
        passDoc = null;
        error = null;
        if (sub0.ProgramBlocks is null || sub0.ProgramBlocks.Count == 0)
            return false;

        HLSLProgramBlock block = sub0.ProgramBlocks[0];
        return TryExtractPassFromProgramBlock(
            block,
            passIndex: 0,
            subShaderTags,
            sub0.Commands ?? new List<ShaderLabCommandNode>(),
            extractPragmas,
            passName: null,
            out passDoc,
            out error);
    }

    private static IReadOnlyList<string> DeduplicateSorted(List<string> lines)
    {
        lines.Sort(StringComparer.Ordinal);
        var unique = new List<string>();
        string? prev = null;
        foreach (string line in lines)
        {
            if (line == prev)
                continue;
            unique.Add(line);
            prev = line;
        }

        return unique;
    }

    private static bool TryExtractPass(
        ShaderCodePassNode codePass,
        int passIndex,
        IReadOnlyDictionary<string, string> subShaderTags,
        Func<string, IReadOnlyList<string>> extractPragmas,
        out ShaderPassDocument? passDoc,
        out string? error)
    {
        passDoc = null;
        error = null;
        HLSLProgramBlock? blockNullable = codePass.ProgramBlock;
        if (blockNullable is null)
        {
            error = "Pass has no program block.";
            return false;
        }

        string? passName = null;
        foreach (ShaderLabCommandNode? cmd in codePass.Commands ?? new List<ShaderLabCommandNode>())
        {
            if (cmd is ShaderLabCommandNameNode nameNode && !string.IsNullOrEmpty(nameNode.Name))
            {
                passName = nameNode.Name;
                break;
            }
        }

        List<ShaderLabCommandNode> cmdList = codePass.Commands ?? new List<ShaderLabCommandNode>();
        return TryExtractPassFromProgramBlock(
            blockNullable.Value,
            passIndex,
            subShaderTags,
            cmdList,
            extractPragmas,
            passName,
            out passDoc,
            out error);
    }

    /// <summary>
    /// Builds a <see cref="ShaderPassDocument"/> from a parsed <see cref="HLSLProgramBlock"/> plus ShaderLab commands for fixed-function state.
    /// </summary>
    private static bool TryExtractPassFromProgramBlock(
        HLSLProgramBlock block,
        int passIndex,
        IReadOnlyDictionary<string, string> subShaderTags,
        IReadOnlyList<ShaderLabCommandNode> commandNodesForRenderState,
        Func<string, IReadOnlyList<string>> extractPragmas,
        string? passName,
        out ShaderPassDocument? passDoc,
        out string? error)
    {
        passDoc = null;
        error = null;
        string program = block.CodeWithoutIncludes;

        if (PragmaParser.HasGeometryStage(program))
        {
            error = GeometryShaderNotSupportedMessage;
            return false;
        }

        if (!PragmaParser.TryGetVertexEntry(program, out string vert))
        {
            error = $"Pass {passIndex} must declare #pragma vertex for this converter.";
            return false;
        }

        if (!PragmaParser.TryGetFragmentEntry(program, out string frag))
        {
            error = $"Pass {passIndex} must declare #pragma fragment for this converter.";
            return false;
        }

        float? pragmaTarget = null;
        if (PragmaParser.TryGetShaderTarget(program, out float tgt))
            pragmaTarget = tgt;

        var cmdList = commandNodesForRenderState as List<ShaderLabCommandNode> ?? commandNodesForRenderState.ToList();
        passDoc = new ShaderPassDocument
        {
            PassName = passName,
            PassIndex = passIndex,
            ProgramSource = program,
            Pragmas = extractPragmas(program),
            VertexEntry = vert,
            FragmentEntry = frag,
            RenderStateSummary = RenderStateFormatter.Summarize(cmdList),
            FixedFunctionState = RenderStateExtractor.Extract(cmdList, subShaderTags),
            PragmaShaderTarget = pragmaTarget,
        };
        return true;
    }

    private static List<string> ExtractPragmaLines(string program)
    {
        var list = new List<string>();
        foreach (string line in program.Split('\n'))
        {
            string t = line.TrimStart();
            if (t.StartsWith("#pragma", StringComparison.Ordinal))
                list.Add(t);
        }

        return list;
    }

    private static Dictionary<string, string> ExtractSubShaderTags(SubShaderNode sub)
    {
        var dict = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (ShaderLabCommandNode? cmd in sub.Commands ?? new List<ShaderLabCommandNode>())
        {
            if (cmd is ShaderLabCommandTagsNode tagsNode && tagsNode.Tags is not null)
            {
                foreach (KeyValuePair<string, string> kv in tagsNode.Tags)
                    dict[kv.Key] = kv.Value;
            }
        }

        return dict;
    }

    private static List<ShaderPropertyRecord> ExtractProperties(List<ShaderPropertyNode> nodes)
    {
        var list = new List<ShaderPropertyRecord>();
        foreach (ShaderPropertyNode prop in nodes)
        {
            list.Add(new ShaderPropertyRecord
            {
                Name = prop.Uniform ?? "_unnamed",
                DisplayLabel = prop.Name ?? "",
                Kind = prop.Kind,
                Range = prop.RangeMinMax,
                DefaultSummary = SummarizeDefault(prop),
            });
        }

        return list;
    }

    private static string SummarizeDefault(ShaderPropertyNode prop)
    {
        if (prop.Value is ShaderPropertyValueFloatNode f)
            return f.Number.ToString(System.Globalization.CultureInfo.InvariantCulture);
        if (prop.Value is ShaderPropertyValueIntegerNode i)
            return i.Number.ToString(System.Globalization.CultureInfo.InvariantCulture);
        if (prop.Value is ShaderPropertyValueVectorNode v)
            return v.HasWChannel
                ? $"({v.Vector.x},{v.Vector.y},{v.Vector.z},{v.Vector.w})"
                : $"({v.Vector.x},{v.Vector.y},{v.Vector.z})";
        if (prop.Value is ShaderPropertyValueColorNode c)
            return c.HasAlphaChannel
                ? $"({c.Color.r},{c.Color.g},{c.Color.b},{c.Color.a})"
                : $"({c.Color.r},{c.Color.g},{c.Color.b})";
        if (prop.Value is ShaderPropertyValueTextureNode t)
            return t.TextureName ?? "";
        return "";
    }

    private static IEnumerable<string> ExtractMultiCompileLines(string program)
    {
        foreach (string line in program.Split('\n'))
        {
            string trimmed = line.TrimStart();
            if (trimmed.StartsWith("#pragma multi_compile", StringComparison.Ordinal) ||
                trimmed.StartsWith("#pragma shader_feature", StringComparison.Ordinal) ||
                trimmed.StartsWith("#pragma multi_compile_", StringComparison.Ordinal))
                yield return trimmed;
        }
    }
}
