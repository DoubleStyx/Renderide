using System.Linq;
using UnityShaderParser.Common;
using UnityShaderParser.HLSL.PreProcessor;
using UnityShaderParser.ShaderLab;

namespace UnityShaderConverter.Analysis;

/// <summary>Parses ShaderLab via UnityShaderParser and builds <see cref="ShaderFileDocument"/>.</summary>
public static class ShaderLabAnalyzer
{
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

        SubShaderNode sub0 = root.SubShaders[0];
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

                ShaderPassDocument addedPass = passDoc!;
                passes.Add(addedPass);
                foreach (string line in ExtractMultiCompileLines(addedPass.ProgramSource))
                    multiCompiles.Add(line);
                codePassIndex++;
            }
        }

        if (passes.Count == 0)
        {
            errors.Add("No CGPROGRAM/HLSLPROGRAM passes found in first SubShader.");
            return false;
        }

        document = new ShaderFileDocument
        {
            SourcePath = Path.GetFullPath(shaderPath),
            ShaderName = root.Name ?? Path.GetFileNameWithoutExtension(shaderPath),
            Properties = properties,
            SubShaderTags = subShaderTags,
            Passes = passes,
            MultiCompilePragmas = DeduplicateSorted(multiCompiles),
        };
        return true;
    }

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

        HLSLProgramBlock block = blockNullable.Value;
        string program = block.CodeWithoutIncludes;
        if (PragmaParser.HasGeometryStage(program))
        {
            error = "Pass uses #pragma geometry which is not supported by UnityShaderConverter yet.";
            return false;
        }

        if (!PragmaParser.TryGetVertexEntry(program, out string vert))
        {
            error = "Pass must declare #pragma vertex for this converter.";
            return false;
        }

        if (!PragmaParser.TryGetFragmentEntry(program, out string frag))
        {
            error = "Pass must declare #pragma fragment for this converter.";
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
