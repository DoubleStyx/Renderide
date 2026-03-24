using CommandLine;
using Renderide.Generators.Logging;

namespace UnityShaderConverter.Options;

/// <summary>Command-line options for <see cref="ConverterRunner.Run"/>.</summary>
public sealed class ConverterOptions
{
    /// <summary>When true, lowers the log threshold to trace (parser diagnostics, slang command lines, intermediate compile steps).</summary>
    /// <remarks>
    /// The default log file suppresses <see cref="NotEnoughLogs.LogLevel.Warning"/> lines; use <c>--verbose</c> for trace-level detail.
    /// Hard failures (parse, <c>slangc</c>, WGSL post-process, etc.) log full messages at <see cref="NotEnoughLogs.LogLevel.Error"/> or
    /// <see cref="NotEnoughLogs.LogLevel.Info"/> without <c>--verbose</c>.
    /// </remarks>
    [Option('v', "verbose", Required = false, HelpText = "Trace-level logging (parser notes, slang command lines, intermediate diagnostics).")]
    public bool Verbose { get; set; }

    /// <summary>One or more roots to scan recursively for *.shader files.</summary>
    [Option('i', "input", Required = false, HelpText = "Directory containing .shader files (repeatable). Defaults to SampleShaders + Resonite Unity shaders under the repo.")]
    public IEnumerable<string> InputDirectories { get; set; } = Array.Empty<string>();

    /// <summary>Output root: <c>crates/renderide/src/shaders/generated/</c> with one <c>&lt;mod&gt;/</c> folder per converted shader (e.g. <c>pbs_metallic/material.rs</c>).</summary>
    [Option('o', "output", Required = false, HelpText = "Output directory (default: crates/renderide/src/shaders/generated).")]
    public string? OutputDirectory { get; set; }

    /// <summary>Path to <c>slangc</c>; overrides the <c>Slang.Sdk</c> binary next to the tool and <c>SLANGC</c>.</summary>
    [Option("slangc", Required = false, HelpText = "Path to slangc (default: Slang.Sdk under the app, then SLANGC, then PATH).")]
    public string? SlangcPath { get; set; }

    /// <summary>When set, skips invoking <c>slangc</c> even for eligible shaders.</summary>
    [Option("skip-slang", Required = false, HelpText = "Do not run slangc; each passN.wgsl must already exist under the shader folder.")]
    public bool SkipSlang { get; set; }

    /// <summary>Optional JSON file with slang eligibility globs, variant caps, and forced defines.</summary>
    [Option("compiler-config", Required = false, HelpText = "Path to compiler JSON config (merges over built-in defaults).")]
    public string? CompilerConfigPath { get; set; }

    /// <summary>Optional JSON file mapping shader names to explicit variant define lists.</summary>
    [Option("variant-config", Required = false, HelpText = "Path to variant override JSON.")]
    public string? VariantConfigPath { get; set; }

    /// <summary>Directory containing Unity <c>UnityCG.cginc</c> (Unity CGIncludes). Overrides UNITY_SHADER_CONVERTER_CG_INCLUDES.</summary>
    [Option("cg-includes", Required = false, HelpText = "Folder with UnityCG.cginc (Unity CGIncludes). Default: bundled next to the tool, else search from shader path.")]
    public string? UnityCgIncludesDirectory { get; set; }

    /// <summary>Writes generated <c>.slang</c> per pass next to WGSL when slangc runs.</summary>
    [Option("dump-intermediate", Required = false, HelpText = "Directory to write per-pass generated .slang (and optional diagnostics).")]
    public string? DumpIntermediateDirectory { get; set; }

    /// <summary>Skips slangc/Rust refresh when shader and include inputs are unchanged (manifest under the dump or output tree).</summary>
    [Option("compile-cache", Required = false, HelpText = "Directory for compile cache manifests (skip slangc when inputs unchanged).")]
    public string? CompileCacheDirectory { get; set; }

    /// <summary>Extra <c>-I</c> roots for <c>slangc</c> (repeatable); appended after Unity CGIncludes resolution.</summary>
    [Option("extra-cg-include", Required = false, HelpText = "Additional include directory for slangc (repeatable).")]
    public IEnumerable<string> ExtraCgIncludeDirectories { get; set; } = Array.Empty<string>();

    /// <summary>Skips ShaderLab passes whose <c>LightMode</c> is deferred, meta, motion vectors, etc.</summary>
    [Option("skip-non-forward-passes", Required = false, HelpText = "Drop Deferred/Meta/MotionVectors passes before emission.")]
    public bool SkipNonForwardPasses { get; set; }

    /// <summary>Skips <c>ForwardAdd</c> passes (for single-pass clustered forward after base pass uses clusters).</summary>
    [Option("skip-forward-add-passes", Required = false, HelpText = "Drop LightMode ForwardAdd passes.")]
    public bool SkipForwardAddPasses { get; set; }

    /// <summary>Force <c>slangc</c> warning suppression and error-only stderr in logs (overrides compiler config).</summary>
    [Option("suppress-slang-warnings", Required = false, HelpText = "Force slang warning suppression (-warnings-disable + error-only stderr in logs).")]
    public bool ForceSuppressSlangWarnings { get; set; }

    /// <summary>Force slang warnings visible in logs (overrides compiler config and --suppress-slang-warnings).</summary>
    [Option("no-suppress-slang-warnings", Required = false, HelpText = "Show slang warnings in logs (overrides config and --suppress-slang-warnings).")]
    public bool ForceNoSuppressSlangWarnings { get; set; }

    /// <summary>Resolves default input and output paths relative to the git repository root (Renderide parent).</summary>
    public void DetermineDefaultPaths()
    {
        string renderideRoot = RenderidePathResolver.ResolveRenderideRoot(RenderidePathResolver.TryGetGitRepositoryRoot());

        if (OutputDirectory is null)
            OutputDirectory = Path.Combine(renderideRoot, "crates", "renderide", "src", "shaders", "generated");

        if (!InputDirectories.Any())
        {
            InputDirectories = new[]
            {
                Path.Combine(renderideRoot, "generators", "UnityShaderConverter", "SampleShaders"),
                Path.Combine(renderideRoot, "third_party", "Resonite.UnityShaders", "Assets", "Shaders"),
            };
        }
    }
}
