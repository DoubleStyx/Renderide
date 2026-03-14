using System.Diagnostics;
using CommandLine;

namespace SharedTypeGenerator.Options;

public class GeneratorOptions
{
    [Option('v', "verbose", Required = false, HelpText = "Output verbose messaging as Rust types are generated.")]
    public bool Verbose { get; set; }

    [Option("il-verbose", Required = false, HelpText = "Output verbose IL details to the Rust file as Rust types are generated.")]
    public bool IlVerbose { get; set; }

    [Option('i', "assembly-path", Required = true, HelpText = "The absolute path to the Renderite.Shared.dll file.")]
    public string AssemblyPath { get; set; } = null!;

    [Option('o', "output-rust-file", Required = false, Default = null, HelpText = "The destination .rs file to generate.")]
    public string? OutputRustFile { get; set; }

    public void DetermineDefaultOutputPath()
    {
        Process process = Process.Start(new ProcessStartInfo
        {
            FileName = "git",
            Arguments = "rev-parse --show-toplevel",
            UseShellExecute = false,
            RedirectStandardOutput = true,
        })!;
        process.WaitForExit();
        if (process.ExitCode != 0)
            throw new Exception("Git exited with exit code " + process.ExitCode);

        string? output = process.StandardOutput.ReadLine();
        if (output == null)
            throw new Exception("Git returned no output");

        OutputRustFile = Path.Combine(output, "crates", "renderide", "src", "shared", "shared.rs");
    }
}
