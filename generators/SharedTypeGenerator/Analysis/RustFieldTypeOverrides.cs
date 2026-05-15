namespace SharedTypeGenerator.Analysis;

/// <summary>Applies narrow Rust storage type overrides for generated fields whose wire type remains unchanged.</summary>
public static class RustFieldTypeOverrides
{
    /// <summary>Rust type used for string fields that are usually renderer-owned static strings.</summary>
    public const string StaticStringCowOption = "Option<Cow<'static, str>>";

    /// <summary>Returns the Rust type to emit for a C# field after applying field-specific storage overrides.</summary>
    public static string Apply(string csharpTypeName, string csharpFieldName, string rustType)
    {
        if (csharpTypeName == "RendererInitResult"
            && rustType == "Option<String>"
            && csharpFieldName is "rendererIdentifier" or "stereoRenderingMode")
        {
            return StaticStringCowOption;
        }

        return rustType;
    }

    /// <summary>Returns true when the emitted field type is the static string <c>Cow&lt;'static, str&gt;</c> override.</summary>
    public static bool IsStaticStringCowOption(string rustType) =>
        string.Equals(rustType.Trim(), StaticStringCowOption, StringComparison.Ordinal);
}
