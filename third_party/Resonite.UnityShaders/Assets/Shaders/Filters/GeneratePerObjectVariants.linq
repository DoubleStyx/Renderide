<Query Kind="Statements" />

#if !UNITY_EDITOR

var dir = Path.GetDirectoryName(Util.CurrentQueryPath);

foreach(var file in Directory.EnumerateFiles(dir, "*.shader"))
{
	if(file.Contains("PerObject"))
		continue;
		
	var source = File.ReadAllText(file);
	
	source = source.Replace("\"_BackgroundTexture\"", "").Replace("_BackgroundTexture", "_GrabTexture");
	
	int index = 0;
	index = source.IndexOf("Shader");
	index = source.IndexOf("\"", index) + 1;
	index = source.IndexOf("\"", index);
	
	source = source.Insert(index, "_PerObject");
	
	File.WriteAllText(file.Replace(".shader", "_PerObject.shader"), source);
}

#endif