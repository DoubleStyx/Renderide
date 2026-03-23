using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[ExecuteAlways]
public class KeywordTester : MonoBehaviour
{
    [System.Serializable]
    public class ShaderKeywordSetting
    {
        public string name;
        public bool enabled;
    }

    public List<ShaderKeywordSetting> Keywords = new List<ShaderKeywordSetting>();
    
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        var renderer = gameObject.GetComponent<Renderer>();

        foreach (var mat in renderer.sharedMaterials)
        {
            foreach (var keyword in Keywords)
            {
                if (keyword.enabled)
                    mat.EnableKeyword(keyword.name);
                else
                    mat.DisableKeyword(keyword.name);
            }
        }
    }
}
