Shader "Custom/Nosamplers"
{
    Properties
    {
        _Color ("Color", Color) = (1,1,1,1)		
        _Albedo("Albedo 0", 2D) = "white" {}
        _Albedo1("Albedo 1", 2D) = "white" {}
        _Albedo2("Albedo 2", 2D) = "white" {}
        _Albedo3("Albedo 3", 2D) = "white" {}
        _MetallicMap("Metallic 0", 2D) = "white" {}
        _EmissionMap("MapA", 2D) = "white" {}
        _EmissionMap1("MabB", 2D) = "white" {}
        _Glossiness ("Smoothness", Range(0,1)) = 0.5
        _Metallic ("Metallic", Range(0,1)) = 0.0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 200

        CGPROGRAM
        // Physically based Standard lighting model, and enable shadows on all light types
        #pragma surface surf Standard fullforwardshadows

        // Use shader model 3.0 target, to get nicer looking lighting
        #pragma target 3.0

        UNITY_DECLARE_TEX2D(_Albedo);
        UNITY_DECLARE_TEX2D(_Albedo1);
        UNITY_DECLARE_TEX2D(_Albedo2);
        UNITY_DECLARE_TEX2D(_Albedo3);

        UNITY_DECLARE_TEX2D_NOSAMPLER(_MetallicMap);

        sampler2D _EmissionMap;
        sampler2D _EmissionMap1;

        struct Input
        {
            float2 uv_Albedo;
        };

        half _Glossiness;
        half _Metallic;
        fixed4 _Color;

        // Add instancing support for this shader. You need to check 'Enable Instancing' on materials that use the shader.
        // See https://docs.unity3d.com/Manual/GPUInstancing.html for more information about instancing.
        // #pragma instancing_options assumeuniformscaling
        UNITY_INSTANCING_BUFFER_START(Props)
            // put more per-instance properties here
        UNITY_INSTANCING_BUFFER_END(Props)

        void surf (Input IN, inout SurfaceOutputStandard o)
        {
            // Albedo comes from a texture tinted by color
            fixed4 c = UNITY_SAMPLE_TEX2D(_Albedo, IN.uv_Albedo) * _Color;
            o.Albedo = c.rgb;

            fixed4 m = UNITY_SAMPLE_TEX2D_SAMPLER(_MetallicMap, _Albedo, IN.uv_Albedo);
            half3 e0 = 1;
            half3 e1 = 1;
            e0 *= tex2D(_EmissionMap, IN.uv_Albedo).rgb;
            e1 *= tex2D(_EmissionMap1, IN.uv_Albedo).rgb;

            o.Emission = lerp(e0, e1, 0.5);

            // Metallic and smoothness come from slider variables
            o.Metallic = m.r;
            o.Smoothness = m.a;
            o.Alpha = c.a;
        }
        ENDCG
    }
    FallBack "Diffuse"
}
