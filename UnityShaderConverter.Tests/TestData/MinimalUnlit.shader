// Reference shader with no Unity .cginc includes — used for end-to-end Slang → WGSL checks.
Shader "Converter/MinimalUnlit"
{
    Properties
    {
        _Color ("Color", Color) = (1,1,1,1)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            struct appdata
            {
                float4 vertex : POSITION;
            };

            struct v2f
            {
                float4 vertex : SV_POSITION;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = v.vertex;
                return o;
            }

            float4 frag (v2f i) : SV_Target
            {
                return float4(1.0, 0.0, 1.0, 1.0);
            }
            ENDCG
        }
    }
}
