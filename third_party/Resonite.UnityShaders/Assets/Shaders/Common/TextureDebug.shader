Shader "Unlit/TextureDebug"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _TextureChannel ("Texture Channel", Float) = 0
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
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                UNITY_FOG_COORDS(1)
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;
            float _TextureChannel;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                UNITY_TRANSFER_FOG(o,o.vertex);
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                // sample the texture
                fixed4 col = tex2D(_MainTex, i.uv);
                // apply fog
                UNITY_APPLY_FOG(i.fogCoord, col);

                if (_TextureChannel == 0)
                {
                    col.rgb = col.r;
                    col.a = 1;
                } else if (_TextureChannel == 1)
                {
                    col.rgb = col.g;
                    col.a = 1;
                } else if (_TextureChannel == 2)
                {
                    col.rgb = col.b;
                    col.a = 1;
                } else if (_TextureChannel == 3)
                {
                    col.rgb = col.a;
                    col.a = 1;
                }
                
                return col;
            }
            ENDCG
        }
    }
}
