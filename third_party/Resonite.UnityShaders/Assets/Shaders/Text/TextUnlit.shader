Shader "Text/Unlit"
{
    Properties
    {
        _FontAtlas ("Font Atlas", 2D) = "white" {}

		_TintColor ("Tint Color", Color) = (1,1,1,1)
		_OutlineColor ("Outline Color", Color) = (1,1,1,0)
		_BackgroundColor ("Background Color", Color) = (0,0,0,0)

		_Range ("Range", Vector) = (0.001, 0.001, 0, 0)

		_FaceDilate ("Face Dilate", Float) = 0
		_FaceSoftness ("Face Softness", Range(0, 1)) = 0
		_OutlineSize ("Outline Size", Float) = 0

		_SrcBlend("SrcBlend", Float) = 5
		_DstBlend("DstBlend", Float) = 10
		_ZWrite("ZWrite", Float) = 1.0
		_Cull("Cull", Float) = 2.0

		_ZTest("ZTest", Float) = 2
    }
    SubShader
    {
        Tags { "Queue" = "AlphaTest" "RenderType"="Transparent" }
        LOD 100

        Pass
        {
			Blend[_SrcBlend][_DstBlend], One One
			BlendOp Add, Max
			ZWrite[_ZWrite]
			Cull[_Cull]
			ZTest[_ZTest]
			Offset[_OffsetFactor],[_OffsetUnits]

            CGPROGRAM
			

            #pragma vertex vert
            #pragma fragment frag
			#pragma multi_compile_instancing

			#pragma multi_compile RASTER SDF MSDF
			#pragma multi_compile _ OUTLINE

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
				float4 color : COLOR;
				float3 extraData : NORMAL;

				UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
				float4 color : COLOR;
				float3 extraData : NORMAL;

				UNITY_VERTEX_OUTPUT_STEREO
            };

            sampler2D _FontAtlas;

			float4 _TintColor;
			float4 _OutlineColor;
			float4 _BackgroundColor;

			float2 _Range;

			float _FaceDilate;
			float _FaceSoftness;
			float _OutlineSize;

			float median(float r, float g, float b)
			{
				return max(min(r, g), min(max(r, g), b));
			}

            v2f vert (appdata v)
            {
                v2f o;

				UNITY_SETUP_INSTANCE_ID(v);
				UNITY_INITIALIZE_OUTPUT(v2f, o);
				UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
				o.color = v.color;
				o.extraData = v.extraData;

                return o;
            }

            float4 frag (v2f i) : SV_Target
            {
				//UNITY_SETUP_INSTANCE_ID(i);
				UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

				float4 atlasColor = tex2D(_FontAtlas, i.uv);

#if defined(MSDF) || defined(SDF)
				float2 msdfUnit = _Range;

#ifdef MSDF
				float sigDist = median(atlasColor.r, atlasColor.g, atlasColor.b) - 0.5;
#elif SDF
				float sigDist = atlasColor.a - 0.5;
#endif

				sigDist += _FaceDilate + i.extraData.x;

				float antiAliasing = dot(msdfUnit, 0.5 / fwidth(i.uv));
				antiAliasing = max(antiAliasing, 1);

				float glyphLerp = lerp(sigDist * antiAliasing, sigDist, _FaceSoftness);
				glyphLerp = saturate(glyphLerp + 0.5);

				clip(max(glyphLerp, _BackgroundColor.a) - 0.001);

				float4 fillColor = _TintColor * i.color;

#ifdef OUTLINE
				float outlineDist = sigDist - (_OutlineSize + i.extraData.y);
				float outlineLerp = lerp(outlineDist * antiAliasing, outlineDist, _FaceSoftness);

				outlineLerp = saturate(outlineLerp + 0.5);

				fillColor = lerp(_OutlineColor * float4(1,1,1,i.color.a), fillColor, outlineLerp);
#endif

				return lerp(_BackgroundColor * i.color, fillColor, glyphLerp);
#endif

#ifdef RASTER
				float4 c = atlasColor * i.color;

				clip(c.a - 0.001);

				return c;
#endif
            }

            ENDCG
        }
    }
}
