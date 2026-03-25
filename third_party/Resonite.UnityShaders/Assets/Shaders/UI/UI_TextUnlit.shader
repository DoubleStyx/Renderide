Shader "UI/Text/Unlit"
{
    Properties
    {
        _FontAtlas ("Font Atlas", 2D) = "white" {}

		_TintColor ("Tint Color", Color) = (1,1,1,1)
		_OverlayTint("OverlayTint", Color) = (1,1,1,0.5)
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

		_StencilComp("Stencil Comparison", Float) = 8
		_Stencil("Stencil ID", Float) = 0
		_StencilOp("Stencil Operation", Float) = 0
		_StencilWriteMask("Stencil Write Mask", Float) = 255
		_StencilReadMask("Stencil Read Mask", Float) = 255

		_ColorMask("Color Mask", Float) = 15

		_Rect("Rect", Vector) = (0,0,1,1)
    }
    SubShader
    {
		Tags
		{
			"Queue" = "Transparent"
			"IgnoreProjector" = "True"
			"RenderType" = "Transparent"
			"PreviewType" = "Plane"
			"CanUseSpriteAtlas" = "True"
			"DisableBatching" = "True"
		}

		Stencil
		{
			Ref[_Stencil]
			Comp[_StencilComp]
			Pass[_StencilOp]
			ReadMask[_StencilReadMask]
			WriteMask[_StencilWriteMask]
		}

		Lighting Off
		Cull[_Cull]
		ZWrite[_ZWrite]
		ZTest[_ZTest]
		Blend[_SrcBlend][_DstBlend], One One
		BlendOp Add, Max
		Offset[_OffsetFactor],[_OffsetUnits]
		ColorMask[_ColorMask]

        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
			#pragma multi_compile_instancing
			#pragma target 2.0

			#pragma multi_compile RASTER SDF MSDF
			#pragma multi_compile _ OUTLINE
			#pragma multi_compile _ RECTCLIP
			#pragma multi_compile _ OVERLAY

            #include "UnityCG.cginc"
			#include "UnityUI.cginc"

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
                float4 vertex : SV_POSITION;
				float2 uv : TEXCOORD0;
				float4 color : COLOR;
				float3 extraData : NORMAL;

				float2 position : TEXCOORD1;

				float4 projPos : TEXCOORD2;

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

float4 _Rect;

			float4 _OverlayTint;
			UNITY_DECLARE_DEPTH_TEXTURE(_CameraDepthTexture);

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

				o.position = v.vertex.xy;

                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
				o.color = v.color;
				o.extraData = v.extraData;

				o.projPos = ComputeScreenPos(o.vertex);
				COMPUTE_EYEDEPTH(o.projPos.z);

                return o;
            }

            float4 frag (v2f i) : SV_Target
            {
				//UNITY_SETUP_INSTANCE_ID(i);
				UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

				float4 c = float4(0, 0, 0, 0);

#ifdef RECTCLIP
				clip(UnityGet2DClipping(i.position, _Rect) - 0.1);
#endif

				float4 atlasColor = tex2D(_FontAtlas, i.uv);

#if defined(MSDF) || defined(SDF)
				float2 msdfUnit = _Range;

				float sigDist;
#ifdef MSDF
				sigDist = median(atlasColor.r, atlasColor.g, atlasColor.b) - 0.5;
#else
				sigDist = atlasColor.a - 0.5;
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

				c = lerp(_BackgroundColor * i.color, fillColor, glyphLerp);
#elif defined(RASTER)
				c = atlasColor * i.color;
				clip(c.a - 0.001);
#endif

#ifdef OVERLAY
				float sceneZ = LinearEyeDepth(SAMPLE_DEPTH_TEXTURE_PROJ(_CameraDepthTexture, UNITY_PROJ_COORD(i.projPos)));
				float partZ = i.projPos.z;

				if (partZ > sceneZ)
					c *= _OverlayTint;
#endif

				return c;
            }

            ENDCG
        }
    }
}
