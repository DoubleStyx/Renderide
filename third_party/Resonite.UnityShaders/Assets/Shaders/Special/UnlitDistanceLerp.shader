Shader "UnlitDistanceLerp"
{
	Properties
	{
		_Point("Point", Vector) = (0,0,0,0)
		_Distance("Distance", Float) = 1
		_Transition("Transition", Float) = 0.1

		_Cutoff("Alpha cutoff", Range(0,1)) = 0.5

		_NearTex("Near Texture", 2D) = "white" {}
		_FarTex("Near Texture", 2D) = "white" {}

		_NearColor("Near Color", Color) = (1,1,1,1)
		_FarColor("Far Color", Color) = (1,1,1,1)

		_SrcBlend("SrcBlend", Float) = 1.0
		_DstBlend("DstBlend", Float) = 0.0
		_ZWrite("ZWrite", Float) = 1.0
		_Cull("Cull", Float) = 2.0

		_ZTest("ZTest", Float) = 2
	}
		SubShader
		{
			Tags { "Queue" = "AlphaTest+200" "RenderType" = "Transparent" }
		LOD 100

		Pass {

				Blend[_SrcBlend][_DstBlend], One One
				BlendOp Add, Max
				ZWrite[_ZWrite]
				Cull[_Cull]
				ZTest[_ZTest]
				Offset[_OffsetFactor],[_OffsetUnits]

				CGPROGRAM
				

				#pragma vertex evr_vert
				#pragma fragment frag
				#pragma multi_compile_fwdbase_fullshadows
				#pragma multi_compile_instancing

				#pragma multi_compile WORLD_SPACE LOCAL_SPACE
				#pragma multi_compile _ _VERTEXCOLORS
				#pragma multi_compile _ _ALPHATEST

				#define _TEXTURE 1

				#include "UnityCG.cginc"
				#include "../Common.cginc"

				float3 _Point;
				float _Distance;
				float _Transition;

				sampler2D _NearTex;
				sampler2D _FarTex;

				float4 _NearTex_ST;
				float4 _FarTex_ST;

				float4 _NearColor;
				float4 _FarColor;


				fixed4 frag(evr_v2f i) : SV_Target
				{
					EVR_SETUP_INSTANCING_FRAGMENT(i);
					UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

					float _dist = distance(_Point, i.position);

					_dist -= _Distance;

					float _lerp = saturate((_dist / _Transition) + _Transition * 0.5);

					float4 cNear = tex2D(_NearTex, TRANSFORM_TEX(i.texcoord, _NearTex)) * _NearColor;
					float4 cFar = tex2D(_FarTex, TRANSFORM_TEX(i.texcoord, _FarTex)) * _FarColor;

					float4 c = lerp(cNear, cFar, _lerp);

#if defined(_ALPHATEST)
					clip(c.a - _Cutoff);
#endif
					return c;
				}

			ENDCG
		}
		}
}
