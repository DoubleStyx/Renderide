Shader "Matcap"
{
	Properties
	{
		_MainTex ("Matcap Texture", 2D) = "white" {}
		_NormalMap ("Normal Map", 2D) = "bump" {}

		_SrcBlend("SrcBlend", Float) = 1.0
		_DstBlend("DstBlend", Float) = 0.0
		_ZWrite("ZWrite", Float) = 1.0
		_Cull("Cull", Float) = 2.0

		_ZTest("ZTest", Float) = 2
	}

		

	SubShader
	{
		Tags { "RenderType"="Opaque" }
		Offset[_OffsetFactor],[_OffsetUnits]
		LOD 100

		Pass
		{
			Blend[_SrcBlend][_DstBlend], One One
			BlendOp Add, Max
			ZWrite[_ZWrite]
			Cull[_Cull]
			ZTest[_ZTest]

			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			// make fog work
			#pragma multi_compile_fog

			#pragma multi_compile _ _NORMALMAP

#pragma target 3.0
			
			#include "UnityCG.cginc"
			#include "../Common.cginc"

			struct v2f
			{
#ifdef _NORMALMAP
				float2 uv : TEXCOORD0;
				float3 tangent : TANGENT;
				float3 bitangent : TEXCOORD2;
#endif

				UNITY_FOG_COORDS(1)
				float4 vertex : SV_POSITION;
				float3 normal : NORMAL;

				UNITY_VERTEX_OUTPUT_STEREO
			};

			sampler2D _MainTex;
			
			evr_v2f vert (evr_appdata_t v)
			{
				evr_v2f o;

				EVR_SETUP_INSTANCING_VERTEX(v, o);
				UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

				EVR_APPLY_VERTEX_POSITION(v, o);
				EVR_APPLY_NORMAL_INFO(v, o);

				EVR_APPLY_TEXCOORD_INFO_TRANSFORM(texcoord, texcoord, v, _NormalMap, o);
				
				EVR_APPLY_BITANGENT(v, o);

				EVR_SETUP_INSTANCING_VERTEX(v, o);
				
				return o;
			}
			
			fixed4 frag (evr_v2f i) : SV_Target
			{
				EVR_SETUP_INSTANCING_FRAGMENT(i);
				UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

				i.normal = normalize(i.normal);

				EVR_APPLY_NORMALMAP_INFO_FRAG_TRANSFORM(i, texcoord, _NormalMap, 1);
				
				// transform the normal into the screen coordinates
				i.normal = mul((float3x3)UNITY_MATRIX_V, i.normal);

				// sample the materialcap texture by converting normal to UV
				float2 uv = (i.normal.xy * 0.5) + 0.5;

				fixed4 col = tex2D(_MainTex, uv);

				return col;
			}
			ENDCG
		}
	}
}
