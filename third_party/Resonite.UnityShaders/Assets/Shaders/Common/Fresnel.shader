// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

// Upgrade NOTE: replaced '_Object2World' with 'unity_ObjectToWorld'180 equirectangular

Shader "Fresnel" {
Properties {

	_Exp ("Exponent", Float) = 1.0

	_GammaCurve ("Gamma Curve", Float) = 1.0

	_FarTex ("Far Texture", 2D) = "white" {}
	_NearTex ("Near Texture", 2D) = "white" {}

	_FarColor ("FarColor", Color) = (0,0,0,1)
	_NearColor ("NearColor", Color) = (1,1,1,1)
	
	_NormalMap ("Normal Map", 2D) = "bump" {}
	_NormalScale("Normal Scale", Float) = 1

	_Cutoff("Alpha cutoff", Range(0,1)) = 0.5

	_MaskTex("Mask Texture", 2D) = "white" {}

	_SrcBlend ("SrcBlend", Float) = 1.0
	_DstBlend ("DstBlend", Float) = 0.0
	_ZWrite ("ZWrite", Float) = 1.0
	_Cull ("Cull", Float) = 2.0

	_PolarPow("Polar Mapping Power", Float) = 1.0

	_ZTest ("ZTest", Float) = 2
}



SubShader {
	Tags { "Queue" = "AlphaTest+200" "RenderType"="Opaque" }
	Offset[_OffsetFactor],[_OffsetUnits]
	LOD 100
	
	Pass {  

		Blend [_SrcBlend] [_DstBlend], One One
		ZWrite [_ZWrite]
		Cull [_Cull]
		ZTest [_ZTest]
		
		CGPROGRAM
			#pragma target 3.0

			#pragma vertex evr_vert
			#pragma fragment frag
			#pragma multi_compile_instancing

			#pragma multi_compile _ _TEXTURE
			#pragma multi_compile _ _NORMALMAP
			#pragma multi_compile _ _ALPHATEST
			#pragma multi_compile _ _MUL_ALPHA_INTENSITY
			#pragma multi_compile _ _POLARUV
			#pragma multi_compile _ _VERTEXCOLORS
			#pragma multi_compile _ _MASK_TEXTURE_MUL _MASK_TEXTURE_CLIP
			#pragma multi_compile _ _VERTEX_LINEAR_COLOR
			#pragma multi_compile _ _VERTEX_SRGB_COLOR 
			#pragma multi_compile _ _VERTEX_HDRSRGB_COLOR
		 
			#define WORLD_SPACE 1
		
			#include "UnityCG.cginc"
			#include "UnityStandardUtils.cginc"
			#include "../Common.cginc"
 
			half4 _FarColor;
			half4 _NearColor;

			float _Exp;

			#ifdef _TEXTURE
			sampler2D _FarTex;
			float4 _FarTex_ST;

			sampler2D _NearTex;
			float4 _NearTex_ST;
			#endif

#if defined(_MASK_TEXTURE_MUL) || defined(_MASK_TEXTURE_CLIP)
			sampler2D _MaskTex;
			float4 _MaskTex_ST;
#endif

			#ifdef _POLARUV
			float _PolarPow;
			#endif
			
			half4 frag (evr_v2f i) : SV_Target
			{
				EVR_SETUP_INSTANCING_FRAGMENT(i);
				//UNITY_SETUP_INSTANCE_ID(i);
				UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

				i.normal = normalize(i.normal);

				float normScale = UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _NormalScale);

				EVR_APPLY_NORMALMAP_INFO_FRAG_TRANSFORM(i, texcoord, _NormalMap, normScale);

				float3 viewDir = normalize(_WorldSpaceCameraPos.xyz - i.position.xyz);

				float fresnel = pow(1 - abs(dot(i.normal, viewDir)), _Exp);

				half4 farColor = _FarColor;
				half4 nearColor = _NearColor;

				#ifdef _TEXTURE

					#ifdef _POLARUV
					float2 polarUv = PolarUV(i.texcoord * 2 - 1, _PolarPow);
					farColor *= SampleTex2Dpolar(_FarTex, polarUv, _FarTex_ST);
					nearColor *= SampleTex2Dpolar(_NearTex, polarUv, _NearTex_ST);
					#else
					farColor *= tex2D(_FarTex, TRANSFORM_TEX(i.texcoord, _FarTex));
					nearColor *= tex2D(_NearTex, TRANSFORM_TEX(i.texcoord, _NearTex));
					#endif

				#endif

				fresnel = pow(fresnel, _GammaCurve);

				// compute final color by blending between the two based on the fresnel

				half4 col = lerp(nearColor, farColor, fresnel);

#if defined(_MASK_TEXTURE_MUL) || defined(_MASK_TEXTURE_CLIP)
				float4 mask = tex2D(_MaskTex, TRANSFORM_TEX(i.texcoord, _MaskTex));

				float mul = (mask.r + mask.g + mask.b) * 0.3333333 * mask.a;

#ifdef _MASK_TEXTURE_MUL
				col.a *= mul;
#endif

#ifdef _MASK_TEXTURE_CLIP
				if (mul - _Cutoff <= 0)
					discard;
#endif

#endif

#if defined(_ALPHATEST) && !defined(_MASK_TEXTURE_CLIP)
				clip(col.a - _Cutoff);
#endif

				EVR_APPLY_VERTEX_COLORS_FRAG(col, i);

#ifdef _MUL_ALPHA_INTENSITY
				float mulfactor = (col.r + col.g + col.b) * 0.3333333;
				col.a *= mulfactor * mulfactor;
#endif
				
				EVR_APPLY_ALPHA_OUTPUT(col.a, col)
				
				//UNITY_OPAQUE_ALPHA(col.a);
				return col;
			}
		ENDCG
	}
}

}
