// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

// Upgrade NOTE: replaced '_Object2World' with 'unity_ObjectToWorld'

Shader "FresnelLerp" {
Properties {

	_Lerp("Lerp", Float) = 0
	_LerpTex("LerpTexture", 2D) = "black" {}

	_Exp0 ("Exponent0", Float) = 1.0
	_Exp1("Exponent1", Float) = 1.0

	_GammaCurve("Power", Float) = 2.2

	_FarTex0 ("Far Texture 0", 2D) = "white" {}
	_NearTex0 ("Near Texture 0", 2D) = "white" {}

	_FarTex1("Far Texture 1", 2D) = "white" {}
	_NearTex1("Near Texture 1", 2D) = "white" {}

	_FarColor0 ("FarColor 0", Color) = (0,0,0,1)
	_NearColor0 ("NearColor 0", Color) = (1,1,1,1)

	_FarColor1("FarColor 1", Color) = (0.2,0.2,0.2,1)
	_NearColor1("NearColor 1", Color) = (0.8,0.8,0.8,0.8)
	
	_NormalMap0 ("Normal Map 0", 2D) = "bump" {}
	_NormalMap1("Normal Map 1", 2D) = "bump" {}

	_SrcBlend("SrcBlend", Float) = 1.0
	_DstBlend("DstBlend", Float) = 0.0
	_ZWrite("ZWrite", Float) = 1.0
	_Cull("Cull", Float) = 2.0

	_LerpPolarPow("LerpTex Polar Power", Float) = 1.0

	_ZTest("ZTest", Float) = 2
}



SubShader{
	Tags { "RenderType" = "Opaque" }
	Offset[_OffsetFactor],[_OffsetUnits]
	LOD 100

	Pass {

		Blend[_SrcBlend][_DstBlend], One One
		BlendOp Add, Max
		ZWrite[_ZWrite]
		Cull[_Cull]
		ZTest[_ZTest]

		CGPROGRAM
			

			#pragma vertex evr_vert
			#pragma fragment frag
			#pragma multi_compile_fog
			#pragma multi_compile_instancing

			#pragma multi_compile _ _TEXTURE
			#pragma multi_compile _ _NORMALMAP
			#pragma multi_compile _ _LERPTEX_POLARUV _LERPTEX
#pragma multi_compile _ _MULTI_VALUES

#pragma target 3.0
			#define WORLD_SPACE 1

			#include "UnityCG.cginc"
			#include "../Common.cginc"

			fixed4 _FarColor0;
			fixed4 _NearColor0;

			fixed4 _FarColor1;
			fixed4 _NearColor1;

			float _Exp0;
			float _Exp1;
			float _GammaCurve;

			float _LerpPolarPow;
			sampler2D _LerpTex;
			float4 _LerpTex_ST;

			sampler2D _FarTex0;
			sampler2D _FarTex1;
			float4 _FarTex0_ST;
			float4 _FarTex1_ST;

			sampler2D _NearTex0;
			sampler2D _NearTex1;
			float4 _NearTex0_ST;
			float4 _NearTex1_ST;

			sampler2D _NormalMap0;
			sampler2D _NormalMap1;
			float4 _NormalMap0_ST;
			float4 _NormalMap1_ST;
			
			fixed4 frag (evr_v2f i) : SV_Target
			{
				EVR_SETUP_INSTANCING_FRAGMENT(i);
				UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

				i.normal = normalize(i.normal);

				#if defined(_LERPTEX)
				half l = tex2D(_LerpTex, TRANSFORM_TEX(i.texcoord, _LerpTex));
#ifdef _MULTI_VALUES
				l *= UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _Lerp);
#endif
				#elif _LERPTEX_POLARUV
				half l = SampleTex2Dpolar(_LerpTex, PolarUV(i.texcoord*2 - 1, _LerpPolarPow), _LerpTex_ST).r;
#ifdef _MULTI_VALUES
				l *= UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _Lerp);
#endif
				#else
				half l = UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _Lerp);
				#endif

				#ifdef _NORMALMAP

				float3x3 tangentTransform = float3x3(i.tangent, i.bitangent, i.normal);
				float3 bumpNormal = UnpackNormal(
					lerp(tex2D(_NormalMap0, TRANSFORM_TEX(i.texcoord, _NormalMap0)),
						tex2D(_NormalMap1, TRANSFORM_TEX(i.texcoord, _NormalMap1)), l));

				// Compute pertrubed normal, replacing the old one
				i.normal = normalize( mul(bumpNormal, tangentTransform) );

				#endif

				float3 viewDir = normalize(_WorldSpaceCameraPos.xyz - i.position.xyz);

				float fresnel = pow(pow(1 - abs(dot(i.normal, viewDir)), lerp(_Exp0, _Exp1, l)), _GammaCurve);

				fixed4 farColor = lerp(_FarColor0, _FarColor1, l);
				fixed4 nearColor = lerp(_NearColor0, _NearColor1, l);

				#ifdef _TEXTURE

				farColor *= lerp(
					tex2D(_FarTex0, TRANSFORM_TEX(i.texcoord, _FarTex0)),
					tex2D(_FarTex1, TRANSFORM_TEX(i.texcoord, _FarTex1)), l);
				nearColor *= lerp(
					tex2D(_NearTex0, TRANSFORM_TEX(i.texcoord, _NearTex0)),
					tex2D(_NearTex1, TRANSFORM_TEX(i.texcoord, _NearTex1)), l);

				#endif

				// compute final color by blending between the two based on the fresnel

				fixed4 col = lerp(nearColor, farColor, fresnel);

				UNITY_APPLY_FOG(i.fogCoord, col);
				//UNITY_OPAQUE_ALPHA(col.a);
				return col;
			}
		ENDCG
	}
}

}
