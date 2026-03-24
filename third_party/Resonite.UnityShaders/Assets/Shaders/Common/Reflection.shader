// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

Shader "Reflection" {
Properties {
	_ReflectionTex ("Reflection Texture", 2D) = "white" {}
	_NormalMap ("Normal Map", 2D) = "bump" {}

	_Color("Color", Color) = (1,1,1,1)
	_Cutoff("Alpha cutoff", Range(0,1)) = 0.5

	_SrcBlend("SrcBlend", Float) = 1.0
	_DstBlend("DstBlend", Float) = 0.0
	_ZWrite("ZWrite", Float) = 1.0
	_Cull("Cull", Float) = 2.0

	_Distort ("Reflection Distort", Range(0, 2)) = 0

	_ZTest("ZTest", Float) = 2

	_OffsetFactor ("Offset Factor", Float) = 0.0
	_OffsetUnits("Offset Units", Float) = 0.0
}



SubShader{
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
			

			#pragma vertex vert
			#pragma fragment frag
			#pragma multi_compile_fog
			#pragma multi_compile_fwdbase_fullshadows

			#pragma multi_compile _ _COLOR
			#pragma multi_compile _ _NORMALMAP
			#pragma multi_compile _ _ALPHATEST
			#pragma multi_compile _ _MUL_ALPHA_INTENSITY
			#pragma multi_compile _ _OFFSET_TEXTURE
			#pragma multi_compile _ _MUL_RGB_BY_ALPHA

			#include "UnityCG.cginc"
			#include "..\Common.cginc"

		struct appdata_t
			{
				float4 vertex : POSITION;
				float3 normal : NORMAL;
				
				#ifdef _NORMALMAP
				float2 uv : TEXCOORD0;
				#endif

				UNITY_VERTEX_INPUT_INSTANCE_ID
			};

			struct v2f
			{
				float4 vertex : SV_POSITION;

				float4 ref : TEXCOORD0;
				float3 viewDir : TEXCOORD1;

				#ifdef _NORMALMAP
				float2 uv : TEXCOORD2;
				#endif

				float eyeIndex : TEXCOORD3;

				UNITY_FOG_COORDS(4)
				UNITY_VERTEX_OUTPUT_STEREO
			};

			sampler2D _ReflectionTex;

#ifndef UNITY_SINGLE_PASS_STEREO
			uniform float _stereoActiveEye;
#endif
			
			v2f vert (appdata_t v)
			{
				v2f o;

				UNITY_SETUP_INSTANCE_ID(v);
				UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

				o.vertex = UnityObjectToClipPos(v.vertex);
				o.viewDir.xzy = WorldSpaceViewDir(v.vertex);
				o.ref = ComputeNonStereoScreenPos(o.vertex);

				#ifdef _NORMALMAP
				o.uv = TRANSFORM_TEX(v.uv, _NormalMap);
				#endif

#ifdef UNITY_SINGLE_PASS_STEREO
				o.eyeIndex = unity_StereoEyeIndex;
#else
				o.eyeIndex = _stereoActiveEye;
#endif
				// When not using single pass stereo rendering, eye index must be determined by testing the
				// sign of the horizontal skew of the projection matrix.
//				float skew = unity_CameraProjection[0][2];
//				if (skew == 0)
//					o.eyeIndex = -1; // mono
//				else
//					o.eyeIndex = skew > 0;
//#endif

				UNITY_TRANSFER_FOG(o,o.vertex);
				return o;
			}
			
			half4 frag(v2f i) : SV_Target
			{
				//UNITY_SETUP_INSTANCE_ID(i);
				UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

				i.viewDir = normalize(i.viewDir);

				float4 uv = i.ref;

				#ifdef _NORMALMAP
				half3 bump = UnpackNormal(UNITY_SAMPLE_TEX2D(_NormalMap, i.uv)).rgb;
				uv.xy += bump * _Distort;
				#endif

				uv = UNITY_PROJ_COORD(uv);

				uv.xy /= uv.w;

				if (i.eyeIndex >= 0)
				{
					uv.x *= 0.5;
					uv.x += i.eyeIndex * 0.5;
				}

				half4 col = tex2D(_ReflectionTex, uv.xy);

				#if defined(_COLOR)
				col *= _Color;
				#endif

				#if defined(_ALPHATEST)
				clip(col.a - _Cutoff);
				#endif

				#ifdef _MUL_RGB_BY_ALPHA
				col.rgb *= col.a;
				#endif

				// This is for external camera blending, so additive shaders output alpha
				#ifdef _MUL_ALPHA_INTENSITY
				float mulfactor = (col.r + col.g + col.b) * 0.3333333;
				col.a *= mulfactor;
				#endif

				UNITY_APPLY_FOG(i.fogCoord, col);
				//UNITY_OPAQUE_ALPHA(col.a);
				return col;
			}
		ENDCG
	}
}

}
