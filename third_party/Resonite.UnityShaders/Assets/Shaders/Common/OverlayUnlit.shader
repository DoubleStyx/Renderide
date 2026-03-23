// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

Shader "OverlayUnlit" {
Properties {
	_BehindTex ("Behind Texture", 2D) = "white" {}
	_BehindColor("Behind Color", Color) = (0.5,0.5,0.5,0.5)

	_FrontTex("Front Texture", 2D) = "white" {}
	_FrontColor("Front Color", Color) = (1,1,1,1)

	_Cutoff("Alpha cutoff", Range(0,1)) = 0.5

	_SrcBlend("SrcBlend", Float) = 1.0
	_DstBlend("DstBlend", Float) = 0.0
	_ZWrite("ZWrite", Float) = 1.0
	_Cull("Cull", Float) = 2.0

	_PolarPow ("Polar Mapping Power", Float) = 1.0

	_OffsetFactor ("Offset Factor", Float) = 0
	_OffsetUnits("Offset Units", Float) = 0
}

	

SubShader{
	Tags { "RenderType" = "Opaque" }
	LOD 100

	Pass {

		Blend[_SrcBlend][_DstBlend], One One
		BlendOp Add, Max
		ZWrite[_ZWrite]
		Cull[_Cull]
		ZTest Greater
		Offset [_OffsetFactor], [_OffsetUnits]

		CGPROGRAM
			

			#pragma vertex evr_vert
			#pragma fragment frag
			#pragma multi_compile_fog

			#pragma multi_compile _ _TEXTURE
			#pragma multi_compile _ _ALPHATEST
			#pragma multi_compile _ _VERTEXCOLORS
			#pragma multi_compile _ _MUL_ALPHA_INTENSITY
			#pragma multi_compile _ _MUL_RGB_BY_ALPHA
			#pragma multi_compile _ _POLARUV
			#pragma multi_compile _ _VERTEX_LINEAR_COLOR
			#pragma multi_compile _ _VERTEX_SRGB_COLOR
			#pragma multi_compile _ _VERTEX_HDRSRGB_COLOR
			
			#include "UnityCG.cginc"
			#include "../Common.cginc"
		
		#ifdef _TEXTURE
			sampler2D _BehindTex;
			float4 _BehindTex_ST;
			#endif

			fixed4 _BehindColor;

			#ifdef _POLARUV
			float _PolarPow;
			#endif
			
			fixed4 frag (evr_v2f i) : SV_Target
			{
				EVR_SETUP_INSTANCING_FRAGMENT(i);
				UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

				#if defined(_TEXTURE)

					#if defined(_POLARUV)
					float2 uvddx, uvddy;
					float2 uv = PolarMapping(i.texcoord * 2 - 1, _BehindTex_ST, _PolarPow, uvddx, uvddy);
					#else
					float2 uv = TRANSFORM_TEX(i.texcoord, _BehindTex);
					#endif
				
					#if defined(_POLARUV)
					fixed4 col = tex2Dgrad(_BehindTex, uv, uvddx, uvddy);
					#else
					fixed4 col = tex2D(_BehindTex, uv);
					#endif

					col *= _BehindColor;

				#else
				fixed4 col = _BehindColor;
				#endif

				#if defined(_ALPHATEST)
				clip(col.a - _Cutoff);
				#endif

				EVR_APPLY_VERTEX_COLORS_FRAG(col, i);

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

		Pass{

			Blend[_SrcBlend][_DstBlend], One One
			BlendOp Add, Max
			ZWrite[_ZWrite]
			Cull[_Cull]
			ZTest LEqual
			Offset[_OffsetFactor],[_OffsetUnits]

			CGPROGRAM
	

	#pragma vertex evr_vert
	#pragma fragment frag
	#pragma multi_compile_fog

	#pragma multi_compile _ _TEXTURE
	#pragma multi_compile _ _ALPHATEST
	#pragma multi_compile _ _VERTEXCOLORS
	#pragma multi_compile _ _MUL_ALPHA_INTENSITY
	#pragma multi_compile _ _MUL_RGB_BY_ALPHA
	#pragma multi_compile _ _POLARUV
	#pragma multi_compile _ _VERTEX_LINEAR_COLOR
	#pragma multi_compile _ _VERTEX_SRGB_COLOR
	#pragma multi_compile _ _VERTEX_HDRSRGB_COLOR

	#include "UnityCG.cginc"
	#include "../Common.cginc"
			
			#ifdef _TEXTURE
		sampler2D _FrontTex;
		float4 _FrontTex_ST;
	#endif

		fixed4 _FrontColor;
			
	#ifdef _POLARUV
		float _PolarPow;
	#endif

		fixed4 frag(evr_v2f i) : SV_Target
		{
			EVR_SETUP_INSTANCING_FRAGMENT(i);
			UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

	#if defined(_TEXTURE)

	#if defined(_POLARUV)
			float2 uvddx, uvddy;
		float2 uv = PolarMapping(i.texcoord * 2 - 1, _FrontTex_ST, _PolarPow, uvddx, uvddy);
	#else
			float2 uv = TRANSFORM_TEX(i.texcoord, _FrontTex);
	#endif

	#if defined(_POLARUV)
			fixed4 col = tex2Dgrad(_FrontTex, uv, uvddx, uvddy);
	#else
			fixed4 col = tex2D(_FrontTex, uv);
	#endif

			col *= _FrontColor;

	#else
			fixed4 col = _FrontColor;
	#endif

	#if defined(_ALPHATEST)
		clip(col.a - _Cutoff);
	#endif
			
		EVR_APPLY_VERTEX_COLORS_FRAG(col, i);

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
