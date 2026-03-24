Shader "Unlit" {
	Properties{
		_Tex("Texture", 2D) = "white" {}
		_Color("Color", Color) = (1,1,1,1)
		_Cutoff("Alpha cutoff", Range(0,1)) = 0.5

		_SrcBlend("SrcBlend", Float) = 1.0
		_DstBlend("DstBlend", Float) = 0.0
		_ZWrite("ZWrite", Float) = 1.0
		_Cull("Cull", Float) = 2.0

		_OffsetTex("Offset Texture", 2D) = "black" {}
		_OffsetMagnitude("Offset Magnitude", Vector) = (0.1, 0.1, 0, 0)

		_MaskTex("Mask Texture", 2D) = "white" {}

		_PolarPow("Polar Mapping Power", Float) = 1.0

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
					#pragma multi_compile_fwdbase_fullshadows
					#pragma multi_compile_instancing

					#pragma multi_compile _ _TEXTURE _TEXTURE_NORMALMAP
					#pragma multi_compile _ _COLOR
					#pragma multi_compile _ _ALPHATEST
					#pragma multi_compile _ _VERTEXCOLORS
					#pragma multi_compile _ _MUL_ALPHA_INTENSITY
					#pragma multi_compile _ _OFFSET_TEXTURE
					#pragma multi_compile _ _MASK_TEXTURE_MUL _MASK_TEXTURE_CLIP
					#pragma multi_compile _ _MUL_RGB_BY_ALPHA
					#pragma multi_compile _ _POLARUV
					#pragma multi_compile _ _RIGHT_EYE_ST

					#pragma multi_compile _ _VERTEX_LINEAR_COLOR _VERTEX_SRGB_COLOR

					#include "UnityCG.cginc"
					#include "UnityStandardUtils.cginc"
					#include "..\Common.cginc"

					#if defined(_TEXTURE) || defined(_TEXTURE_NORMALMAP) || defined(_MASK_TEXTURE_MUL) || defined(_MASK_TEXTURE_CLIP)
					#define _USES_TEXTURE
					#endif

					struct appdata_t
					{
						float4 vertex : POSITION;

						#ifdef _USES_TEXTURE
						float2 texcoord : TEXCOORD0;
						#endif

						#ifdef _VERTEXCOLORS
						float4 vcolor : COLOR;
						#endif

						UNITY_VERTEX_INPUT_INSTANCE_ID
					};

					struct v2f
					{
						float4 vertex : SV_POSITION;

						#ifdef _USES_TEXTURE
						half2 texcoord : TEXCOORD0;
						#endif

						#ifdef _VERTEXCOLORS
						float4 vcolor : COLOR;
						#endif

						UNITY_VERTEX_OUTPUT_STEREO
					};


					#if defined(_TEXTURE) || defined(_TEXTURE_NORMALMAP)
					sampler2D _Tex;
					float4 _Tex_ST;
					#ifdef _RIGHT_EYE_ST
					float4 _RightEye_ST;
					#endif
					#endif

					#ifdef _OFFSET_TEXTURE
					float2 _OffsetMagnitude;

					sampler2D _OffsetTex;
					float4 _OffsetTex_ST;
					#endif

					#if defined(_MASK_TEXTURE_MUL) || defined(_MASK_TEXTURE_CLIP)
					sampler2D _MaskTex;
					float4 _MaskTex_ST;
					#endif

					#ifdef _POLARUV
					float _PolarPow;
					#endif

					v2f vert(appdata_t v)
					{
						v2f o;

						UNITY_SETUP_INSTANCE_ID(v);
						UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

						o.vertex = UnityObjectToClipPos(v.vertex);

						#ifdef _USES_TEXTURE
						o.texcoord = v.texcoord;
						#endif

						#ifdef _VERTEXCOLORS
						o.vcolor = v.vcolor;
						#endif

						return o;
					}

					float4 frag(v2f i) : SV_Target
					{
						//UNITY_SETUP_INSTANCE_ID(i);
						UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

						#if defined(_TEXTURE) || defined(_TEXTURE_NORMALMAP)

							float4 _ST;
							#if defined(_RIGHT_EYE_ST)
							if (unity_StereoEyeIndex == 0)
								_ST = _Tex_ST;
							else
								_ST = _RightEye_ST;
							#else
							_ST = _Tex_ST;
							#endif

							#if defined(_POLARUV)
							float2 uvddx, uvddy;
							float2 uv = PolarMapping(i.texcoord * 2 - 1, _ST, _PolarPow, uvddx, uvddy);
							#else
							float2 uv = i.texcoord * _ST.xy + _ST.zw;
							#endif

							// TODO!!! Polar mapping for Offset Texture as well?
							// Polar offets?
							#if defined(_OFFSET_TEXTURE)
							float4 offset = tex2D(_OffsetTex, TRANSFORM_TEX(i.texcoord, _OffsetTex));
							uv += offset.xy * _OffsetMagnitude;
							#endif

							#if defined(_POLARUV)
							float4 col = tex2Dgrad(_Tex, uv, uvddx, uvddy);
							#else
							float4 col = tex2D(_Tex, uv);
							#endif

							#if defined(_TEXTURE_NORMALMAP)
							col = float4(UnpackScaleNormal(col, 1) * 0.5 + 0.5, 1);
							#endif

							#if defined(_COLOR)
							col *= UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _Color);
							#endif

						#elif defined(_COLOR)
						float4 col = UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _Color);
						#else
						float4 col = float4(1,1,1,1);
						#endif

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
						if (col.a - _Cutoff <= 0)
							discard;
						//clip(col.a - _Cutoff);
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

						//UNITY_OPAQUE_ALPHA(col.a);
						return col;
					}
				ENDCG
			}
		}

}
