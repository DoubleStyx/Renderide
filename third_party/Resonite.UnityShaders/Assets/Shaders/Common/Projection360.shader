// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

Shader "Projection360"
{
	Properties
	{
		[PerRendererData] _MainTex("Texture", 2D) = "black" {}
		[PerRendererData] _MainCube("Texture Cube", CUBE) = "black" {}

		_SecondTex("Second Texture", 2D) = "black" {}
		_SecondCube("Second Texture Cube", CUBE) = "black" {}
		_TextureLerp("Texture Lerp", Float) = 0
		_CubeLOD("Cube LOD", Float) = 0

		_ProjectionLerp("Projection", Float) = 0
		_Tint("Tint", Color) = (1,1,1,1)

		_FOV("Field of View", Vector) = (6.28318530718, 3.14159265359, 0, 0)
		_SecondTexOffset("Second Texture Offset", Vector) = (0, 0, 0, 0)

		_OutsideColor("Outside Color", Color) = (0, 0, 0, 0)
		_Exposure("Exposure", Float) = 1
		_Gamma("Gamma", Float) = 1

		_TintTex("Tint Texture", 2D) = "white" {}
		_Tint0("Tint Color 0", Color) = (1, 0, 0, 1)
		_Tint1("Tint Color 1", Color) = (0, 1, 0, 1)

		_StencilComp("Stencil Comparison", Float) = 8
		_Stencil("Stencil ID", Float) = 0
		_StencilOp("Stencil Operation", Float) = 0
		_StencilWriteMask("Stencil Write Mask", Float) = 255
		_StencilReadMask("Stencil Read Mask", Float) = 255

		_ColorMask("Color Mask", Float) = 15

		_OffsetTex("Offset Tex", 2D) = "black" {}
		_OffsetMask("Offset Mask", 2D) = "white" {}
		_OffsetMagnitude("Offset Magnitude", Vector) = (0.1, 0.1, 0, 0)

		_SrcBlend("SrcBlend", Float) = 1.0
		_DstBlend("DstBlend", Float) = 0.0
		_ZWrite("ZWrite", Float) = 1.0
		_Cull("Cull", Float) = 2.0

		_MaxIntensity("Max Intensity", Float) = 4.0

		[PerRendererData] _PerspectiveFOV("Perspective FOV", Vector) = (0.785398, 0.785398, 0, 0)
	}
		SubShader
		{
			Tags { "Queue" = "Transparent-100" "RenderType" = "Transparent" "DisableBatching" = "True" }
			LOD 100
			Blend[_SrcBlend][_DstBlend], One One
			BlendOp Add, Max
			ZWrite[_ZWrite]
			Cull[_Cull]
			ZTest[_ZTest]
			Offset[_OffsetFactor],[_OffsetUnits]
			ColorMask[_ColorMask]

			Stencil
			{
				Ref[_Stencil]
				Comp[_StencilComp]
				Pass[_StencilOp]
				ReadMask[_StencilReadMask]
				WriteMask[_StencilWriteMask]
			}

			Pass
			{
				CGPROGRAM
				#pragma vertex vert
				#pragma fragment frag

				//#pragma exclude_renderers d3d11_9x
				

				#pragma multi_compile_instancing

				#pragma multi_compile _VIEW _WORLD_VIEW _NORMAL _PERSPECTIVE
				#pragma multi_compile _ _RIGHT_EYE_ST
				#pragma multi_compile OUTSIDE_CLIP OUTSIDE_COLOR OUTSIDE_CLAMP
				#pragma multi_compile TINT_TEX_NONE TINT_TEX_DIRECT TINT_TEX_LERP
				#pragma multi_compile _ _CLAMP_INTENSITY
				#pragma multi_compile _ SECOND_TEXTURE
				#pragma multi_compile EQUIRECTANGULAR CUBEMAP CUBEMAP_LOD
				#pragma multi_compile _ _OFFSET
				//#pragma multi_compile EQUIRECTANGULAR EQUIDISTANT

				#pragma multi_compile _ RECTCLIP

				#include "UnityCG.cginc"
				#include "UnityUI.cginc"
				#include "UnityStandardUtils.cginc"

				struct appdata
				{
					float4 vertex : POSITION;

	#ifdef _NORMAL
					float3 normal : NORMAL;
	#endif

	#ifdef _PERSPECTIVE
					float2 texcoord : TEXCOORD0;
	#endif

					UNITY_VERTEX_INPUT_INSTANCE_ID
				};

				struct v2f
				{
					float4 vertex : SV_POSITION;
					float4 pos: TEXCOORD1;
					float dist : TEXCOORD2;

	#ifdef _NORMAL
					float3 normal : NORMAL;
	#endif

	#ifdef _PERSPECTIVE
					float2 texcoord : TEXCOORD0;
	#endif

					float2 position : TEXCOORD3;

					UNITY_VERTEX_OUTPUT_STEREO
				};

	#ifdef CUBEMAP_LOD
				float _CubeLOD;
	#endif

	#if defined(CUBEMAP) || defined(CUBEMAP_LOD)
				samplerCUBE _MainCube;
	#else
				sampler2D _MainTex;
				float4 _MainTex_ST;
	#endif

	#ifdef _RIGHT_EYE_ST
				float4 _RightEye_ST;
	#endif

	#ifdef SECOND_TEXTURE
	#if defined(CUBEMAP) || defined(CUBEMAP_LOD)
				samplerCUBE _SecondCube;
	#else
				sampler2D _SecondTex;
	#endif
				float _TextureLerp;
				float2 _SecondTexOffset;
	#endif

				float _ProjectionLerp;
				float4 _Tint;
				float4 _FOV;
				float _Exposure;
				float _Gamma;
	#ifdef OUTSIDE_COLOR
				float4 _OutsideColor;
	#endif

	#ifdef _CLAMP_INTENSITY
				float _MaxIntensity;
	#endif

	#if defined(TINT_TEX_DIRECT) || defined(TINT_TEX_LERP)
				sampler2D _TintTex;
				float4 _TintTex_ST;
	#endif
	#if defined(TINT_TEX_LERP)
				float4 _Tint0;
				float4 _Tint1;
	#endif

	#ifdef _PERSPECTIVE
				float4 _PerspectiveFOV;
	#endif

	#ifdef _OFFSET
				sampler2D _OffsetTex;
				sampler2D _OffsetMask;
				float4 _OffsetTex_ST;
				float4 _OffsetMagnitude;
	#endif

				float4 _Rect;

				static const float PI = 3.14159265359;
				static const float TAU = 6.283185307;

				float2 dir_to_uv(float3 viewDir)
				{
					float2 angle = float2(
						atan2(viewDir.x, viewDir.z),
						acos(dot(viewDir, float3(0, 1, 0))) - PI * 0.5
						);

					// remap it to normalized UV
					float2 maxAngle = _FOV.xy * 0.5;

					angle += maxAngle;

					angle += _FOV.zw;
					angle = fmod(angle, float2(PI * 2, PI));
					angle += float2(PI * 2, PI);
					angle = fmod(angle, float2(PI * 2, PI));

					angle /= _FOV.xy;

					return angle;
				}

				void rotate_dir(inout float3 viewDir, float2 rotate)
				{
					float _sin, _cos;
					float3x3 rot;

					_sin = sin(rotate.y);
					_cos = cos(rotate.y);

					rot = float3x3(
						1, 0, 0,
						0, _cos, -_sin,
						0, _sin, _cos
						);

					viewDir = mul(rot, viewDir);

					_sin = sin(rotate.x);
					_cos = cos(rotate.x);

					rot = float3x3(
						_cos, 0, _sin,
						0, 1, 0,
						-_sin, 0, _cos
						);

					viewDir = mul(rot, viewDir);
				}

				v2f vert(appdata v)
				{
					v2f o;

					UNITY_SETUP_INSTANCE_ID(v);
					UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

					o.vertex = UnityObjectToClipPos(v.vertex);
					o.pos = v.vertex;
					o.dist = o.vertex.w; // some shader profiles disallow reading from position semantics in fragment
	#ifdef _NORMAL
					o.normal = -v.normal;
	#endif

	#ifdef _PERSPECTIVE
					o.texcoord = v.texcoord;
	#endif


	#ifdef RECTCLIP
					o.position = v.vertex.xy;
	#endif

					return o;
				}

				half4 frag(v2f i) : SV_Target
				{
					//UNITY_SETUP_INSTANCE_ID(i);
					UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

	#ifdef RECTCLIP
				clip(UnityGet2DClipping(i.position, _Rect) - 0.1);
	#endif

					float3 viewDir = float3(0.0, 0.0, 1.0);
	#ifdef _VIEW
					viewDir = normalize(ObjSpaceViewDir(i.pos));
	#elif _WORLD_VIEW
					viewDir = normalize(WorldSpaceViewDir(i.pos));
	#elif _NORMAL
					viewDir = normalize(i.normal);
	#endif

	#ifdef _PERSPECTIVE
					float2 planePos = (i.texcoord - 0.5) * 2;
					planePos.y *= -1;

					float2 planeDir = tan(_PerspectiveFOV.xy * 0.5) * planePos;

					viewDir = normalize(float3(planeDir, 1));

					rotate_dir(viewDir, _PerspectiveFOV.zw);
	#endif

	#ifdef _OFFSET
					float2 offset_uv = dir_to_uv(viewDir);

					float2 offset = tex2Dlod(_OffsetTex, float4(offset_uv * _OffsetTex_ST.xy + _OffsetTex_ST.zw, 0, 0)).rg;
					float2 offsetMask = tex2Dlod(_OffsetMask, float4(offset_uv, 0, 0)).rg;
					float2 offsetMagnitude = offsetMask * _OffsetMagnitude.xy;

					offset = offset * 2 - 1;
					offset *= offsetMagnitude;

					rotate_dir(viewDir, offset);
	#endif

	#if defined(CUBEMAP) || defined(CUBEMAP_LOD)

					viewDir *= -1;

	#ifdef CUBEMAP
					float4 c = texCUBE(_MainCube, viewDir);
	#else
					float4 c = texCUBElod(_MainCube, float4(viewDir, _CubeLOD));
	#endif

	#ifdef SECOND_TEXTURE

	#ifdef CUBEMAP
					float4 sc = texCUBE(_SecondCube, viewDir);
	#else
					float4 sc = texCUBElod(_SecondCube, float4(viewDir, _CubeLOD));
	#endif

					c = lerp(c, sc, _TextureLerp);
	#endif

	#else // CUBEMAP
					float2 uv = dir_to_uv(viewDir);

	#ifdef OUTSIDE_CLIP
					if (uv.x < 0 || uv.x > 1 || uv.y < 0 || uv.y > 1)
						discard;
	#elif defined(OUTSIDE_COLOR)
					if (uv.x < 0 || uv.x > 1 || uv.y < 0 || uv.y > 1)
						return _OutsideColor;
	#else
					uv = saturate(uv);
	#endif

					//#ifdef EQUIRECTANGULAR

					//#elif defined(EQUIDISTANT)
									// This is wrong!!! TODO!!!
									//float2 uv = sin((uv - 0.5)*PI)*0.5 + 0.5;
					//#endif

									float4 _ST;
					#if defined(_RIGHT_EYE_ST)
									if (unity_StereoEyeIndex == 0)
										_ST = _MainTex_ST;
									else
										_ST = _RightEye_ST;
					#else
									_ST = _MainTex_ST;
					#endif

									uv = uv * _ST.xy + _ST.zw;

									//return fixed4(u, v, 0, 1);
									float4 c = tex2Dlod(_MainTex, float4(uv, 0, 0));

					#ifdef SECOND_TEXTURE
									float4 sc = tex2Dlod(_SecondTex, float4(uv + _SecondTexOffset, 0, 0));
									c = lerp(c, sc, _TextureLerp);
					#endif

					#if defined(TINT_TEX_DIRECT)
									c *= tex2Dlod(_TintTex, float4(uv, 0, 0));
					#elif defined(TINT_TEX_LERP)
									float l = tex2Dlod(_TintTex, float4(uv * _TintTex_ST.xy + _TintTex_ST.wz, 0, 0));
									c *= lerp(_Tint0, _Tint1, l);
					#endif

					#endif // CUBEMAP

									float fade = saturate(clamp(i.dist - 0.05, 0, 0.1) * 10);

									half4 tint = _Tint;
									tint.a *= fade;

									c = float4(pow(c.xyz, _Gamma) * _Exposure, c.a) * tint;

					#ifdef _CLAMP_INTENSITY
									float m = max(c.x, max(c.y, c.z));
									if (m > _MaxIntensity)
										c.xyz *= _MaxIntensity / m;
					#endif

									return c;
								}
								ENDCG
							}
		}
}
