Shader "UI/Unlit"
{
	Properties
	{
		[PerRendererData] _MainTex("Sprite Texture", 2D) = "white" {}
		_Tint("Tint", Color) = (1,1,1,1)
		_OverlayTint ("OverlayTint", Color) = (1,1,1,0.5)
		_Cutoff("Alpha cutoff", Range(0,1)) = 0.98
		_Rect("Rect", Vector) = (0,0,1,1)

		_SrcBlend("SrcBlend", Float) = 1.0
		_DstBlend("DstBlend", Float) = 0.0
		_ZWrite("ZWrite", Float) = 1.0
		_Cull("Cull", Float) = 2.0

		[PerRendererData] _MaskTex("Mask Texture", 2D) = "white" {}

		_StencilComp("Stencil Comparison", Float) = 8
		_Stencil("Stencil ID", Float) = 0
		_StencilOp("Stencil Operation", Float) = 0
		_StencilWriteMask("Stencil Write Mask", Float) = 255
		_StencilReadMask("Stencil Read Mask", Float) = 255

		_ColorMask("Color Mask", Float) = 15
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

			Pass
			{
			CGPROGRAM
				#pragma vertex vert
				#pragma fragment frag
				#pragma multi_compile_instancing
				#pragma target 2.0

				#include "UnityCG.cginc"
				#include "UnityUI.cginc"
				#include "UnityStandardUtils.cginc"

				#pragma multi_compile _ ALPHACLIP
				#pragma multi_compile _ RECTCLIP
				#pragma multi_compile _ OVERLAY
				#pragma multi_compile _ TEXTURE_NORMALMAP TEXTURE_LERPCOLOR
				#pragma multi_compile _ _MASK_TEXTURE_MUL _MASK_TEXTURE_CLIP

				struct appdata_t
				{
					float4 vertex   : POSITION;
					float4 color    : COLOR;
					float2 texcoord : TEXCOORD0;

					float4 lerpColor : TANGENT;

					UNITY_VERTEX_INPUT_INSTANCE_ID
				};

				struct v2f
				{
					float4 vertex   : SV_POSITION;
					float4 color : COLOR;
					float2 texcoord  : TEXCOORD0;

					float4 lerpColor : TANGENT;

					float2 position : TEXCOORD1;

					float4 projPos : TEXCOORD2;

					UNITY_VERTEX_OUTPUT_STEREO
				};

				sampler2D _MainTex;
				float4 _MainTex_ST;

				half4 _Tint;

				float _Cutoff;		

				sampler2D _MaskTex;
				float4 _MaskTex_ST;

float4 _Rect;

				float4 _OverlayTint;
				UNITY_DECLARE_DEPTH_TEXTURE(_CameraDepthTexture);

				v2f vert(appdata_t v)
				{
					v2f OUT;

					UNITY_SETUP_INSTANCE_ID(v);
					UNITY_INITIALIZE_OUTPUT(v2f, OUT);
					UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);

					OUT.position = v.vertex.xy;
					OUT.vertex = UnityObjectToClipPos(v.vertex);

					OUT.texcoord = TRANSFORM_TEX(v.texcoord, _MainTex);

					OUT.color = v.color * _Tint;

					OUT.lerpColor = v.lerpColor * _Tint;

					OUT.projPos = ComputeScreenPos(OUT.vertex);
					COMPUTE_EYEDEPTH(OUT.projPos.z);

					return OUT;
				}

				fixed4 frag(v2f IN) : SV_Target
				{
					//UNITY_SETUP_INSTANCE_ID(IN);
					UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);

#ifdef RECTCLIP
					clip(UnityGet2DClipping(IN.position, _Rect) - 0.1);
#endif

					half4 color = tex2D(_MainTex, IN.texcoord);

#ifdef TEXTURE_NORMALMAP
					color = half4(UnpackScaleNormal(color, 1) * 0.5 + 0.5, 1);
#endif

#ifdef TEXTURE_LERPCOLOR
					half l = (color.r + color.g + color.b) * 0.3333333333;
					half4 lerpColor = lerp(IN.color, IN.lerpColor, l);

					color = half4(lerpColor.rgb, lerpColor.a * color.a);
#else
					color *= IN.color;
#endif

#if defined(_MASK_TEXTURE_MUL) || defined(_MASK_TEXTURE_CLIP)
					float4 mask = tex2D(_MaskTex, TRANSFORM_TEX(IN.texcoord, _MaskTex));

					float mul = (mask.r + mask.g + mask.b) * 0.3333333 * mask.a;

#ifdef _MASK_TEXTURE_MUL
					color.a *= mul;
#endif

#ifdef _MASK_TEXTURE_CLIP
					if (mul - _Cutoff <= 0)
						discard;
#endif

#endif // MASK TEXTURE

#if defined(ALPHACLIP) && !defined(_MASK_TEXTURE_CLIP)
					clip(color.a - _Cutoff);
#endif

#ifdef OVERLAY
					float sceneZ = LinearEyeDepth(SAMPLE_DEPTH_TEXTURE_PROJ(_CameraDepthTexture, UNITY_PROJ_COORD(IN.projPos)));
					float partZ = IN.projPos.z;

					if (partZ > sceneZ)
						color *= _OverlayTint;
#endif

					return color;
				}
			ENDCG
			}
		}
}
