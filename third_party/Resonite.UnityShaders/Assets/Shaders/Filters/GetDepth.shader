Shader "Filters/Get Depth"
{
	Properties
	{
		_Multiply("Multiply", Float) = 1
		_Offset("Offset", Float) = 0

		_ClipMin("Clip Min", Float) = 0
		_ClipMax("Clip Max", Float) = 1

		_SrcBlend("SrcBlend", Float) = 1.0
		_DstBlend("DstBlend", Float) = 0.0
		_ZWrite("ZWrite", Float) = 0
		_Cull("Cull", Float) = 2.0

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
			"Queue" = "Transparent+500"
			"IgnoreProjector" = "True"
			"RenderType" = "Transparent"
			"PreviewType" = "Plane"
			"CanUseSpriteAtlas" = "True"
		}

		Stencil
		{
			Ref[_Stencil]
			Comp[_StencilComp]
			Pass[_StencilOp]
			ReadMask[_StencilReadMask]
			WriteMask[_StencilWriteMask]
		}

		Offset[_OffsetFactor],[_OffsetUnits]
		Blend[_SrcBlend][_DstBlend], One One
		BlendOp Add, Max
		ZWrite[_ZWrite]
		Cull[_Cull]
		ZTest[_ZTest]
		ColorMask[_ColorMask]

		// Render the object with the texture generated above, and invert the colors
		Pass
	{
		CGPROGRAM
#pragma vertex vert
#pragma fragment frag
#include "UnityCG.cginc"
#include "UnityUI.cginc"
#include "UnityStandardUtils.cginc"


#pragma exclude_renderers d3d11_9x

#pragma multi_compile _ CLIP
#pragma multi_compile _ RECTCLIP

			float _Multiply;
			float _Offset;

#ifdef CLIP
			float _ClipMin;
			float _ClipMax;
#endif

#ifdef RECTCLIP
			float4 _Rect;
#endif

			UNITY_DECLARE_DEPTH_TEXTURE(_CameraDepthTexture);

			static const float PI = 3.14159265359;
			static const float TAU = 6.283185307;

			struct v2f
			{
				float2 uv : TEXCOORD0;
				float4 projPos : TEXCOORD1;
				float4 pos : SV_POSITION;

#ifdef RECTCLIP
				float2 position : TEXCOORD2;
#endif
			};

			v2f vert(appdata_full v)
			{
				v2f o;
				// use UnityObjectToClipPos from UnityCG.cginc to calculate 
				// the clip-space of the vertex
				o.pos = UnityObjectToClipPos(v.vertex);
				o.uv = v.texcoord.xy;

				// use ComputeGrabScreenPos function from UnityCG.cginc
				// to get the correct texture coordinate
				o.projPos = ComputeScreenPos(o.pos);
				COMPUTE_EYEDEPTH(o.projPos.z);


#ifdef RECTCLIP
				o.position = v.vertex.xy;
#endif

				return o;
			}

			sampler2D _GrabTexture;

			half4 frag(v2f i) : SV_Target
			{
#ifdef RECTCLIP
				clip(UnityGet2DClipping(i.position, _Rect) - 0.1);
#endif

				float depth = LinearEyeDepth(SAMPLE_DEPTH_TEXTURE_PROJ(_CameraDepthTexture, UNITY_PROJ_COORD(i.projPos)));

#ifdef CLIP
				depth -= _ClipMin;
				depth /= _ClipMax - _ClipMin;
#endif

				depth *= _Multiply;
				depth += _Offset;

				depth = saturate(depth);

				return half4(depth.xxx, 1);
			}
			ENDCG
		}

	}
}