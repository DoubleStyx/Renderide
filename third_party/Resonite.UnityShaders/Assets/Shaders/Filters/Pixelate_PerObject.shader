Shader "Filters/Pixelate_PerObject"
{
	Properties
	{
		_Resolution("Resolution", Vector) = (100, 100, 0, 0)
		_ResolutionTex ("Resolution Tex", 2D) = "white" {}

		_SrcBlend("SrcBlend", Float) = 1.0
		_DstBlend("DstBlend", Float) = 0.0
		_ZWrite("ZWrite", Float) = 1.0
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

		// Grab the screen behind the object into _GrabTexture
		GrabPass
		{
			
		}

		// Render the object with the texture generated above, and invert the colors
		Pass
	{
		CGPROGRAM


#pragma multi_compile_instancing

#pragma vertex vert
#pragma fragment frag
#include "UnityCG.cginc"
#include "UnityUI.cginc"
#include "UnityStandardUtils.cginc"

#pragma multi_compile _ RESOLUTION_TEX
#pragma multi_compile _ RECTCLIP

		float2 _Resolution;

#ifdef RESOLUTION_TEX
		sampler2D _ResolutionTex;
		float4 _ResolutionTex_ST;
#endif

#ifdef RECTCLIP
		float4 _Rect;
#endif

		static const float PI = 3.14159265359;
		static const float TAU = 6.283185307;

		struct v2f
		{
			float2 uv : TEXCOORD0;
			float4 grabPos : TEXCOORD1;
			float4 pos : SV_POSITION;

#ifdef RECTCLIP
			float2 position : TEXCOORD2;
#endif
		};

		v2f vert(appdata_base v)
		{
			v2f o;

			UNITY_SETUP_INSTANCE_ID(v);
			// use UnityObjectToClipPos from UnityCG.cginc to calculate 
			// the clip-space of the vertex
			o.pos = UnityObjectToClipPos(v.vertex);
			o.uv = v.texcoord.xy;
			// use ComputeGrabScreenPos function from UnityCG.cginc
			// to get the correct texture coordinate
			o.grabPos = ComputeGrabScreenPos(o.pos);

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

			float2 grabUv = i.grabPos.xy / i.grabPos.w;

#ifdef RESOLUTION_TEX
			float2 size = _Resolution * tex2D(_ResolutionTex, TRANSFORM_TEX(i.uv, _ResolutionTex)).rg;
#else
			float2 size = _Resolution;
#endif
			grabUv = round(grabUv * size) / size;

			return tex2D(_GrabTexture, grabUv);
		}
		ENDCG
	}

	}
}