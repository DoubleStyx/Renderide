Shader "Filters/Gamma"
{
	Properties
	{
		_Gamma("Gamma", Float) = 2.2

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
#pragma vertex vert
#pragma fragment frag
#include "UnityCG.cginc"
#include "UnityUI.cginc"
#include "UnityStandardUtils.cginc"


#pragma exclude_renderers d3d11_9x

#pragma multi_compile _ RECTCLIP

			uniform float _Gamma;

			static const float PI = 3.14159265359;
			static const float TAU = 6.283185307;

#ifdef RECTCLIP
			float4 _Rect;
#endif

			struct v2f
			{
				float2 uv : TEXCOORD0;
				float4 grabPos : TEXCOORD1;
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

				half4 c = tex2D(_GrabTexture, grabUv);

				c.rgb = pow(c.rgb, _Gamma);

				return c;
			}
			ENDCG
		}

	}
}