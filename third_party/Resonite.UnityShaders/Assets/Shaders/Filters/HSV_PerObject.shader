 Shader "Filters/HSV_PerObject"
{
	Properties
	{
		_HSVOffset("HSV Offset", Vector) = (0.2, 0.2, 0.2, 0)
		_HSVMul("HSV Mul", Vector) = (1, 1, 1, 1)

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

			uniform float4 _HSVOffset;
			uniform float4 _HSVMul;

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

			// conversion from: https://forum.unity.com/threads/different-blending-modes-like-add-screen-overlay-changing-hue-tint.62507/#post-413034

			float3 rgb_to_hsv_no_clip(float3 RGB)
			{
				float3 HSV;

				float minChannel, maxChannel;
				if (RGB.x > RGB.y) {
					maxChannel = RGB.x;
					minChannel = RGB.y;
				}
				else {
					maxChannel = RGB.y;
					minChannel = RGB.x;
				}

				if (RGB.z > maxChannel) maxChannel = RGB.z;
				if (RGB.z < minChannel) minChannel = RGB.z;

				HSV.xy = 0;
				HSV.z = maxChannel;
				float delta = maxChannel - minChannel;             //Delta RGB value
				if (delta != 0) {                    // If gray, leave H  S at zero
					HSV.y = delta / HSV.z;
					float3 delRGB;
					delRGB = (HSV.zzz - RGB + 3 * delta) / (6.0*delta);
					if (RGB.x == HSV.z) HSV.x = delRGB.z - delRGB.y;
					else if (RGB.y == HSV.z) HSV.x = (1.0 / 3.0) + delRGB.x - delRGB.z;
					else if (RGB.z == HSV.z) HSV.x = (2.0 / 3.0) + delRGB.y - delRGB.x;
				}
				return (HSV);
			}

			float3 hsv_to_rgb(float3 HSV)
			{
				float3 RGB = HSV.z;

				float var_h = HSV.x * 6;
				float var_i = floor(var_h);   // Or ... var_i = floor( var_h )
				float var_1 = HSV.z * (1.0 - HSV.y);
				float var_2 = HSV.z * (1.0 - HSV.y * (var_h - var_i));
				float var_3 = HSV.z * (1.0 - HSV.y * (1 - (var_h - var_i)));
				if (var_i == 0) { RGB = float3(HSV.z, var_3, var_1); }
				else if (var_i == 1) { RGB = float3(var_2, HSV.z, var_1); }
				else if (var_i == 2) { RGB = float3(var_1, HSV.z, var_3); }
				else if (var_i == 3) { RGB = float3(var_1, var_2, HSV.z); }
				else if (var_i == 4) { RGB = float3(var_3, var_1, HSV.z); }
				else { RGB = float3(HSV.z, var_1, var_2); }

				return (RGB);
			}

			half4 frag(v2f i) : SV_Target
			{
#ifdef RECTCLIP
				clip(UnityGet2DClipping(i.position, _Rect) - 0.1);
#endif

				float2 grabUv = i.grabPos.xy / i.grabPos.w;

				half4 c = tex2D(_GrabTexture, grabUv);

				float3 hsv = rgb_to_hsv_no_clip(c.rgb);

				hsv *= _HSVMul.xyz;
				hsv += _HSVOffset.xyz;
				hsv.x = frac(hsv.x);
				hsv.y = saturate(hsv.y);

				c.rgb = hsv_to_rgb(hsv);

				return c;
			}
			ENDCG
		}

	}
}