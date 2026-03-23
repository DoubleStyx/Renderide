// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

Shader "CubemapProjection"
{
	Properties
	{
		_Cube ("Cubemap", CUBE) = "" {}
	}
	SubShader
	{
		Tags { "RenderType"="Opaque" }
		LOD 100

		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag

			#pragma multi_compile _ FLIP
			
			#include "UnityCG.cginc"

			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
			};

			struct v2f
			{
				float2 uv : TEXCOORD0;
				float4 vertex : SV_POSITION;
			};

			samplerCUBE _Cube;
			float4x4 _Rotation;
			
			v2f vert(appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.uv = v.uv;
				return o;
			}
			
			half4 frag (v2f i) : SV_Target
			{
				// convert UV coordinates to directional vector using equirectangular projection
				float hAngle = i.uv.x * 6.283185307;
				float vAngle = ((1-i.uv.y) - 0.5) * 3.14159265359;

				float3 dir = float3(0,0,1);

				float3x3 mat = float3x3(
					1, 0, 0,
					0, cos(vAngle), -sin(vAngle),
					0, sin(vAngle), cos(vAngle)
					);

				dir = mul(mat, dir);

				mat = float3x3(
					cos(hAngle), 0, sin(hAngle),
					0, 1, 0,
					-sin(hAngle), 0, cos(hAngle)
					);

				dir = mul(mat, dir);

				dir = mul((float3x3)_Rotation, dir);

#ifdef FLIP
				dir *= -1;
#endif

				//return fixed4((dir/2) + 0.5, 1);

				return texCUBE(_Cube, dir);
			}
			ENDCG
		}
	}
}
