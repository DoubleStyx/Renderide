Shader "Unlit/PolarGrid"
{
	Properties
	{
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
			// make fog work
			#pragma multi_compile_fog
			
			#include "UnityCG.cginc"

			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
			};

			struct v2f
			{
				float2 uv : TEXCOORD0;
				UNITY_FOG_COORDS(1)
				float4 vertex : SV_POSITION;
			};

			v2f vert (appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.uv = v.uv;
				UNITY_TRANSFER_FOG(o,o.vertex);
				return o;
			}
			
			fixed4 frag (v2f i) : SV_Target
			{
				// shift UV coordinates so they cover -1 to 1 range
				i.uv = (i.uv * 2) - 1;

				// compute polar coordinates
				float radius = length(i.uv);
				float angle = atan2(i.uv.x, i.uv.y);

				float refRadius = round(radius * 100) / 100;
				float dist = abs(radius - refRadius) * 100;
				float aaf = fwidth(dist);
				float raaf = fwidth(radius * 100);

				dist = 1 - smoothstep(0.05 - aaf, 0.05, dist);

				float debug = smoothstep(0.15, 0.25, raaf);

				dist -= debug;

				float4 col = float4(dist * debug, dist * (1-debug), 0, 1);

				// apply fog
				//UNITY_APPLY_FOG(i.fogCoord, col);
				return col;
			}
			ENDCG
		}
	}
}
