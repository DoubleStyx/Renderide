Shader "Invisible"
{
	Properties
	{
	}
	SubShader
	{
		Tags { "RenderType"="Transparent" "Queue" = "Transparent" }
		LOD 100

		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag

			#pragma multi_compile_instancing
			
			#include "UnityCG.cginc"

			struct v2f
			{
				fixed4 vertex : SV_POSITION;
			};
			
			v2f vert ()
			{
				v2f o;
				o.vertex = fixed4(0,0,0,0);
				return o;
			}
			
			fixed4 frag (v2f i) : SV_Target
			{
				discard;
				return fixed4(0,0,0,0);
			}
			ENDCG
		}
	}
}
