Shader "FurFX/1.0/FurFX Basic 40 Layer"
{
	Properties
	{
		_Color ("Color (RGB)", Color) = (1,1,1,1)
	  	_SpecColor ("Specular Material Color (RGB)", Color) = (1,1,1,1) 
	  	_Shininess ("Shininess", Range (0.01, 10)) = 8		
		_FurLength ("Fur Length", Range (.0002, 0.5)) = .05
		_MainTex ("Base (RGB) Mask(A)", 2D) = "white" { }
		_NoiseTex ("Noise (RGB)", 2D) = "white" { }
		_Cutoff ("Alpha Cutoff", Range (0, 1)) = .0001
		_EdgeFade ("Edge Fade", Range(0,1)) = 0.15
		_HairHardness ("Fur Hardness", Range(0.1,1)) = 1
		_HairThinness ("Fur Thinness", Range(0.01,10)) = 2
		_HairShading ("Fur Shading", Range(0.0,1)) = 0.25
		_HairColoring ("Fur Coloring", Range(0.0,1)) = 0.1
		_SkinAlpha ("Mask Alpha Factor", Range(0.0,1)) = 0.5
		_ForceGlobal ("Force Global",Vector) = (0,0,0,0)		
		_ForceLocal ("Force Local",Vector) = (0,0,0,0)	
	}
	Category
	{
		ZWrite On
		//Tags {"Queue" = "Transparent"}
		Tags { "QUEUE"="Transparent" "RenderType"="Opaque" " IgnoreProjector"="True"}	
		Tags { "LightMode" = "ForwardBase" }
		Blend SrcAlpha OneMinusSrcAlpha
		Alphatest Greater [_Cutoff]
		
		SubShader
		{
			Pass
			{
				//ZWrite On
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment fragBase
				#define FUR_MULTIPLIER 0.0
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.25
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.05
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.075
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.1
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.125
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.15
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.175
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.2
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.225
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.25
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.275
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.3
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.325
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.35
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.375
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.4
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.425
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.45
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.475
				#include "FurFX.cginc"
				ENDCG

			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.5
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.525
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.55
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.575
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.6
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.625
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.65
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.675
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.7
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.725
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.75
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.775
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.8
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.825
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.85
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.875
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.9
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.925
				#include "FurFX.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.95
				#include "FurFX.cginc"
				ENDCG

			}
			Pass
			{
								CGPROGRAM
				

				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.975
				#include "FurFX.cginc"
				ENDCG

			}
			
		}		Fallback "Diffuse", 1
	}
}
