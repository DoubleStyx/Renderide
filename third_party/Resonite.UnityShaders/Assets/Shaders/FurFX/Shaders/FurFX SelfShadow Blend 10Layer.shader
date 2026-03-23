Shader "FurFX/!Test Shaders/FurFX SelfShadow Blend 10 Layer"
{
	Properties
	{
		_Color ("Color (RGB)", Color) = (1,1,1,1)
	  	_SpecColor ("Specular Material Color (RGB)", Color) = (1,1,1,1) 
	  	_RimColor ("Rim Color", Color) = (0.0,0.0,0.0,0.0)
      	_RimPower ("Rim Power", Range(0.5,8.0)) = 4.0
	  	_Shininess ("Shininess", Range (0.01, 10)) = 8		
		_FurLength ("Fur Length", Range (.0002, 0.5)) = .05
		_MainTex ("Base (RGB) Mask(A)", 2D) = "white" { }
		_NoiseTex ("Noise (RGB)", 2D) = "white" { }
		_Cube ("Reflection Map", Cube) = "" {}
		_Reflection("Reflection Power", Range (0.00, 1)) = 0.0
		_Cutoff ("Alpha Cutoff", Range (0, 1)) = .0001
		_EdgeFade ("Edge Fade", Range(0,1)) = 0.15
		_HairHardness ("Fur Hardness", Range(0.1,1)) = 1
		_HairThinness ("Fur Thinness", Range(0.01,10)) = 2
		_HairShading ("Fur Shading", Range(0.0,1)) = 0.25
		_HairColoring ("Fur Coloring", Range(0.0,1)) = 0.1
		_SkinAlpha ("Mask Alpha Factor", Range(0.0,1)) = 0.5
		_ForceGlobal ("Force Global",Vector) = (0,0,0,0)		
		_ForceLocal ("Force Local",Vector) = (0,0,0,0)		
		_ShadowStrength ("Shadow Strength", Range(0.0,1)) = 1
	}
	Category
	{

		ZWrite On
		Tags { "QUEUE"="AlphaTest" "RenderType"="TransparentCutout" " IgnoreProjector"="True"}	
		Tags { "LightMode" = "ForwardBase" }
		Blend SrcAlpha OneMinusSrcAlpha
		Alphatest Greater [_Cutoff]
		//AlphaTest GEqual [_Cutoff]
		  		  
		SubShader
		{
				
      	UsePass "VertexLit/SHADOWCOLLECTOR"    
      	UsePass "VertexLit/SHADOWCASTER"
       
			Pass
			{
				ZWrite On
								CGPROGRAM
								#pragma only_renderers d3d9 opengl
				#pragma target 2.0
				#pragma multi_compile_fwdbase
				#pragma vertex VertexProgram
				#pragma fragment fragBase
				#define FUR_MULTIPLIER 0.0
				#include "FurFX Pixel.cginc"
				ENDCG
			}
			
			
			Pass
			{
								CGPROGRAM
								#pragma only_renderers d3d9 opengl
				#pragma target 2.0
#pragma multi_compile_fwdbase
				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.1
				#include "FurFX Pixel.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
								#pragma only_renderers d3d9 opengl
				#pragma target 2.0
#pragma multi_compile_fwdbase
				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.2
				#include "FurFX Pixel.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
								#pragma only_renderers d3d9 opengl
				#pragma target 2.0
#pragma multi_compile_fwdbase
				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.3
				#include "FurFX Pixel.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
								#pragma only_renderers d3d9 opengl
				#pragma target 2.0
#pragma multi_compile_fwdbase
				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.4
				#include "FurFX Pixel.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
								#pragma only_renderers d3d9 opengl
				#pragma target 2.0
#pragma multi_compile_fwdbase
				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.5
				#include "FurFX Pixel.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
								#pragma only_renderers d3d9 opengl
				#pragma target 2.0
#pragma multi_compile_fwdbase
				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.6
				#include "FurFX Pixel.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
								#pragma only_renderers d3d9 opengl
				#pragma target 2.0
#pragma multi_compile_fwdbase
				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.7
				#include "FurFX Pixel.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
								#pragma only_renderers d3d9 opengl
				#pragma target 2.0
#pragma multi_compile_fwdbase
				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.8
				#include "FurFX Pixel.cginc"
				ENDCG
			}
			Pass
			{
								CGPROGRAM
								#pragma only_renderers d3d9 opengl
				#pragma target 2.0
#pragma multi_compile_fwdbase
				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 0.9
				#include "FurFX Pixel.cginc"
				ENDCG
			}
			
			Pass
			{
								CGPROGRAM
								#pragma only_renderers d3d9 opengl
				#pragma target 2.0
#pragma multi_compile_fwdbase
				#pragma vertex VertexProgram
				#pragma fragment frag
				#pragma fragmentoption ARB_fog_exp2
				#pragma fragmentoption ARB_precision_hint_fastest
				#define FUR_MULTIPLIER 1.00
				#include "FurFX Pixel.cginc"
				ENDCG

			}
		}		
		Fallback "VertexLit",1
		//FallBack "Transparent/Cutout/Diffuse"
	}
}
