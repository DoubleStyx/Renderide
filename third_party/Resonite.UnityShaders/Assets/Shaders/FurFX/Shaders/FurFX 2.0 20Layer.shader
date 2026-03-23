Shader "FurFX/2.0/FurFX 20 Layer"
{
	Properties
	{
		_Color ("Color (RGB)", Color) = (1,1,1,1)
	  	_SpecColor ("Specular Material Color (RGB)", Color) = (1,1,1,1) 
	  	_BonusAmbient ("Bonus Ambient", Color) = (0,0,0,1)


	  	_Shininess ("Shininess", Range (0.01, 10)) = 8		
	  	_Gloss ("Gloss", Range (0.0, 3)) = 1        
	  		  	
	  	_RimColor ("Rim Color", Color) = (0.0,0.0,0.0,0.0)
      	_RimPower ("Rim Power", Range(0.5,8.0)) = 4.0	  	
	  	
		_FurLength ("Fur Length", Range (.0002, 0.5)) = .05
		_MainTex ("Base (RGB) Mask(A)", 2D) = "white" { }
		_BumpMap ("Normalmap", 2D) = "bump" {}
		_NoiseTex ("Noise (RGB)", 2D) = "white" { }
		
		_Cube ("Reflection Map", Cube) = "" {}
		_ReflColor ("Reflection Color (RGB)", Color) = (1,1,1,1)
		_Reflection("Reflection Power", Range (0.00, 1)) = 0.0
		_ReflMinLevel("Reflection Minimal Length", Range (0.00, 1)) = 0.0
				
		_Cutoff ("Alpha Cutoff", Range (0, 1)) = 0.2
		
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
		Tags {"Queue"="AlphaTest" "IgnoreProjector"="True" "RenderType"="TransparentCutout"}
		LOD 200
		
		SubShader
		{
		
      	UsePass "VertexLit/SHADOWCOLLECTOR"    
      	UsePass "VertexLit/SHADOWCASTER"		
		

				CGPROGRAM
				#pragma surface surfBase BlinnPhong addshadow
				#define FUR_MULTIPLIER 0.0
				#include "FurFX BrandNew.cginc"
				ENDCG
				
				CGPROGRAM
				#pragma surface surfFur BlinnPhong vertex:vert alphatest:_Cutoff addshadow
				#pragma target 3.0
				#define FUR_MULTIPLIER 0.05
				#include "FurFX BrandNew.cginc"
				ENDCG
				
				CGPROGRAM
				#pragma surface surfFur BlinnPhong vertex:vert alphatest:_Cutoff addshadow
				#pragma target 3.0
				#define FUR_MULTIPLIER 0.1
				#include "FurFX BrandNew.cginc"
				ENDCG	
				
				CGPROGRAM
				#pragma surface surfFur BlinnPhong vertex:vert alphatest:_Cutoff addshadow
				#pragma target 3.0
				#define FUR_MULTIPLIER 0.15
				#include "FurFX BrandNew.cginc"
				ENDCG	
				
				CGPROGRAM
				#pragma surface surfFur BlinnPhong vertex:vert alphatest:_Cutoff addshadow
				#pragma target 3.0
				#define FUR_MULTIPLIER 0.2
				#include "FurFX BrandNew.cginc"
				ENDCG	
				
				CGPROGRAM 
				#pragma surface surfFur BlinnPhong vertex:vert alphatest:_Cutoff addshadow
				#pragma target 3.0
				#define FUR_MULTIPLIER 0.25
				#include "FurFX BrandNew.cginc"
				ENDCG			
				
				CGPROGRAM
				#pragma surface surfFur BlinnPhong vertex:vert alphatest:_Cutoff addshadow
				#pragma target 3.0
				#define FUR_MULTIPLIER 0.3
				#include "FurFX BrandNew.cginc"
				ENDCG
				
				CGPROGRAM
				#pragma surface surfFur BlinnPhong vertex:vert alphatest:_Cutoff addshadow
				#pragma target 3.0
				#define FUR_MULTIPLIER 0.35
				#include "FurFX BrandNew.cginc"
				ENDCG	
				
				CGPROGRAM
				#pragma surface surfFur BlinnPhong vertex:vert alphatest:_Cutoff addshadow
				#pragma target 3.0
				#define FUR_MULTIPLIER 0.4
				#include "FurFX BrandNew.cginc"
				ENDCG	
				
				CGPROGRAM
				#pragma surface surfFur BlinnPhong vertex:vert alphatest:_Cutoff addshadow
				#pragma target 3.0
				#define FUR_MULTIPLIER 0.45
				#include "FurFX BrandNew.cginc"
				ENDCG	
				
				CGPROGRAM
				#pragma surface surfFur BlinnPhong vertex:vert alphatest:_Cutoff addshadow
				#pragma target 3.0
				#define FUR_MULTIPLIER 0.5
				#include "FurFX BrandNew.cginc"
				ENDCG	
				
				CGPROGRAM 
				#pragma surface surfFur BlinnPhong vertex:vert alphatest:_Cutoff addshadow
				#pragma target 3.0
				#define FUR_MULTIPLIER 0.55
				#include "FurFX BrandNew.cginc"
				ENDCG
				
				CGPROGRAM
				#pragma surface surfFur BlinnPhong vertex:vert alphatest:_Cutoff addshadow
				#pragma target 3.0
				#define FUR_MULTIPLIER 0.6
				#include "FurFX BrandNew.cginc"
				ENDCG	
				
				CGPROGRAM
				#pragma surface surfFur BlinnPhong vertex:vert alphatest:_Cutoff addshadow
				#pragma target 3.0
				#define FUR_MULTIPLIER 0.65
				#include "FurFX BrandNew.cginc"
				ENDCG	
				
				CGPROGRAM
				#pragma surface surfFur BlinnPhong vertex:vert alphatest:_Cutoff addshadow
				#pragma target 3.0
				#define FUR_MULTIPLIER 0.7
				#include "FurFX BrandNew.cginc"
				ENDCG	
				
				CGPROGRAM
				#pragma surface surfFur BlinnPhong vertex:vert alphatest:_Cutoff addshadow
				#pragma target 3.0
				#define FUR_MULTIPLIER 0.75
				#include "FurFX BrandNew.cginc"
				ENDCG			
				
				CGPROGRAM
				#pragma surface surfFur BlinnPhong vertex:vert alphatest:_Cutoff addshadow
				#pragma target 3.0
				#define FUR_MULTIPLIER 0.8
				#include "FurFX BrandNew.cginc"
				ENDCG
				
				CGPROGRAM
				#pragma surface surfFur BlinnPhong vertex:vert alphatest:_Cutoff addshadow
				#pragma target 3.0
				#define FUR_MULTIPLIER 0.85
				#include "FurFX BrandNew.cginc"
				ENDCG	
				
				CGPROGRAM
				#pragma surface surfFur BlinnPhong vertex:vert alphatest:_Cutoff addshadow
				#pragma target 3.0
				#define FUR_MULTIPLIER 0.9
				#include "FurFX BrandNew.cginc"
				ENDCG	
				
				CGPROGRAM
				#pragma surface surfFur BlinnPhong vertex:vert alphatest:_Cutoff addshadow
				#pragma target 3.0
				#define FUR_MULTIPLIER 0.95
				#include "FurFX BrandNew.cginc"
				ENDCG	
				
				CGPROGRAM
				#pragma surface surfFur BlinnPhong vertex:vert alphatest:_Cutoff addshadow
				#pragma target 3.0
				#define FUR_MULTIPLIER 1.0
				#include "FurFX BrandNew.cginc"
				ENDCG	
																																																																																																																																																																																																																																																																																																																																																																																																																																										
				
		}		
		//FallBack "Transparent/Cutout/VertexLit"
		Fallback "VertexLit",1
	}
}
