// Made with Amplify Shader Editor
// Available at the Unity Asset Store - http://u3d.as/y3X 
Shader "Depth Finder AmplifyShader"
{
	Properties
	{
		[HDR]_PlayerOnlyCamera("Player Only Camera", 2D) = "white" {}
		[HDR]_OverlayStatic("Overlay Static", 2D) = "white" {}
		_ScrollSpeed("Scroll Speed", Vector) = (5,2,0,0)
		_StaticStrength("Static Strength", Range( 0 , 1)) = 0.5
		[HDR]_PlayerColour("Player Colour", Color) = (0,1,0.006896496,0)
		_GlitchTiling("Glitch Tiling ", Range( 0 , 5)) = 0
		_Smoothness("Smoothness", Range( 0 , 1)) = 0
		[HDR]_PixelTexture("Pixel Texture", 2D) = "white" {}
		_Backgroundcolour("Background colour", Color) = (0.4117647,0.4117647,0.4117647,0)
		_Resolution("Resolution", Vector) = (1,1,0,0)
		[HideInInspector] _texcoord( "", 2D ) = "white" {}
		[HideInInspector] __dirty( "", Int ) = 1
	}

	SubShader
	{
		Tags{ "RenderType" = "Opaque"  "Queue" = "Geometry+0" "IsEmissive" = "true"  }
		Cull Off
		CGPROGRAM
		#include "UnityShaderVariables.cginc"
		#pragma target 3.0
		#pragma surface surf Standard keepalpha noshadow 
		struct Input
		{
			float2 uv_texcoord;
		};

		uniform float4 _PlayerColour;
		uniform sampler2D _PlayerOnlyCamera;
		uniform float4 _PlayerOnlyCamera_ST;
		uniform float _StaticStrength;
		uniform sampler2D _OverlayStatic;
		uniform float2 _ScrollSpeed;
		uniform float _GlitchTiling;
		uniform float4 _Backgroundcolour;
		uniform sampler2D _PixelTexture;
		uniform float2 _Resolution;
		uniform float _Smoothness;


		float4 CalculateContrast( float contrastValue, float4 colorTarget )
		{
			float t = 0.5 * ( 1.0 - contrastValue );
			return mul( float4x4( contrastValue,0,0,t, 0,contrastValue,0,t, 0,0,contrastValue,t, 0,0,0,1 ), colorTarget );
		}

		void surf( Input i , inout SurfaceOutputStandard o )
		{
			float2 uv_PlayerOnlyCamera = i.uv_texcoord * _PlayerOnlyCamera_ST.xy + _PlayerOnlyCamera_ST.zw;
			float mulTime90 = _Time.y * 1;
			float2 temp_cast_0 = (_GlitchTiling).xx;
			float2 uv_TexCoord70 = i.uv_texcoord * temp_cast_0 + float2( 0,0 );
			float2 panner88 = ( uv_TexCoord70 + mulTime90 * _ScrollSpeed);
			float4 blendOpSrc91 = tex2D( _PlayerOnlyCamera, uv_PlayerOnlyCamera );
			float4 blendOpDest91 = CalculateContrast(_StaticStrength,tex2D( _OverlayStatic, panner88 ));
			float4 blendOpSrc150 = ( saturate( ( 1.0 - ( ( 1.0 - blendOpDest91) / blendOpSrc91) ) ));
			float4 blendOpDest150 = _Backgroundcolour;
			float2 uv_TexCoord140 = i.uv_texcoord * _Resolution + float2( 0,0 );
			float4 temp_output_139_0 = ( ( _PlayerColour * CalculateContrast(1.0,( saturate( 	max( blendOpSrc150, blendOpDest150 ) ))) ) * tex2D( _PixelTexture, uv_TexCoord140 ) );
			o.Albedo = temp_output_139_0.rgb;
			o.Emission = temp_output_139_0.rgb;
			o.Smoothness = _Smoothness;
			o.Alpha = 1;
		}

		ENDCG
	}
	CustomEditor "ASEMaterialInspector"
}
/*ASEBEGIN
Version=15001
124;386;1133;744;1482.22;978.4114;1;True;True
Node;AmplifyShaderEditor.RangedFloatNode;136;-1656.166,-251.548;Float;False;Property;_GlitchTiling;Glitch Tiling ;5;0;Create;True;0;0;False;0;0;0;0;5;0;1;FLOAT;0
Node;AmplifyShaderEditor.TextureCoordinatesNode;70;-1264.002,-240.0515;Float;False;0;-1;2;3;2;SAMPLER2D;;False;0;FLOAT2;1,1;False;1;FLOAT2;0,0;False;5;FLOAT2;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.SimpleTimeNode;90;-1449.936,-105.9447;Float;False;1;0;FLOAT;1;False;1;FLOAT;0
Node;AmplifyShaderEditor.Vector2Node;75;-1548.956,-406.6747;Float;False;Property;_ScrollSpeed;Scroll Speed;2;0;Create;True;0;0;False;0;5,2;5,2;0;3;FLOAT2;0;FLOAT;1;FLOAT;2
Node;AmplifyShaderEditor.PannerNode;88;-1263.607,-411.866;Float;False;3;0;FLOAT2;0,0;False;2;FLOAT2;0,0;False;1;FLOAT;0;False;1;FLOAT2;0
Node;AmplifyShaderEditor.SamplerNode;61;-1250.402,-619.6207;Float;True;Property;_OverlayStatic;Overlay Static;1;1;[HDR];Create;True;0;0;False;0;None;None;True;0;False;white;Auto;False;Object;-1;Auto;Texture2D;6;0;SAMPLER2D;;False;1;FLOAT2;0,0;False;2;FLOAT;0;False;3;FLOAT2;0,0;False;4;FLOAT2;0,0;False;5;FLOAT;0;False;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.RangedFloatNode;126;-885.7765,-270.4325;Float;False;Property;_StaticStrength;Static Strength;3;0;Create;True;0;0;False;0;0.5;0.509;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleContrastOpNode;125;-590.7095,-612.9009;Float;False;2;1;COLOR;0,0,0,0;False;0;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.SamplerNode;2;-1425.307,-823.0458;Float;True;Property;_PlayerOnlyCamera;Player Only Camera;0;1;[HDR];Create;True;0;0;False;0;None;2e2ed4787e7c50148a8bde251b54d7a3;True;0;False;white;Auto;False;Object;-1;Auto;Texture2D;6;0;SAMPLER2D;;False;1;FLOAT2;0,0;False;2;FLOAT;0;False;3;FLOAT2;0,0;False;4;FLOAT2;0,0;False;5;FLOAT;1;False;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;153;-810.9119,-642.9088;Float;False;Property;_Backgroundcolour;Background colour;8;0;Create;True;0;0;False;0;0.4117647,0.4117647,0.4117647,0;0,0,0,0;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.BlendOpsNode;91;-782.8251,-807.1982;Float;False;ColorBurn;True;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.BlendOpsNode;150;-547.211,-810.6104;Float;False;Lighten;True;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.RangedFloatNode;128;-332.0972,-607.9476;Float;False;Constant;_Float1;Float 1;3;0;Create;True;0;0;False;0;1;0;0;0;0;1;FLOAT;0
Node;AmplifyShaderEditor.Vector2Node;155;-821.5461,-1088.677;Float;False;Property;_Resolution;Resolution;9;0;Create;True;0;0;False;0;1,1;500,500;0;3;FLOAT2;0;FLOAT;1;FLOAT;2
Node;AmplifyShaderEditor.TextureCoordinatesNode;140;-621.9002,-940.3093;Float;False;0;-1;2;3;2;SAMPLER2D;;False;0;FLOAT2;1,1;False;1;FLOAT2;0,0;False;5;FLOAT2;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.ColorNode;133;-367.728,-518.9161;Float;False;Property;_PlayerColour;Player Colour;4;1;[HDR];Create;True;0;0;False;0;0,1,0.006896496,0;1,1,1,0;0;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.SimpleContrastOpNode;127;-308.9974,-770.2018;Float;False;2;1;COLOR;0,0,0,0;False;0;FLOAT;0;False;1;COLOR;0
Node;AmplifyShaderEditor.SamplerNode;138;-309.5473,-962.4976;Float;True;Property;_PixelTexture;Pixel Texture;7;1;[HDR];Create;True;0;0;False;0;None;0a690bd550b83874c9bbd852c2cd721e;True;0;False;white;Auto;False;Object;-1;Auto;Texture2D;6;0;SAMPLER2D;;False;1;FLOAT2;0,0;False;2;FLOAT;0;False;3;FLOAT2;0,0;False;4;FLOAT2;0,0;False;5;FLOAT;1;False;5;COLOR;0;FLOAT;1;FLOAT;2;FLOAT;3;FLOAT;4
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;134;-123.6048,-773.0251;Float;False;2;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.RangedFloatNode;137;-7.665367,-323.0614;Float;False;Property;_Smoothness;Smoothness;6;0;Create;True;0;0;False;0;0;0.948;0;1;0;1;FLOAT;0
Node;AmplifyShaderEditor.SimpleMultiplyOpNode;139;174.2848,-781.9967;Float;False;2;2;0;COLOR;0,0,0,0;False;1;COLOR;0,0,0,0;False;1;COLOR;0
Node;AmplifyShaderEditor.StandardSurfaceOutputNode;0;318.3661,-414.1124;Float;False;True;2;Float;ASEMaterialInspector;0;0;Standard;Depth Finder AmplifyShader;False;False;False;False;False;False;False;False;False;False;False;False;False;False;False;False;False;False;False;Off;0;False;-1;0;False;-1;False;0;0;False;0;Opaque;0.5;True;False;0;False;Opaque;;Geometry;All;True;True;True;True;True;True;True;True;True;True;True;True;True;True;True;True;True;0;False;-1;False;0;False;-1;255;False;-1;255;False;-1;0;False;-1;0;False;-1;0;False;-1;0;False;-1;0;False;-1;0;False;-1;0;False;-1;0;False;-1;False;2;15;10;25;False;0.5;False;0;Zero;Zero;0;Zero;Zero;OFF;OFF;0;False;0;0,0,0,0;VertexOffset;True;False;Cylindrical;False;Relative;0;;-1;-1;-1;-1;0;0;0;False;0;0;0;False;-1;-1;16;0;FLOAT3;0,0,0;False;1;FLOAT3;0,0,0;False;2;FLOAT3;0,0,0;False;3;FLOAT;0;False;4;FLOAT;0;False;5;FLOAT;0;False;6;FLOAT3;0,0,0;False;7;FLOAT3;0,0,0;False;8;FLOAT;0;False;9;FLOAT;0;False;10;FLOAT;0;False;13;FLOAT3;0,0,0;False;11;FLOAT3;0,0,0;False;12;FLOAT3;0,0,0;False;14;FLOAT4;0,0,0,0;False;15;FLOAT3;0,0,0;False;0
WireConnection;70;0;136;0
WireConnection;88;0;70;0
WireConnection;88;2;75;0
WireConnection;88;1;90;0
WireConnection;61;1;88;0
WireConnection;125;1;61;0
WireConnection;125;0;126;0
WireConnection;91;0;2;0
WireConnection;91;1;125;0
WireConnection;150;0;91;0
WireConnection;150;1;153;0
WireConnection;140;0;155;0
WireConnection;127;1;150;0
WireConnection;127;0;128;0
WireConnection;138;1;140;0
WireConnection;134;0;133;0
WireConnection;134;1;127;0
WireConnection;139;0;134;0
WireConnection;139;1;138;0
WireConnection;0;0;139;0
WireConnection;0;2;139;0
WireConnection;0;4;137;0
ASEEND*/
//CHKSM=C3C353ED4895A8939E0E5E65F56EADAA6152943F