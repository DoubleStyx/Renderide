furFX - Physics-based Fur Shaders 3.0
-------------------------------------

Information
-----------
Pack contains multipass FUR shaders with many extra features like physics based fur movement, fur gravity, custom coloring etc.

Main features
-------------
PC and MAC compatible, Unity 3.5.x , 4.x and Unity 5.x compatible
Works on SM2.0
Fur Wind Simulation
Fur Gravity Simulation
Fur Rigidbody Forces Simulation
Fur Thickness, Fur Coloring, Fur Shading, Fur Length, Hair Hardness
Fur Length Masking
Fur Rim Light
Fur Cubemap Reflection
Directional Light + up to 4 vertex lights
6 sample fur textures with alpha mask

FuFX 1.x Shaders :
5,10,20,40 Layer Basic Shaders
5,10,20,40 Layer Advanced Shaders
5,10,20,40 Layer Advanced Shell Shaders

FurFX 2.x Shaders :
5,10,20,40 Layer Shader

FurFX 3.x Shaders (for Unity 5.x) :
5,10,20,40 Layer Shader

Directional Light, Ambient Light and Specular
Model Cast Shadow

Use instruction
---------------
Assign fur shader for your material. If you wanna extra effects like fur gravity, movement with rigidbody velocity or wind - you need to add <FurForce.js> script to object. 
You can also control fur movement by yourself - look into FurForce.js to see how to pass fur movement Vector into shader.

DirectX 11
----------
If you wanna use shaders under DX11 - unpack ZIP from FurFX directory and use DX11 shaders.
Shaders are zipped because Unity 3.5 rise error since it dont recognise Unity 4 shaders - sorry for that but we wanna keep compatiblity for both Unity versions.

Shader Parameters Description
-----------------------------
Color (RGB) - material color
Specular Material Color (RGB) - specular light color
Shininess - specular shininess factor
Fur Length - length of fur
Base (RGB) - main texture (RGB). Alpha works like fur length mask.
Noise (RGB) - noise texture. Assign one of prepared 2 or make yours.
Alpha Cutoff - alpha cutoff factor
Edge Fade - how much fur go into transparent when more far from core
Fur Hardness - how much fur hold on place after applying gravity or other forces
Fur Thinness - thinness of fur
Fur Shading - how much fur color go darker when closer to fur core
Fur Coloring - adding color mixing from noise texture
Mask Alpha Factor - factor of alpha mask taked from main texture. Can override alpha mask from texture alpha.
Force Global - global force working of fur - keep inside (-1,1) - NOT affected by object rotation - use for gravity, wind etc
Force Local - local force working of fur - keep inside (-1,1) - AFFECTED by object rotation - use to shape fur on objects
	
Advanced Shaders Bonus Parameters 
-------------------
Rim Color - color of rim lightning (more dark = less visible)
Rim Power - power of rim
Reflection Map - cubemap reflection
Reflection Power - alpha of reflection cubemap

Self Shadow Shaders
-------------------
There are 2 types of Self Shadow Shaders. Blend and Not Blended (need Cutout param to be less than 1.0).
Not Blended shaders will work with all type of objects behind fur object. Blended will work only in front of non transparent objects. Choose shader that will fit for your game.
			
FurForce.js
-----------
smoothing - higher value makes your fur faster rotate in desire direction
addRigidbodyForce - add force from rigidbody velocity
rigidbodyForceFactor - factor for above
addGravityToForce - add gravity force to fur
gravityFactor - factor for above
addWindForce - add wind force to fur
windForceFactor - scale of wind - if you wanna have wind not blowing from up and down - set it as (1,0,1)
windForceSpeed - speed of wind changing direction
		
Version History
---------------
1.0 Initial Version
1.1 Fixed fur alpha problems
1.2 Added DX11 shaders
1.3 Added Local Fur Direction
	Fixed some bugs
1.4 Added Advanced FurFX Shaders :
	- rim light
	- cubemap mapping
	- directional + up to 4 vertex lights support
	All 1.3 shaders are now called Basic FurFX
	Fixed Texture Alpha Mask - now works like it suppose to work - by limiting max fur length (can be override by Mask Alpha Factor)
	Added sample torus scene
	Fixed grass scene - less wind for better effect (see Wind Factor Force in <furforce.js>)
	Changed some textures for illustrating how alpha mask works
1.5 Added Shell Shader 5,10,20 and 40 layer Advanced
	Added Cartoon Fur Example with noise and fur texture
1.6 Added Self Shadow Test Shaders (Blend and Cutout versions) for DX9 and DX11
	Added Alpha Mask Sample Scene
	Added Shell DX11 Shaders
	Fixed problem with Unity 4.x + Point Lights + Fog
2.0 Added FurFX 2.0 shaders (works best with Deferred Lightning)
	Shaders are now full source
3.0 Added FurFX 3.0 shader (for Unity 5.x)	
				
Requests
--------
This pack of shaders is constantly updated by shaders that you have requested. If you got some request - feel free to contact me or write on Unity3d forum.		

Unity Forums thread
-------------------
http://forum.unity3d.com/threads/184856-furFX-Physics-based-Fur-Shaders

Special Thanks
------
Seith for lightbox model
Eric Wilkinson for Werewolf model

Credits
-------
Red Dot Games 2013
http://www.reddotgames.pl