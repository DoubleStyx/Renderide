# What is this?
This repo contains the shaders used in Resonite. Resonite is a free social VR sandbox platform, which allows for socialization and collaborative in-game building.

You can get Resonite free on Steam: https://store.steampowered.com/app/2519830/Resonite/

Currently Resonite offers a fixed set of shaders to build content with, which are based on Unity shaders. This repository contains the source code of those shaders for reference purposes

# What's the purpose of this repo?
There three primary reasons we made this repository:
1) Reference for implementing a new official Unity renderer as well as custom ones
    - Our goal is to move away from Unity as a renderer and switch to one that we fully control
    - All existing shaders need to be rewritten for the new renderer
    - Our hope is to potentially get community help with rewriting the shaders when the time comes - hence this reference
    - We also want to encourage tinkering with custom renderers in the community as well!
    - Read more (and potentially contribute) here: https://github.com/Yellow-Dog-Man/Resonite-Issues/discussions/5830
2) Inclusion in our new Unity SDK
    - This SDK allows converting existing Unity content (or creating new) to Resonite using the Unity Editor
    - The SDK has a converter system for materials, that will try to map any shader into Resonite's set
    - We can include the actual Resonite shaders in the Unity SDK, so they can be used to build content that will map 1:1 to Resonite's materials
    - The sources also help serve as a reference for converting other shaders to figure out the best way to map them
    - Unity SDK repo here: https://github.com/Yellow-Dog-Man/Resonite.UnitySDK
3) Serve as reference for using the materials in Resonite itself and to understand their behaviors better

# Do you accept contributions?
> [!IMPORTANT]
> We do not accept issues or PR's for this repository. It is meant for reference purposes only.

While we'd love community help and contributions, we are not at stage where we have capacity for an overhead this would need right now. There are a few reasons for this:
- This repo is only half the equation - the shaders need FrooxEngine code to update them, which is not part of this repo
- It's relatively cumbersome to update and test shaders with our current pipeline, so we don't do it often
- We need to ensure that any changes don't break existing content, which requires more care for reviewing the changes
- We'd likely need to reject any new major additions (both to existing shaders & adding brand new ones) outright - all of the shaders we have will need to be reworked for the new renderer, so this would require a lot more work on our end at later point
- Even changes to existing shaders we might need to evaluate if it would interfere with the work on new Renderer

## What if I find a bug in the shader?
Please report it on the main repo with replication case as a Resonite bug!

https://github.com/Yellow-Dog-Man/Resonite-Issues/issues

# Why not just allow custom shaders?
While on technical level we could allow everyone to just use Unity to compile custom shaders for Resonite, this presents a major practical problem - long term content compatibility.

One of our core philosophies for Resonite is long term content compatibility - we avoid breaking user content as much as possible.

Allowing everyone to compile their own shaders would make this impossible, because we would not be able to port these to a new renderer, breaking any content that uses them.

Once we have a new renderer though, where we have full control over the shader pipeline, we can introduce a custom shader system which will allow for long term compatibility.

# Why are these shaders a bit of a mess?
The shaders in this repo have grown and been changed organically over many years, with a number of quick fixes, partial reworks, leading the state of them a bit of a mess.

Since our plan is to fully move away from these shaders and replace them with new implementations for the new renderer, we didn't invest much time in cleaning this repo up.

# What's the license of the shaders?
Unless specified otherwise in the shaders themselves, the license is MIT as specified in the repo. Feel free to use them in other projects, even if they're unrelated to Resonite!
