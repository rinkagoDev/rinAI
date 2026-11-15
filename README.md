# rinAI
A hand-rolled C++/OpenGL VRM viewer where a local LLM poses the character's own skeleton, joint by joint - no animation clips, no engine.

rinAI is a from-scratch C++/OpenGL VRM and glTF model viewer where a locally-run LLM controls the character's own skeleton directly, frame by frame. There is no animation system, no canned clips, no scripted behavior tree - the model gets a text description of its own body and its surroundings, and every pose you see on screen is a joint rotation the LLM decided to set.

This is a single-file Win32 application (`rinAI.cpp` + a couple of small headers), built with a raw `cl.exe` command line. No CMake, no Visual Studio project file, no game engine underneath - fixed-function OpenGL 1.x, hand-rolled 4x4 matrix math, and the skinning pipeline is a plain CPU loop (optionally OpenMP-parallelized) over `cgltf`-parsed buffers.

## What's actually novel here

- **Direct joint-space control by an LLM.** The model doesn't pick from a library of animations - it emits raw per-axis rotations (`{"Head": {"y": 0.3}}`) for whichever joints it wants to move, read out of a whitelist of the current model's real bone names. Everything else about the pose stays exactly where it was.
- **A canonical bone-name whitelist, not just a blacklist.** Picking which bones in an arbitrary VRM/glTF rig are "real" posable joints (hips, spine, arms, fingers...) versus rig plumbing (twist helpers, IK targets, spring colliders) matters a lot more once an LLM is the one naming joints by string - a bone the AI can't name correctly is a bone it effectively can't use.
- **No game engine, on purpose.** Skinning, lighting, the room, the shadow pass - all fixed-function GL calls and manual matrix multiplies. It's a viewer, not an engine, and stays legible as one.

## Architecture

- **glTF/VRM loading** (`cgltf.h`, `LoadVRMModel`) - parses the model, flattens the node hierarchy, extracts skins/joints/textures.
- **Skinning** (`ProcessAndDrawComponent`) - CPU linear-blend skinning per vertex, OpenMP-parallelized over large meshes, straight into immediate-mode `glBegin/glEnd` triangles.
- **Joint rig & pose filtering** (`IsImportantJoint`, `MatchesCanonicalBoneName`) - decides which bones are worth giving the AI control over, and smooths every joint's current angle toward whatever target the AI last set (`UpdateJointSmoothing`), at a rate controlled by the Rig Smoothness setting.
- **Environment / room** (`rinAI.h`: `EnvProp`, `DrawEnvironment`, `ResolveCollision`) - a small bounded room with a few pieces of furniture, drawn in world space, independent of the character's own transform.
- **Agent loop** (`UpdateAgent`, `ParseAgentCommand`, `ApplyPoseCommand`, `BuildSpatialContext`) - advances the character's walk/turn queue, applies whatever pose the AI last requested, and builds the "where am I" context string sent back to the model each turn.
- **Ollama client** (`ollama_client.h`) - a compact WinINet-based HTTP client talking to a local Ollama server's `/api/chat` and `/api/tags`, run on a detached thread so the render loop never blocks on a network call.
- **Chat UI** (`ShowChatWindow`, `ChatDlgProc`) - a second hand-rolled modeless Win32 dialog (no ImGui anywhere in this project) for talking to the model, picking which Ollama model to use, and optionally attaching a screenshot for vision models.

## Building

Requirements:
- Windows 10/11
- Visual Studio Build Tools with the **Desktop development with C++** workload (for `cl.exe` and the Windows SDK - WinINet and OpenGL/GLU headers come from the SDK, nothing extra to install)
- A `.vrm`, `.glb`, or `.gltf` file. Drop one named `avatar.vrm` next to the `.exe`, or point the path box / Browse button at any model after launch.

From a "x64 Native Tools Command Prompt for VS", in the project directory:

```
cl.exe /EHsc /O2 /openmp rinAI.cpp /link user32.lib gdi32.lib opengl32.lib glu32.lib comdlg32.lib comctl32.lib wininet.lib
```

That's the whole build - one source file, no project generation step. `json.hpp` (nlohmann/json, header-only) and `stb_image_write.h` need to sit next to `rinAI.cpp`; both are vendored in this repo already, same as `cgltf.h` and `stb_image.h`.

## Running the AI

rinAI talks to a locally-run [Ollama](https://ollama.com) server - nothing is sent anywhere else, and the app does zero networking beyond `http://127.0.0.1:11434`. ("openllama" in casual conversation almost always means the Ollama tool itself, not the separate OpenLLaMA model family - if someone told you to install "openllama," this is almost certainly what they meant.)

1. Install Ollama for Windows from **https://ollama.com/download** - it has a native Windows installer, no WSL required, and runs as a background service once installed.
2. Pull a text model to drive the character's chat and posing, e.g.:
   ```
   ollama pull llama3.2:3b
   ```
   That's small enough to run comfortably on CPU. If you've got a decent GPU, a bigger model (`llama3.2`, `llama3.1:8b`, etc.) will give noticeably better pose/movement decisions.
3. If you want to use the optional "send screenshot" option in the chat window (letting the model actually see the viewport instead of just reading position text), pull a vision model:
   ```
   ollama pull llava
   ```
   `llava` comes in `llava:7b`, `llava:13b`, and `llava:34b` variants - on a CPU-only machine with no dedicated GPU/NPU, `llava:7b` is the practical choice; the bigger ones will be very slow without hardware acceleration.
4. Run `ollama list` any time to see what you've actually got pulled - that's the same list the Chat window's model dropdown populates from (hit Refresh in the window if you pull something new while it's open).
5. Launch rinAI, click **Chat**, pick a model from the dropdown, and start talking. The character will reply in text and, on its own line, emit a movement/pose command you won't see in the transcript - that's what drives the body.

## Controls

Mouse only, no keyboard shortcuts:

- **Left-click drag** - orbit the camera
- **Right-click drag** - pan the camera target
- **Scroll wheel** - zoom

There is no manual joint dragging and no WASD character movement - the AI drives both the pose and the walk/turn, which is the whole point of this version. The colored spheres on the rig are a read-only visualization: cyan means that joint is near its neutral pose, magenta means the AI currently has a real target set on it.

## Known limitations

- Fixed-function OpenGL 1.x only - no shaders, no PBR, no modern rendering pipeline. Texturing and lighting are both about as basic as GL gets.
- No cloth/hair physics. An earlier version of this project had Verlet-integrated cloth and hair with capsule collision; it was removed because it actively fought with the AI-driven joint control (the AI sets a joint target, physics simulation independently nudges vertices around it - the two systems have no reason to agree on what a skirt or a strand of hair should be doing).
- No canned/keyframe animation. Also removed, for the same reason: this project's whole premise is that the LLM is the only thing driving joints, and a looping animation clip would just be fighting it for control of the same bones.
- WinINet's request model has no clean mid-transfer cancel, so "Interrupt" can only stop the *next* request from being sent and drop the result of one already in flight - it can't abort an HTTP call partway through.
- Windows-only. The rig-filtering logic, the agent protocol, and the environment system are all portable in principle, but the UI layer is Win32 dialogs and WinINet top to bottom.
