#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX

#include <windows.h>
#include <windowsx.h>
#include <commdlg.h>
#include <commctrl.h>
#include <gl/gl.h>
#include <gl/glu.h>

#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

#include <cstdlib>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <deque>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <mutex>

#include "ollama_client.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "rinAI.h"

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glu32.lib")
#pragma comment(lib, "comdlg32.lib")
#pragma comment(lib, "comctl32.lib")

// --- Resource Identifiers ---
#define IDC_PATH_EDIT         101
#define IDC_BROWSE_BTN        102
#define IDC_IMPORT_BTN        103
#define IDC_SETTINGS_BTN      104
#define IDC_CHAT_BTN          105
#define IDC_STATUS_TEXT       106
#define IDC_STOP_BTN          107
#define IDC_INTERRUPT_BTN     108
#define TOOLBAR_HEIGHT        80

#define IDT_CHAT_TIMER        1

// Settings Dialog Component IDs
#define IDC_CHECK_MULTICORE   202
#define IDC_CHECK_CULLING     203
#define IDC_CHECK_LIGHTING    204
#define IDC_SLIDER_CAM_SPD    205
#define IDC_SLIDER_RIG_SMO    206
#define IDC_RADIO_BG_DARK     207
#define IDC_RADIO_BG_LIGHT    208
#define IDC_CHECK_INVERT_CAM  209
#define IDC_CHECK_SHADOWS     210
#define IDC_CHECK_WIREFRAME   211
#define IDC_COMBO_TEX_QUALITY 212
#define IDC_TAB_MAIN          250

// Grouping IDs for Tab 0
#define IDC_STATIC_GRP2       252
#define IDC_STATIC_GRP3       253
#define IDC_STATIC_GRP4       254
#define IDC_STATIC_GRP5       255
#define IDC_STATIC_TXT1       256
#define IDC_STATIC_TXT2       257
#define IDC_STATIC_TXT3       258

// IDs for Tab 1 (Debug)
#define IDC_DEBUG_GRP1        270
#define IDC_CHECK_DB_BONES    271
#define IDC_CHECK_DB_NOTEX    273

// Chat window IDs
#define IDC_CHAT_TRANSCRIPT     301
#define IDC_CHAT_INPUT          302
#define IDC_CHAT_SEND_BTN       303
#define IDC_CHAT_MODEL_COMBO    304
#define IDC_CHAT_REFRESH_BTN    305
#define IDC_CHAT_SCREENSHOT_CHK 306
#define IDC_CHAT_MODEL_LABEL    307


enum MeshClassification {
    MESH_SKELETAL_BODY,
    MESH_CLOTHING_LAYER,
    MESH_HAIR_STRANDS
};

struct SegregatedMeshComponent {
    cgltf_mesh* meshRef;
    cgltf_skin* skinRef;
    size_t skinIndex;
    float globalFallbackMatrix[16];
    MeshClassification classification;
};

// What the AI agent can ask the character's root to do. Each queued move
// runs for a fixed duration (see DurationForMove) rather than until some
// target is reached - simple, predictable, and good enough for a character
// that re-decides what to do every chat turn anyway.
enum class MoveAction { None, WalkForward, WalkBackward, TurnLeft, TurnRight };
struct QueuedMove { MoveAction action; float duration; };

// --- Global Engine Core Configurations ---
bool  g_SettingUseMulticore = true;
bool  g_SettingUseCulling = true;
bool  g_SettingUseLighting = true;
bool  g_SettingInvertCam = true;
float g_SettingCamSmooth = 35.0f;
float g_SettingRigSmooth = 40.0f;
int   g_SettingBgStyle = 0;
bool  g_SettingRenderShadows = true;
bool  g_SettingWireframeMode = false;
int   g_SettingTextureQuality = 2;
bool g_DebugShowBones = false;
bool g_DebugDisableTextures = false;

// Joints are now a pure visualization of what the AI is doing - purely
// cosmetic sphere radius, no picking math attached to it any more.
const float kJointSphereRadius = 0.025f;
GLUquadric* g_JointQuadric = nullptr;

struct JointAngles {
    float curX = 0.0f, tgtX = 0.0f;
    float curY = 0.0f, tgtY = 0.0f;
    float curZ = 0.0f, tgtZ = 0.0f;
};

struct FlattenedNode {
    cgltf_node* node;
    size_t parentIdx;
    float localMatrix[16];
    float globalMatrix[16];
    bool isDirty;
};

cgltf_data* g_GltfData = nullptr;
std::vector<GLuint> g_Textures;
std::vector<FlattenedNode> g_FlatNodes;
std::unordered_map<cgltf_node*, size_t> g_NodeToFlatIdx;

std::vector<cgltf_node*> g_JointList;
std::unordered_map<cgltf_node*, JointAngles> g_SkeletalRig;
// Joints that actually deform a mesh vertex with meaningful weight, found
// by scanning real skinning data at load time - used to filter out
// helper/IK/twist bones that would otherwise clutter the viewer with
// spheres that don't do anything useful.
std::unordered_set<cgltf_node*> g_InfluencedJoints;
// Lowercased joint name -> node, rebuilt on every model load. This is how
// the agent's JSON pose commands (which name joints by string) get mapped
// back onto actual rig nodes.
std::unordered_map<std::string, cgltf_node*> g_JointNameToNode;
int g_SelectedJointIdx = 0;
bool g_RigHierarchyDirty = true;

struct CachedVertex {
    float p[3];
    float n[3];
};
std::vector<CachedVertex> g_VertexCache;
std::vector<float> g_SkinCombinedMatricesCache;
std::vector<SegregatedMeshComponent> g_SeparatedComponents;

// --- Room + character agent state ---
std::vector<EnvProp> g_EnvProps;
const float g_EnvHalfSize = 5.0f;

std::deque<QueuedMove> g_MoveQueue;
MoveAction g_CurrentMove = MoveAction::None;
float g_CurrentMoveTimeLeft = 0.0f;
float g_CharacterX = 0.0f, g_CharacterZ = 0.0f, g_CharacterFacing = 0.0f;
std::string g_LastActionStatus = "idle";

OllamaClient g_Ollama;

HDC   hDC;
HGLRC hRC;
HWND  hWnd;
HWND  hEditPath;
HWND  hBrowseBtn;
HWND  hImportBtn;
HWND  hChatBtn;
HWND  hStatusText;
HWND  hStopBtn;
HWND  hInterruptBtn;
HWND  g_hSettingsDlg = nullptr;
HWND  g_hChatDlg = nullptr;
HWND  g_hChatTranscript = nullptr;
HWND  g_hChatInput = nullptr;
HWND  g_hChatModelCombo = nullptr;
size_t g_ChatRenderedHistoryCount = 0;
size_t g_ChatRenderedModelsCount = (size_t)-1;

int winWidth = 900;
int winHeight = 640;

float g_InAngleX = 3.14159f, g_InAngleY = 0.0f, g_InDistance = 2.5f;
float g_InTargetX = 0.0f, g_InTargetY = 1.1f, g_InTargetZ = 0.0f;
float g_CamAngleX = 3.14159f, g_CamAngleY = 0.0f, g_CamDistance = 2.5f;
float g_TargetX = 0.0f, g_TargetY = 1.1f, g_TargetZ = 0.0f;

bool g_IsDragging = false, g_IsPanning = false;
int  g_LastMouseX = 0, g_LastMouseY = 0;

static LARGE_INTEGER g_PerfFreq = {};
static LARGE_INTEGER g_LastTick = {};

typedef void (APIENTRY* PFNGLGENERATEMIPMAPEXTPROC_t)(GLenum);
static PFNGLGENERATEMIPMAPEXTPROC_t g_glGenerateMipmapEXT = nullptr;

// --- High-Performance Matrix Operations ---
inline void MultiplyMat4(float* out, const float* a, const float* b) {
    for (int c = 0; c < 4; ++c) {
        float b0 = b[c * 4 + 0], b1 = b[c * 4 + 1], b2 = b[c * 4 + 2], b3 = b[c * 4 + 3];
        out[c * 4 + 0] = a[0] * b0 + a[4] * b1 + a[8] * b2 + a[12] * b3;
        out[c * 4 + 1] = a[1] * b0 + a[5] * b1 + a[9] * b2 + a[13] * b3;
        out[c * 4 + 2] = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14] * b3;
        out[c * 4 + 3] = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15] * b3;
    }
}

inline void TransformVec3(float* out, const float* m, const float* v) {
    float x = v[0], y = v[1], z = v[2];
    out[0] = m[0] * x + m[4] * y + m[8] * z + m[12];
    out[1] = m[1] * x + m[5] * y + m[9] * z + m[13];
    out[2] = m[2] * x + m[6] * y + m[10] * z + m[14];
}

inline void TransformNormal(float* out, const float* m, const float* n) {
    float x = n[0], y = n[1], z = n[2];
    out[0] = m[0] * x + m[4] * y + m[8] * z;
    out[1] = m[1] * x + m[5] * y + m[9] * z;
    out[2] = m[2] * x + m[6] * y + m[10] * z;
}

inline void BuildRotationXYZMat4(float* m, float rx, float ry, float rz) {
    float cx = cosf(rx), sx = sinf(rx);
    float cy = cosf(ry), sy = sinf(ry);
    float cz = cosf(rz), sz = sinf(rz);
    m[0] = cy * cz;                m[4] = -cy * sz;               m[8] = sy;       m[12] = 0;
    m[1] = cx * sz + sx * sy * cz; m[5] = cx * cz - sx * sy * sz; m[9] = -sx * cy; m[13] = 0;
    m[2] = sx * sz - cx * sy * cz; m[6] = sx * cz + cx * sy * sz; m[10] = cx * cy; m[14] = 0;
    m[3] = 0;                      m[7] = 0;                      m[11] = 0;       m[15] = 1;
}

inline void QuatToMat4(float* m, const float* q) {
    float x = q[0], y = q[1], z = q[2], w = q[3];
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;
    m[0] = 1 - 2 * (y2 + z2); m[4] = 2 * (xy - wz);     m[8] = 2 * (xz + wy);     m[12] = 0;
    m[1] = 2 * (xy + wz);   m[5] = 1 - 2 * (x2 + z2);   m[9] = 2 * (yz - wx);     m[13] = 0;
    m[2] = 2 * (xz - wy);   m[6] = 2 * (yz + wx);     m[10] = 1 - 2 * (x2 + y2);  m[14] = 0;
    m[3] = 0;           m[7] = 0;              m[11] = 0;            m[15] = 1;
}

static inline float Lerp(float cur, float tgt, float speed, float dt) {
    return tgt + (cur - tgt) * expf(-speed * dt);
}

bool IsSimulationNode(const char* name) {
    if (!name) return false;
    std::string lowerStr(name);
    std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(),
        [](unsigned char c) { return std::tolower(c); });
    if (lowerStr.find("hair") != std::string::npos || lowerStr.find("cloth") != std::string::npos ||
        lowerStr.find("dress") != std::string::npos || lowerStr.find("skirt") != std::string::npos ||
        lowerStr.find("sleeve") != std::string::npos || lowerStr.find("jiggle") != std::string::npos) {
        return true;
    }
    return false;
}

// Scans the actual JOINTS_0/WEIGHTS_0 vertex data (the same data
// ProcessAndDrawComponent uses for skinning) to find which joints genuinely
// deform some part of the mesh. Bones that never show up here (IK targets,
// unused twist helpers, control-only bones, etc.) get excluded from the
// joint-sphere list, since posing them wouldn't visibly do anything anyway.
void GatherInfluencedJoints(cgltf_node* node) {
    if (node->mesh && node->skin) {
        cgltf_skin* skin = node->skin;
        for (size_t p = 0; p < node->mesh->primitives_count; ++p) {
            cgltf_primitive& prim = node->mesh->primitives[p];
            cgltf_accessor* jointsAcc = nullptr, * weightsAcc = nullptr;
            for (size_t a = 0; a < prim.attributes_count; ++a) {
                if (prim.attributes[a].type == cgltf_attribute_type_joints) jointsAcc = prim.attributes[a].data;
                else if (prim.attributes[a].type == cgltf_attribute_type_weights) weightsAcc = prim.attributes[a].data;
            }
            if (!jointsAcc || !weightsAcc) continue;

            const char* jointPtr = (const char*)jointsAcc->buffer_view->buffer->data + jointsAcc->buffer_view->offset + jointsAcc->offset;
            size_t jointStride = jointsAcc->stride;
            const char* weightPtr = (const char*)weightsAcc->buffer_view->buffer->data + weightsAcc->buffer_view->offset + weightsAcc->offset;
            size_t weightStride = weightsAcc->stride;
            int componentType = jointsAcc->component_type;
            size_t count = jointsAcc->count;

            for (size_t v = 0; v < count; ++v) {
                const float* w = (const float*)(weightPtr + v * weightStride);
                unsigned int jAr[4] = { 0 };
                if (componentType == cgltf_component_type_r_8u) {
                    const unsigned char* jb = (const unsigned char*)(jointPtr + v * jointStride);
                    jAr[0] = jb[0]; jAr[1] = jb[1]; jAr[2] = jb[2]; jAr[3] = jb[3];
                }
                else if (componentType == cgltf_component_type_r_16u) {
                    const unsigned short* js = (const unsigned short*)(jointPtr + v * jointStride);
                    jAr[0] = js[0]; jAr[1] = js[1]; jAr[2] = js[2]; jAr[3] = js[3];
                }
                for (int k = 0; k < 4; ++k) {
                    if (w[k] > 0.02f && jAr[k] < skin->joints_count) {
                        g_InfluencedJoints.insert(skin->joints[jAr[k]]);
                    }
                }
            }
        }
    }
    for (size_t i = 0; i < node->children_count; ++i) GatherInfluencedJoints(node->children[i]);
}

// Canonical humanoid bone name fragments. A joint has to match at least one
// of these (after clearing the blacklist below) to get a pose handle - this
// is what keeps oddly-named model-specific rig junk out of the AI's reach
// without needing a full VRM humanoid-bone-map lookup.
static const char* kCanonicalBoneKeywords[] = {
    "hips", "pelvis", "spine", "chest", "neck", "head",
    "shoulder", "clavicle", "collar",
    "upperarm", "upper_arm", "forearm", "lowerarm", "lower_arm", "arm",
    "hand", "wrist",
    "upperleg", "upper_leg", "thigh", "lowerleg", "lower_leg", "shin", "calf", "leg", "knee",
    "foot", "ankle", "toe",
    "thumb", "index", "middle", "ring", "little", "pinky"
};

bool MatchesCanonicalBoneName(const std::string& lowerName) {
    for (const char* kw : kCanonicalBoneKeywords) {
        if (lowerName.find(kw) != std::string::npos) return true;
    }
    return false;
}

// Decides whether a joint gets a pose-sphere and is eligible for the AI to
// target. Filters out:
//  - bones with no real skin influence (see GatherInfluencedJoints)
//  - common auxiliary/helper bone naming patterns (twist, roll, IK, etc.)
//  - anything that doesn't also match a canonical humanoid bone name
//  - near-zero-length bones (fingertip/end markers with negligible offset
//    from their parent) which add clutter without giving useful control
bool IsImportantJoint(cgltf_node* node) {
    if (!node) return false;
    if (g_InfluencedJoints.find(node) == g_InfluencedJoints.end()) return false;
    if (!node->name) return false;

    std::string n(node->name);
    std::transform(n.begin(), n.end(), n.begin(), [](unsigned char c) { return std::tolower(c); });
    static const char* skipSubstrings[] = {
        "twist", "roll", "adjust", "helper", "dummy", "ik_", "_ik", "tip", "nub", "target",
        "spring", "secondary", "collider", "aux", "ctrl", "control", "null", "end", "leaf", "fix", "const"
    };
    for (const char* s : skipSubstrings) {
        if (n.find(s) != std::string::npos) return false;
    }
    if (!MatchesCanonicalBoneName(n)) return false;

    auto it = g_NodeToFlatIdx.find(node);
    if (it != g_NodeToFlatIdx.end()) {
        // Use the LOCAL matrix's translation (offset from parent), not the
        // global matrix - at the point this is called during model load,
        // FastUpdateGlobalTransforms hasn't run yet, so globalMatrix isn't
        // truly global. Local translation magnitude is a correct proxy for
        // bone length regardless of that.
        const float* lm = g_FlatNodes[it->second].localMatrix;
        float distSq = lm[12] * lm[12] + lm[13] * lm[13] + lm[14] * lm[14];
        if (distSq < 0.0004f) return false; // < 2cm from its parent
    }

    return true;
}

void UpdateWindowTitle() {
    SetWindowTextA(hWnd, "rinAI ver2.0 - AI-Driven Posing");
}

void ApplyTextureQualitySettings(GLuint texID) {
    if (texID == 0) return;
    glBindTexture(GL_TEXTURE_2D, texID);
    if (g_SettingTextureQuality == 0) {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
    else if (g_SettingTextureQuality == 1) {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }
    else {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        if (g_glGenerateMipmapEXT) {
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            g_glGenerateMipmapEXT(GL_TEXTURE_2D);
        }
        else {
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        }
    }
}

void FlattenHierarchy(cgltf_node* node, size_t parentIdx) {
    size_t currentIdx = g_FlatNodes.size();
    FlattenedNode fn;
    fn.node = node;
    fn.parentIdx = parentIdx;
    cgltf_node_transform_local(node, fn.localMatrix);
    memcpy(fn.globalMatrix, fn.localMatrix, sizeof(float) * 16);
    fn.isDirty = true;

    g_FlatNodes.push_back(fn);
    g_NodeToFlatIdx[node] = currentIdx;

    for (size_t i = 0; i < node->children_count; ++i) {
        FlattenHierarchy(node->children[i], currentIdx);
    }
}

MeshClassification ClassifyMeshNode(const char* name) {
    if (!name) return MESH_SKELETAL_BODY;
    std::string upperStr(name);
    std::transform(upperStr.begin(), upperStr.end(), upperStr.begin(),
        [](unsigned char c) { return std::tolower(c); });

    if (upperStr.find("hair") != std::string::npos) return MESH_HAIR_STRANDS;
    if (upperStr.find("cloth") != std::string::npos || upperStr.find("dress") != std::string::npos ||
        upperStr.find("skirt") != std::string::npos || upperStr.find("sleeve") != std::string::npos) {
        return MESH_CLOTHING_LAYER;
    }
    return MESH_SKELETAL_BODY;
}

void GatherSeparatedComponents(cgltf_node* node) {
    if (node->mesh) {
        SegregatedMeshComponent comp = {};
        comp.meshRef = node->mesh;
        comp.skinRef = node->skin;
        comp.classification = ClassifyMeshNode(node->name ? node->name : (node->mesh->name ? node->mesh->name : ""));

        auto it = g_NodeToFlatIdx.find(node);
        if (it != g_NodeToFlatIdx.end()) {
            memcpy(comp.globalFallbackMatrix, g_FlatNodes[it->second].globalMatrix, sizeof(float) * 16);
        }
        else {
            float id[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
            memcpy(comp.globalFallbackMatrix, id, sizeof(id));
        }

        comp.skinIndex = 0;
        if (node->skin && g_GltfData) {
            for (size_t s = 0; s < g_GltfData->skins_count; ++s) {
                if (&g_GltfData->skins[s] == node->skin) { comp.skinIndex = s; break; }
            }
        }

        // Clothing/hair used to get a separate cloth-sim treatment here;
        // with the physics system gone they're just ordinary skinned
        // geometry now. classification is kept only for wireframe coloring.
        g_SeparatedComponents.push_back(comp);
    }
    for (size_t i = 0; i < node->children_count; ++i) {
        GatherSeparatedComponents(node->children[i]);
    }
}

void FastUpdateGlobalTransforms() {
    size_t nodeCount = g_FlatNodes.size();
    for (size_t i = 0; i < nodeCount; ++i) {
        FlattenedNode& fn = g_FlatNodes[i];
        if (fn.parentIdx != (size_t)-1) {
            if (g_FlatNodes[fn.parentIdx].isDirty) fn.isDirty = true;
        }

        if (fn.isDirty || g_RigHierarchyDirty) {
            float local[16];
            memcpy(local, fn.localMatrix, sizeof(local));

            auto rigIt = g_SkeletalRig.find(fn.node);
            if (rigIt != g_SkeletalRig.end()) {
                JointAngles& ja = rigIt->second;
                if (fabs(ja.curX) > 0.0001f || fabs(ja.curY) > 0.0001f || fabs(ja.curZ) > 0.0001f) {
                    float rotMat[16], modifiedLocal[16];
                    BuildRotationXYZMat4(rotMat, ja.curX, ja.curY, ja.curZ);
                    MultiplyMat4(modifiedLocal, local, rotMat);
                    memcpy(local, modifiedLocal, sizeof(local));
                }
            }

            if (fn.parentIdx != (size_t)-1) {
                MultiplyMat4(fn.globalMatrix, g_FlatNodes[fn.parentIdx].globalMatrix, local);
            }
            else {
                memcpy(fn.globalMatrix, local, sizeof(local));
            }
            fn.isDirty = false;
        }
    }

    if (g_GltfData) {
        size_t maxJoints = 0;
        for (size_t s = 0; s < g_GltfData->skins_count; ++s) {
            if (g_GltfData->skins[s].joints_count > maxJoints) maxJoints = g_GltfData->skins[s].joints_count;
        }
        for (size_t s = 0; s < g_GltfData->skins_count; ++s) {
            cgltf_skin& skin = g_GltfData->skins[s];
            float* ibmBase = (skin.inverse_bind_matrices) ? (float*)((char*)skin.inverse_bind_matrices->buffer_view->buffer->data + skin.inverse_bind_matrices->buffer_view->offset + skin.inverse_bind_matrices->offset) : nullptr;

            for (size_t j = 0; j < skin.joints_count; ++j) {
                size_t cacheOffset = (s * maxJoints + j) * 16;
                auto it = g_NodeToFlatIdx.find(skin.joints[j]);
                if (it == g_NodeToFlatIdx.end()) continue;
                const float* globalMat = g_FlatNodes[it->second].globalMatrix;

                if (ibmBase) {
                    MultiplyMat4(&g_SkinCombinedMatricesCache[cacheOffset], globalMat, ibmBase + j * 16);
                }
                else {
                    float identity[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
                    MultiplyMat4(&g_SkinCombinedMatricesCache[cacheOffset], globalMat, identity);
                }
            }
        }
    }

    g_RigHierarchyDirty = false;
}

bool LoadVRMModel(const std::string& filename) {
    if (!g_Textures.empty()) { glDeleteTextures((GLsizei)g_Textures.size(), g_Textures.data()); g_Textures.clear(); }
    g_FlatNodes.clear();
    g_NodeToFlatIdx.clear();
    g_SkeletalRig.clear();
    g_JointList.clear();
    g_SeparatedComponents.clear();
    g_SelectedJointIdx = 0;
    g_InfluencedJoints.clear();
    g_JointNameToNode.clear();

    // A freshly loaded model gets a clean slate in the room too - otherwise
    // an in-progress walk or a stale pose target from the previous model
    // would carry over onto the new rig.
    g_MoveQueue.clear();
    g_CurrentMove = MoveAction::None;
    g_CharacterX = 0.0f; g_CharacterZ = 0.0f; g_CharacterFacing = 0.0f;

    if (g_GltfData) { cgltf_free(g_GltfData); g_GltfData = nullptr; }

    cgltf_options options = {};
    cgltf_result result = cgltf_parse_file(&options, filename.c_str(), &g_GltfData);
    if (result != cgltf_result_success) return false;

    result = cgltf_load_buffers(&options, g_GltfData, filename.c_str());
    if (result != cgltf_result_success) return false;

    for (size_t i = 0; i < g_GltfData->scenes_count; ++i) {
        for (size_t j = 0; j < g_GltfData->scenes[i].nodes_count; ++j) {
            FlattenHierarchy(g_GltfData->scenes[i].nodes[j], (size_t)-1);
        }
    }

    for (size_t i = 0; i < g_GltfData->scenes_count; ++i) {
        for (size_t j = 0; j < g_GltfData->scenes[i].nodes_count; ++j) {
            GatherInfluencedJoints(g_GltfData->scenes[i].nodes[j]);
        }
    }

    size_t maxJointsCount = 0;
    bool hadSkinJoints = false;
    for (size_t i = 0; i < g_GltfData->skins_count; ++i) {
        cgltf_skin& skin = g_GltfData->skins[i];
        if (skin.joints_count > maxJointsCount) maxJointsCount = skin.joints_count;
        for (size_t j = 0; j < skin.joints_count; ++j) {
            cgltf_node* jNode = skin.joints[j];
            hadSkinJoints = true;
            if (jNode && g_SkeletalRig.find(jNode) == g_SkeletalRig.end()) {
                if (!IsSimulationNode(jNode->name)) {
                    g_SkeletalRig[jNode] = JointAngles{};
                    // Only add a pose sphere for joints that pass the
                    // "important" filter (real skin influence, canonical
                    // bone name, not a twist/IK/helper bone, not a
                    // near-zero-length end marker). Every non-simulation
                    // joint still gets a JointAngles entry above so the rig
                    // itself stays fully correct even for joints without a
                    // visible handle.
                    if (IsImportantJoint(jNode)) g_JointList.push_back(jNode);
                }
            }
        }
    }

    g_SkinCombinedMatricesCache.assign(g_GltfData->skins_count * maxJointsCount * 16, 0.0f);

    if (!hadSkinJoints) {
        // No skin at all in this glTF - GatherInfluencedJoints has nothing
        // to find in that case, so IsImportantJoint would filter out
        // everything. Fall back to the old "every non-simulation node is a
        // joint" behavior instead, just for this edge case.
        for (auto& fn : g_FlatNodes) {
            if (!IsSimulationNode(fn.node->name)) {
                g_SkeletalRig[fn.node] = JointAngles{};
                g_JointList.push_back(fn.node);
            }
        }
    }
    else if (g_JointList.empty()) {
        // The model DOES have skin joints, but the importance filter
        // (influence scan + name patterns + canonical whitelist + bone-
        // length check) rejected every single one - e.g. its bone names use
        // conventions the filter doesn't expect. Rather than silently show
        // zero handles and leave the AI with nothing to pose, fall back to
        // every non-simulation skin joint so posing still works, just
        // without the decluttering.
        for (const auto& rigEntry : g_SkeletalRig) {
            g_JointList.push_back(rigEntry.first);
        }
    }

    for (size_t i = 0; i < g_GltfData->scenes_count; ++i) {
        for (size_t j = 0; j < g_GltfData->scenes[i].nodes_count; ++j) {
            GatherSeparatedComponents(g_GltfData->scenes[i].nodes[j]);
        }
    }

    UpdateWindowTitle();
    g_RigHierarchyDirty = true;

    // Rebuild the name->node lookup and the joint list the system prompt
    // advertises to the model, now that g_JointList is final for this rig.
    std::string jointsCsv;
    for (auto* jNode : g_JointList) {
        if (!jNode->name) continue;
        std::string lower(jNode->name);
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
        g_JointNameToNode[lower] = jNode;
        if (!jointsCsv.empty()) jointsCsv += ", ";
        jointsCsv += jNode->name;
    }
    g_Ollama.availableJointsCSV = jointsCsv.empty() ? "(no joints loaded yet)" : jointsCsv;

    for (size_t i = 0; i < g_GltfData->images_count; ++i) {
        cgltf_image& img = g_GltfData->images[i];
        if (img.buffer_view) {
            unsigned char* bufStart = (unsigned char*)img.buffer_view->buffer->data + img.buffer_view->offset;
            int w, h, c;
            unsigned char* pixels = stbi_load_from_memory(bufStart, (int)img.buffer_view->size, &w, &h, &c, 4);
            if (pixels) {
                GLuint texID; glGenTextures(1, &texID);
                glBindTexture(GL_TEXTURE_2D, texID);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
                g_Textures.push_back(texID);
                ApplyTextureQualitySettings(texID);
                stbi_image_free(pixels);
            }
            else g_Textures.push_back(0);
        }
        else g_Textures.push_back(0);
    }
    return true;
}

bool g_IsRenderingShadowPass = false;

void ProcessAndDrawComponent(SegregatedMeshComponent& comp) {
    cgltf_mesh* mesh = comp.meshRef;
    cgltf_skin* skin = comp.skinRef;

    if (g_SettingWireframeMode && !g_IsRenderingShadowPass) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glDisable(GL_LIGHTING);
    }
    else {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        if (g_SettingUseLighting && !g_IsRenderingShadowPass) glEnable(GL_LIGHTING);
    }

    size_t strideJoints = 0;
    if (g_GltfData) {
        for (size_t i = 0; i < g_GltfData->skins_count; ++i) {
            if (g_GltfData->skins[i].joints_count > strideJoints) strideJoints = g_GltfData->skins[i].joints_count;
        }
    }
    size_t skinCacheBaseOffset = comp.skinIndex * strideJoints * 16;

    for (size_t j = 0; j < mesh->primitives_count; ++j) {
        cgltf_primitive& prim = mesh->primitives[j];

        int texIndex = -1;
        if (!g_SettingWireframeMode && !g_IsRenderingShadowPass && prim.material && prim.material->has_pbr_metallic_roughness) {
            cgltf_texture* tex = prim.material->pbr_metallic_roughness.base_color_texture.texture;
            if (tex && tex->image) {
                for (size_t imgIdx = 0; imgIdx < g_GltfData->images_count; ++imgIdx) {
                    if (&g_GltfData->images[imgIdx] == tex->image) { texIndex = (int)imgIdx; break; }
                }
            }
        }

        if (texIndex >= 0 && texIndex < (int)g_Textures.size() && !g_SettingWireframeMode && !g_IsRenderingShadowPass && !g_DebugDisableTextures) {
            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, g_Textures[texIndex]);
        }
        else glDisable(GL_TEXTURE_2D);

        cgltf_accessor* posAcc = nullptr, * normAcc = nullptr, * uvAcc = nullptr, * jointsAcc = nullptr, * weightsAcc = nullptr;
        for (size_t k = 0; k < prim.attributes_count; ++k) {
            switch (prim.attributes[k].type) {
            case cgltf_attribute_type_position: posAcc = prim.attributes[k].data; break;
            case cgltf_attribute_type_normal:   normAcc = prim.attributes[k].data; break;
            case cgltf_attribute_type_texcoord: uvAcc = prim.attributes[k].data; break;
            case cgltf_attribute_type_joints:   jointsAcc = prim.attributes[k].data; break;
            case cgltf_attribute_type_weights:  weightsAcc = prim.attributes[k].data; break;
            default: break;
            }
        }
        if (!posAcc) continue;

        size_t count = posAcc->count;
        if (g_VertexCache.size() < count) g_VertexCache.resize(count);

        const char* posPtr = (const char*)posAcc->buffer_view->buffer->data + posAcc->buffer_view->offset + posAcc->offset;
        size_t posStride = posAcc->stride;
        const char* normPtr = normAcc ? (const char*)normAcc->buffer_view->buffer->data + normAcc->buffer_view->offset + normAcc->offset : nullptr;
        size_t normStride = normAcc ? normAcc->stride : 0;
        const char* jointPtr = jointsAcc ? (const char*)jointsAcc->buffer_view->buffer->data + jointsAcc->buffer_view->offset + jointsAcc->offset : nullptr;
        size_t jointStride = jointsAcc ? jointsAcc->stride : 0;
        const char* weightPtr = weightsAcc ? (const char*)weightsAcc->buffer_view->buffer->data + weightsAcc->buffer_view->offset + weightsAcc->offset : nullptr;
        size_t weightStride = weightsAcc ? weightsAcc->stride : 0;
        int componentType = jointsAcc ? jointsAcc->component_type : 0;
        bool processSkinning = (skin && jointPtr && weightPtr);

#pragma omp parallel for if(g_SettingUseMulticore && count > 500) schedule(static)
        for (int v = 0; v < (int)count; ++v) {
            const float* p = (const float*)(posPtr + v * posStride);
            const float* n = normPtr ? (const float*)(normPtr + v * normStride) : nullptr;

            float animatedP[3] = { 0 }, finalN[3] = { 0 };
            float srcN[3] = { 0.0f, 1.0f, 0.0f };
            if (n) { srcN[0] = n[0]; srcN[1] = n[1]; srcN[2] = n[2]; }

            if (processSkinning) {
                const float* weights = (const float*)(weightPtr + v * weightStride);
                unsigned int jAr[4] = { 0 };
                if (componentType == cgltf_component_type_r_8u) {
                    const unsigned char* jb = (const unsigned char*)(jointPtr + v * jointStride);
                    jAr[0] = jb[0]; jAr[1] = jb[1]; jAr[2] = jb[2]; jAr[3] = jb[3];
                }
                else if (componentType == cgltf_component_type_r_16u) {
                    const unsigned short* js = (const unsigned short*)(jointPtr + v * jointStride);
                    jAr[0] = js[0]; jAr[1] = js[1]; jAr[2] = js[2]; jAr[3] = js[3];
                }

                float weightSum = weights[0] + weights[1] + weights[2] + weights[3];
                if (weightSum < 0.01f) {
                    TransformVec3(animatedP, comp.globalFallbackMatrix, p);
                    TransformNormal(finalN, comp.globalFallbackMatrix, srcN);
                }
                else {
                    for (int w = 0; w < 4; ++w) {
                        float weight = weights[w];
                        if (weight <= 0.0001f || jAr[w] >= skin->joints_count) continue;
                        size_t cachedMatIdx = skinCacheBaseOffset + (jAr[w] * 16);
                        if (cachedMatIdx + 15 >= g_SkinCombinedMatricesCache.size()) continue;
                        const float* jointMat = &g_SkinCombinedMatricesCache[cachedMatIdx];

                        float lP[3], lN[3];
                        TransformVec3(lP, jointMat, p);
                        TransformNormal(lN, jointMat, srcN);

                        animatedP[0] += lP[0] * weight; animatedP[1] += lP[1] * weight; animatedP[2] += lP[2] * weight;
                        finalN[0] += lN[0] * weight; finalN[1] += lN[1] * weight; finalN[2] += lN[2] * weight;
                    }
                }
            }
            else {
                TransformVec3(animatedP, comp.globalFallbackMatrix, p);
                TransformNormal(finalN, comp.globalFallbackMatrix, srcN);
            }

            g_VertexCache[v].p[0] = animatedP[0];
            g_VertexCache[v].p[1] = animatedP[1];
            g_VertexCache[v].p[2] = animatedP[2];
            g_VertexCache[v].n[0] = finalN[0];
            g_VertexCache[v].n[1] = finalN[1];
            g_VertexCache[v].n[2] = finalN[2];
        }

        const char* uvPtr = uvAcc ? (const char*)uvAcc->buffer_view->buffer->data + uvAcc->buffer_view->offset + uvAcc->offset : nullptr;
        size_t uvStride = uvAcc ? uvAcc->stride : 0;

        glBegin(GL_TRIANGLES);
        if (prim.indices) {
            const char* idxPtr = (const char*)prim.indices->buffer_view->buffer->data + prim.indices->buffer_view->offset + prim.indices->offset;
            size_t idxStride = prim.indices->stride;
            size_t indicesCount = prim.indices->count;

            for (size_t idx = 0; idx < indicesCount; ++idx) {
                size_t v;
                if (idxStride == 2)       v = *(const unsigned short*)(idxPtr + idx * 2);
                else if (idxStride == 4)  v = *(const unsigned int*)(idxPtr + idx * 4);
                else                      v = *(const unsigned char*)(idxPtr + idx);

                if (g_IsRenderingShadowPass) glColor4f(0.08f, 0.08f, 0.11f, 0.55f);
                else if (g_SettingWireframeMode) {
                    if (comp.classification == MESH_HAIR_STRANDS)       glColor3f(1.0f, 0.85f, 0.0f);
                    else if (comp.classification == MESH_CLOTHING_LAYER) glColor3f(0.0f, 1.0f, 0.3f);
                    else                                                glColor3f(0.0f, 0.8f, 1.0f);
                }
                else glColor3f(1.0f, 1.0f, 1.0f);

                if (uvPtr && !g_SettingWireframeMode && !g_IsRenderingShadowPass) glTexCoord2fv((const float*)(uvPtr + v * uvStride));
                glNormal3fv(g_VertexCache[v].n);
                glVertex3fv(g_VertexCache[v].p);
            }
        }
        else {
            for (size_t v = 0; v < count; ++v) {
                if (g_IsRenderingShadowPass) glColor4f(0.08f, 0.08f, 0.11f, 0.55f);
                else if (g_SettingWireframeMode) {
                    if (comp.classification == MESH_HAIR_STRANDS)       glColor3f(1.0f, 0.85f, 0.0f);
                    else if (comp.classification == MESH_CLOTHING_LAYER) glColor3f(0.0f, 1.0f, 0.3f);
                    else                                                glColor3f(0.0f, 0.8f, 1.0f);
                }
                else glColor3f(1.0f, 1.0f, 1.0f);

                if (uvPtr && !g_SettingWireframeMode && !g_IsRenderingShadowPass) glTexCoord2fv((const float*)(uvPtr + v * uvStride));
                glNormal3fv(g_VertexCache[v].n);
                glVertex3fv(g_VertexCache[v].p);
            }
        }
        glEnd();
    }
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void RenderGroundShadowGrid() {
    glDisable(GL_LIGHTING); glDisable(GL_TEXTURE_2D);
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    float shadowMat[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        -0.3f, 0.0f, -0.4f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.001f, 0.0f, 1.0f
    };

    glPushMatrix(); glMultMatrixf(shadowMat);
    g_IsRenderingShadowPass = true;
    for (auto& comp : g_SeparatedComponents) ProcessAndDrawComponent(comp);
    g_IsRenderingShadowPass = false;
    glPopMatrix(); glDisable(GL_BLEND);
}

void DrawVRMModel() {
    if (g_SeparatedComponents.empty()) {
        if (g_SettingUseCulling) glEnable(GL_CULL_FACE); else glDisable(GL_CULL_FACE);
        if (g_SettingUseLighting) glEnable(GL_LIGHTING); else glDisable(GL_LIGHTING);
        glPushMatrix();
        glScalef(0.5f, 0.5f, 0.5f);
        glTranslatef(0.0f, 1.0f, 0.0f);
        glBegin(GL_QUADS);
        glNormal3f(0.0f, 0.0f, 1.0f); glColor3f(0.8f, 0.2f, 0.2f);
        glVertex3f(-1.0f, -1.0f, 1.0f);
        glVertex3f(1.0f, -1.0f, 1.0f);
        glVertex3f(1.0f, 1.0f, 1.0f);
        glVertex3f(-1.0f, 1.0f, 1.0f);
        glEnd();
        glPopMatrix();
        return;
    }

    FastUpdateGlobalTransforms();

    if (g_SettingUseCulling) glEnable(GL_CULL_FACE); else glDisable(GL_CULL_FACE);

    glPushMatrix();

    // Room is drawn in world space, before the character root transform -
    // it doesn't move with the character.
    DrawGround();
    DrawEnvironment(g_EnvProps, g_EnvHalfSize);

    glPushMatrix();
    glTranslatef(g_CharacterX, 0.0f, g_CharacterZ);
    glRotatef(g_CharacterFacing * 57.2957795f, 0.0f, 1.0f, 0.0f);

    if (g_SettingRenderShadows) RenderGroundShadowGrid();
    if (g_SettingUseLighting && !g_SettingWireframeMode) glEnable(GL_LIGHTING); else glDisable(GL_LIGHTING);

    for (auto& comp : g_SeparatedComponents) {
        ProcessAndDrawComponent(comp);
    }

    // Debug
    if (g_DebugShowBones) {
        glDisable(GL_LIGHTING);
        glDisable(GL_DEPTH_TEST);
        glLineWidth(3.0f);
        glBegin(GL_LINES);
        glColor3f(1.0f, 0.8f, 0.1f);
        for (const auto& fn : g_FlatNodes) {
            if (fn.parentIdx != (size_t)-1) {
                const float* pMat = g_FlatNodes[fn.parentIdx].globalMatrix;
                const float* cMat = fn.globalMatrix;
                glVertex3f(pMat[12], pMat[13], pMat[14]);
                glVertex3f(cMat[12], cMat[13], cMat[14]);
            }
        }
        glEnd();
        glEnable(GL_DEPTH_TEST);
    }

    // Joint spheres - a pure visualization now, not something the user can
    // grab: magenta means the AI currently has a non-neutral target on that
    // joint, cyan means it's sitting at (or returning to) neutral.
    if (g_JointQuadric) {
        glDisable(GL_LIGHTING);
        glDisable(GL_TEXTURE_2D);
        for (auto* jNode : g_JointList) {
            auto it = g_NodeToFlatIdx.find(jNode);
            if (it == g_NodeToFlatIdx.end()) continue;
            const float* m = g_FlatNodes[it->second].globalMatrix;
            glPushMatrix();
            glTranslatef(m[12], m[13], m[14]);

            bool hasTarget = false;
            auto rigIt = g_SkeletalRig.find(jNode);
            if (rigIt != g_SkeletalRig.end()) {
                const JointAngles& ja = rigIt->second;
                hasTarget = fabsf(ja.tgtX) > 0.02f || fabsf(ja.tgtY) > 0.02f || fabsf(ja.tgtZ) > 0.02f;
            }
            if (hasTarget) glColor3f(1.0f, 0.2f, 0.9f);
            else           glColor3f(0.1f, 0.9f, 1.0f);

            gluSphere(g_JointQuadric, kJointSphereRadius, 10, 8);
            glPopMatrix();
        }
        if (g_SettingUseLighting && !g_SettingWireframeMode) glEnable(GL_LIGHTING);
    }

    glPopMatrix(); // character root transform
    glPopMatrix();
}

// --- Agent command protocol -------------------------------------------
// Finds the matching outer-brace span of the JSON object that follows
// "[AGENT:" in a chat reply, tracking string/brace depth rather than just
// hunting for the next ']' - a joint name or stray punctuation in the
// model's own prose could otherwise close the scan early.
static bool FindAgentJsonSpan(const std::string& text, size_t& tagStart, size_t& jsonStart, size_t& jsonEnd) {
    tagStart = text.find("[AGENT:");
    if (tagStart == std::string::npos) return false;
    jsonStart = text.find('{', tagStart);
    if (jsonStart == std::string::npos) return false;

    int depth = 0;
    bool inString = false;
    char prev = 0;
    for (size_t i = jsonStart; i < text.size(); ++i) {
        char c = text[i];
        if (inString) {
            if (c == '"' && prev != '\\') inString = false;
        }
        else {
            if (c == '"') inString = true;
            else if (c == '{') depth++;
            else if (c == '}') {
                depth--;
                if (depth == 0) { jsonEnd = i; return true; }
            }
        }
        prev = c;
    }
    return false;
}

bool ParseAgentCommand(const std::string& text, MoveAction& outMove, std::unordered_map<std::string, std::array<float, 3>>& outJointTargets) {
    outMove = MoveAction::None;
    outJointTargets.clear();

    size_t tagStart, jsonStart, jsonEnd;
    if (!FindAgentJsonSpan(text, tagStart, jsonStart, jsonEnd)) return false;

    std::string jsonBlob = text.substr(jsonStart, jsonEnd - jsonStart + 1);
    try {
        auto j = nlohmann::json::parse(jsonBlob);

        if (j.contains("move") && j["move"].is_string()) {
            std::string mv = j["move"].get<std::string>();
            if (mv == "walk_forward")       outMove = MoveAction::WalkForward;
            else if (mv == "walk_backward") outMove = MoveAction::WalkBackward;
            else if (mv == "turn_left")     outMove = MoveAction::TurnLeft;
            else if (mv == "turn_right")    outMove = MoveAction::TurnRight;
        }

        if (j.contains("pose") && j["pose"].is_object()) {
            for (auto it = j["pose"].begin(); it != j["pose"].end(); ++it) {
                std::array<float, 3> axes = { NAN, NAN, NAN };
                const auto& v = it.value();
                if (v.is_object()) {
                    if (v.contains("x") && v["x"].is_number()) axes[0] = v["x"].get<float>();
                    if (v.contains("y") && v["y"].is_number()) axes[1] = v["y"].get<float>();
                    if (v.contains("z") && v["z"].is_number()) axes[2] = v["z"].get<float>();
                }
                outJointTargets[it.key()] = axes;
            }
        }
        return true;
    }
    catch (...) {
        return false;
    }
}

// Display-only: cuts the [AGENT: {...}] block out of a reply so the chat
// transcript shows what the character "said" without the raw command JSON.
std::string StripActionTagForDisplay(const std::string& text) {
    size_t tagStart, jsonStart, jsonEnd;
    if (!FindAgentJsonSpan(text, tagStart, jsonStart, jsonEnd)) return text;

    size_t closeBracket = text.find(']', jsonEnd);
    size_t stripEnd = (closeBracket != std::string::npos) ? closeBracket + 1 : jsonEnd + 1;

    std::string result = text.substr(0, tagStart) + text.substr(stripEnd);
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r' || result.back() == ' ')) result.pop_back();
    return result;
}

// Applies a parsed pose command to the rig: sets joint targets (curX/Y/Z
// chase these every frame via UpdateJointSmoothing) and reports which
// joint names, if any, didn't match anything on the current model - the AI
// occasionally invents or misspells a joint name despite the whitelist in
// the system prompt, and this makes that visible in the status bar instead
// of silently doing nothing.
std::string ApplyPoseCommand(const std::unordered_map<std::string, std::array<float, 3>>& jointTargets) {
    int posed = 0;
    std::vector<std::string> unknown;
    for (auto& kv : jointTargets) {
        std::string lowerName = kv.first;
        std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), [](unsigned char c) { return std::tolower(c); });

        auto it = g_JointNameToNode.find(lowerName);
        if (it == g_JointNameToNode.end()) { unknown.push_back(kv.first); continue; }
        auto rigIt = g_SkeletalRig.find(it->second);
        if (rigIt == g_SkeletalRig.end()) continue;

        const auto& axes = kv.second;
        if (!std::isnan(axes[0])) rigIt->second.tgtX = axes[0];
        if (!std::isnan(axes[1])) rigIt->second.tgtY = axes[1];
        if (!std::isnan(axes[2])) rigIt->second.tgtZ = axes[2];
        posed++;
    }

    std::string result = "posed " + std::to_string(posed) + " joint(s)";
    if (!unknown.empty()) {
        result += " (unknown:";
        for (auto& u : unknown) result += " " + u;
        result += ")";
    }
    return result;
}

// Builds the "[Current state: ...]" note appended to the user's message so
// the model has some idea where its own body is without needing real vision.
std::string BuildSpatialContext() {
    float f = fmodf(g_CharacterFacing, 6.2831853f);
    if (f < 0.0f) f += 6.2831853f;
    const char* facingLabel;
    if (f < 0.785f || f >= 5.498f)   facingLabel = "north";
    else if (f < 2.356f)             facingLabel = "east";
    else if (f < 3.927f)             facingLabel = "south";
    else                              facingLabel = "west";

    float distFromCenter = sqrtf(g_CharacterX * g_CharacterX + g_CharacterZ * g_CharacterZ);
    std::string region = (distFromCenter > g_EnvHalfSize * 0.6f) ? "a wall" : "the center of the room";

    std::string ctx = "[Current state: standing near " + region + ", facing " + facingLabel + ". Nearby:";
    bool first = true;
    for (auto& p : g_EnvProps) {
        float dx = p.x - g_CharacterX, dz = p.z - g_CharacterZ;
        char buf[32];
        snprintf(buf, sizeof(buf), "%.1fm", sqrtf(dx * dx + dz * dz));
        ctx += (first ? " " : ", ");
        ctx += std::string(p.label) + " (" + buf + " away)";
        first = false;
    }
    ctx += ".]";
    return ctx;
}

float DurationForMove(MoveAction a) {
    return (a == MoveAction::TurnLeft || a == MoveAction::TurnRight) ? 1.0f : 2.5f;
}

// Every joint's current angle chases its AI-set target at g_SettingRigSmooth
// - this is what makes that slider (previously dead - its control wasn't
// even wired to a real trackbar) actually matter: higher values snap to a
// new pose faster, lower values drift into it.
void UpdateJointSmoothing(float dt) {
    for (auto& kv : g_SkeletalRig) {
        JointAngles& ja = kv.second;
        float nx = Lerp(ja.curX, ja.tgtX, g_SettingRigSmooth, dt);
        float ny = Lerp(ja.curY, ja.tgtY, g_SettingRigSmooth, dt);
        float nz = Lerp(ja.curZ, ja.tgtZ, g_SettingRigSmooth, dt);
        if (fabsf(nx - ja.curX) > 0.0001f || fabsf(ny - ja.curY) > 0.0001f || fabsf(nz - ja.curZ) > 0.0001f) {
            g_RigHierarchyDirty = true;
        }
        ja.curX = nx; ja.curY = ny; ja.curZ = nz;
    }
}

// Runs once per frame: picks up whatever the Ollama worker thread last
// decided, advances any in-progress walk/turn, and chases joint targets.
void UpdateAgent(float dt) {
    if (g_Ollama.hasPendingAction) {
        std::string raw;
        { std::lock_guard<std::mutex> lock(g_Ollama.actionMutex); raw = g_Ollama.pendingActionRaw; }

        MoveAction move;
        std::unordered_map<std::string, std::array<float, 3>> jointTargets;
        if (ParseAgentCommand(raw, move, jointTargets)) {
            if (move != MoveAction::None) g_MoveQueue.push_back({ move, DurationForMove(move) });
            g_LastActionStatus = ApplyPoseCommand(jointTargets);
        }
        else {
            g_LastActionStatus = "no action parsed";
        }
        g_Ollama.hasPendingAction = false;
    }

    if (g_CurrentMove == MoveAction::None && !g_MoveQueue.empty()) {
        QueuedMove qm = g_MoveQueue.front(); g_MoveQueue.pop_front();
        g_CurrentMove = qm.action;
        g_CurrentMoveTimeLeft = qm.duration;
    }

    if (g_CurrentMove != MoveAction::None) {
        const float walkSpeed = 0.8f, turnSpeed = 1.2f;
        switch (g_CurrentMove) {
        case MoveAction::WalkForward:
            g_CharacterX += sinf(g_CharacterFacing) * walkSpeed * dt;
            g_CharacterZ += cosf(g_CharacterFacing) * walkSpeed * dt;
            break;
        case MoveAction::WalkBackward:
            g_CharacterX -= sinf(g_CharacterFacing) * walkSpeed * dt;
            g_CharacterZ -= cosf(g_CharacterFacing) * walkSpeed * dt;
            break;
        case MoveAction::TurnLeft:  g_CharacterFacing -= turnSpeed * dt; break;
        case MoveAction::TurnRight: g_CharacterFacing += turnSpeed * dt; break;
        default: break;
        }
        ResolveCollision(g_CharacterX, g_CharacterZ, g_EnvProps, g_EnvHalfSize, 0.25f);

        g_CurrentMoveTimeLeft -= dt;
        if (g_CurrentMoveTimeLeft <= 0.0f) g_CurrentMove = MoveAction::None;
    }

    UpdateJointSmoothing(dt);
}

void initOpenGL(HWND hwnd) {
    hDC = GetDC(hwnd);
    PIXELFORMATDESCRIPTOR pfd = {}; pfd.nSize = sizeof(pfd); pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA; pfd.cColorBits = 32; pfd.cDepthBits = 24; pfd.iLayerType = PFD_MAIN_PLANE;
    int pf = ChoosePixelFormat(hDC, &pfd); SetPixelFormat(hDC, pf, &pfd);
    hRC = wglCreateContext(hDC); wglMakeCurrent(hDC, hRC);

    glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING); glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL); glEnable(GL_CULL_FACE);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

    GLfloat lightPos[] = { 1.5f, 4.0f, 3.5f, 1.0f };
    GLfloat lightAmb[] = { 0.40f, 0.40f, 0.42f, 1.0f };
    GLfloat lightDif[] = { 0.75f, 0.75f, 0.75f, 1.0f };
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);
    glLightfv(GL_LIGHT0, GL_AMBIENT, lightAmb);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDif);

    glClearColor(0.12f, 0.12f, 0.15f, 1.0f);
    glEnable(GL_SCISSOR_TEST);
    g_glGenerateMipmapEXT = (PFNGLGENERATEMIPMAPEXTPROC_t)wglGetProcAddress("glGenerateMipmapEXT");

    g_JointQuadric = gluNewQuadric();
    SetupEnvironment(g_EnvProps);
}

void resizeGL(int width, int height) {
    winWidth = width; winHeight = height;
    int glH = (height > TOOLBAR_HEIGHT) ? height - TOOLBAR_HEIGHT : 1;
    glViewport(0, TOOLBAR_HEIGHT, width, glH);
    glScissor(0, TOOLBAR_HEIGHT, width, glH);
    glMatrixMode(GL_PROJECTION); glLoadIdentity();
    gluPerspective(45.0, (double)width / (double)(glH > 0 ? glH : 1), 0.1, 100.0);
    glMatrixMode(GL_MODELVIEW);
}

void UpdateTabVisibility(HWND hDlg) {
    int tab = TabCtrl_GetCurSel(GetDlgItem(hDlg, IDC_TAB_MAIN));

    const int tab0_IDs[] = {
        IDC_CHECK_MULTICORE, IDC_CHECK_CULLING, IDC_CHECK_LIGHTING,
        IDC_CHECK_SHADOWS, IDC_CHECK_WIREFRAME, IDC_COMBO_TEX_QUALITY,
        IDC_SLIDER_CAM_SPD, IDC_SLIDER_RIG_SMO,
        IDC_CHECK_INVERT_CAM, IDC_RADIO_BG_DARK, IDC_RADIO_BG_LIGHT,
        IDC_STATIC_GRP2, IDC_STATIC_GRP3, IDC_STATIC_GRP4, IDC_STATIC_GRP5,
        IDC_STATIC_TXT1, IDC_STATIC_TXT2, IDC_STATIC_TXT3
    };

    const int tab1_IDs[] = {
        IDC_DEBUG_GRP1, IDC_CHECK_DB_BONES, IDC_CHECK_DB_NOTEX
    };

    for (int id : tab0_IDs) {
        HWND h = GetDlgItem(hDlg, id);
        if (h) ShowWindow(h, (tab == 0) ? SW_SHOW : SW_HIDE);
    }
    for (int id : tab1_IDs) {
        HWND h = GetDlgItem(hDlg, id);
        if (h) ShowWindow(h, (tab == 1) ? SW_SHOW : SW_HIDE);
    }
}

INT_PTR CALLBACK SettingsDlgProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_INITDIALOG:
        CheckDlgButton(hDlg, IDC_CHECK_MULTICORE, g_SettingUseMulticore ? BST_CHECKED : BST_UNCHECKED);
        CheckDlgButton(hDlg, IDC_CHECK_CULLING, g_SettingUseCulling ? BST_CHECKED : BST_UNCHECKED);
        CheckDlgButton(hDlg, IDC_CHECK_LIGHTING, g_SettingUseLighting ? BST_CHECKED : BST_UNCHECKED);
        CheckDlgButton(hDlg, IDC_CHECK_INVERT_CAM, g_SettingInvertCam ? BST_CHECKED : BST_UNCHECKED);
        CheckDlgButton(hDlg, IDC_CHECK_SHADOWS, g_SettingRenderShadows ? BST_CHECKED : BST_UNCHECKED);
        CheckDlgButton(hDlg, IDC_CHECK_WIREFRAME, g_SettingWireframeMode ? BST_CHECKED : BST_UNCHECKED);
        CheckRadioButton(hDlg, IDC_RADIO_BG_DARK, IDC_RADIO_BG_LIGHT, (g_SettingBgStyle == 0) ? IDC_RADIO_BG_DARK : IDC_RADIO_BG_LIGHT);
        SendMessage(GetDlgItem(hDlg, IDC_SLIDER_CAM_SPD), TBM_SETRANGE, TRUE, MAKELONG(2, 120));
        SendMessage(GetDlgItem(hDlg, IDC_SLIDER_CAM_SPD), TBM_SETPOS, TRUE, (int)g_SettingCamSmooth);
        SendMessage(GetDlgItem(hDlg, IDC_SLIDER_RIG_SMO), TBM_SETRANGE, TRUE, MAKELONG(2, 120));
        SendMessage(GetDlgItem(hDlg, IDC_SLIDER_RIG_SMO), TBM_SETPOS, TRUE, (int)g_SettingRigSmooth);

        {
            HWND hCombo = GetDlgItem(hDlg, IDC_COMBO_TEX_QUALITY);
            SendMessageA(hCombo, CB_ADDSTRING, 0, (LPARAM)"Low (Nearest Filtering)");
            SendMessageA(hCombo, CB_ADDSTRING, 0, (LPARAM)"Medium (Linear Filtering)");
            SendMessageA(hCombo, CB_ADDSTRING, 0, (LPARAM)"High Ultra (Mipmapped Profiles)");
            SendMessage(hCombo, CB_SETCURSEL, g_SettingTextureQuality, 0);
        }

        CheckDlgButton(hDlg, IDC_CHECK_DB_BONES, g_DebugShowBones ? BST_CHECKED : BST_UNCHECKED);
        CheckDlgButton(hDlg, IDC_CHECK_DB_NOTEX, g_DebugDisableTextures ? BST_CHECKED : BST_UNCHECKED);
        UpdateTabVisibility(hDlg);
        return (INT_PTR)TRUE;

    case WM_NOTIFY: {
        LPNMHDR lpnmhdr = (LPNMHDR)lParam;
        if (lpnmhdr->code == TCN_SELCHANGE && lpnmhdr->idFrom == IDC_TAB_MAIN) {
            UpdateTabVisibility(hDlg);
            return (INT_PTR)TRUE;
        }
        break;
    }

    case WM_COMMAND:
        if (LOWORD(wParam) == IDC_COMBO_TEX_QUALITY && HIWORD(wParam) == CBN_SELCHANGE) {
            g_SettingTextureQuality = (int)SendMessage(GetDlgItem(hDlg, IDC_COMBO_TEX_QUALITY), CB_GETCURSEL, 0, 0);
            for (GLuint tex : g_Textures) ApplyTextureQualitySettings(tex);
            return (INT_PTR)TRUE;
        }
        if (LOWORD(wParam) == IDOK || HIWORD(wParam) == BN_CLICKED) {
            g_SettingUseMulticore = (IsDlgButtonChecked(hDlg, IDC_CHECK_MULTICORE) == BST_CHECKED);
            g_SettingUseCulling = (IsDlgButtonChecked(hDlg, IDC_CHECK_CULLING) == BST_CHECKED);
            g_SettingUseLighting = (IsDlgButtonChecked(hDlg, IDC_CHECK_LIGHTING) == BST_CHECKED);
            g_SettingInvertCam = (IsDlgButtonChecked(hDlg, IDC_CHECK_INVERT_CAM) == BST_CHECKED);
            g_SettingRenderShadows = (IsDlgButtonChecked(hDlg, IDC_CHECK_SHADOWS) == BST_CHECKED);
            g_SettingWireframeMode = (IsDlgButtonChecked(hDlg, IDC_CHECK_WIREFRAME) == BST_CHECKED);
            g_SettingBgStyle = (IsDlgButtonChecked(hDlg, IDC_RADIO_BG_DARK) == BST_CHECKED) ? 0 : 1;
            g_DebugShowBones = (IsDlgButtonChecked(hDlg, IDC_CHECK_DB_BONES) == BST_CHECKED);
            g_DebugDisableTextures = (IsDlgButtonChecked(hDlg, IDC_CHECK_DB_NOTEX) == BST_CHECKED);

            if (g_SettingBgStyle == 0) glClearColor(0.12f, 0.12f, 0.15f, 1.0f);
            else                       glClearColor(0.40f, 0.40f, 0.42f, 1.0f);

            // This used to call SetFocus(hWnd) unconditionally here, which
            // ran on EVERY checkbox/radio click (not just when closing via
            // "Close") - meaning every single toggle bounced focus back to
            // the main window mid-interaction. Only hand focus back when
            // the dialog is actually being destroyed.
            if (LOWORD(wParam) == IDOK) {
                DestroyWindow(hDlg);
                g_hSettingsDlg = nullptr;
                SetFocus(hWnd);
            }
            return (INT_PTR)TRUE;
        }
        if (LOWORD(wParam) == IDCANCEL) { DestroyWindow(hDlg); g_hSettingsDlg = nullptr; SetFocus(hWnd); return (INT_PTR)TRUE; }
        break;

    case WM_HSCROLL:
        g_SettingCamSmooth = (float)SendMessage(GetDlgItem(hDlg, IDC_SLIDER_CAM_SPD), TBM_GETPOS, 0, 0);
        g_SettingRigSmooth = (float)SendMessage(GetDlgItem(hDlg, IDC_SLIDER_RIG_SMO), TBM_GETPOS, 0, 0);
        return (INT_PTR)TRUE;
    }
    return (INT_PTR)FALSE;
}

void ShowSettingsMenu(HWND parent) {
    if (g_hSettingsDlg) { SetActiveWindow(g_hSettingsDlg); return; }
    std::vector<WORD> dialogTemplate(1024);
    LPDLGTEMPLATE lpdt = (LPDLGTEMPLATE)dialogTemplate.data();
    lpdt->style = DS_SETFONT | DS_MODALFRAME | WS_POPUP | WS_CAPTION | WS_SYSMENU;

    lpdt->cdit = 0; lpdt->x = 10; lpdt->y = 10; lpdt->cx = 300; lpdt->cy = 420;

    g_hSettingsDlg = CreateDialogIndirectParamA(GetModuleHandle(nullptr), lpdt, parent, SettingsDlgProc, 0);
    if (!g_hSettingsDlg) return;

    SetWindowTextA(g_hSettingsDlg, "Engine Configuration");
    HFONT hFont = (HFONT)GetStockObject(DEFAULT_GUI_FONT);

    HWND hTab = CreateWindowExA(0, WC_TABCONTROLA, "", WS_CHILD | WS_CLIPSIBLINGS | WS_VISIBLE, 5, 5, 280, 370, g_hSettingsDlg, (HMENU)IDC_TAB_MAIN, nullptr, nullptr);
    TCITEMA tie = { 0 }; tie.mask = TCIF_TEXT;
    tie.pszText = (LPSTR)"Core Settings"; SendMessageA(hTab, TCM_INSERTITEM, 0, (LPARAM)&tie);
    tie.pszText = (LPSTR)"Debug Tools";   SendMessageA(hTab, TCM_INSERTITEM, 1, (LPARAM)&tie);

    CreateWindowA("BUTTON", "Hardware Architecture", WS_CHILD | WS_VISIBLE | BS_GROUPBOX, 15, 35, 255, 75, g_hSettingsDlg, (HMENU)IDC_STATIC_GRP2, nullptr, nullptr);
    CreateWindowA("BUTTON", "Enable OpenMP Threads", WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 25, 50, 235, 18, g_hSettingsDlg, (HMENU)IDC_CHECK_MULTICORE, nullptr, nullptr);
    CreateWindowA("BUTTON", "Enable Backface Culling", WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 25, 68, 235, 18, g_hSettingsDlg, (HMENU)IDC_CHECK_CULLING, nullptr, nullptr);
    CreateWindowA("BUTTON", "Enable Dynamic Lighting", WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 25, 86, 235, 18, g_hSettingsDlg, (HMENU)IDC_CHECK_LIGHTING, nullptr, nullptr);

    CreateWindowA("BUTTON", "Graphics Pipeline", WS_CHILD | WS_VISIBLE | BS_GROUPBOX, 15, 120, 255, 90, g_hSettingsDlg, (HMENU)IDC_STATIC_GRP3, nullptr, nullptr);
    CreateWindowA("BUTTON", "Render Shadows", WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 25, 135, 235, 18, g_hSettingsDlg, (HMENU)IDC_CHECK_SHADOWS, nullptr, nullptr);
    CreateWindowA("BUTTON", "Wireframe Mode", WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 25, 153, 235, 18, g_hSettingsDlg, (HMENU)IDC_CHECK_WIREFRAME, nullptr, nullptr);
    CreateWindowA("STATIC", "Texture Quality:", WS_CHILD | WS_VISIBLE, 25, 177, 100, 15, g_hSettingsDlg, (HMENU)IDC_STATIC_TXT1, nullptr, nullptr);
    CreateWindowA("COMBOBOX", "", WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST | WS_VSCROLL, 125, 174, 135, 100, g_hSettingsDlg, (HMENU)IDC_COMBO_TEX_QUALITY, nullptr, nullptr);

    CreateWindowA("BUTTON", "Camera && Rig Smoothing", WS_CHILD | WS_VISIBLE | BS_GROUPBOX, 15, 220, 255, 70, g_hSettingsDlg, (HMENU)IDC_STATIC_GRP4, nullptr, nullptr);
    CreateWindowA("STATIC", "Camera Speed", WS_CHILD | WS_VISIBLE, 25, 235, 100, 15, g_hSettingsDlg, (HMENU)IDC_STATIC_TXT2, nullptr, nullptr);
    CreateWindowA("msctls_trackbar32", "", WS_CHILD | WS_VISIBLE | TBS_HORZ, 130, 233, 130, 22, g_hSettingsDlg, (HMENU)IDC_SLIDER_CAM_SPD, nullptr, nullptr);
    CreateWindowA("STATIC", "Rig Smoothness", WS_CHILD | WS_VISIBLE, 25, 259, 100, 15, g_hSettingsDlg, (HMENU)IDC_STATIC_TXT3, nullptr, nullptr);
    CreateWindowA("msctls_trackbar32", "", WS_CHILD | WS_VISIBLE | TBS_HORZ, 130, 257, 130, 22, g_hSettingsDlg, (HMENU)IDC_SLIDER_RIG_SMO, nullptr, nullptr);

    CreateWindowA("BUTTON", "Camera & Background", WS_CHILD | WS_VISIBLE | BS_GROUPBOX, 15, 300, 255, 65, g_hSettingsDlg, (HMENU)IDC_STATIC_GRP5, nullptr, nullptr);
    CreateWindowA("BUTTON", "Invert Camera", WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 25, 315, 235, 18, g_hSettingsDlg, (HMENU)IDC_CHECK_INVERT_CAM, nullptr, nullptr);
    CreateWindowA("BUTTON", "Dark BG", WS_CHILD | WS_VISIBLE | BS_AUTORADIOBUTTON | WS_GROUP, 30, 335, 100, 20, g_hSettingsDlg, (HMENU)IDC_RADIO_BG_DARK, nullptr, nullptr);
    CreateWindowA("BUTTON", "Light BG", WS_CHILD | WS_VISIBLE | BS_AUTORADIOBUTTON, 145, 335, 100, 20, g_hSettingsDlg, (HMENU)IDC_RADIO_BG_LIGHT, nullptr, nullptr);

    CreateWindowA("BUTTON", "Render Overlays", WS_CHILD | WS_VISIBLE | BS_GROUPBOX, 15, 35, 255, 70, g_hSettingsDlg, (HMENU)IDC_DEBUG_GRP1, nullptr, nullptr);
    CreateWindowA("BUTTON", "Show Bones (Yellow)", WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 25, 55, 235, 18, g_hSettingsDlg, (HMENU)IDC_CHECK_DB_BONES, nullptr, nullptr);
    CreateWindowA("BUTTON", "Disable Textures", WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 25, 75, 235, 18, g_hSettingsDlg, (HMENU)IDC_CHECK_DB_NOTEX, nullptr, nullptr);

    HWND hOk = CreateWindowA("BUTTON", "Close", WS_CHILD | WS_VISIBLE | BS_DEFPUSHBUTTON, 100, 385, 100, 26, g_hSettingsDlg, (HMENU)IDOK, nullptr, nullptr);

    const HWND elements[] = { hTab, hOk };
    for (HWND h : elements) SendMessage(h, WM_SETFONT, (WPARAM)hFont, TRUE);

    SetWindowPos(g_hSettingsDlg, nullptr, 0, 0, 305, 470, SWP_NOMOVE | SWP_NOZORDER);
    SendMessage(g_hSettingsDlg, WM_INITDIALOG, 0, 0);
    ShowWindow(g_hSettingsDlg, SW_SHOW);
    SetForegroundWindow(g_hSettingsDlg);
    SetFocus(g_hSettingsDlg);
}

// --- Chat window ---------------------------------------------------------
// A second modeless dialog, built the same hand-rolled way as the Settings
// dialog above (CreateDialogIndirectParamA + a raw DLGTEMPLATE + child
// CreateWindowA controls) since this project has no ImGui or any other UI
// toolkit to reach for.

static void AppendToTranscript(const std::string& text) {
    if (!g_hChatTranscript) return;
    int len = GetWindowTextLengthA(g_hChatTranscript);
    SendMessageA(g_hChatTranscript, EM_SETSEL, (WPARAM)len, (LPARAM)len);
    SendMessageA(g_hChatTranscript, EM_REPLACESEL, FALSE, (LPARAM)text.c_str());
    SendMessageA(g_hChatTranscript, EM_SCROLLCARET, 0, 0);
}

INT_PTR CALLBACK ChatDlgProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_INITDIALOG:
        g_hChatTranscript = GetDlgItem(hDlg, IDC_CHAT_TRANSCRIPT);
        g_hChatInput = GetDlgItem(hDlg, IDC_CHAT_INPUT);
        g_hChatModelCombo = GetDlgItem(hDlg, IDC_CHAT_MODEL_COMBO);
        g_ChatRenderedHistoryCount = 0;
        g_ChatRenderedModelsCount = (size_t)-1;
        g_Ollama.RefreshModels();
        return (INT_PTR)TRUE;

    case WM_COMMAND:
        if (LOWORD(wParam) == IDC_CHAT_REFRESH_BTN && HIWORD(wParam) == BN_CLICKED) {
            g_Ollama.RefreshModels();
            return (INT_PTR)TRUE;
        }
        // Send is the dialog's default pushbutton, so IsDialogMessage
        // routes Enter in the input box here too - no key-event plumbing
        // needed for that.
        if (LOWORD(wParam) == IDC_CHAT_SEND_BTN && HIWORD(wParam) == BN_CLICKED) {
            char buf[2048] = {};
            GetWindowTextA(g_hChatInput, buf, sizeof(buf));
            std::string text(buf);
            if (!text.empty() && !g_Ollama.busy) {
                int sel = (int)SendMessage(g_hChatModelCombo, CB_GETCURSEL, 0, 0);
                if (sel >= 0 && sel < (int)g_Ollama.models.size()) g_Ollama.selectedModel = g_Ollama.models[sel];

                g_Ollama.spatialContext = BuildSpatialContext();
                bool wantScreenshot = (IsDlgButtonChecked(hDlg, IDC_CHAT_SCREENSHOT_CHK) == BST_CHECKED);
                g_Ollama.screenshotBase64PNG = wantScreenshot ? CaptureScreenshotBase64PNG() : "";

                g_Ollama.SendMessage(text);
                SetWindowTextA(g_hChatInput, "");
            }
            return (INT_PTR)TRUE;
        }
        if (LOWORD(wParam) == IDCANCEL) {
            DestroyWindow(hDlg);
            g_hChatDlg = nullptr;
            g_hChatTranscript = nullptr;
            g_hChatInput = nullptr;
            g_hChatModelCombo = nullptr;
            SetFocus(hWnd);
            return (INT_PTR)TRUE;
        }
        break;
    }
    return (INT_PTR)FALSE;
}

void ShowChatWindow(HWND parent) {
    if (g_hChatDlg) { SetActiveWindow(g_hChatDlg); return; }
    std::vector<WORD> dialogTemplate(1024);
    LPDLGTEMPLATE lpdt = (LPDLGTEMPLATE)dialogTemplate.data();
    lpdt->style = DS_SETFONT | DS_MODALFRAME | WS_POPUP | WS_CAPTION | WS_SYSMENU;
    lpdt->cdit = 0; lpdt->x = 20; lpdt->y = 20; lpdt->cx = 320; lpdt->cy = 300;

    g_hChatDlg = CreateDialogIndirectParamA(GetModuleHandle(nullptr), lpdt, parent, ChatDlgProc, 0);
    if (!g_hChatDlg) return;

    SetWindowTextA(g_hChatDlg, "rinAI - Chat");
    HFONT hFont = (HFONT)GetStockObject(DEFAULT_GUI_FONT);

    HWND hTranscript = CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", "",
        WS_CHILD | WS_VISIBLE | WS_VSCROLL | ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL,
        10, 10, 380, 260, g_hChatDlg, (HMENU)IDC_CHAT_TRANSCRIPT, nullptr, nullptr);

    CreateWindowA("STATIC", "Model:", WS_CHILD | WS_VISIBLE, 10, 280, 45, 15, g_hChatDlg, (HMENU)IDC_CHAT_MODEL_LABEL, nullptr, nullptr);
    HWND hModelCombo = CreateWindowA("COMBOBOX", "", WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST | WS_VSCROLL,
        58, 277, 200, 100, g_hChatDlg, (HMENU)IDC_CHAT_MODEL_COMBO, nullptr, nullptr);
    HWND hRefreshBtn = CreateWindowA("BUTTON", "Refresh", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
        264, 276, 60, 22, g_hChatDlg, (HMENU)IDC_CHAT_REFRESH_BTN, nullptr, nullptr);
    HWND hScreenshotChk = CreateWindowA("BUTTON", "Send screenshot each turn (vision models only)",
        WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 10, 305, 330, 18, g_hChatDlg, (HMENU)IDC_CHAT_SCREENSHOT_CHK, nullptr, nullptr);

    HWND hInput = CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", "", WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL,
        10, 330, 300, 24, g_hChatDlg, (HMENU)IDC_CHAT_INPUT, nullptr, nullptr);
    HWND hSendBtn = CreateWindowA("BUTTON", "Send", WS_CHILD | WS_VISIBLE | BS_DEFPUSHBUTTON,
        316, 329, 74, 26, g_hChatDlg, (HMENU)IDC_CHAT_SEND_BTN, nullptr, nullptr);

    const HWND elements[] = { hTranscript, hModelCombo, hRefreshBtn, hScreenshotChk, hInput, hSendBtn };
    for (HWND h : elements) SendMessage(h, WM_SETFONT, (WPARAM)hFont, TRUE);

    SetWindowPos(g_hChatDlg, nullptr, 0, 0, 410, 410, SWP_NOMOVE | SWP_NOZORDER);
    ShowWindow(g_hChatDlg, SW_SHOW);
    SetForegroundWindow(g_hChatDlg);
    SetFocus(hInput);
}

static std::string BuildStatusText() {
    if (g_Ollama.busy) return "Status: thinking...";
    static const char* moveNames[] = { "idle", "walking forward", "walking backward", "turning left", "turning right" };
    return "Status: " + std::string(moveNames[(int)g_CurrentMove]) + " | " + g_LastActionStatus;
}

void renderFrame() {
    LARGE_INTEGER now; QueryPerformanceCounter(&now);
    float dt = (g_LastTick.QuadPart == 0) ? 0.016f :
        (float)(now.QuadPart - g_LastTick.QuadPart) / (float)g_PerfFreq.QuadPart;
    g_LastTick = now;
    if (dt > 0.1f) dt = 0.1f;

    UpdateAgent(dt);

    // Camera smoothing
    g_CamAngleX = Lerp(g_CamAngleX, g_InAngleX, g_SettingCamSmooth, dt);
    g_CamAngleY = Lerp(g_CamAngleY, g_InAngleY, g_SettingCamSmooth, dt);
    g_CamDistance = Lerp(g_CamDistance, g_InDistance, g_SettingCamSmooth, dt);
    g_TargetX = Lerp(g_TargetX, g_InTargetX, g_SettingCamSmooth, dt);
    g_TargetY = Lerp(g_TargetY, g_InTargetY, g_SettingCamSmooth, dt);
    g_TargetZ = Lerp(g_TargetZ, g_InTargetZ, g_SettingCamSmooth, dt);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    float camX = g_TargetX + g_CamDistance * cosf(g_CamAngleY) * sinf(g_CamAngleX);
    float camY = g_TargetY + g_CamDistance * sinf(g_CamAngleY);
    float camZ = g_TargetZ + g_CamDistance * cosf(g_CamAngleY) * cosf(g_CamAngleX);

    gluLookAt(camX, camY, camZ, g_TargetX, g_TargetY, g_TargetZ, 0.0f, 1.0f, 0.0f);

    DrawVRMModel();
    SwapBuffers(hDC);
}

static bool IsInGLArea(int y) { return y < (winHeight - TOOLBAR_HEIGHT); }

static std::string BrowseForModel(HWND owner) {
    char buf[MAX_PATH] = {}; OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn); ofn.hwndOwner = owner;
    ofn.lpstrFilter = "VRM / glTF Files\0*.vrm;*.glb;*.gltf\0All Files\0*.*\0";
    ofn.lpstrFile = buf; ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetOpenFileNameA(&ofn)) return std::string(buf);
    return {};
}

// Two toolbar rows now: model path/import/settings/chat on top, agent
// status/Stop/Interrupt underneath.
static void LayoutToolbar(int w, int h) {
    if (!hEditPath) return;
    int topY = h - TOOLBAR_HEIGHT;
    int pad = 8;
    int row1Y = topY + pad;
    int row2Y = topY + pad + 32;

    int setW = 80, impW = 110, btnW = 90, chatW = 70;
    int editW = w - btnW - impW - setW - chatW - pad * 6;
    if (editW < 10) editW = 10;

    int x = pad;
    SetWindowPos(hEditPath, nullptr, x, row1Y + 2, editW, 24, SWP_NOZORDER); x += editW + pad;
    SetWindowPos(hBrowseBtn, nullptr, x, row1Y, btnW, 28, SWP_NOZORDER); x += btnW + pad;
    SetWindowPos(hImportBtn, nullptr, x, row1Y, impW, 28, SWP_NOZORDER); x += impW + pad;
    SetWindowPos(GetDlgItem(hWnd, IDC_SETTINGS_BTN), nullptr, x, row1Y, setW, 28, SWP_NOZORDER); x += setW + pad;
    SetWindowPos(hChatBtn, nullptr, x, row1Y, chatW, 28, SWP_NOZORDER);

    int stopW = 70, interruptW = 90;
    int statusW = w - stopW - interruptW - pad * 3;
    if (statusW < 10) statusW = 10;
    int x2 = pad;
    SetWindowPos(hStatusText, nullptr, x2, row2Y + 6, statusW, 20, SWP_NOZORDER); x2 += statusW + pad;
    SetWindowPos(hStopBtn, nullptr, x2, row2Y, stopW, 24, SWP_NOZORDER); x2 += stopW + pad;
    SetWindowPos(hInterruptBtn, nullptr, x2, row2Y, interruptW, 24, SWP_NOZORDER);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_CREATE: {
        HFONT hFont = (HFONT)GetStockObject(DEFAULT_GUI_FONT);
        HINSTANCE hInst = ((LPCREATESTRUCT)lParam)->hInstance;
        hEditPath = CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", "avatar.vrm", WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL, 0, 0, 0, 0, hwnd, (HMENU)IDC_PATH_EDIT, hInst, nullptr);
        SendMessage(hEditPath, WM_SETFONT, (WPARAM)hFont, TRUE);
        hBrowseBtn = CreateWindowA("BUTTON", "Browse...", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 0, 0, 0, 0, hwnd, (HMENU)IDC_BROWSE_BTN, hInst, nullptr);
        SendMessage(hBrowseBtn, WM_SETFONT, (WPARAM)hFont, TRUE);
        hImportBtn = CreateWindowA("BUTTON", "Import VRM", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 0, 0, 0, 0, hwnd, (HMENU)IDC_IMPORT_BTN, hInst, nullptr);
        SendMessage(hImportBtn, WM_SETFONT, (WPARAM)hFont, TRUE);
        HWND hSetBtn = CreateWindowA("BUTTON", "Settings", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 0, 0, 0, 0, hwnd, (HMENU)IDC_SETTINGS_BTN, hInst, nullptr);
        SendMessage(hSetBtn, WM_SETFONT, (WPARAM)hFont, TRUE);
        hChatBtn = CreateWindowA("BUTTON", "Chat", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 0, 0, 0, 0, hwnd, (HMENU)IDC_CHAT_BTN, hInst, nullptr);
        SendMessage(hChatBtn, WM_SETFONT, (WPARAM)hFont, TRUE);
        hStatusText = CreateWindowA("STATIC", "Status: idle", WS_CHILD | WS_VISIBLE, 0, 0, 0, 0, hwnd, (HMENU)IDC_STATUS_TEXT, hInst, nullptr);
        SendMessage(hStatusText, WM_SETFONT, (WPARAM)hFont, TRUE);
        hStopBtn = CreateWindowA("BUTTON", "Stop", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 0, 0, 0, 0, hwnd, (HMENU)IDC_STOP_BTN, hInst, nullptr);
        SendMessage(hStopBtn, WM_SETFONT, (WPARAM)hFont, TRUE);
        hInterruptBtn = CreateWindowA("BUTTON", "Interrupt", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 0, 0, 0, 0, hwnd, (HMENU)IDC_INTERRUPT_BTN, hInst, nullptr);
        SendMessage(hInterruptBtn, WM_SETFONT, (WPARAM)hFont, TRUE);
        EnableWindow(hInterruptBtn, FALSE);

        // Drives both the status text and the chat transcript - see
        // WM_TIMER below for why this is how the UI finds out about work
        // happening on the Ollama worker thread.
        SetTimer(hwnd, IDT_CHAT_TIMER, 250, nullptr);
        return 0;
    }
    case WM_SIZE: {
        int w = LOWORD(lParam), h = HIWORD(lParam);
        LayoutToolbar(w, h); resizeGL(w, h); return 0;
    }
    case WM_TIMER: {
        if (wParam == IDT_CHAT_TIMER) {
            SetWindowTextA(hStatusText, BuildStatusText().c_str());
            EnableWindow(hInterruptBtn, g_Ollama.busy ? TRUE : FALSE);

            if (g_hChatDlg) {
                std::vector<ChatMessage> snapshot;
                { std::lock_guard<std::mutex> lock(g_Ollama.historyMutex); snapshot = g_Ollama.history; }
                if (snapshot.size() > g_ChatRenderedHistoryCount) {
                    for (size_t i = g_ChatRenderedHistoryCount; i < snapshot.size(); ++i) {
                        std::string line = (snapshot[i].role == "user" ? "You: " : "rinAI: ") +
                            StripActionTagForDisplay(snapshot[i].content) + "\r\n";
                        AppendToTranscript(line);
                    }
                    g_ChatRenderedHistoryCount = snapshot.size();
                }

                if (g_Ollama.models.size() != g_ChatRenderedModelsCount) {
                    HWND hCombo = GetDlgItem(g_hChatDlg, IDC_CHAT_MODEL_COMBO);
                    if (hCombo) {
                        SendMessageA(hCombo, CB_RESETCONTENT, 0, 0);
                        for (auto& m : g_Ollama.models) SendMessageA(hCombo, CB_ADDSTRING, 0, (LPARAM)m.c_str());
                        if (!g_Ollama.models.empty()) SendMessage(hCombo, CB_SETCURSEL, 0, 0);
                    }
                    g_ChatRenderedModelsCount = g_Ollama.models.size();
                }
            }
        }
        return 0;
    }
    case WM_COMMAND: {
        int id = LOWORD(wParam);
        if (id == IDC_BROWSE_BTN) {
            std::string path = BrowseForModel(hwnd);
            if (!path.empty()) SetWindowTextA(hEditPath, path.c_str());
            SetFocus(hwnd);
        }
        else if (id == IDC_IMPORT_BTN) {
            char buf[MAX_PATH] = {}; GetWindowTextA(hEditPath, buf, MAX_PATH);
            if (buf[0] != '\0') LoadVRMModel(std::string(buf));
            SetFocus(hwnd);
        }
        else if (id == IDC_SETTINGS_BTN) {
            // Previously this fell through to an unconditional
            // SetFocus(hwnd) below, which ran even after opening the
            // dialog - stealing focus back to the main window in the same
            // instant Settings appeared. Just open it and stop here.
            ShowSettingsMenu(hwnd);
        }
        else if (id == IDC_CHAT_BTN) {
            ShowChatWindow(hwnd);
        }
        else if (id == IDC_STOP_BTN) {
            g_MoveQueue.clear();
            g_CurrentMove = MoveAction::None;
            for (auto& kv : g_SkeletalRig) { kv.second.tgtX = 0.0f; kv.second.tgtY = 0.0f; kv.second.tgtZ = 0.0f; }
            g_LastActionStatus = "stopped";
            SetFocus(hwnd);
        }
        else if (id == IDC_INTERRUPT_BTN) {
            g_Ollama.interruptRequested = true;
            SetFocus(hwnd);
        }
        return 0;
    }
    case WM_MOUSEMOVE: {
        int mx = GET_X_LPARAM(lParam), my = GET_Y_LPARAM(lParam);
        if (g_IsDragging) {
            float modifier = g_SettingInvertCam ? -0.01f : 0.01f;
            g_InAngleX += (mx - g_LastMouseX) * modifier;
            g_InAngleY += (my - g_LastMouseY) * modifier;
            if (g_InAngleY > 1.5f) g_InAngleY = 1.5f;
            if (g_InAngleY < -1.5f) g_InAngleY = -1.5f;
            g_LastMouseX = mx; g_LastMouseY = my;
        }
        else if (g_IsPanning) {
            int dx = mx - g_LastMouseX, dy = my - g_LastMouseY;
            float ps = 0.002f * g_InDistance;
            g_InTargetY += dy * ps;
            g_InTargetX += dx * ps * cosf(g_InAngleX);
            g_InTargetZ -= dx * ps * sinf(g_InAngleX);
            g_LastMouseX = mx; g_LastMouseY = my;
        }
        return 0;
    }
    case WM_LBUTTONDOWN: {
        int mx = GET_X_LPARAM(lParam);
        int my = GET_Y_LPARAM(lParam);
        if (IsInGLArea(my)) {
            // Joint spheres are no longer pickable - left button is pure
            // camera orbit now.
            g_IsDragging = true;
            g_LastMouseX = mx; g_LastMouseY = my;
            SetCapture(hwnd); SetFocus(hwnd);
        }
        return 0;
    }
    case WM_RBUTTONDOWN: {
        int my = GET_Y_LPARAM(lParam);
        if (IsInGLArea(my)) { g_IsPanning = true; g_LastMouseX = GET_X_LPARAM(lParam); g_LastMouseY = my; SetCapture(hwnd); SetFocus(hwnd); }
        return 0;
    }
    case WM_LBUTTONUP:
        g_IsDragging = false;
        if (!g_IsPanning) ReleaseCapture();
        return 0;
    case WM_RBUTTONUP: g_IsPanning = false; if (!g_IsDragging) ReleaseCapture(); return 0;
    case WM_MOUSEWHEEL: {
        POINT pt = { GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam) }; ScreenToClient(hwnd, &pt);
        if (IsInGLArea(pt.y)) {
            short delta = GET_WHEEL_DELTA_WPARAM(wParam);
            g_InDistance -= (delta / 120.0f) * 0.4f * (g_InDistance / 2.5f);
            if (g_InDistance < 0.1f)  g_InDistance = 0.1f;
            if (g_InDistance > 20.0f) g_InDistance = 20.0f;
        }
        return 0;
    }
    case WM_ERASEBKGND: {
        RECT rc; GetClientRect(hwnd, &rc); rc.top = rc.bottom - TOOLBAR_HEIGHT;
        HBRUSH br = CreateSolidBrush(RGB(45, 45, 50)); FillRect((HDC)wParam, &rc, br); DeleteObject(br);
        return 1;
    }
    case WM_DESTROY:
        KillTimer(hwnd, IDT_CHAT_TIMER);
        if (g_GltfData) { cgltf_free(g_GltfData); g_GltfData = nullptr; }
        if (g_JointQuadric) { gluDeleteQuadric(g_JointQuadric); g_JointQuadric = nullptr; }
        wglMakeCurrent(nullptr, nullptr); wglDeleteContext(hRC); ReleaseDC(hwnd, hDC);
        PostQuitMessage(0); return 0;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    InitCommonControls();
    WNDCLASSW wc = {};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = L"VRMViewer";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassW(&wc);

    hWnd = CreateWindowExW(0, L"VRMViewer", L"rinAI ver2.0 - AI-Driven Posing",
        WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN,
        CW_USEDEFAULT, CW_USEDEFAULT, winWidth, winHeight,
        nullptr, nullptr, hInstance, nullptr);

    initOpenGL(hWnd);
    resizeGL(winWidth, winHeight);
    LoadVRMModel("avatar.vrm");
    QueryPerformanceFrequency(&g_PerfFreq);
    ShowWindow(hWnd, nCmdShow);
    UpdateWindow(hWnd);

    MSG msg = {};
    while (true) {
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) goto done;
            bool handled = (g_hSettingsDlg && IsDialogMessage(g_hSettingsDlg, &msg)) ||
                (g_hChatDlg && IsDialogMessage(g_hChatDlg, &msg));
            if (!handled) {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
        }
        renderFrame();
    }
done:
    return (int)msg.wParam;
}
