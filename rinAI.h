#pragma once
// rinAI.h
// Small support header for rinAI.cpp. Holds pure-logic/rendering helpers that
// don't need to live next to the Win32 plumbing: the checkered ground plane,
// the bounded room the character stands in, and a screenshot-to-base64
// helper used when sending a frame to a vision model.

#include <gl/gl.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <cstdint>

#include "stb_image_write.h"

inline void DrawGround() {
    const int   halfTiles = 10;      // 20x20 grid
    const float tileSize = 0.5f;

    glDisable(GL_TEXTURE_2D);
    if (glIsEnabled(GL_LIGHTING)) glDisable(GL_LIGHTING);

    glBegin(GL_QUADS);
    for (int x = -halfTiles; x < halfTiles; ++x) {
        for (int z = -halfTiles; z < halfTiles; ++z) {
            bool even = ((x + z) % 2 == 0);
            if (even) glColor3f(0.20f, 0.20f, 0.23f);
            else      glColor3f(0.16f, 0.16f, 0.19f);

            float x0 = x * tileSize, x1 = x0 + tileSize;
            float z0 = z * tileSize, z1 = z0 + tileSize;

            glNormal3f(0.0f, 1.0f, 0.0f);
            glVertex3f(x0, 0.0f, z0);
            glVertex3f(x1, 0.0f, z0);
            glVertex3f(x1, 0.0f, z1);
            glVertex3f(x0, 0.0f, z1);
        }
    }
    glEnd();
}

// --- Bounded room / environment -------------------------------------------
// A handful of fixed furniture props plus four walls, all in world space.
// This is deliberately dumb - no instancing, no meshes, just colored boxes -
// it only needs to exist, give the agent something to describe and bump
// into, and read as "a room" from a distance.
struct EnvProp {
    float x, z, halfW, halfD, height;
    float r, g, b;
    const char* label;
};

inline void SetupEnvironment(std::vector<EnvProp>& out) {
    out.clear();
    out.push_back({ 1.4f,  1.2f, 0.45f, 0.30f, 0.45f, 0.45f, 0.32f, 0.20f, "table" });
    out.push_back({ 1.4f,  0.3f, 0.22f, 0.22f, 0.50f, 0.30f, 0.22f, 0.16f, "chair" });
    out.push_back({ -1.6f, -1.3f, 0.30f, 0.60f, 1.30f, 0.28f, 0.18f, 0.10f, "bookshelf" });
    out.push_back({ -1.5f,  1.5f, 0.35f, 0.35f, 0.50f, 0.55f, 0.42f, 0.22f, "crate" });
}

inline void DrawBox(float cx, float cz, float halfW, float halfD, float height, float r, float g, float b) {
    float x0 = cx - halfW, x1 = cx + halfW;
    float z0 = cz - halfD, z1 = cz + halfD;

    glColor3f(r, g, b);
    glBegin(GL_QUADS);
    glNormal3f(0.0f, 1.0f, 0.0f);
    glVertex3f(x0, height, z0); glVertex3f(x1, height, z0); glVertex3f(x1, height, z1); glVertex3f(x0, height, z1);

    glNormal3f(0.0f, 0.0f, -1.0f);
    glVertex3f(x0, 0.0f, z0); glVertex3f(x1, 0.0f, z0); glVertex3f(x1, height, z0); glVertex3f(x0, height, z0);

    glNormal3f(0.0f, 0.0f, 1.0f);
    glVertex3f(x0, 0.0f, z1); glVertex3f(x0, height, z1); glVertex3f(x1, height, z1); glVertex3f(x1, 0.0f, z1);

    glNormal3f(-1.0f, 0.0f, 0.0f);
    glVertex3f(x0, 0.0f, z0); glVertex3f(x0, height, z0); glVertex3f(x0, height, z1); glVertex3f(x0, 0.0f, z1);

    glNormal3f(1.0f, 0.0f, 0.0f);
    glVertex3f(x1, 0.0f, z0); glVertex3f(x1, 0.0f, z1); glVertex3f(x1, height, z1); glVertex3f(x1, height, z0);
    glEnd();
}

inline void DrawEnvironment(const std::vector<EnvProp>& props, float halfSize) {
    const float wallHeight = 2.2f;
    glColor3f(0.24f, 0.22f, 0.26f);
    glBegin(GL_QUADS);
    glNormal3f(0.0f, 0.0f, 1.0f);
    glVertex3f(-halfSize, 0.0f, -halfSize); glVertex3f(halfSize, 0.0f, -halfSize);
    glVertex3f(halfSize, wallHeight, -halfSize); glVertex3f(-halfSize, wallHeight, -halfSize);

    glNormal3f(0.0f, 0.0f, -1.0f);
    glVertex3f(-halfSize, 0.0f, halfSize); glVertex3f(-halfSize, wallHeight, halfSize);
    glVertex3f(halfSize, wallHeight, halfSize); glVertex3f(halfSize, 0.0f, halfSize);

    glNormal3f(1.0f, 0.0f, 0.0f);
    glVertex3f(-halfSize, 0.0f, -halfSize); glVertex3f(-halfSize, wallHeight, -halfSize);
    glVertex3f(-halfSize, wallHeight, halfSize); glVertex3f(-halfSize, 0.0f, halfSize);

    glNormal3f(-1.0f, 0.0f, 0.0f);
    glVertex3f(halfSize, 0.0f, -halfSize); glVertex3f(halfSize, 0.0f, halfSize);
    glVertex3f(halfSize, wallHeight, halfSize); glVertex3f(halfSize, wallHeight, -halfSize);
    glEnd();

    for (const auto& p : props) {
        DrawBox(p.x, p.z, p.halfW, p.halfD, p.height, p.r, p.g, p.b);
    }
}

// Clamps (x, z) inside the room walls, then pushes it out of any prop's
// AABB it's overlapping (expanded by radius). Props are checked independently
// rather than resolved as a combined field, which is fine at the scale of a
// handful of boxes and one slow-moving character root.
inline void ResolveCollision(float& x, float& z, const std::vector<EnvProp>& props, float halfSize, float radius) {
    float lim = halfSize - radius;
    if (x > lim) x = lim; if (x < -lim) x = -lim;
    if (z > lim) z = lim; if (z < -lim) z = -lim;

    for (const auto& p : props) {
        float minX = p.x - p.halfW - radius, maxX = p.x + p.halfW + radius;
        float minZ = p.z - p.halfD - radius, maxZ = p.z + p.halfD + radius;
        if (x <= minX || x >= maxX || z <= minZ || z >= maxZ) continue;

        float pushLeft = x - minX, pushRight = maxX - x;
        float pushBack = z - minZ, pushFwd = maxZ - z;
        float minPush = std::min(std::min(pushLeft, pushRight), std::min(pushBack, pushFwd));
        if (minPush == pushLeft) x = minX;
        else if (minPush == pushRight) x = maxX;
        else if (minPush == pushBack) z = minZ;
        else z = maxZ;
    }
}

// --- Screenshot -> base64 PNG ----------------------------------------------
// Used only when the "send screenshot" chat option is on, to hand the
// current frame to a vision model as an Ollama "images" entry.
inline std::string Base64EncodeBytes(const unsigned char* data, size_t len) {
    static const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    size_t i = 0;
    for (; i + 2 < len; i += 3) {
        uint32_t n = (uint32_t(data[i]) << 16) | (uint32_t(data[i + 1]) << 8) | data[i + 2];
        out += tbl[(n >> 18) & 0x3F]; out += tbl[(n >> 12) & 0x3F];
        out += tbl[(n >> 6) & 0x3F];  out += tbl[n & 0x3F];
    }
    size_t rem = len - i;
    if (rem == 1) {
        uint32_t n = uint32_t(data[i]) << 16;
        out += tbl[(n >> 18) & 0x3F]; out += tbl[(n >> 12) & 0x3F]; out += "==";
    }
    else if (rem == 2) {
        uint32_t n = (uint32_t(data[i]) << 16) | (uint32_t(data[i + 1]) << 8);
        out += tbl[(n >> 18) & 0x3F]; out += tbl[(n >> 12) & 0x3F]; out += tbl[(n >> 6) & 0x3F]; out += '=';
    }
    return out;
}

inline std::string CaptureScreenshotBase64PNG() {
    GLint vp[4]; glGetIntegerv(GL_VIEWPORT, vp);
    int w = vp[2], h = vp[3];
    if (w <= 0 || h <= 0) return "";

    std::vector<unsigned char> raw(size_t(w) * h * 3);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, raw.data());

    // glReadPixels returns rows bottom-up; flip them so the PNG isn't upside down.
    std::vector<unsigned char> flipped(raw.size());
    size_t rowBytes = size_t(w) * 3;
    for (int y = 0; y < h; ++y) {
        memcpy(&flipped[y * rowBytes], &raw[(h - 1 - y) * rowBytes], rowBytes);
    }

    std::string png;
    stbi_write_png_to_func([](void* ctx, void* data, int size) {
        reinterpret_cast<std::string*>(ctx)->append(reinterpret_cast<const char*>(data), size);
        }, &png, w, h, 3, flipped.data(), (int)rowBytes);

    return Base64EncodeBytes(reinterpret_cast<const unsigned char*>(png.data()), png.size());
}
