#pragma once
// ollama_client.h
// Talks to a local Ollama server over WinINet - no libcurl, nothing to
// download, just the HTTP client that already ships with the Windows SDK.
// Everything here runs the request on a detached std::thread and hands the
// result back through atomics/mutexes the main loop polls on a timer; see
// the "chat window" code in rinAI.cpp for the other half of that handoff.

#include <windows.h>
#include <wininet.h>
#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <thread>
#include <cstring>
#include "json.hpp"

#pragma comment(lib, "wininet.lib")

struct ChatMessage {
    std::string role;
    std::string content;
};

namespace Http {

inline bool ParseUrl(const std::string& url, std::string& host, INTERNET_PORT& port, std::string& path) {
    char hostBuf[256] = {}, pathBuf[2048] = {};
    URL_COMPONENTSA uc = {};
    uc.dwStructSize = sizeof(uc);
    uc.lpszHostName = hostBuf; uc.dwHostNameLength = sizeof(hostBuf);
    uc.lpszUrlPath = pathBuf; uc.dwUrlPathLength = sizeof(pathBuf);
    if (!InternetCrackUrlA(url.c_str(), (DWORD)url.size(), 0, &uc)) return false;
    host = hostBuf;
    port = uc.nPort;
    path = pathBuf[0] ? pathBuf : "/";
    return true;
}

inline std::string ReadAll(HINTERNET hRequest) {
    std::string result;
    char buf[4096];
    DWORD bytesRead = 0;
    while (InternetReadFile(hRequest, buf, sizeof(buf), &bytesRead) && bytesRead > 0) {
        result.append(buf, bytesRead);
    }
    return result;
}

inline std::string HttpGet(const std::string& url) {
    std::string host, path; INTERNET_PORT port;
    if (!ParseUrl(url, host, port, path)) return "";

    std::string out;
    HINTERNET hSession = InternetOpenA("rinAI", INTERNET_OPEN_TYPE_DIRECT, nullptr, nullptr, 0);
    if (!hSession) return out;
    HINTERNET hConnect = InternetConnectA(hSession, host.c_str(), port, nullptr, nullptr, INTERNET_SERVICE_HTTP, 0, 0);
    if (hConnect) {
        HINTERNET hRequest = HttpOpenRequestA(hConnect, "GET", path.c_str(), nullptr, nullptr, nullptr, INTERNET_FLAG_RELOAD, 0);
        if (hRequest) {
            if (HttpSendRequestA(hRequest, nullptr, 0, nullptr, 0)) out = ReadAll(hRequest);
            InternetCloseHandle(hRequest);
        }
        InternetCloseHandle(hConnect);
    }
    InternetCloseHandle(hSession);
    return out;
}

inline std::string HttpPostJson(const std::string& url, const std::string& body) {
    std::string host, path; INTERNET_PORT port;
    if (!ParseUrl(url, host, port, path)) return "";

    std::string out;
    HINTERNET hSession = InternetOpenA("rinAI", INTERNET_OPEN_TYPE_DIRECT, nullptr, nullptr, 0);
    if (!hSession) return out;
    HINTERNET hConnect = InternetConnectA(hSession, host.c_str(), port, nullptr, nullptr, INTERNET_SERVICE_HTTP, 0, 0);
    if (hConnect) {
        static const char* headers = "Content-Type: application/json\r\n";
        HINTERNET hRequest = HttpOpenRequestA(hConnect, "POST", path.c_str(), nullptr, nullptr, nullptr, INTERNET_FLAG_RELOAD, 0);
        if (hRequest) {
            if (HttpSendRequestA(hRequest, headers, (DWORD)strlen(headers), (LPVOID)body.data(), (DWORD)body.size())) {
                out = ReadAll(hRequest);
            }
            InternetCloseHandle(hRequest);
        }
        InternetCloseHandle(hConnect);
    }
    InternetCloseHandle(hSession);
    return out;
}

} // namespace Http

class OllamaClient {
public:
    std::string host = "http://127.0.0.1:11434";
    std::string selectedModel;
    std::vector<std::string> models;
    std::atomic<bool> busy{ false };
    std::atomic<bool> modelsLoading{ false };
    std::atomic<bool> interruptRequested{ false };
    std::atomic<bool> hasPendingAction{ false };

    std::string availableJointsCSV = "(no joints loaded yet)";
    std::string spatialContext;
    std::string screenshotBase64PNG;

    std::mutex historyMutex;
    std::vector<ChatMessage> history;

    std::mutex actionMutex;
    std::string pendingActionRaw;

    std::string BuildSystemPrompt() const {
        return
            "You are rinAI, a small 3D virtual character living in a bounded room with a few pieces of "
            "furniture. You control your own body directly - there is no animation system, only your own "
            "joints. After your written reply, on its own line, include exactly one JSON command in this "
            "exact form:\n"
            "[AGENT: {\"move\": \"walk_forward\", \"pose\": {\"Head\": {\"y\": 0.3}}}]\n"
            "\"move\" is optional and must be exactly one of: walk_forward, walk_backward, turn_left, "
            "turn_right. Omit \"move\" entirely if you don't want to move this turn.\n"
            "\"pose\" is optional. Its keys must be chosen only from this exact list of joint names that "
            "exist on your current body: " + availableJointsCSV + ". For each joint you include, give any "
            "of \"x\", \"y\", \"z\" as a rotation in radians (roughly -1.2 to 1.2 is a natural range per "
            "axis) - omit axes you don't want to change. Joints you don't mention keep whatever pose "
            "they're already in. Always include exactly one [AGENT: {...}] block, even if both \"move\" "
            "and \"pose\" are empty or omitted.\n"
            "Each message you receive ends with a bracketed note like [Current state: ...] describing "
            "your position, facing direction, and what's nearby right now - use it to decide whether to "
            "move toward or away from something, but it isn't part of what the person actually said to you.";
    }

    void RefreshModels() {
        modelsLoading = true;
        std::thread([this]() {
            std::string resp = Http::HttpGet(host + "/api/tags");
            std::vector<std::string> found;
            try {
                auto j = nlohmann::json::parse(resp);
                if (j.contains("models")) {
                    for (auto& m : j["models"]) {
                        if (m.contains("name")) found.push_back(m["name"].get<std::string>());
                    }
                }
            }
            catch (...) {}
            models = found;
            if (selectedModel.empty() && !models.empty()) selectedModel = models[0];
            modelsLoading = false;
            }).detach();
    }

    void SendMessage(const std::string& userText) {
        if (busy) return;
        busy = true;
        // A new send always clears a prior interrupt - otherwise clicking
        // Interrupt once would silently drop every future message too,
        // since the in-flight check below never resets the flag itself.
        interruptRequested = false;
        std::thread([this, userText]() {
            if (interruptRequested) { busy = false; return; }

            std::string content = userText;
            if (!spatialContext.empty()) content += " " + spatialContext;

            nlohmann::json msgs = nlohmann::json::array();
            msgs.push_back({ {"role", "system"}, {"content", BuildSystemPrompt()} });
            {
                std::lock_guard<std::mutex> lock(historyMutex);
                for (auto& m : history) msgs.push_back({ {"role", m.role}, {"content", m.content} });
            }

            nlohmann::json userMsg = { {"role", "user"}, {"content", content} };
            if (!screenshotBase64PNG.empty()) userMsg["images"] = nlohmann::json::array({ screenshotBase64PNG });
            msgs.push_back(userMsg);

            nlohmann::json body;
            body["model"] = selectedModel.empty() ? "llama3.2" : selectedModel;
            body["stream"] = false;
            body["messages"] = msgs;

            std::string resp = Http::HttpPostJson(host + "/api/chat", body.dump());

            // WinINet has no mid-transfer cancel the way libcurl's progress
            // callback does, so Interrupt can't abort a request already in
            // flight - it can only stop the NEXT one and drop this result.
            if (interruptRequested) { busy = false; return; }

            std::string replyText;
            try {
                auto j = nlohmann::json::parse(resp);
                if (j.contains("message") && j["message"].contains("content")) {
                    replyText = j["message"]["content"].get<std::string>();
                }
                else {
                    replyText = "[AGENT: {}] (empty response from Ollama)";
                }
            }
            catch (...) {
                replyText = "[AGENT: {}] (could not parse Ollama's response - is the model name right?)";
            }

            {
                std::lock_guard<std::mutex> lock(historyMutex);
                history.push_back({ "user", userText });
                history.push_back({ "assistant", replyText });
            }
            {
                std::lock_guard<std::mutex> lock(actionMutex);
                pendingActionRaw = replyText;
            }
            hasPendingAction = true;
            busy = false;
            }).detach();
    }
};
