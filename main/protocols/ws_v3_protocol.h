/**
 * WebSocket v3 Protocol — Communication layer matching backend ws_protocol.py
 *
 * Replaces xiaozhi's MCP JSON-RPC protocol with our simplified v3 protocol:
 *   - No hello handshake (just connect and heartbeat)
 *   - Binary audio: 4-byte LE length prefix (matching backend's encode_audio_frame)
 *   - JSON messages: heartbeat, speech_segment, state, speech, character_sync, etc.
 */

#ifndef WS_V3_PROTOCOL_H
#define WS_V3_PROTOCOL_H

#include "protocol.h"

#include <freertos/FreeRTOS.h>
#include <freertos/event_groups.h>
#include <esp_timer.h>

#include <string>
#include <memory>
#include <functional>

// Forward declare WebSocket from managed component
class WebSocket;

#define WS_V3_PROTOCOL_VERSION 3

// Message types (aligns with backend core/ws_protocol.py)
#define WS_V3_TYPE_HEARTBEAT        "heartbeat"
#define WS_V3_TYPE_SPEECH_SEGMENT   "speech_segment"
#define WS_V3_TYPE_STATE            "state"
#define WS_V3_TYPE_SPEECH           "speech"
#define WS_V3_TYPE_SUBTITLE         "subtitle"
#define WS_V3_TYPE_CHARACTER_SYNC   "character_sync"
#define WS_V3_TYPE_SWITCH_MODEL     "switch_model"
#define WS_V3_TYPE_INITIATIVE       "initiative"
#define WS_V3_TYPE_ERROR            "error"
#define WS_V3_TYPE_PING             "ping"
#define WS_V3_TYPE_PONG             "pong"

// Device states (aligns with backend DeviceState enum)
#define WS_V3_STATE_IDLE        "idle"
#define WS_V3_STATE_LISTENING   "listening"
#define WS_V3_STATE_THINKING    "thinking"
#define WS_V3_STATE_SPEAKING    "speaking"
#define WS_V3_STATE_ERROR       "error"

/**
 * Parsed WebSocket v3 message from backend
 */
struct WsV3Message {
    int version = 0;
    std::string type;

    // state message
    std::string state;
    std::string emotion;

    // speech / subtitle message
    std::string text;
    float emotion_intensity = 0.5f;
    std::string voice_name;
    std::string character_name;
    int audio_chunks = 0;

    // character_sync message
    std::string sprites_url;
    std::string voice_id;

    // switch_model message
    std::string model_type;
    std::string model_id;
    std::string model_name;
    std::string download_url;

    // error message
    std::string code;
    std::string message;
};

/**
 * Callbacks for handling v3 protocol messages
 */
struct WsV3Callbacks {
    // Backend sent a state change (drives animation)
    std::function<void(const std::string& state, const std::string& emotion)> on_state;

    // Backend sent TTS audio (JSON header received, binary PCM chunks follow)
    std::function<void(const std::string& text, const std::string& emotion, int audio_chunks)> on_speech_start;

    // Backend sent a subtitle
    std::function<void(const std::string& text)> on_subtitle;

    // Backend sent character sync (full character state)
    std::function<void(const std::string& sprites_url, const std::string& voice_id)> on_character_sync;

    // Backend sent switch model command
    std::function<void(const std::string& model_type, const std::string& model_id,
                       const std::string& download_url)> on_switch_model;

    // Backend sent initiative (character-initiated speech)
    std::function<void(const std::string& text, const std::string& emotion)> on_initiative;

    // Backend sent error
    std::function<void(const std::string& code, const std::string& message)> on_error;

    // Audio channel opened
    std::function<void()> on_connected;

    // Audio channel closed
    std::function<void()> on_disconnected;
};


class WsV3Protocol : public Protocol {
public:
    WsV3Protocol();
    ~WsV3Protocol() override;

    // Protocol interface
    bool Start() override;
    bool OpenAudioChannel() override;
    void CloseAudioChannel(bool send_goodbye = true) override;
    bool IsAudioChannelOpened() const override;
    bool SendAudio(std::unique_ptr<AudioStreamPacket> packet) override;
    bool SendText(const std::string& text) override;
    void SendStartListening(ListeningMode mode) override;
    void SendStopListening() override;
    void SendWakeWordDetected(const std::string& wake_word) override;

    // V3-specific: Send heartbeat
    void SendHeartbeat(const std::string& state, int wifi_rssi = 0);

    // V3-specific: Send speech segment (VAD detected voice)
    void SendSpeechSegment(int duration_ms, float vad_confidence, const std::string& audio_format = "opus");

    // V3-specific: Set server URL (format: ws://host:port/ws/chat/device_id)
    void SetServerUrl(const std::string& url);

    // V3-specific: Start heartbeat timer
    void StartHeartbeat(int interval_ms = 30000);

    // V3-specific: Stop heartbeat timer
    void StopHeartbeat();

    // V3-specific: Set callbacks
    void SetCallbacks(WsV3Callbacks callbacks);

private:
    static constexpr int HEARTBEAT_INTERVAL_MS = 30000;
    static constexpr int HEARTBEAT_TIMEOUT_MS = 90000;

    std::shared_ptr<WebSocket> websocket_;  // The actual websocket connection
    WsV3Callbacks callbacks_;
    esp_timer_handle_t heartbeat_timer_ = nullptr;
    std::string server_url_;
    std::string current_state_;

    // Incoming speech state (for multi-chunk assembly)
    bool speech_in_progress_ = false;
    int speech_chunks_remaining_ = 0;

    // Heartbeat timer callback
    static void HeartbeatTimerCallback(void* arg);

    // Parse incoming JSON message
    void ParseJsonMessage(const cJSON* root);

    // Encode binary audio frame (4-byte LE length prefix)
    static std::vector<uint8_t> EncodeAudioFrame(const uint8_t* data, size_t len);
};

#endif // WS_V3_PROTOCOL_H
