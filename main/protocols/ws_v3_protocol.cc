#include "ws_v3_protocol.h"
#include "board.h"
#include "system_info.h"
#include "settings.h"

#include <cstring>
#include <cJSON.h>
#include <esp_log.h>

#define TAG "WS_V3"

WsV3Protocol::WsV3Protocol() {
}

WsV3Protocol::~WsV3Protocol() {
    StopHeartbeat();
    CloseAudioChannel(false);
}

// ── Protocol Interface ──────────────────────────────────────────

bool WsV3Protocol::Start() { return true; }

bool WsV3Protocol::OpenAudioChannel() {
    if (server_url_.empty()) {
        ESP_LOGE(TAG, "Server URL not set");
        return false;
    }

    error_occurred_ = false;
    ESP_LOGI(TAG, "Connecting to backend: %s", server_url_.c_str());

    auto network = Board::GetInstance().GetNetwork();
    websocket_ = network->CreateWebSocket(1);
    if (websocket_ == nullptr) {
        ESP_LOGE(TAG, "Failed to create websocket");
        return false;
    }

    // Set headers
    websocket_->SetHeader("Device-Id", SystemInfo::GetMacAddress().c_str());
    websocket_->SetHeader("Client-Id", Board::GetInstance().GetUuid().c_str());
    websocket_->SetHeader("Protocol-Version", std::to_string(WS_V3_PROTOCOL_VERSION).c_str());

    // Handle text (JSON) messages
    websocket_->OnData([this](const char* data, size_t len, bool binary) {
        if (binary) {
            // Binary audio frame: 4-byte LE length prefix + PCM data
            size_t offset = 0;
            while (offset + 4 <= len) {
                uint32_t frame_len = (uint32_t)(uint8_t)data[offset] |
                                     ((uint32_t)(uint8_t)data[offset+1] << 8) |
                                     ((uint32_t)(uint8_t)data[offset+2] << 16) |
                                     ((uint32_t)(uint8_t)data[offset+3] << 24);
                offset += 4;

                if (frame_len == 0) {
                    // Terminator — audio stream complete
                    speech_in_progress_ = false;
                    break;
                }

                if (offset + frame_len <= len) {
                    if (on_incoming_audio_) {
                        on_incoming_audio_(std::make_unique<AudioStreamPacket>(AudioStreamPacket{
                            .sample_rate = server_sample_rate_,
                            .frame_duration = server_frame_duration_,
                            .timestamp = 0,
                            .payload = std::vector<uint8_t>(
                                (uint8_t*)(data + offset),
                                (uint8_t*)(data + offset) + frame_len)
                        }));
                    }
                    offset += frame_len;
                } else {
                    ESP_LOGE(TAG, "Incomplete audio frame: want %lu, have %zu",
                             frame_len, len - offset);
                    break;
                }
            }
        } else {
            // JSON message
            auto root = cJSON_ParseWithLength(data, len);
            if (root) {
                ParseJsonMessage(root);
                cJSON_Delete(root);
            } else {
                ESP_LOGE(TAG, "Bad JSON: %.*s", (int)len, data);
            }
        }
        last_incoming_time_ = std::chrono::steady_clock::now();
    });

    websocket_->OnDisconnected([this]() {
        ESP_LOGI(TAG, "WebSocket disconnected from backend");
        StopHeartbeat();
        if (callbacks_.on_disconnected) callbacks_.on_disconnected();
        if (on_audio_channel_closed_) on_audio_channel_closed_();
    });

    if (!websocket_->Connect(server_url_.c_str())) {
        ESP_LOGE(TAG, "Failed to connect to %s", server_url_.c_str());
        websocket_.reset();
        return false;
    }

    ESP_LOGI(TAG, "Connected to backend");
    StartHeartbeat();

    if (callbacks_.on_connected) callbacks_.on_connected();
    if (on_audio_channel_opened_) on_audio_channel_opened_();

    return true;
}

void WsV3Protocol::CloseAudioChannel(bool send_goodbye) {
    (void)send_goodbye;
    StopHeartbeat();
    websocket_.reset();
}

bool WsV3Protocol::IsAudioChannelOpened() const {
    return websocket_ != nullptr && websocket_->IsConnected() && !error_occurred_;
}

bool WsV3Protocol::SendAudio(std::unique_ptr<AudioStreamPacket> packet) {
    if (!IsAudioChannelOpened()) return false;
    auto frame = EncodeAudioFrame(packet->payload.data(), packet->payload.size());
    return websocket_->Send(frame.data(), frame.size(), true);
}

bool WsV3Protocol::SendText(const std::string& text) {
    if (!IsAudioChannelOpened()) return false;
    return websocket_->Send(text);
}

// ── V3 Methods ──────────────────────────────────────────────────

void WsV3Protocol::SetServerUrl(const std::string& url) { server_url_ = url; }

void WsV3Protocol::SetCallbacks(WsV3Callbacks callbacks) { callbacks_ = std::move(callbacks); }

void WsV3Protocol::SendHeartbeat(const std::string& state, int wifi_rssi) {
    if (!IsAudioChannelOpened()) return;
    current_state_ = state;

    cJSON* root = cJSON_CreateObject();
    cJSON_AddNumberToObject(root, "v", WS_V3_PROTOCOL_VERSION);
    cJSON_AddStringToObject(root, "type", WS_V3_TYPE_HEARTBEAT);
    cJSON_AddStringToObject(root, "state", state.c_str());
    cJSON_AddNumberToObject(root, "wifi_rssi", wifi_rssi);
    char* json_str = cJSON_PrintUnformatted(root);
    SendText(std::string(json_str));
    cJSON_free(json_str);
    cJSON_Delete(root);
}

void WsV3Protocol::SendSpeechSegment(int duration_ms, float vad_confidence,
                                      const std::string& audio_format) {
    if (!IsAudioChannelOpened()) return;

    cJSON* root = cJSON_CreateObject();
    cJSON_AddNumberToObject(root, "v", WS_V3_PROTOCOL_VERSION);
    cJSON_AddStringToObject(root, "type", WS_V3_TYPE_SPEECH_SEGMENT);
    cJSON_AddNumberToObject(root, "duration_ms", duration_ms);
    cJSON_AddNumberToObject(root, "vad_confidence", vad_confidence);
    cJSON_AddStringToObject(root, "audio_format", audio_format.c_str());
    char* json_str = cJSON_PrintUnformatted(root);
    SendText(std::string(json_str));
    cJSON_free(json_str);
    cJSON_Delete(root);
}

void WsV3Protocol::SendStartListening(ListeningMode mode) {
    if (!IsAudioChannelOpened()) return;

    cJSON* root = cJSON_CreateObject();
    cJSON_AddNumberToObject(root, "v", WS_V3_PROTOCOL_VERSION);
    cJSON_AddStringToObject(root, "type", WS_V3_TYPE_STATE);
    cJSON_AddStringToObject(root, "state", WS_V3_STATE_LISTENING);
    const char* mode_str = (mode == kListeningModeRealtime) ? "realtime" :
                           (mode == kListeningModeAutoStop) ? "auto" : "manual";
    cJSON_AddStringToObject(root, "mode", mode_str);
    char* json_str = cJSON_PrintUnformatted(root);
    SendText(std::string(json_str));
    cJSON_free(json_str);
    cJSON_Delete(root);
    ESP_LOGI(TAG, "SendStartListening mode=%s", mode_str);
}

void WsV3Protocol::SendStopListening() {
    if (!IsAudioChannelOpened()) return;

    cJSON* root = cJSON_CreateObject();
    cJSON_AddNumberToObject(root, "v", WS_V3_PROTOCOL_VERSION);
    cJSON_AddStringToObject(root, "type", WS_V3_TYPE_STATE);
    cJSON_AddStringToObject(root, "state", WS_V3_STATE_IDLE);
    char* json_str = cJSON_PrintUnformatted(root);
    SendText(std::string(json_str));
    cJSON_free(json_str);
    cJSON_Delete(root);
    ESP_LOGI(TAG, "SendStopListening");
}

void WsV3Protocol::SendWakeWordDetected(const std::string& wake_word) {
    // In our protocol, wake word triggers start of listening
    ESP_LOGI(TAG, "Wake word detected: %s", wake_word.c_str());
    SendStartListening(kListeningModeAutoStop);
}

void WsV3Protocol::StartHeartbeat(int interval_ms) {
    StopHeartbeat();
    esp_timer_create_args_t args = {};
    args.callback = HeartbeatTimerCallback;
    args.arg = this;
    args.dispatch_method = ESP_TIMER_TASK;
    args.name = "ws_v3_hb";
    esp_timer_create(&args, &heartbeat_timer_);
    esp_timer_start_periodic(heartbeat_timer_, interval_ms * 1000);
}

void WsV3Protocol::StopHeartbeat() {
    if (heartbeat_timer_) {
        esp_timer_stop(heartbeat_timer_);
        esp_timer_delete(heartbeat_timer_);
        heartbeat_timer_ = nullptr;
    }
}

void WsV3Protocol::HeartbeatTimerCallback(void* arg) {
    auto* self = static_cast<WsV3Protocol*>(arg);
    self->SendHeartbeat(self->current_state_);
}

// ── JSON Parser ─────────────────────────────────────────────────

void WsV3Protocol::ParseJsonMessage(const cJSON* root) {
    auto type_item = cJSON_GetObjectItem(root, "type");
    if (!cJSON_IsString(type_item)) return;

    std::string type(type_item->valuestring);

    if (type == WS_V3_TYPE_STATE) {
        auto state = cJSON_GetObjectItem(root, "state");
        auto emotion = cJSON_GetObjectItem(root, "emotion");
        auto text = cJSON_GetObjectItem(root, "text");
        std::string s = cJSON_IsString(state) ? state->valuestring : "";
        std::string e = cJSON_IsString(emotion) ? emotion->valuestring : "neutral";
        ESP_LOGI(TAG, "State: %s (%s)", s.c_str(), e.c_str());
        if (callbacks_.on_state) callbacks_.on_state(s, e);
        if (cJSON_IsString(text) && callbacks_.on_subtitle)
            callbacks_.on_subtitle(text->valuestring);

    } else if (type == WS_V3_TYPE_SPEECH) {
        auto text = cJSON_GetObjectItem(root, "text");
        auto emotion = cJSON_GetObjectItem(root, "emotion");
        auto chunks = cJSON_GetObjectItem(root, "audio_chunks");
        std::string t = cJSON_IsString(text) ? text->valuestring : "";
        std::string e = cJSON_IsString(emotion) ? emotion->valuestring : "neutral";
        int n = cJSON_IsNumber(chunks) ? chunks->valueint : 0;
        speech_in_progress_ = true;
        speech_chunks_remaining_ = n;
        if (callbacks_.on_speech_start) callbacks_.on_speech_start(t, e, n);

    } else if (type == WS_V3_TYPE_SUBTITLE) {
        auto text = cJSON_GetObjectItem(root, "text");
        if (cJSON_IsString(text) && callbacks_.on_subtitle)
            callbacks_.on_subtitle(text->valuestring);

    } else if (type == WS_V3_TYPE_CHARACTER_SYNC) {
        auto sprites = cJSON_GetObjectItem(root, "sprites_url");
        auto voice = cJSON_GetObjectItem(root, "voice_id");
        std::string s = cJSON_IsString(sprites) ? sprites->valuestring : "";
        std::string v = cJSON_IsString(voice) ? voice->valuestring : "";
        if (callbacks_.on_character_sync) callbacks_.on_character_sync(s, v);

    } else if (type == WS_V3_TYPE_SWITCH_MODEL) {
        auto mt = cJSON_GetObjectItem(root, "model_type");
        auto mi = cJSON_GetObjectItem(root, "model_id");
        auto du = cJSON_GetObjectItem(root, "download_url");
        std::string t = cJSON_IsString(mt) ? mt->valuestring : "";
        std::string i = cJSON_IsString(mi) ? mi->valuestring : "";
        std::string u = cJSON_IsString(du) ? du->valuestring : "";
        if (callbacks_.on_switch_model) callbacks_.on_switch_model(t, i, u);

    } else if (type == WS_V3_TYPE_INITIATIVE) {
        auto text = cJSON_GetObjectItem(root, "text");
        auto emotion = cJSON_GetObjectItem(root, "emotion");
        std::string t = cJSON_IsString(text) ? text->valuestring : "";
        std::string e = cJSON_IsString(emotion) ? emotion->valuestring : "neutral";
        if (callbacks_.on_initiative) callbacks_.on_initiative(t, e);

    } else if (type == WS_V3_TYPE_ERROR) {
        auto code = cJSON_GetObjectItem(root, "code");
        auto msg = cJSON_GetObjectItem(root, "message");
        std::string c = cJSON_IsString(code) ? code->valuestring : "";
        std::string m = cJSON_IsString(msg) ? msg->valuestring : "";
        ESP_LOGE(TAG, "Backend error: [%s] %s", c.c_str(), m.c_str());
        if (callbacks_.on_error) callbacks_.on_error(c, m);

    } else if (type == WS_V3_TYPE_PONG) {
        // Heartbeat acknowledged
    } else {
        ESP_LOGW(TAG, "Unknown type: %s", type.c_str());
    }
}

// ── Binary Encoding ────────────────────────────────────────────

std::vector<uint8_t> WsV3Protocol::EncodeAudioFrame(const uint8_t* data, size_t len) {
    std::vector<uint8_t> frame(4 + len);
    frame[0] = len & 0xFF;
    frame[1] = (len >> 8) & 0xFF;
    frame[2] = (len >> 16) & 0xFF;
    frame[3] = (len >> 24) & 0xFF;
    if (len > 0) memcpy(frame.data() + 4, data, len);
    return frame;
}
