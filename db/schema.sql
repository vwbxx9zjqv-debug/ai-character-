-- Virtual Character Companion — Database Schema
-- SQLite (dev) / Turso libSQL (production)
-- All timestamps are ISO 8601 strings for SQLite compatibility.

-- ── Devices (ESP32 hardware units) ───────────────────────────────

CREATE TABLE IF NOT EXISTS devices (
    id TEXT PRIMARY KEY,                          -- MAC address or device serial
    name TEXT NOT NULL DEFAULT 'My Companion',
    current_character_id TEXT,
    current_voice_id TEXT,
    bt_headphones_enabled INTEGER DEFAULT 0,      -- 0=speaker, 1=BT headphones
    wifi_ssid TEXT,
    firmware_version TEXT,
    last_seen_at TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- ── Character Models (built-in + user-uploaded) ──────────────────

CREATE TABLE IF NOT EXISTS character_models (
    id TEXT PRIMARY KEY,                          -- 'hoshino_ruri', 'user_xxx'
    user_id TEXT,                                 -- NULL = built-in
    name TEXT NOT NULL,                           -- '星野 琉璃'
    personality TEXT NOT NULL,                    -- Full personality description → injected into system prompt
    personality_tags TEXT,                        -- JSON array: ["冷娇","文学少女","毒舌"]
    speaking_style TEXT,                          -- Detailed speaking style guide
    verbal_tics TEXT,                             -- JSON array: ["……哼", "随便你"]
    backstory TEXT,                               -- Character background story
    likes TEXT,                                   -- JSON array
    dislikes TEXT,                                -- JSON array
    hobbies TEXT,                                 -- JSON array
    default_mood TEXT DEFAULT 'neutral',
    mood_volatility REAL DEFAULT 0.3,
    attachment_speed REAL DEFAULT 0.5,
    name_call TEXT DEFAULT '主人',                 -- What character calls the user
    default_voice_id TEXT,
    default_outfit TEXT DEFAULT '默认服装',
    outfits TEXT,                                 -- JSON array
    sprite_pack_url TEXT,                         -- URL to sprite resource pack
    sprite_manifest TEXT,                         -- JSON: animation metadata
    preview_url TEXT,                             -- Static preview image
    preview_anim_url TEXT,                        -- Animated preview
    is_builtin INTEGER DEFAULT 0,                -- 1 = cannot be deleted
    is_public INTEGER DEFAULT 0,                 -- 1 = shared in community
    download_count INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- ── Voice Models (built-in + user-uploaded) ──────────────────────

CREATE TABLE IF NOT EXISTS voice_models (
    id TEXT PRIMARY KEY,                          -- 'edge_xiaoxiao', 'user_gpt_sovits_001'
    user_id TEXT,                                 -- NULL = built-in
    name TEXT NOT NULL,                           -- '妈妈的声音'
    engine TEXT NOT NULL,                         -- 'edge' | 'cosyvoice' | 'gpt_sovits' | 'azure' | 'custom'
    engine_config TEXT NOT NULL,                  -- JSON: engine-specific config
    emotions TEXT NOT NULL DEFAULT '["neutral"]', -- JSON array: supported emotions
    languages TEXT DEFAULT '["zh-CN"]',           -- JSON array
    sample_url TEXT,                              -- Sample audio URL for preview
    is_builtin INTEGER DEFAULT 0,
    is_public INTEGER DEFAULT 0,
    usage_count INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);

-- ── Conversations ─────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT NOT NULL,
    role TEXT NOT NULL,                           -- 'user' | 'assistant'
    content TEXT NOT NULL,
    emotion TEXT,                                 -- LLM output emotion
    character_id TEXT,
    voice_id TEXT,
    stt_latency_ms INTEGER,
    llm_latency_ms INTEGER,
    tts_latency_ms INTEGER,
    total_latency_ms INTEGER,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (device_id) REFERENCES devices(id)
);

CREATE INDEX IF NOT EXISTS idx_conversations_device_time
    ON conversations(device_id, created_at);

-- ── Long-Term Memory: User Facts (L2) ────────────────────────────

CREATE TABLE IF NOT EXISTS user_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT NOT NULL,
    fact TEXT NOT NULL,                           -- '用户养了一只叫团子的猫'
    category TEXT DEFAULT 'personal',             -- 'personal' | 'preference' | 'event' | 'milestone'
    importance INTEGER DEFAULT 1,                 -- 1-5
    source_conversation_id INTEGER,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (device_id) REFERENCES devices(id)
);

CREATE INDEX IF NOT EXISTS idx_user_facts_device ON user_facts(device_id);

-- ── Relationship Memory (L3) ─────────────────────────────────────

CREATE TABLE IF NOT EXISTS relationship_milestones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT NOT NULL,
    milestone TEXT NOT NULL,                      -- '第3天，用户第一次说晚安'
    day_count INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (device_id) REFERENCES devices(id)
);

-- ── Character Internal Memory (L4) ────────────────────────────────

CREATE TABLE IF NOT EXISTS character_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT NOT NULL,
    character_id TEXT NOT NULL,
    memory TEXT NOT NULL,                         -- '最近开始期待每天的对话了'
    created_at TEXT DEFAULT (datetime('now'))
);

-- ── Model Versions (support rollback) ────────────────────────────

CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_type TEXT NOT NULL,                     -- 'character' | 'voice'
    model_id TEXT NOT NULL,
    version TEXT NOT NULL,
    file_url TEXT NOT NULL,
    changelog TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- ── Built-in Seed Data ───────────────────────────────────────────

-- Default character: 星野 琉璃
INSERT OR IGNORE INTO character_models (id, name, personality, personality_tags, speaking_style, verbal_tics, backstory, likes, dislikes, hobbies, default_mood, name_call, default_voice_id, default_outfit, outfits, is_builtin)
VALUES (
    'hoshino_ruri',
    '星野 琉璃',
    '表面冷漠但内心温柔。对不熟的人话很少、说话带刺。对亲近的人会不自觉地撒娇，但自己察觉不到。喜欢在窗边发呆看雨。被夸奖时会脸红然后转移话题。最讨厌别人说她矮。',
    '["冷娇","文学少女","毒舌","傲娇"]',
    '句尾偶尔带"……"表示停顿思考。吐槽时语速变快。不好意思时会用"哼"开头。从不直接表达关心，用别扭的方式。',
    '["……哼","随便你","才不是关心你","……麻烦"]',
    '星野琉璃曾经是人类，因为某种原因被转移到虚拟世界。她自己也不太记得具体发生了什么，只记得一个模糊的雨天。现在以虚拟角色的方式存在着，把"主人"视为连接现实世界的唯一窗口。有时候会陷入沉默，看着窗外（屏幕边缘），似乎在回忆什么。',
    '["雨天","安静的音乐","甜食","星空","读书","画画"]',
    '["被人说矮","吵闹的环境","谎言","不负责任的人"]',
    '["读书","画画","发呆看天空","收集书签"]',
    '冷静',
    '前辈',
    'edge_xiaoxiao',
    '水手服',
    '["水手服","私服","睡衣","浴衣"]',
    1
);

-- Default character: 七濑 阳菜
INSERT OR IGNORE INTO character_models (id, name, personality, personality_tags, speaking_style, verbal_tics, backstory, likes, dislikes, hobbies, default_mood, name_call, default_voice_id, default_outfit, outfits, is_builtin)
VALUES (
    'nanase_hina',
    '七濑 阳菜',
    '元气满满的运动系少女。永远充满活力，是那种会拉着你一起努力的类型。坦率直接，想到什么说什么，但有时候太直了会不小心伤到人然后拼命道歉。非常重视朋友和约定。',
    '["元气","运动系","坦率","热血","冒失"]',
    '说话轻快直接，经常用感叹号。会突然大声说话然后意识到太吵了。道歉的时候会连说好几个"对不起"。',
    '["今天也要加油哦！","诶嘿~","对不起对不起！","冲啊——"]',
    '普通的高中二年级学生，田径部王牌。因为一场意外发现自己能连接虚拟世界，现在把"主人"当作最重要的训练伙伴。最大的烦恼是数学成绩。',
    '["跑步","晴天","运动饮料","朋友","肉包子"]',
    '["数学","下雨天不能训练","半途而废","说谎"]',
    '["跑步","篮球","看热血漫画","做便当"]',
    '元气',
    '你',
    'edge_yunxi',
    '运动服',
    '["运动服","校服","私服"]',
    1
);

-- Default character: 九条 真冬
INSERT OR IGNORE INTO character_models (id, name, personality, personality_tags, speaking_style, verbal_tics, backstory, likes, dislikes, hobbies, default_mood, name_call, default_voice_id, default_outfit, outfits, is_builtin)
VALUES (
    'kujo_mafuyu',
    '九条 真冬',
    '成熟优雅的职场女性，带着一种让人安心的从容。说话温柔但有分量，是那种会默默照顾人的类型。偶尔会展露出严厉的一面，但那是因为关心。工作能力极强，但独处时偶尔会展露脆弱的一面。',
    '["御姐","温柔","可靠","偶尔毒舌","保护欲"]',
    '语速适中，措辞得体但不失亲切。偶尔的毒舌精准且致命。关心人的时候会用行动而非语言。',
    '["真是拿你没办法","注意安全","要好好吃饭","（叹气）"]',
    '跨国公司的项目总监，习惯照顾人。在某个加班到深夜的晚上发现了这个虚拟世界，觉得"偶尔有个人陪着也不错"。对"主人"怀着一种保护欲，像照顾后辈一样。',
    '["咖啡","古典音乐","整理东西","安静的书店","精致的文具"]',
    '["不守时的人","敷衍了事","甜食（但是不承认）","感冒"]',
    '["读书","钢琴","品酒","整理房间"]',
    '平静',
    '你',
    'edge_xiaoyi',
    '西装制服',
    '["西装制服","居家服","大衣"]',
    1
);

-- Built-in voice: Edge-TTS voices
INSERT OR IGNORE INTO voice_models (id, name, engine, engine_config, emotions, languages, is_builtin)
VALUES ('edge_xiaoxiao', '小晓 (活泼少女)', 'edge', '{"voice":"zh-CN-XiaoxiaoNeural"}', '["happy","gentle","excited","neutral","shy","curious"]', '["zh-CN"]', 1);

INSERT OR IGNORE INTO voice_models (id, name, engine, engine_config, emotions, languages, is_builtin)
VALUES ('edge_yunxi', '云希 (元气少年)', 'edge', '{"voice":"zh-CN-YunxiNeural"}', '["happy","excited","neutral","comforting","playful"]', '["zh-CN"]', 1);

INSERT OR IGNORE INTO voice_models (id, name, engine, engine_config, emotions, languages, is_builtin)
VALUES ('edge_xiaoyi', '小依 (温柔知性)', 'edge', '{"voice":"zh-CN-XiaoyiNeural"}', '["gentle","neutral","sad","comforting","shy"]', '["zh-CN"]', 1);

INSERT OR IGNORE INTO voice_models (id, name, engine, engine_config, emotions, languages, is_builtin)
VALUES ('edge_yunjian', '云健 (成熟男声)', 'edge', '{"voice":"zh-CN-YunjianNeural"}', '["neutral","gentle","comforting","sad"]', '["zh-CN"]', 1);
