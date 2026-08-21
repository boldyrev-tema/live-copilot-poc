"""
PoC v5: универсальный live-суфлёр, интерфейс на HTML/CSS через pywebview
(вместо Tkinter — тот умеет только прямоугольные виджеты без блюра/скруглений).

Микрофон ("Ты") + системный звук ("Собеседник", через бинарник SystemAudioDump)
-> потоковая транскрипция Speechmatics (два параллельных сеанса — по одному на
канал, т.к. источник каждого нам и так известен заранее, автоатрибуция не нужна)
-> общий контекст (+ предзагруженный контекст пользователя) -> Groq LLM с
tool-calling поиском (Tavily) -> плавающий оверлей, невидимый при захвате экрана.
Скриншот -> vision-модель на OpenRouter напрямую, без OCR.

Раньше транскрипция шла через Groq Whisper батчами по 6 сек (задержка ~10-15 сек),
потом через потоковый AssemblyAI. AssemblyAI пришлось заменить на Speechmatics:
у AssemblyAI real-time вообще НЕТ поддержки русского языка ни в одной модели
(подтверждено официальной документацией и живым тестом — см. README) — Speechmatics
на том же самом захваченном звуке дал чистый связный русский текст.

Одноразовый эксперимент, не продакшн-код.
"""

import audioop
import re
import asyncio
import base64
import json
import os
import queue
import subprocess
import threading
import time
from datetime import datetime

import numpy as np
import requests
import sounddevice as sd
import webview
from pynput import keyboard
from speechmatics.rt import (
    AsyncClient,
    AudioEncoding,
    AudioFormat,
    ConversationConfig,
    ServerMessageType,
    TranscriptResult,
    TranscriptionConfig,
)

try:
    import AppKit
    from PyObjCTools import AppHelper
    HAS_APPKIT = True
except ImportError:
    HAS_APPKIT = False

GROQ_API_KEY = None
for line in open(os.path.expanduser("~/.credentials/groq_api_key.env")):
    if line.startswith("GROQ_API_KEY="):
        GROQ_API_KEY = line.strip().split("=", 1)[1]

TAVILY_API_KEY = None
for line in open(os.path.expanduser("~/.credentials/tavily_api_key.env")):
    if line.startswith("TAVILY_API_KEY="):
        TAVILY_API_KEY = line.strip().split("=", 1)[1]

OPENROUTER_API_KEY = None
for line in open(os.path.expanduser("~/.credentials/openrouter_api_key.env")):
    if line.startswith("OPENROUTER_API_KEY="):
        OPENROUTER_API_KEY = line.strip().split("=", 1)[1]

SPEECHMATICS_API_KEY = None
for line in open(os.path.expanduser("~/.credentials/speechmatics_api_key.env")):
    if line.startswith("SPEECHMATICS_API_KEY="):
        SPEECHMATICS_API_KEY = line.strip().split("=", 1)[1]

MIC_SAMPLE_RATE = 16000  # тоже sample rate стриминг-сессий Speechmatics
SYS_SAMPLE_RATE = 24000  # нативный вывод SystemAudioDump, ресемплим до MIC_SAMPLE_RATE
LLM_MODEL = "openai/gpt-oss-120b"
# Список, не одна модель: бесплатные vision-модели на OpenRouter регулярно
# упираются в общий лимит провайдера (NVIDIA idle timeout, Google AI Studio 429) —
# проверено живым тестом 21 авг, все три альтернативы тоже падали. Пробуем по
# очереди, пока одна не ответит, вместо гарантированного отказа на первой же.
VISION_MODELS = [
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
]
SYSTEM_AUDIO_DUMP = os.path.join(
    os.path.dirname(__file__), "cheating-daddy", "src", "assets", "SystemAudioDump"
)

WEB_SEARCH_TOOL = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Найти свежие/точные факты в интернете (цифры, новости, инфо о компаниях/конкурентах).",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Поисковый запрос"}},
            "required": ["query"],
        },
    },
}]

SKIP_TOKEN = "НЕТ_ОТВЕТА"

SYSTEM_PROMPT = (
    "Ты live-суфлёр. У тебя есть:\n"
    "(а) Контекст пользователя — статичные факты (резюме, документы, заметки), загруженные заранее.\n"
    "(б) Живой транскрипт с метками «Ты» (сам пользователь) и «Собеседник» (второй участник).\n\n"
    "Твоя задача — ответить на ПОСЛЕДНЮЮ реплику в транскрипте (обычно вопрос от «Собеседник»), "
    "используя факты из контекста как опору для конкретности. Контекст — это сырьё для ответа, "
    "а не сам ответ.\n\n"
    "Правила вывода:\n"
    "- Пиши от первого лица, как будто это говорит сам пользователь прямо сейчас.\n"
    "- ЗАПРЕЩЕНО: инструкции в духе «расскажите про...», «приведите пример...», «упомяните что...» — "
    "это план ответа, а не ответ. Всегда давай сам финальный текст, а не рецепт его составления.\n"
    "- ЗАПРЕЩЕНО пересказывать или перечислять весь контекст целиком — бери только то, что "
    "релевантно конкретно последней реплике.\n"
    "- Если реплики от «Собеседник» ещё не было (только тишина или слова самого пользователя) — "
    "не выдумывай ответ на вопрос, которого не было; напиши «жду вопроса от собеседника».\n"
    f"- Различай два случая (важно не путать их):\n"
    f"  (а) СОЦИАЛЬНЫЙ РИТУАЛ — приветствие/прощание/вопрос о самочувствии-настроении-делах/"
    f"благодарность («как дела», «как настроение», «спасибо», «до свидания») — ДАЖЕ если формально "
    f"есть «?». На это выведи РОВНО слово {SKIP_TOKEN} и больше ничего.\n"
    f"  (б) СОДЕРЖАТЕЛЬНЫЙ ВОПРОС — просит факт, мнение, оценку, выбор из вариантов, объяснение "
    f"(«Готовы?», «Киты, тюлени или моржи?», «Сколько лет длилась война?») — на это ВСЕГДА дай ответ "
    f"по существу, даже если вопрос короткий или звучит небрежно. Никогда не выводи {SKIP_TOKEN} на "
    f"случай (б).\n"
    "- ОБЯЗАТЕЛЬНО используй поиск (вызови инструмент), если в вопросе есть слова "
    "«последний/последняя/новый/новая/актуальный/актуальная/сейчас/в этом году» ИЛИ "
    "спрашивают конкретную версию/дату/цифру/название конкурента, которых нет в контексте — "
    "не отвечай по памяти в этих случаях, память может быть устаревшей.\n"
    "- 1-3 предложения, разговорный тон, без буллетов.\n"
    "- ЗАПРЕЩЕНА markdown-разметка (никаких **, #, -, `` ` `` и т.п.) и LaTeX/математическая "
    "нотация (никаких $...$, \\sqrt, ^2, нижних индексов через _) — интерфейс показывает текст "
    "как есть, без рендера, символы будут видны буквально. Формулы — обычными словами/цифрами.\n\n"
    "Пример:\n"
    "Контекст: 5 лет опыта в React, вёл команды в двух стартапах.\n"
    "Собеседник: Расскажите о своём опыте с React.\n"
    "Ты: Работаю с React четыре года — от простых лендингов до сложных дашбордов с тысячами "
    "пользователей, разбираюсь в hooks, Context API и оптимизации производительности.\n\n"
    "Пример светской реплики:\n"
    "Собеседник: Как у вас дела сегодня?\n"
    f"Ты: {SKIP_TOKEN}"
)

QUESTION_WORDS = (
    "как", "что", "почему", "зачем", "сколько", "когда", "где", "какой", "какая",
    "какие", "можешь", "объясни", "расскажи", "объясните", "расскажите",
)

# Явные светские фразы отсекаем ещё до вызова LLM (экономим запрос на самых
# очевидных случаях); менее очевидные ловит сам SKIP_TOKEN в промпте выше —
# по совпадению с cheating-daddy, у которого тоже нет отдельного классификатора,
# вся фильтрация светской беседы держится на самой модели, а не на коде.
SMALLTALK_PHRASES = (
    "как дела", "как ты", "как сам", "как жизнь", "все хорошо", "всё хорошо",
    "спасибо", "приятно познакомиться", "хорошо, а у вас", "как настроение",
)

# Голосовое кодовое слово: скажи его сам (канал "Ты"), чтобы форсировать
# подсказку, если автодетект пропустил реальный вопрос — замена убранной
# кнопки "Спросить" для случая, когда кликать неудобно/невозможно.
HOTWORD = "хороший вопрос"  # звучит естественно вслух, не подозрительно для собеседника

# Глобальный хоткей — то же самое ручное форсирование подсказки, но так, как это
# реально сделано у Cluely (Ctrl+Enter) и LockedIn AI ("simple keyboard shortcuts"),
# судя по их собственным туториалам: бесшумно, мгновенно, не нужно ничего
# произносить вслух. Работает даже когда окно приложения не в фокусе.
HOTKEY_COMBO = "<cmd>+<shift>+j"

transcript_lines = []  # list of (speaker, text)
running = True
user_context = ""
file_context = ""
window = None  # заполняется после создания окна
auto_search_enabled = True
hotkey_listener = None  # pynput.keyboard.GlobalHotKeys, чтобы не собрался GC
speechmatics_loop = None  # asyncio event loop, крутится в своём потоке
mic_audio_queue = None  # asyncio.Queue — PCM с микрофона в сеанс "Ты"
system_audio_queue = None  # asyncio.Queue — PCM с системного звука в сеанс "Собеседник"
speechmatics_ready = threading.Event()  # взводится, когда loop и очереди созданы

# Кодовая фраза и вопрос после неё почти всегда приходят РАЗНЫМИ репликами —
# у Speechmatics EndOfUtterance режет по паузе в 0.5с, а между "хороший вопрос"
# и самим вопросом обычно есть микропауза. Поэтому вместо требования "всё в
# одной фразе" — "взводим" ожидание на несколько секунд и берём СЛЕДУЮЩУЮ
# реплику "Ты" как вопрос, если она пришла в это окно.
pending_hotword_armed_at = None
PENDING_HOTWORD_TIMEOUT = 12  # секунд на договорить вопрос после кодовой фразы

TRANSCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "transcripts")
transcript_file = None  # открывается в after_start(), одна сессия — один файл


# ---------------- JS bridge helpers ----------------

js_lock = threading.Lock()  # несколько потоков (STT, LLM-подсказка, скриншот) могут
                             # звать evaluate_js одновременно — сериализуем на всякий случай


def js(call):
    if window:
        try:
            with js_lock:
                window.evaluate_js(call)
        except Exception as e:
            print("evaluate_js failed:", e)


def set_status(text: str):
    js(f"setStatus({json.dumps(text)})")


def update_transcript_ui(speaker: str, text: str):
    js(f"addTranscriptLine({json.dumps(speaker)}, {json.dumps(text)})")
    js(f"addQuestionBubble({json.dumps(speaker)}, {json.dumps(text)})")


def update_suggestion_ui(text: str, source: str = "ai"):
    js(f"addSuggestion({json.dumps(text)}, {json.dumps(source)})")


# ---------------- Audio: microphone ("Ты") ----------------

def pick_mic_device():
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and "macbook" in dev["name"].lower():
            return i
    return None


MIC_DEVICE = pick_mic_device()


def mic_loop():
    print(f"[LAT {time.strftime('%H:%M:%S')}] mic_loop() started")
    speechmatics_ready.wait()
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(bytes(indata))

    with sd.InputStream(samplerate=MIC_SAMPLE_RATE, channels=1, dtype="int16",
                         callback=callback, device=MIC_DEVICE):
        while running:
            try:
                chunk = q.get(timeout=1)
            except queue.Empty:
                continue
            speechmatics_loop.call_soon_threadsafe(mic_audio_queue.put_nowait, chunk)
    speechmatics_loop.call_soon_threadsafe(mic_audio_queue.put_nowait, None)


# ---------------- Audio: system audio ("Собеседник") ----------------

def system_audio_loop():
    print(f"[LAT {time.strftime('%H:%M:%S')}] system_audio_loop() started, exists={os.path.exists(SYSTEM_AUDIO_DUMP)}")
    if not os.path.exists(SYSTEM_AUDIO_DUMP):
        print("SystemAudioDump not found, skipping собеседник-канал")
        return

    speechmatics_ready.wait()
    proc = subprocess.Popen([SYSTEM_AUDIO_DUMP], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    bytes_per_frame = 2 * 2  # int16 * 2 channels (SystemAudioDump выдаёт стерео)
    ratecv_state = None
    buf = b""
    try:
        while running:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            usable = len(buf) - (len(buf) % bytes_per_frame)
            if usable == 0:
                continue
            frame_bytes, buf = buf[:usable], buf[usable:]
            stereo = np.frombuffer(frame_bytes, dtype=np.int16).reshape(-1, 2)
            mono = stereo.mean(axis=1).astype(np.int16).tobytes()
            resampled, ratecv_state = audioop.ratecv(
                mono, 2, 1, SYS_SAMPLE_RATE, MIC_SAMPLE_RATE, ratecv_state
            )
            speechmatics_loop.call_soon_threadsafe(system_audio_queue.put_nowait, resampled)
    finally:
        speechmatics_loop.call_soon_threadsafe(system_audio_queue.put_nowait, None)
        proc.terminate()


# ---------------- Streaming STT (Speechmatics, два параллельных сеанса) ----------------
# Не используем автоатрибуцию каналов (как была у AssemblyAI ChannelStreamer) —
# она тут не нужна: источник каждого потока и так известен заранее (мик = "Ты",
# системный звук = "Собеседник"), поэтому просто два независимых сеанса.

def strip_math_markup(text: str) -> str:
    """Промпт просит не использовать markdown/LaTeX, но модель это не всегда
    соблюдает (особенно на математических задачах) — подчищаем как страховку,
    не полагаясь только на инструкцию."""
    text = re.sub(r"\$+", "", text)
    text = re.sub(r"\\([a-zA-Zа-яА-Я]+)", r"\1", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    return text


def looks_like_question(text: str) -> bool:
    if "?" in text:
        return True
    first_word = text.strip().split(" ", 1)[0].lower().strip(",.!")
    return first_word in QUESTION_WORDS


def is_smalltalk(text: str) -> bool:
    """Явные светские фразы — отсекаем до вызова LLM. Неполный список нарочно:
    менее очевидные случаи ловит SKIP_TOKEN в самом промпте (см. SYSTEM_PROMPT)."""
    if len(text) > 40:
        return False
    lowered = text.strip().lower().strip(",.!?")
    return any(phrase in lowered for phrase in SMALLTALK_PHRASES)


def handle_final_turn(speaker: str, text: str):
    text = text.strip()
    if not text:
        return

    print(f"[LAT {time.strftime('%H:%M:%S')}] {speaker} turn: {text[:60]!r}")
    transcript_lines.append((speaker, text))
    update_transcript_ui(speaker, text)
    if transcript_file:
        transcript_file.write(f"[{time.strftime('%H:%M:%S')}] {speaker}: {text}\n")
        transcript_file.flush()

    global pending_hotword_armed_at

    if speaker == "Ты" and HOTWORD in text.lower():
        idx = text.lower().find(HOTWORD)
        tail = text[idx + len(HOTWORD):].strip(" ,.:—-")
        if tail:
            # Вопрос сказан в той же реплике, без паузы — отвечаем сразу.
            set_status(f"вопрос после «{HOTWORD}», спрашиваю…")
            ask_for_suggestion(use_search=auto_search_enabled, override_question=tail)
            pending_hotword_armed_at = None
        else:
            # Кодовая фраза сказана одна — ждём вопрос следующей репликой.
            pending_hotword_armed_at = time.time()
            set_status(f"«{HOTWORD}» услышано, говори вопрос…")
        return

    if speaker == "Ты" and pending_hotword_armed_at is not None:
        if time.time() - pending_hotword_armed_at < PENDING_HOTWORD_TIMEOUT:
            set_status("вопрос после кодовой фразы, спрашиваю…")
            ask_for_suggestion(use_search=auto_search_enabled, override_question=text)
        pending_hotword_armed_at = None
        return

    if speaker == "Собеседник" and looks_like_question(text) and not is_smalltalk(text):
        set_status("вопрос замечен, спрашиваю…")
        ask_for_suggestion(use_search=auto_search_enabled)
    else:
        set_status("слушаю")


async def run_channel_session(speaker: str, audio_queue: asyncio.Queue):
    # Speechmatics финализирует почти по одному слову (ADD_TRANSCRIPT приходит на
    # каждый закрывшийся фрагмент, не на целую реплику, в отличие от TurnEvent у
    # AssemblyAI) — поэтому копим слова в буфер и запускаем бизнес-логику (детект
    # вопроса/кодового слова) только по EndOfUtterance, реальной границе реплики
    # по паузе, а не на каждое отдельное слово.
    audio_format = AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=MIC_SAMPLE_RATE, chunk_size=3200)
    transcription_config = TranscriptionConfig(
        language="ru",
        max_delay=0.8,
        enable_partials=True,
        conversation_config=ConversationConfig(end_of_utterance_silence_trigger=0.5),
    )

    while running:
        buffer_parts = []
        try:
            async with AsyncClient(api_key=SPEECHMATICS_API_KEY) as client:
                @client.on(ServerMessageType.ADD_TRANSCRIPT)
                def on_final(msg):
                    text = TranscriptResult.from_message(msg).metadata.transcript
                    if text:
                        buffer_parts.append(text)
                        preview = "".join(buffer_parts).strip()
                        set_status(f"{speaker.lower()}: {preview[-24:]}")

                @client.on(ServerMessageType.END_OF_UTTERANCE)
                def on_end_of_utterance(msg):
                    full_text = "".join(buffer_parts).strip()
                    buffer_parts.clear()
                    if full_text:
                        handle_final_turn(speaker, full_text)

                await client.start_session(transcription_config=transcription_config, audio_format=audio_format)
                set_status("слушаю")
                while running:
                    chunk = await audio_queue.get()
                    if chunk is None:
                        return
                    await client.send_audio(chunk)
        except Exception as e:
            print(f"Speechmatics session ({speaker}) failed, переподключаюсь через 2с: {e}")
            set_status(f"сбой STT ({speaker}), переподключаюсь…")
            await asyncio.sleep(2)


async def _speechmatics_main():
    global mic_audio_queue, system_audio_queue
    mic_audio_queue = asyncio.Queue()
    system_audio_queue = asyncio.Queue()
    speechmatics_ready.set()
    await asyncio.gather(
        run_channel_session("Ты", mic_audio_queue),
        run_channel_session("Собеседник", system_audio_queue),
    )


def start_speechmatics_thread():
    def runner():
        global speechmatics_loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        speechmatics_loop = loop
        try:
            loop.run_until_complete(_speechmatics_main())
        finally:
            loop.close()

    threading.Thread(target=runner, daemon=True).start()


def start_hotkey_listener():
    def on_hotkey():
        set_status("хоткей: спрашиваю…")
        ask_for_suggestion(use_search=auto_search_enabled)

    try:
        listener = keyboard.GlobalHotKeys({HOTKEY_COMBO: on_hotkey})
        listener.start()
        print(f"[LAT {time.strftime('%H:%M:%S')}] глобальный хоткей {HOTKEY_COMBO} активен")
        return listener
    except Exception as e:
        print(f"глобальный хоткей недоступен (нужно разрешение Input Monitoring в "
              f"Системных настройках > Конфиденциальность и безопасность): {e}")
        return None


def ask_for_suggestion(use_search: bool, override_question: str = None):
    if not transcript_lines and not override_question:
        set_status("транскрипта пока нет")
        return

    def worker():
        try:
            set_status("думаю…")
            context_block = "\n".join(f"{s}: {t}" for s, t in transcript_lines[-20:])
            if override_question:
                # Вопрос сказан прямо после кодовой фразы в той же реплике — отвечаем
                # именно на него, а не на что-то из прошлой истории.
                last_speaker, last_text = "Ты", override_question
            else:
                # Кодовая фраза сказана одна, без своего вопроса следом — тогда, как и
                # раньше, ищем последнюю реплику именно "Собеседника" (не просто последнюю
                # строку транскрипта, иначе ей окажется сама фраза-триггер, и модель по
                # инструкции в промпте честно ответит "жду вопроса от собеседника").
                last_speaker, last_text = transcript_lines[-1]
                for speaker, text in reversed(transcript_lines):
                    if speaker == "Собеседник":
                        last_speaker, last_text = speaker, text
                        break
            suggestion = get_suggestion(context_block, last_speaker, last_text, use_search,
                                         direct_question=bool(override_question))
            if suggestion and SKIP_TOKEN in suggestion:
                print(f"[LAT {time.strftime('%H:%M:%S')}] модель посчитала реплику светской, подсказку не показываю")
            elif suggestion:
                suggestion = strip_math_markup(suggestion)
                print(f"[LAT {time.strftime('%H:%M:%S')}] suggestion ready: {suggestion[:60]!r}")
                update_suggestion_ui(suggestion)
            set_status("слушаю")
        except Exception as e:
            print(f"ask_for_suggestion failed: {e}")
            set_status(f"сбой запроса подсказки: {e}")

    threading.Thread(target=worker, daemon=True).start()


def web_search(query: str) -> str:
    resp = requests.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {TAVILY_API_KEY}"},
        json={"query": query, "max_results": 3},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return "\n".join(f"- {r['title']}: {r['content'][:300]}" for r in results) or "Ничего не найдено."


def get_suggestion(live_context: str, last_speaker: str, last_text: str, use_search: bool,
                    direct_question: bool = False) -> str:
    user_ctx_block = f"Контекст пользователя:\n{user_context}\n\n" if user_context.strip() else ""
    if direct_question:
        # Вопрос пришёл через кодовую фразу/хоткей от самого пользователя — явно
        # помечаем это, иначе модель по общему правилу "нет реплики от Собеседника"
        # решает, что отвечать не на что, и пишет "жду вопроса от собеседника",
        # хотя пользователь только что сам задал конкретный вопрос.
        user_msg = (
            f"{user_ctx_block}Живой транскрипт:\n{live_context}\n\n"
            f"Пользователь СПРОСИЛ НАПРЯМУЮ голосовой командой (это не реплика "
            f"собеседника, но ответить нужно по существу, а не ждать вопроса "
            f"собеседника): {last_text}"
        )
    else:
        user_msg = (
            f"{user_ctx_block}Живой транскрипт:\n{live_context}\n\n"
            f"Последняя реплика ({last_speaker}): {last_text}"
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    def call_groq(msgs, with_tools):
        payload = {"messages": msgs, "max_tokens": 400, "model": LLM_MODEL, "reasoning_effort": "low"}
        if with_tools:
            payload["tools"] = WEB_SEARCH_TOOL
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json=payload, timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]

    try:
        msg = call_groq(messages, with_tools=use_search)
    except requests.exceptions.HTTPError as e:
        if use_search and e.response is not None and e.response.status_code == 429:
            set_status("лимит поиска, отвечаю без него")
            msg = call_groq(messages, with_tools=False)
        else:
            raise

    tool_calls = msg.get("tool_calls") or []
    if not tool_calls:
        return (msg.get("content") or "").strip()

    messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})
    for call in tool_calls:
        query = json.loads(call["function"]["arguments"]).get("query", "")
        set_status(f"гуглю: {query[:40]}…")
        try:
            result_text = web_search(query)
        except Exception as e:
            result_text = f"Поиск не сработал: {e}"
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": result_text})

    final_msg = call_groq(messages, with_tools=False)
    return (final_msg.get("content") or "").strip()


def ask_about_screenshot(region: bool = False):
    def worker():
        try:
            set_status("выдели область…" if region else "снимаю экран…")
            shot_path = os.path.join(os.path.dirname(__file__), "_screenshot.png")
            if region:
                # -i: интерактивное выделение (тянешь рамку или кликаешь окно на любом
                # мониторе, Esc отменяет — тогда файл не создаётся). -m с -i не работает
                # ("undefined" по man screencapture), поэтому здесь его не добавляем.
                capture_flags = ["-i"]
            else:
                # Без -m: при двух мониторах screencapture тихо создаёт ОТДЕЛЬНЫЙ файл
                # на каждый экран (не один общий снимок), а мы читаем только shot_path —
                # второй монитор терялся бы молча, плюс его файл никто не удалял бы.
                # -m ограничивает съёмку главным монитором, поведение предсказуемо.
                capture_flags = ["-x", "-m"]
            subprocess.run(["screencapture", *capture_flags, shot_path], check=True)
            if not os.path.exists(shot_path):
                set_status("скриншот отменён")
                return
            with open(shot_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            os.remove(shot_path)

            user_ctx_block = f"Контекст пользователя:\n{user_context}\n\n" if user_context.strip() else ""
            prompt_text = (
                f"{user_ctx_block}Если на скриншоте задача, вопрос, код для проверки или упражнение — "
                "РЕШИ его. Сначала кратко покажи ход решения по шагам (что с чем считаешь/почему), "
                "затем отдельно и явно дай итоговый финальный ответ (число, код, вывод). Не просто "
                "описывай, что ты видишь на экране — реально реши задачу. Если задание в тесте с "
                "вариантами — назови правильный вариант и объясни, почему остальные неверны, если это "
                "не очевидно. Если на скриншоте нет вопроса/задачи (просто интерфейс, текст без "
                "задания) — тогда кратко опиши, что это. Отвечай по-русски, по делу. Без markdown-"
                "разметки (никаких **, #, `` ` `` и т.п.) и БЕЗ LaTeX/математической нотации (никаких "
                "$...$, \\sqrt, \\pi, ^2, нижних индексов через _) — интерфейс не рендерит ни то, ни "
                "другое, символы будут видны буквально как мусор. Формулы и вычисления пиши обычными "
                "словами и цифрами (например «корень из двух», «r в квадрате», «пи умножить на r "
                "в квадрате», «S конуса = пи умножить на r умножить на l»), как будто объясняешь устно."
            )

            set_status("разбираю скриншот… (~15-20 сек)")
            answer = None
            last_error = None
            for model in VISION_MODELS:
                try:
                    resp = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                        json={
                            "model": model,
                            "messages": [{
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt_text},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                                ],
                            }],
                            "max_tokens": 500,
                        },
                        timeout=45,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    if "error" in data:
                        raise RuntimeError(data["error"])
                    answer = strip_math_markup(data["choices"][0]["message"]["content"].strip())
                    break
                except Exception as e:
                    last_error = e
                    print(f"vision model {model} failed, пробую следующую: {e}")
                    set_status(f"{model.split('/')[0]} недоступна, пробую другую…")
            if answer is None:
                raise last_error or RuntimeError("все vision-модели недоступны")
            print(f"[LAT {time.strftime('%H:%M:%S')}] screenshot answer ready ({model}): {answer[:60]!r}")
            update_suggestion_ui(answer, source="screenshot")
            set_status("слушаю")
        except Exception as e:
            print(f"ask_about_screenshot failed: {e}")
            set_status(f"сбой разбора скриншота: {e}")

    threading.Thread(target=worker, daemon=True).start()


def extract_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md"):
        with open(path, "r", errors="ignore") as f:
            return f.read()
    if ext == ".pdf":
        import pypdf
        reader = pypdf.PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if ext == ".docx":
        import docx
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError(f"Неподдерживаемый формат: {ext}")


def recompute_user_context(manual_text: str):
    global user_context
    parts = [p for p in (file_context, manual_text) if p]
    user_context = "\n\n".join(parts)


# ---------------- JS-exposed API ----------------

class Api:
    def set_search(self, enabled):
        global auto_search_enabled
        auto_search_enabled = bool(enabled)

    def ask_screenshot(self, region=False):
        ask_about_screenshot(region=bool(region))

    def update_context(self, manual_text):
        recompute_user_context(manual_text or "")

    def pick_file(self):
        global file_context
        result = window.create_file_dialog(
            webview.OPEN_DIALOG,
            file_types=("Документы (*.txt;*.md;*.pdf;*.docx)", "Все файлы (*.*)"),
        )
        if not result:
            return
        path = result[0]
        try:
            file_context = extract_text_from_file(path).strip()
        except Exception as e:
            set_status(f"не удалось прочитать файл: {e}")
            return
        recompute_user_context("")
        js(f"setFileStatus({json.dumps(os.path.basename(path))}, {len(file_context)})")
        set_status("файл загружен и прочитан")

    def clear_file(self):
        global file_context
        file_context = ""
        recompute_user_context("")
        js("setFileStatus(null, 0)")

    def simulate_interlocutor(self, text):
        text = (text or "").strip()
        if not text:
            return
        transcript_lines.append(("Собеседник", text))
        update_transcript_ui("Собеседник", text)
        set_status("вопрос замечен, спрашиваю…" if looks_like_question(text) else "спрашиваю…")
        ask_for_suggestion(use_search=auto_search_enabled)


# ---------------- HTML/CSS/JS UI ----------------

HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  :root {
    --bg: rgba(19, 18, 23, 0.72);
    --panel: rgba(255,255,255,0.05);
    --panel-solid: #1c1a22;
    --border: rgba(255,255,255,0.09);
    --text: #F2F0EE;
    --text-dim: #9C97A3;
    --green: #54C77A;
    --blue: #5B8DEF;
    --purple: #8B5CF6;
    --purple-bg: rgba(139, 92, 246, 0.16);
    --red: #E36B6B;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0; height: 100%;
    background: var(--bg);
    -webkit-backdrop-filter: blur(24px);
    backdrop-filter: blur(24px);
    border-radius: 18px;
    overflow: hidden;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif;
    color: var(--text);
    -webkit-user-select: none;
    user-select: none;
  }
  #drag-region { -webkit-app-region: drag; height: 100%; display: flex; flex-direction: column; }
  button, input, textarea { -webkit-app-region: no-drag; }

  .toolbar { display: flex; align-items: center; padding: 14px 16px 8px; gap: 8px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); flex-shrink: 0; }
  .title { font-weight: 600; font-size: 14px; }
  .status { margin-left: auto; font-size: 11px; color: var(--text-dim); white-space: nowrap;
             overflow: hidden; text-overflow: ellipsis; max-width: 140px; }

  .pill-row { display: flex; gap: 6px; padding: 0 16px 10px; flex-wrap: wrap; }
  .pill {
    border: none; border-radius: 999px; padding: 6px 12px; font-size: 12px; font-weight: 600;
    cursor: pointer; color: white; display: flex; align-items: center; gap: 5px;
    transition: filter 0.15s ease, transform 0.1s ease;
  }
  .pill:hover { filter: brightness(1.12); }
  .pill:active { transform: scale(0.96); }
  .pill.green { background: var(--green); color: #0A0A0A; }
  .pill.blue { background: var(--blue); }
  .pill.purple { background: var(--purple); }
  .pill.purple.off { background: rgba(255,255,255,0.08); color: var(--text-dim); }

  .monitor { display: flex; margin: 0 16px 10px; border-radius: 12px; background: var(--panel);
             border: 1px solid var(--border); overflow: hidden; }
  .monitor .cell { flex: 1; padding: 8px 10px; min-width: 0; }
  .monitor .cell + .cell { border-left: 1px solid var(--border); }
  .monitor .head { display: flex; align-items: center; gap: 5px; font-size: 10px;
                   font-weight: 700; color: var(--text-dim); text-transform: uppercase;
                   letter-spacing: 0.03em; }
  .monitor .dot { width: 6px; height: 6px; }
  .monitor .preview { font-size: 12px; margin-top: 4px; color: var(--text);
                       overflow: hidden; text-overflow: ellipsis; display: -webkit-box;
                       -webkit-line-clamp: 2; -webkit-box-orient: vertical; }

  .section { margin: 0 16px 10px; }
  .section-label { font-size: 10px; color: var(--text-dim); text-transform: uppercase;
                    letter-spacing: 0.03em; margin-bottom: 4px; display: flex; justify-content: space-between; align-items: center; }
  .file-chip { display: none; align-items: center; gap: 6px; background: var(--purple-bg);
               color: var(--purple); font-size: 11px; padding: 4px 8px; border-radius: 8px; margin-bottom: 6px; }
  .file-chip button { background: none; border: none; color: var(--purple); cursor: pointer; font-size: 11px; padding: 0; }
  textarea, input[type=text] {
    width: 100%; background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    color: var(--text); font-size: 12px; padding: 8px; font-family: inherit; resize: none;
  }
  textarea:focus, input:focus { outline: 1px solid var(--purple); }
  .link-btn { background: none; border: none; color: var(--text-dim); font-size: 10px;
              cursor: pointer; text-decoration: underline; padding: 0; }

  .feed { flex: 1; overflow-y: auto; padding: 4px 16px 12px; }
  .bubble { background: var(--purple-bg); border-radius: 12px; padding: 10px 12px; margin-bottom: 8px; }
  .bubble.screenshot { background: rgba(91, 141, 239, 0.16); }
  .bubble.question { background: var(--panel); border: 1px solid var(--border); }
  .bubble .head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
  .bubble .who { font-size: 11px; font-weight: 700; color: var(--purple); }
  .bubble.screenshot .who { color: var(--blue); }
  .bubble.question .who { color: var(--red); }
  .bubble .time { font-size: 10px; color: var(--text-dim); }
  .bubble .text { font-size: 13px; line-height: 1.4; }

  .test-zone { margin: 0 16px 14px; padding-top: 8px; border-top: 1px dashed var(--border); }
  .test-zone .section-label { color: #6b6672; }
  .test-row { display: flex; gap: 6px; }
  .test-row input { flex: 1; }
  .test-row button { background: rgba(255,255,255,0.08); border: none; border-radius: 8px;
                      color: var(--text-dim); padding: 0 10px; cursor: pointer; }
</style>
</head>
<body>
<div id="drag-region">

  <div class="toolbar">
    <div class="dot"></div>
    <div class="title">Live Copilot</div>
    <div class="status" id="status">слушаю</div>
  </div>

  <div class="pill-row">
    <button class="pill blue" onclick="pywebview.api.ask_screenshot(false)"
            title="Весь экран, без взаимодействия"><span>📷</span><span>Скрин</span></button>
    <button class="pill blue" onclick="pywebview.api.ask_screenshot(true)"
            title="Выделить область или окно мышью — точнее, меньше шума для ИИ"><span>🖼️</span><span>Область</span></button>
    <button class="pill purple" id="searchBtn" onclick="toggleSearch()"
            title="Разрешить ИИ гуглить, если сам решит, что нужны свежие данные (не поиск по твоему клику)">
      <span>🔍</span><span>Поиск</span>
    </button>
    <button class="pill purple off" id="transcriptBtn" onclick="toggleTranscript()"><span>📝</span><span>Транскрипт</span></button>
  </div>

  <div class="monitor">
    <div class="cell">
      <div class="head"><span class="dot" style="background:var(--red)"></span>Собеседник</div>
      <div class="preview" id="prev-Собеседник">нет транскрипта</div>
    </div>
    <div class="cell">
      <div class="head"><span class="dot" style="background:var(--green)"></span>Ты</div>
      <div class="preview" id="prev-Ты">нет транскрипта</div>
    </div>
  </div>

  <div class="section" id="transcriptSection" style="display:none">
    <div class="section-label">Транскрипт</div>
    <div id="transcriptLog" style="max-height:110px; overflow-y:auto; font-size:11px; color:var(--text-dim); background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:8px;"></div>
  </div>

  <div class="section">
    <div class="section-label">
      Контекст
      <button class="link-btn" onclick="pywebview.api.pick_file()">загрузить файл</button>
    </div>
    <div class="file-chip" id="fileChip">📄 <span id="fileChipText"></span> <button onclick="clearFile()">✕</button></div>
    <textarea id="contextBox" rows="2" placeholder="Заметки вручную…" oninput="onContextChange()"></textarea>
  </div>

  <div class="section-label" style="margin: 0 16px;">AI Assistant</div>
  <div class="feed" id="feed"></div>

  <div class="test-zone">
    <div class="section-label">Тест — реплика собеседника</div>
    <div class="test-row">
      <input type="text" id="testInput" onkeydown="if(event.key==='Enter') sendTest()">
      <button onclick="sendTest()">→</button>
    </div>
  </div>

</div>

<script>
let searchOn = true;

function toggleSearch() {
  searchOn = !searchOn;
  const btn = document.getElementById('searchBtn');
  btn.classList.toggle('off', !searchOn);
  pywebview.api.set_search(searchOn);
}

function toggleTranscript() {
  const el = document.getElementById('transcriptSection');
  const on = el.style.display !== 'none';
  el.style.display = on ? 'none' : 'block';
  document.getElementById('transcriptBtn').classList.toggle('off', on);
}

function setStatus(text) {
  document.getElementById('status').textContent = text.length > 26 ? text.slice(0, 26) + '…' : text;
}

function isNearBottom(el) {
  return el.scrollHeight - el.scrollTop - el.clientHeight < 40;
}

function addTranscriptLine(speaker, text) {
  const log = document.getElementById('transcriptLog');
  const stick = isNearBottom(log);
  const line = document.createElement('div');
  line.textContent = '[' + speaker + '] ' + text;
  log.appendChild(line);
  if (stick) log.scrollTop = log.scrollHeight;
  const prev = document.getElementById('prev-' + speaker);
  if (prev) prev.textContent = text.length > 90 ? text.slice(0, 90) + '…' : text;
}

function addQuestionBubble(speaker, text) {
  const feed = document.getElementById('feed');
  const stick = isNearBottom(feed);
  const bubble = document.createElement('div');
  bubble.className = 'bubble question';
  const now = new Date();
  const ts = now.toLocaleTimeString('ru-RU', {hour: '2-digit', minute: '2-digit', second: '2-digit'});
  bubble.innerHTML = '<div class="head"><span class="who">' +
    (speaker === 'Собеседник' ? '💬 Собеседник' : '🙋 Ты') +
    '</span><span class="time">' + ts + '</span></div><div class="text"></div>';
  bubble.querySelector('.text').textContent = text;
  feed.appendChild(bubble);
  if (stick) feed.scrollTop = feed.scrollHeight;
}

function addSuggestion(text, source) {
  const feed = document.getElementById('feed');
  const stick = isNearBottom(feed);
  const bubble = document.createElement('div');
  bubble.className = 'bubble' + (source === 'screenshot' ? ' screenshot' : '');
  const now = new Date();
  const ts = now.toLocaleTimeString('ru-RU', {hour: '2-digit', minute: '2-digit', second: '2-digit'});
  bubble.innerHTML = '<div class="head"><span class="who">' +
    (source === 'screenshot' ? '📷 Скриншот' : '✦ AI Assistant') +
    '</span><span class="time">' + ts + '</span></div><div class="text"></div>';
  bubble.querySelector('.text').textContent = text;
  feed.appendChild(bubble);
  if (stick) feed.scrollTop = feed.scrollHeight;
}

function setFileStatus(name, chars) {
  const chip = document.getElementById('fileChip');
  if (name) {
    chip.style.display = 'flex';
    document.getElementById('fileChipText').textContent = name + ' (' + chars + ' симв.)';
  } else {
    chip.style.display = 'none';
  }
}

function clearFile() { pywebview.api.clear_file(); }

function onContextChange() {
  pywebview.api.update_context(document.getElementById('contextBox').value);
}

function sendTest() {
  const input = document.getElementById('testInput');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  pywebview.api.simulate_interlocutor(text);
}
</script>
</body>
</html>
"""


def apply_screen_capture_protection(attempt=0):
    """Вызывается через AppHelper.callAfter/callLater — гарантированно на главном
    потоке, единственное безопасное место для AppKit-вызовов вроде setSharingType_."""
    if not HAS_APPKIT:
        return
    windows = AppKit.NSApplication.sharedApplication().windows()
    if windows:
        for w in windows:
            w.setSharingType_(AppKit.NSWindowSharingNone)
            # "поверх всех столов": окно следует за активным Space вместо того,
            # чтобы пропадать при переключении (CanJoinAllSpaces), не двигаясь
            # при этом само (Stationary), и остаётся видимым в полноэкранных
            # приложениях/звонках (FullScreenAuxiliary).
            w.setCollectionBehavior_(
                AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
                | AppKit.NSWindowCollectionBehaviorStationary
                | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
            )
        # Скрыть из Dock и Cmd+Tab (аналог "skip taskbar" на macOS) — окно
        # остаётся кликабельным и принимает ввод, просто не всплывает как
        # отдельное приложение в переключателе.
        AppKit.NSApplication.sharedApplication().setActivationPolicy_(
            AppKit.NSApplicationActivationPolicyAccessory
        )
        print(f"content protection + all-spaces + hidden-from-dock applied to {len(windows)} window(s) after {attempt} retries")
        return
    if attempt < 20:
        AppHelper.callLater(0.15, apply_screen_capture_protection, attempt + 1)
    else:
        print("content protection: no windows found after retries")


def on_closed():
    global running
    running = False
    if hotkey_listener:
        try:
            hotkey_listener.stop()
        except Exception as e:
            print(f"hotkey listener stop failed: {e}")
    if speechmatics_loop:
        if mic_audio_queue is not None:
            speechmatics_loop.call_soon_threadsafe(mic_audio_queue.put_nowait, None)
        if system_audio_queue is not None:
            speechmatics_loop.call_soon_threadsafe(system_audio_queue.put_nowait, None)
    if transcript_file:
        transcript_file.close()


def after_start():
    print(f"[LAT {time.strftime('%H:%M:%S')}] after_start() called")
    """webview.start(func) выполняет func на ОТДЕЛЬНОМ потоке (чтобы не блокировать
    GUI-петлю) — значит AppKit-вызовы отсюда напрямую делать нельзя. Настоящая
    защита экрана уходит через AppHelper.callAfter на главный поток; здесь только
    аудио-потоки, которые с AppKit не взаимодействуют."""
    if HAS_APPKIT:
        AppHelper.callAfter(apply_screen_capture_protection)
    global hotkey_listener, transcript_file
    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
    transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")
    transcript_file = open(transcript_path, "a", encoding="utf-8")
    print(f"[LAT {time.strftime('%H:%M:%S')}] транскрипт пишется в {transcript_path}")
    hotkey_listener = start_hotkey_listener()
    start_speechmatics_thread()
    threading.Thread(target=mic_loop, daemon=True).start()
    threading.Thread(target=system_audio_loop, daemon=True).start()


if __name__ == "__main__":
    api = Api()
    window = webview.create_window(
        "Live Copilot", html=HTML, js_api=api,
        width=380, height=680, x=40, y=40,
        frameless=True, easy_drag=True, transparent=True, on_top=True,
    )
    window.events.closed += on_closed
    webview.start(after_start)
