"""
Регрессионные тесты для live_copilot_poc.py. Гоняй после ЛЮБОЙ правки
live_copilot_poc.py (по аналогии с test_server.py у idea_bot / test_all.py
у ~/scripts/).

Запуск:
    venv/bin/python3 test_poc.py

Две части:
1. Бэкенд (Python-функции) — чанкинг, поиск по базе знаний, детект
   светской беседы, очистка markdown/LaTeX, и интеграционные тесты с
   РЕАЛЬНЫМИ вызовами Groq/OpenRouter API (нужны настоящие ключи в
   ~/.credentials/, тратят немного бесплатного лимита).
2. UI (HTML/JS) — через Playwright с настоящим Chromium: вытаскивает HTML
   прямо из модуля, подменяет pywebview.api на заглушку-логгер и проверяет
   JS-функции (кнопки, чипы, пузыри подсказок). Требует `pip install
   playwright` + `playwright install chromium` в venv — если не установлено,
   эта часть пропускается с понятным сообщением, бэкенд-тесты всё равно
   отработают.

НЕ проверяет: реальный микрофон/системный звук, macOS-разрешения,
невидимость окна в живом звонке — для этого нужен живой тест человеком,
см. README.
"""
import http.server
import json
import os
import shutil
import socketserver
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))
import live_copilot_poc as m

passed, failed = [], []


def check(name, cond, detail=""):
    if cond:
        passed.append(name)
        print(f"PASS  {name}")
    else:
        failed.append(name)
        print(f"FAIL  {name}  {detail}")


# ==================== Часть 1: бэкенд ====================
print("=== Бэкенд: юнит-тесты (без API) ===")

doc = "абв " * 300
chunks = m.chunk_text(doc, size=700, overlap=100)
check("chunk_text: несколько чанков на длинном тексте", len(chunks) >= 2, f"got {len(chunks)}")
check("chunk_text: пустая строка -> пусто", m.chunk_text("") == [])
check("chunk_text: короткий текст -> один чанк", len(m.chunk_text("привет мир")) == 1)

print()
print("=== База знаний: векторный поиск (грузит embedding-модель, ~1с из кэша) ===")

def add_kb(name, text):
    for chunk, vec in zip(m.chunk_text(text), m.embed_texts(m.chunk_text(text))):
        m.knowledge_base_chunks.append((name, chunk, vec))

m.knowledge_base_chunks.clear()
add_kb("прайс.txt", "Стоимость подписки высокая по сравнению с рынком, но окупается за счёт экономии времени.")
add_kb("faq.txt", "Погода в Москве переменчивая, часто идут дожди.")

rel_direct = m.retrieve_relevant_chunks("какая стоимость подписки")
check("retrieve: прямой вопрос находит прайс-чанк", any("прайс" in s for s, _ in rel_direct), rel_direct)

rel_paraphrase = m.retrieve_relevant_chunks("почему так дорого стоит ваш продукт")
check("retrieve: ПЕРЕФРАЗ без общих слов тоже находит прайс-чанк (векторный поиск, не keyword)",
      any("прайс" in s for s, _ in rel_paraphrase), rel_paraphrase)

irr = m.retrieve_relevant_chunks("расскажи про свою собаку пожалуйста")
check("retrieve: нерелевантный вопрос -> пусто", irr == [], irr)

kb_save = list(m.knowledge_base_chunks)
m.knowledge_base_chunks.clear()
check("retrieve: пустая база знаний -> пусто", m.retrieve_relevant_chunks("скидка") == [])
m.knowledge_base_chunks.extend(kb_save)

check("smalltalk: 'как дела?' -> True", m.is_smalltalk("как дела?"))
check("smalltalk: содержательный вопрос -> False", not m.is_smalltalk("сколько лет длилась столетняя война"))
check("question: '?' в тексте -> True", m.looks_like_question("это точно так?"))
check("question: начинается с вопросного слова -> True", m.looks_like_question("расскажите о себе"))
check("question: обычное утверждение -> False", not m.looks_like_question("я работал в компании X"))

check("math: убирает $...$", m.strip_math_markup("$x=5$") == "x=5")
check("math: убирает \\command", m.strip_math_markup("\\sqrt{5}") == "sqrt5")
check("math: ^ -> 'в степени'", m.strip_math_markup("8^2") == "8 в степени 2")
check("math: _ -> 'с индексом'", m.strip_math_markup("x_1") == "x с индексом 1")
check("math: убирает **bold**", m.strip_math_markup("**важно**") == "важно")

m.battlecards.clear()
m.battlecards.append(("дорого", "У нас гибкая система скидок при годовой оплате."))
check("battlecard: триггер найден по подстроке", m.match_battlecard("ваш продукт слишком дорого стоит") is not None)
check("battlecard: правильный ответ", "скидок" in m.match_battlecard("это дорого"))
check("battlecard: нет триггера -> None", m.match_battlecard("расскажите о себе") is None)
m.battlecards.clear()

print()
print("=== Бэкенд: handle_final_turn — battlecard перехватывает ДО обычного LLM-вопроса ===")

captured = {}
orig_ask = m.ask_for_suggestion
orig_update_ui = m.update_suggestion_ui
m.ask_for_suggestion = lambda *a, **kw: captured.setdefault("llm_called", True)
m.update_suggestion_ui = lambda text, source="ai": captured.setdefault("ui", (text, source))
m.transcript_lines.clear()
m.battlecards.clear()
m.battlecards.append(("дешевле", "У конкурентов нет нашей интеграции с CRM — это и есть разница в цене."))
m.handle_final_turn("Собеседник", "у конкурентов дешевле, почему у вас так")
check("battlecard: сработал вместо LLM (LLM не вызван)", "llm_called" not in captured, captured)
check("battlecard: ответ ушёл в UI с source=battlecard", captured.get("ui", (None, None))[1] == "battlecard", captured)
m.ask_for_suggestion = orig_ask
m.update_suggestion_ui = orig_update_ui
m.battlecards.clear()
m.transcript_lines.clear()

print()
print("=== Бэкенд: интеграционные тесты (реальные вызовы API) ===")

m.user_context = "Пять лет опыта в React."
m.knowledge_base_chunks.clear()

ans = m.get_suggestion(
    live_context="Собеседник: Расскажите о вашем опыте с React.",
    last_speaker="Собеседник", last_text="Расскажите о вашем опыте с React.",
)
check("get_suggestion: использует контекст (упомянут React)", "react" in ans.lower(), ans)
time.sleep(8)  # Groq free tier: 8000 токенов/мин — разносим реальные вызовы, чтобы не упереться

ans2 = m.get_suggestion(
    live_context="Ты: хороший вопрос сколько будет два плюс два",
    last_speaker="Ты", last_text="сколько будет два плюс два",
    direct_question=True,
)
check("get_suggestion: direct_question отвечает по существу", "четыре" in ans2.lower() or "4" in ans2, ans2)
time.sleep(8)

called = {}
orig_search = m.web_search
m.web_search = lambda q: (called.setdefault("q", q), orig_search(q))[1]
ans3 = m.get_suggestion(
    live_context="Собеседник: Какая сейчас последняя версия Python?",
    last_speaker="Собеседник", last_text="Какая сейчас последняя версия Python?",
)
check("get_suggestion: поиск реально вызван на маркере свежести (поиск теперь всегда доступен, "
      "без ручного тумблера)", "q" in called, called)
m.web_search = orig_search

captured_payload = {}
orig_post = m.requests.post
def spy_post(url, **kw):
    if "tavily" in url:
        captured_payload.update(kw.get("json", {}))
    return orig_post(url, **kw)
m.requests.post = spy_post
m.web_search("тестовый запрос")
m.requests.post = orig_post
check("web_search: запрашивает 6 результатов, не 3 (живой тест поймал: с 3 топ-результаты "
      "были устаревшим блогом, правильный источник был только 5-м)",
      captured_payload.get("max_results") == 6, captured_payload)
time.sleep(8)

ans4 = m.get_suggestion(
    live_context="Собеседник: Как у вас дела сегодня?",
    last_speaker="Собеседник", last_text="Как у вас дела сегодня?",
)
check("get_suggestion: светская реплика -> SKIP_TOKEN", m.SKIP_TOKEN in ans4, ans4)
time.sleep(8)

print()
print("=== Живой баг: 429 от Groq на tool-calling -> откат на 'поиска нет' крашил на 400 "
      "(модель галлюцинировала несуществующий инструмент, раз промпт требовал поиск безусловно) ===")

class FakeResp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
    def raise_for_status(self):
        if self.status_code >= 400:
            e = m.requests.exceptions.HTTPError(f"{self.status_code}")
            e.response = self
            raise e
    def json(self):
        return self._payload

orig_post = m.requests.post
def fallback_spy_post(url, json=None, **kw):
    if "groq" in url and json and json.get("tools"):
        return FakeResp(429, {})
    return orig_post(url, json=json, **kw)
m.requests.post = fallback_spy_post
m.user_context = ""
ans_fallback = m.get_suggestion(
    live_context="Собеседник: Какая сейчас последняя версия React?",
    last_speaker="Собеседник", last_text="Какая сейчас последняя версия React?",
)
m.requests.post = orig_post
check("get_suggestion: откат на search-disabled после 429 -> не падает, есть ответ",
      bool(ans_fallback), ans_fallback)
time.sleep(8)

print()
print("=== Живой баг: пустой контекст + личный вопрос -> модель выдумывала биографию ===")
ans_nobio = m.get_suggestion(
    live_context="Собеседник: Расскажите о своём опыте с Python.",
    last_speaker="Собеседник", last_text="Расскажите о своём опыте с Python.",
)
check("get_suggestion: пустой контекст, личный вопрос -> признаёт нехватку данных, не выдумывает",
      any(p in ans_nobio.lower() for p in
          ("нет данных", "нет информации", "недостаточно данных", "не хватает данных", "нет сведений")),
      ans_nobio)
time.sleep(8)

m.user_context = "Пять лет опыта в продажах SaaS. Не забыть упомянуть кейс с ростом выручки на 40%."
m.knowledge_base_chunks.clear()
add_kb("прайс.txt", "Продукт стоит 5000 рублей в месяц. Скидка 20% при годовой оплате.")
ans5 = m.get_suggestion(
    live_context="Собеседник: Ваш продукт слишком дорогой.",
    last_speaker="Собеседник", last_text="Ваш продукт слишком дорогой, у конкурентов дешевле.",
)
check("get_suggestion: подтянул скидку из базы знаний", "20" in ans5, ans5)
check("get_suggestion: подтянул кейс про 40% из контекста", "40" in ans5, ans5)
time.sleep(8)

m.transcript_lines.clear()
m.transcript_lines.append(("Собеседник", "Расскажите о своём опыте с Python."))
m.transcript_lines.append(("Ты", "Ну, я писал скрипты иногда."))
m.transcript_lines.append(("Собеседник", "Какой у вас опыт с базами данных?"))
m.transcript_lines.append(("Ты", "Работал пять лет с PostgreSQL, снизил время ответа на 30%."))
report_holder = {}
orig_update_ui = m.update_suggestion_ui
m.update_suggestion_ui = lambda text, source="ai": report_holder.update(text=text, source=source)
m.generate_session_report()
time.sleep(6)
m.update_suggestion_ui = orig_update_ui
check("report: сгенерирован и дошёл до UI-колбэка", "text" in report_holder, report_holder)
if "text" in report_holder:
    check("report: source == 'report'", report_holder["source"] == "report")
    check("report: упоминает PostgreSQL (сильная сторона)", "postgres" in report_holder["text"].lower())

print()
print("=== Отчёт: слишком короткий транскрипт НЕ должен идти в LLM (живой баг: модель выдумывала цитаты) ===")
m.transcript_lines.clear()
m.transcript_lines.append(("Собеседник", "Привет"))
short_holder = {}
m.update_suggestion_ui = lambda text, source="ai": short_holder.update(text=text)
m.generate_session_report()
time.sleep(2)
m.update_suggestion_ui = orig_update_ui
check("report: короткий транскрипт (мало реплик 'Ты') -> LLM не вызывается, отчёт не генерируется",
      "text" not in short_holder, short_holder)
m.transcript_lines.clear()

m.user_context = ""
m.knowledge_base_chunks.clear()
m.transcript_lines.clear()

print()
print("=== Кодовое слово: изменяемо в рантайме ===")
orig_hotword = m.HOTWORD
m.HOTWORD = "тестовое слово"
m.transcript_lines.clear()
called = {}
m.ask_for_suggestion = lambda *a, **kw: called.setdefault("called", (a, kw))
m.handle_final_turn("Ты", "тестовое слово какой сегодня день")
check("HOTWORD: новое слово реально ловится в handle_final_turn", "called" in called, called)
m.ask_for_suggestion = orig_ask
m.HOTWORD = orig_hotword
m.transcript_lines.clear()

print()
print("=== Авто-запуск meeting_copilot/run.py при закрытии окна ===")
check(
    "_build_auto_run_command: венв-python meeting_copilot + run.py",
    m._build_auto_run_command() == [m.MEETING_COPILOT_VENV_PYTHON, "run.py"],
    m._build_auto_run_command(),
)
check(
    "MEETING_COPILOT_DIR: указывает на сестринский проект",
    m.MEETING_COPILOT_DIR.endswith("meeting_copilot"),
    m.MEETING_COPILOT_DIR,
)

# _trigger_meeting_copilot_run() не должно падать, даже если сам venv не существует —
# подменяем путь на заведомо отсутствующий, и лог-файл на временный (чтобы не писать
# в реальный ~/Desktop/meeting_copilot/auto_run.log при прогоне тестов), и проверяем,
# что вызов не бросает исключение.
orig_venv_python = m.MEETING_COPILOT_VENV_PYTHON
orig_auto_run_log = m.MEETING_COPILOT_AUTO_RUN_LOG
trigger_test_dir = tempfile.mkdtemp()
m.MEETING_COPILOT_VENV_PYTHON = "/nonexistent/path/python3"
m.MEETING_COPILOT_AUTO_RUN_LOG = os.path.join(trigger_test_dir, "auto_run.log")
try:
    m._trigger_meeting_copilot_run()
    trigger_did_not_raise = True
except Exception as e:
    trigger_did_not_raise = False
    trigger_exception = e
check(
    "_trigger_meeting_copilot_run: не бросает исключение даже при отсутствующем venv",
    trigger_did_not_raise,
    None if trigger_did_not_raise else trigger_exception,
)
m.MEETING_COPILOT_VENV_PYTHON = orig_venv_python
m.MEETING_COPILOT_AUTO_RUN_LOG = orig_auto_run_log
shutil.rmtree(trigger_test_dir, ignore_errors=True)

print()
print("=== Чтение прошлых саммари из meeting_copilot/summaries ===")
check(
    "_format_meeting_label: дата и время из имени файла-транскрипта",
    m._format_meeting_label("2026-08-28_10-00-00") == "2026-08-28 10:00",
    m._format_meeting_label("2026-08-28_10-00-00"),
)

summaries_test_dir = tempfile.mkdtemp()
orig_summaries_dir = m.MEETING_COPILOT_SUMMARIES_DIR
m.MEETING_COPILOT_SUMMARIES_DIR = summaries_test_dir
try:
    check(
        "list_past_summaries: пустая папка -> пустой список",
        m.list_past_summaries() == [],
    )

    with open(os.path.join(summaries_test_dir, "2026-08-27_09-00-00.md"), "w") as f:
        f.write("# старое саммари")
    with open(os.path.join(summaries_test_dir, "2026-08-28_10-00-00.md"), "w") as f:
        f.write("# новое саммари")
    with open(os.path.join(summaries_test_dir, "not_a_summary.txt"), "w") as f:
        f.write("игнорируется — не .md")

    listing = m.list_past_summaries()
    check(
        "list_past_summaries: только .md-файлы, 2 штуки",
        len(listing) == 2,
        listing,
    )
    check(
        "list_past_summaries: новые сверху",
        listing[0]["filename"] == "2026-08-28_10-00-00.md",
        listing,
    )
    check(
        "list_past_summaries: label читаемый",
        listing[0]["label"] == "2026-08-28 10:00",
        listing,
    )

    # Файл с именем не в ожидаемом формате не должен ронять весь список —
    # тот же класс бага, что уже один раз ловили в meeting_copilot/run.py.
    # Отдельно от проверки сортировки выше: позиция такого файла в списке не
    # гарантируется (лексикографический порядок кривого имени непредсказуем),
    # важно только что список не падает и что-то разумное показывает.
    with open(os.path.join(summaries_test_dir, "bad-name.md"), "w") as f:
        f.write("# файл с именем не по формату транскрипта")
    listing_with_bad_name = m.list_past_summaries()
    check(
        "list_past_summaries: файл с кривым именем не роняет список (теперь 3 штуки)",
        len(listing_with_bad_name) == 3,
        listing_with_bad_name,
    )
    check(
        "list_past_summaries: для кривого имени label = само имя файла (не исключение)",
        any(
            item["filename"] == "bad-name.md" and item["label"] == "bad-name.md"
            for item in listing_with_bad_name
        ),
        listing_with_bad_name,
    )

    check(
        "read_summary: возвращает реальное содержимое файла",
        m.read_summary("2026-08-28_10-00-00.md") == "# новое саммари",
    )
    check(
        "read_summary: несуществующий файл -> понятная ошибка, не исключение",
        "не найден" in m.read_summary("нет_такого.md").lower(),
    )
    check(
        "read_summary: path traversal через ../ отклонён",
        "недопустимое" in m.read_summary("../../etc/passwd").lower(),
    )
    check(
        "read_summary: абсолютный путь отклонён",
        "недопустимое" in m.read_summary("/etc/passwd").lower(),
    )
finally:
    m.MEETING_COPILOT_SUMMARIES_DIR = orig_summaries_dir
    shutil.rmtree(summaries_test_dir, ignore_errors=True)

backend_passed, backend_failed = len(passed), len(failed)
print()
print(f"=== Бэкенд итого: {backend_passed} passed, {backend_failed} failed ===")


# ==================== Часть 2: UI через Playwright ====================
print()
print("=== UI (Playwright + Chromium) ===")
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Playwright не установлен — пропускаю UI-тесты.")
    print("Установить: venv/bin/pip install playwright && venv/bin/playwright install chromium")
else:
    PORT = 8934
    html = m.HTML
    mock_script = """
<script>
window.__calls = [];
window.pywebview = { api: new Proxy({}, {
  get(target, prop) {
    return function(...args) {
      window.__calls.push([prop, args]);
      return Promise.resolve();
    };
  }
})};
</script>
"""
    html = html.replace("<script>", mock_script + "<script>", 1)
    test_dir = os.path.join(os.path.dirname(__file__), "_ui_test_tmp")
    os.makedirs(test_dir, exist_ok=True)
    html_path = os.path.join(test_dir, "ui_test.html")
    with open(html_path, "w") as f:
        f.write(html)

    handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(*args, directory=test_dir, **kwargs)
    httpd = socketserver.TCPServer(("", PORT), handler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            console_errors = []
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" and "favicon" not in msg.text else None)
            page.goto(f"http://localhost:{PORT}/ui_test.html")

            results = page.evaluate("""() => {
  const results = [];
  function check(name, cond, detail) { results.push({name, pass: !!cond, detail: detail !== undefined ? String(detail) : ''}); }


  const hotwordBox = document.getElementById('hotwordBox');
  hotwordBox.value = 'моё слово';
  onHotwordChange();
  check('onHotwordChange: api.update_hotword вызван с новым словом', window.__calls.some(c => c[0]==='update_hotword' && c[1][0]==='моё слово'));

  const ctxBox = document.getElementById('contextBox');
  ctxBox.value = 'тестовый контекст';
  onContextChange();
  check('onContextChange: api.update_context вызван', window.__calls.some(c => c[0]==='update_context' && c[1][0]==='тестовый контекст'));

  const transcriptSection = document.getElementById('transcriptSection');
  toggleTranscript();
  check('toggleTranscript: секция стала видимой', transcriptSection.style.display === 'block', transcriptSection.style.display);

  addKbFile('прайс.txt', 3);
  const kbChip = document.getElementById('kbChip');
  check('addKbFile: чип видим', kbChip.style.display === 'flex', kbChip.style.display);
  check('addKbFile: текст чипа корректный', document.getElementById('kbChipText').textContent.includes('прайс.txt') && document.getElementById('kbChipText').textContent.includes('3'));
  addKbFile(null, 0);
  check('addKbFile(null): чип скрыт', kbChip.style.display === 'none');

  const feedBefore = document.getElementById('feed').children.length;
  addSuggestion('Сильные стороны: ...', 'report');
  const feed = document.getElementById('feed');
  const lastBubble = feed.lastElementChild;
  check('addSuggestion(report): добавился пузырь', feed.children.length === feedBefore + 1);
  check('addSuggestion(report): label содержит Отчёт', lastBubble.querySelector('.who').textContent.includes('Отчёт'));
  check('addSuggestion(report): класс screenshot применён', lastBubble.classList.contains('screenshot'));

  addSuggestion('Ответ: 472', 'screenshot');
  check('addSuggestion(screenshot): label корректный', document.getElementById('feed').lastElementChild.querySelector('.who').textContent.includes('Скриншот'));

  addSuggestion('обычная подсказка', 'ai');
  const lastBubble3 = document.getElementById('feed').lastElementChild;
  check('addSuggestion(ai): label AI Assistant', lastBubble3.querySelector('.who').textContent.includes('AI Assistant'));
  check('addSuggestion(ai): НЕТ класса screenshot', !lastBubble3.classList.contains('screenshot'));

  const testInput = document.getElementById('testInput');
  testInput.value = 'тестовый вопрос от собеседника';
  sendTest();
  check('sendTest: api.simulate_interlocutor вызван', window.__calls.some(c => c[0]==='simulate_interlocutor' && c[1][0]==='тестовый вопрос от собеседника'));
  check('sendTest: инпут очищен после отправки', testInput.value === '');

  setStatus('это очень длинный статус который точно длиннее двадцати шести символов');
  check('setStatus: текст обрезан с многоточием', document.getElementById('status').textContent.endsWith('…'));

  document.getElementById('bcTrigger').value = 'дорого';
  document.getElementById('bcResponse').value = 'Скидка 20% при годовой оплате.';
  addBattlecard();
  check('addBattlecard: api.add_battlecard вызван с триггером и ответом',
        window.__calls.some(c => c[0]==='add_battlecard' && c[1][0]==='дорого' && c[1][1]==='Скидка 20% при годовой оплате.'));
  check('addBattlecard: поля очищены после добавления', document.getElementById('bcTrigger').value === '' && document.getElementById('bcResponse').value === '');
  renderBattlecards([['дорого', 'Скидка 20% при годовой оплате.']]);
  check('renderBattlecards: карточка отрисована в списке', document.getElementById('bcList').textContent.includes('дорого'));

  renderBattlecards([['<script>x</script>', 'ответ с <b>тегом</b> внутри']]);
  const bcListEl = document.getElementById('bcList');
  check('renderBattlecards: HTML в тексте карточки экранирован, не выполняется как разметка',
        bcListEl.querySelectorAll('script, b').length === 0 && bcListEl.textContent.includes('<script>'));

  addSuggestion('Скидка 20%', 'battlecard');
  check('addSuggestion(battlecard): label корректный', document.getElementById('feed').lastElementChild.querySelector('.who').textContent.includes('Карточка'));

  const pastList = [
    {filename: '2026-08-28_10-00-00.md', label: '2026-08-28 10:00'},
    {filename: '2026-08-27_09-00-00.md', label: '2026-08-27 09:00'},
  ];
  renderPastSummaries(pastList);
  const pastListEl = document.getElementById('pastCallsList');
  check('renderPastSummaries: оба файла отрисованы', pastListEl.querySelectorAll('.past-call-item').length === 2, pastListEl.innerHTML);
  check('renderPastSummaries: label первого файла виден', pastListEl.textContent.includes('2026-08-28 10:00'));

  renderPastSummaries([]);
  check('renderPastSummaries: пустой список -> нейтральное сообщение', pastListEl.textContent.includes('нет прошлых созвонов'), pastListEl.textContent);

  renderSummaryContent('# Саммари\\n\\nТестовое содержимое.');
  check('renderSummaryContent: текст саммари отображён', document.getElementById('pastCallContent').textContent.includes('Тестовое содержимое.'));

  loadPastSummary('2026-08-28_10-00-00.md');
  check('loadPastSummary: api.read_summary вызван с именем файла', window.__calls.some(c => c[0]==='read_summary' && c[1][0]==='2026-08-28_10-00-00.md'));

  const pastSection = document.getElementById('pastCallsSection');
  const wasVisible = pastSection.style.display !== 'none';
  togglePastCalls();
  check('togglePastCalls: переключает видимость секции', (pastSection.style.display !== 'none') !== wasVisible);
  togglePastCalls();

  return results;
}""")

            for r in results:
                check("UI: " + r["name"], r["pass"], r["detail"])
            check("UI: нет ошибок в консоли браузера (кроме favicon)", len(console_errors) == 0, console_errors)

            browser.close()
    finally:
        httpd.shutdown()
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)


print()
print(f"=== ИТОГО: {len(passed)} passed, {len(failed)} failed ===")
if failed:
    print("Провалились:", failed)
    sys.exit(1)
