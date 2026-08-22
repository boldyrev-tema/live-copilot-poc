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
import socketserver
import sys
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

m.knowledge_base_chunks.clear()
m.knowledge_base_chunks.append(("прайс.txt", "Продукт стоит 5000 рублей. Скидка 20% при годовой оплате."))
m.knowledge_base_chunks.append(("faq.txt", "Погода в Москве переменчивая, часто идут дожди."))
rel = m.retrieve_relevant_chunks("а если скажут что дорого, какая скидка есть?")
check("retrieve: релевантный вопрос находит прайс-чанк", any("прайс" in s for s, _ in rel), rel)
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

print()
print("=== Бэкенд: интеграционные тесты (реальные вызовы API) ===")

m.user_context = "Пять лет опыта в React."
m.user_notes = ""
m.knowledge_base_chunks.clear()

ans = m.get_suggestion(
    live_context="Собеседник: Расскажите о вашем опыте с React.",
    last_speaker="Собеседник", last_text="Расскажите о вашем опыте с React.",
    use_search=False,
)
check("get_suggestion: использует контекст (упомянут React)", "react" in ans.lower(), ans)

ans2 = m.get_suggestion(
    live_context="Ты: хороший вопрос сколько будет два плюс два",
    last_speaker="Ты", last_text="сколько будет два плюс два",
    use_search=False, direct_question=True,
)
check("get_suggestion: direct_question отвечает по существу", "четыре" in ans2.lower() or "4" in ans2, ans2)

called = {}
orig_search = m.web_search
m.web_search = lambda q: (called.setdefault("q", q), orig_search(q))[1]
ans3 = m.get_suggestion(
    live_context="Собеседник: Какая сейчас последняя версия Python?",
    last_speaker="Собеседник", last_text="Какая сейчас последняя версия Python?",
    use_search=True,
)
check("get_suggestion: поиск реально вызван на маркере свежести", "q" in called, called)
m.web_search = orig_search

ans4 = m.get_suggestion(
    live_context="Собеседник: Как у вас дела сегодня?",
    last_speaker="Собеседник", last_text="Как у вас дела сегодня?",
    use_search=False,
)
check("get_suggestion: светская реплика -> SKIP_TOKEN", m.SKIP_TOKEN in ans4, ans4)

m.user_context = "Пять лет опыта в продажах SaaS."
m.user_notes = "Не забыть упомянуть кейс с ростом выручки на 40%."
m.knowledge_base_chunks.clear()
m.knowledge_base_chunks.append(("прайс.txt", "Продукт стоит 5000 рублей в месяц. Скидка 20% при годовой оплате."))
ans5 = m.get_suggestion(
    live_context="Собеседник: Ваш продукт слишком дорогой.",
    last_speaker="Собеседник", last_text="Ваш продукт слишком дорогой, у конкурентов дешевле.",
    use_search=False,
)
check("get_suggestion: подтянул скидку из базы знаний", "20" in ans5, ans5)
check("get_suggestion: подтянул заметку про 40%", "40" in ans5, ans5)

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

m.user_context = ""
m.user_notes = ""
m.knowledge_base_chunks.clear()
m.transcript_lines.clear()

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

  const searchBtn = document.getElementById('searchBtn');
  toggleSearch();
  check('toggleSearch: класс off появился после клика', searchBtn.classList.contains('off') === true);
  check('toggleSearch: api.set_search вызван с false', window.__calls.some(c => c[0]==='set_search' && c[1][0]===false));
  toggleSearch();

  const notesBox = document.getElementById('notesBox');
  notesBox.value = 'тестовая заметка';
  onNotesChange();
  check('onNotesChange: api.update_notes вызван с текстом', window.__calls.some(c => c[0]==='update_notes' && c[1][0]==='тестовая заметка'));

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
