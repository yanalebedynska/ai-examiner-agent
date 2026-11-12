"""AI Examiner Agent: standalone Gradio deployment for Render.

This is a deployment copy of the Colab project. API keys are read from
server environment variables, never from public source code.
"""

import json
import os
import random
import threading
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
from groq import Groq


MODEL_NAME = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
RESULTS_PATH = Path(os.environ.get("RESULTS_FILE", "exam_results.json"))

STUDENTS = {
    "student1@example.com": "Студент 1",
    "student2@example.com": "Студент 2",
    "student3@example.com": "Студент 3",
}

EXAM_TOPICS = [
    "Токенізація тексту",
    "Стемінг та лематизація",
    "Bag-of-Words та TF-IDF",
    "Word Embeddings та GloVe",
    "Модель BERT та її застосування",
    "Великі мовні моделі (LLM)",
    "Prompt Engineering",
    "LLM-агенти та виклик інструментів",
]

active_exams = {}
exam_results = []
state_lock = threading.RLock()


def load_results():
    if not RESULTS_PATH.exists():
        return []
    try:
        content = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        return content if isinstance(content, list) else []
    except (OSError, ValueError):
        return []


exam_results = load_results()


def save_results():
    temp_path = RESULTS_PATH.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(exam_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(RESULTS_PATH)


def groq_client():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("У Render ще не задано GROQ_API_KEY.")
    return Groq(api_key=key)


def ask_llm(topic):
    response = groq_client().chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Ти екзаменатор з основ NLP. Постав рівно одне конкретне "
                    "екзаменаційне запитання українською мовою на задану тему. "
                    "Не пиши відповідь та не переходь до іншої теми."
                ),
            },
            {"role": "user", "content": f"Тема: {topic}"},
        ],
        temperature=0.3,
        reasoning_effort="low",
        max_completion_tokens=450,
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("Модель повернула порожнє запитання.")
    return text


GRADE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "exam_feedback",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "score": {"type": "integer"},
                "correct": {"type": "string"},
                "missing": {"type": "string"},
            },
            "required": ["score", "correct", "missing"],
            "additionalProperties": False,
        },
    },
}


def grade_answer(topic, question, answer):
    response = groq_client().chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Ти викладач з NLP. Оціни відповідь студента від 0 до 10, "
                    "спираючись лише на фактично поставлене запитання. "
                    "Будь уважним: не вимагай зайвої інформації, яка не стосується питання. "
                    "Не став наступних запитань. Поверни score (ціле число 0..10), "
                    "correct (що правильно) та missing (чого бракує) у JSON. "
                    "Пояснення пиши українською."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Тема: {topic}\n"
                    f"Запитання: {question}\n"
                    f"Відповідь студента: {answer}"
                ),
            },
        ],
        response_format=GRADE_SCHEMA,
        temperature=0.2,
        reasoning_effort="low",
        max_completion_tokens=800,
    )
    data = json.loads(response.choices[0].message.content or "{}")
    score = data.get("score")
    if not isinstance(score, int) or not 0 <= score <= 10:
        raise ValueError("Некоректна оцінка від моделі.")
    correct = str(data.get("correct") or "").strip()
    missing = str(data.get("missing") or "").strip()
    feedback = (
        f"### Оцінка: {score}/10\n\n"
        f"**Що правильно:** {correct}\n\n"
        f"**Чого бракує:** {missing}"
    )
    return float(score), feedback


def outputs(status, question="", feedback="", answer="", check=False, next_=False):
    return (
        status,
        question,
        feedback,
        answer,
        gr.update(interactive=check),
        gr.update(interactive=next_),
    )


def start_or_resume(email):
    email = (email or "").strip().lower()
    if email not in STUDENTS:
        return outputs("Перевірте email: такого студента немає у тестовому списку.")

    with state_lock:
        existing = active_exams.get(email)
        if existing:
            topic_no = existing["current_topic"] + 1
            topic = existing["topics"][existing["current_topic"]]
            if existing.get("pending_question"):
                return outputs(
                    f"Тема {topic_no}: {topic}",
                    existing["pending_question"],
                    check=True,
                )
            if existing["history"]:
                record = existing["history"][-1]
                return outputs(
                    "Відповідь уже оцінено. Натисніть «Наступна тема / Завершити».",
                    record["question"],
                    record.get("feedback", ""),
                    next_=True,
                )

    topics = random.sample(EXAM_TOPICS, 3)
    try:
        question = ask_llm(topics[0])
    except Exception as exc:
        return outputs(f"Не вдалося почати іспит: {exc}")

    with state_lock:
        if email not in active_exams:
            active_exams[email] = {
                "student": STUDENTS[email],
                "topics": topics,
                "current_topic": 0,
                "pending_question": question,
                "history": [],
            }
        actual = active_exams[email]
        return outputs(
            f"Іспит розпочато. Студент: {actual['student']}. "
            f"Тема {actual['current_topic'] + 1}: "
            f"{actual['topics'][actual['current_topic']]}",
            actual["pending_question"],
            check=True,
        )


def check_answer(email, answer):
    email = (email or "").strip().lower()
    answer = (answer or "").strip()
    with state_lock:
        exam = active_exams.get(email)
        if exam is None:
            return outputs("Спочатку розпочніть іспит.")
        question = exam.get("pending_question")
        if not question:
            if exam["history"]:
                record = exam["history"][-1]
                return outputs(
                    "Цю відповідь уже оцінено.", record["question"],
                    record["feedback"], next_=True,
                )
            return outputs("Немає активного запитання.")
        if not answer:
            return outputs("Введіть відповідь.", question, check=True)
        topic = exam["topics"][exam["current_topic"]]

    try:
        score, feedback = grade_answer(topic, question, answer)
    except Exception as exc:
        return outputs(f"Помилка оцінювання: {exc}", question, answer=answer, check=True)

    with state_lock:
        latest = active_exams.get(email)
        if not latest or latest.get("pending_question") != question:
            return outputs("Стан іспиту змінився. Натисніть «Продовжити іспит».")
        latest["history"].append(
            {
                "topic": topic,
                "question": question,
                "answer": answer,
                "score": score,
                "feedback": feedback,
            }
        )
        latest["pending_question"] = None

    return outputs(
        f"Відповідь оцінено: {score:g}/10. Можна переходити далі.",
        question, feedback, next_=True,
    )


def next_topic(email):
    email = (email or "").strip().lower()
    with state_lock:
        exam = active_exams.get(email)
        if exam is None:
            return outputs("Активний іспит не знайдено.")
        if exam.get("pending_question"):
            return outputs(
                "Спочатку дайте відповідь на запитання.",
                exam["pending_question"], check=True,
            )
        if not exam["history"] or "score" not in exam["history"][-1]:
            return outputs("Спочатку потрібно оцінити відповідь.")
        next_index = exam["current_topic"] + 1

        if next_index >= len(exam["topics"]):
            if len(exam["history"]) != len(exam["topics"]):
                return outputs("Не всі теми пройдено.")
            scores = [rec["score"] for rec in exam["history"]]
            final_score = round(sum(scores) / len(scores), 2)
            result = {
                "student": exam["student"],
                "email": email,
                "topics": exam["topics"],
                "score": final_score,
                "feedback": f"Іспит завершено. Оцінки: {scores}.",
                "history": exam["history"],
                "date": datetime.now(timezone.utc).isoformat(),
            }
            exam_results.append(result)
            try:
                save_results()
            except OSError as exc:
                exam_results.pop()
                return outputs(f"Не вдалося зберегти результат: {exc}", next_=True)
            del active_exams[email]
            report = (
                "### Результати іспиту\n\n"
                f"Оцінки за темами: {scores}\n\n"
                f"**Підсумкова оцінка: {final_score}/10**\n\n"
                "Результат збережено в локальний файл сервера."
            )
            return outputs("Іспит завершено!", feedback=report)
        new_topic = exam["topics"][next_index]

    try:
        question = ask_llm(new_topic)
    except Exception as exc:
        return outputs(f"Не вдалося створити наступне запитання: {exc}", next_=True)

    with state_lock:
        latest = active_exams.get(email)
        if latest is None:
            return outputs("Активний іспит уже завершено.")
        if latest["current_topic"] != next_index - 1 or latest.get("pending_question"):
            return outputs("Стан іспиту змінився. Натисніть «Продовжити іспит».")
        latest["current_topic"] = next_index
        latest["pending_question"] = question

    return outputs(f"Тема {next_index + 1}: {new_topic}", question, check=True)


with gr.Blocks(title="AI Examiner Agent") as demo:
    gr.Markdown("# AI Examiner Agent")
    gr.Markdown("Інтелектуальна система для проведення іспиту з основ NLP")
    gr.Markdown("Тестові email: `student1@example.com`, `student2@example.com`, `student3@example.com`.")

    email_input = gr.Textbox(label="Електронна пошта студента", placeholder="student2@example.com")
    start_button = gr.Button("Розпочати / Продовжити іспит")
    status_output = gr.Markdown()
    question_output = gr.Textbox(label="Екзаменаційне запитання", lines=3, interactive=False)
    answer_input = gr.Textbox(label="Ваша відповідь", lines=5, placeholder="Введіть відповідь...")
    check_button = gr.Button("Перевірити відповідь", variant="primary", interactive=False)
    feedback_output = gr.Markdown()
    next_button = gr.Button("Наступна тема / Завершити", interactive=False)

    ui_outputs = [
        status_output,
        question_output,
        feedback_output,
        answer_input,
        check_button,
        next_button,
    ]
    start_button.click(start_or_resume, inputs=email_input, outputs=ui_outputs)
    check_button.click(check_answer, inputs=[email_input, answer_input], outputs=ui_outputs)
    next_button.click(next_topic, inputs=email_input, outputs=ui_outputs)


if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "10000")),
        share=False,
        show_error=True,
    )
