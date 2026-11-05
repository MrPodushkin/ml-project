"use strict";
const byId = (id) => document.getElementById(id);
const form = byId("predict-form");
const message = byId("message");
const token = byId("token");
const card = document.querySelector(".result-card");
const counter = () => { byId("counter").textContent = `${Array.from(message.value).length} / 1000`; };
message.addEventListener("input", counter);
document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => { message.value = button.dataset.example; counter(); message.focus(); });
});
async function health() {
  try {
    const response = await fetch("/health", {cache: "no-store"});
    const data = await response.json();
    byId("health").textContent = response.ok ? "● Модель готова к работе" : "● Модель временно недоступна";
    byId("health").classList.toggle("unavailable", !response.ok);
    byId("model-version").textContent = data.model_version;
  } catch {
    byId("health").textContent = "● Нет связи с сервером";
    byId("health").classList.add("unavailable");
  }
}
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const submit = byId("submit");
  submit.disabled = true;
  submit.textContent = "Проверяем…";
  card.className = "card result-card";
  byId("result-title").textContent = "Проверяем сообщение…";
  byId("result-message").textContent = "Ожидаем ответ модели.";
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json", "Authorization": `Bearer ${token.value}`},
      body: JSON.stringify({message: message.value})
    });
    const data = await response.json();
    byId("http-status").textContent = String(response.status);
    byId("request-id").textContent = data.request_id || response.headers.get("X-Request-ID") || "—";
    byId("json-output").textContent = JSON.stringify(data, null, 2);
    if (response.ok) {
      card.classList.add(data.prediction === "spam" ? "spam" : "ham");
      byId("result-title").textContent = data.prediction === "spam" ? "Похоже на спам" : "Обычное сообщение";
      byId("result-message").textContent = data.prediction === "spam" ? "Модель отнесла сообщение к классу spam." : "Модель отнесла сообщение к классу ham.";
      byId("model-version").textContent = data.model_version;
    } else {
      card.classList.add("error");
      const errors = {401: ["Нет доступа", "Проверьте API-токен."], 413: ["Запрос слишком большой", "Уменьшите объём данных."], 415: ["Неверный формат", "API принимает JSON с текстом сообщения."], 422: ["Проверьте сообщение", "Нужна непустая строка до 1000 символов без запрещённых символов."], 429: ["Слишком много запросов", "Дождитесь окончания периода ограничения и повторите попытку."], 503: ["Сервис занят", "Модель пока не готова или обрабатывает другой запрос."], 504: ["Время ожидания истекло", "Сервис восстанавливает процесс модели. Попробуйте позже."]};
      const error = errors[response.status] || ["Ошибка сервиса", "Повторите попытку позже."];
      byId("result-title").textContent = error[0];
      byId("result-message").textContent = error[1];
    }
  } catch {
    card.classList.add("error");
    byId("result-title").textContent = "Нет связи с сервером";
    byId("result-message").textContent = "Проверьте подключение и SSH-туннель.";
    byId("http-status").textContent = "—";
    byId("request-id").textContent = "—";
    byId("json-output").textContent = "{}";
  } finally {
    submit.disabled = false;
    submit.textContent = "Проверить сообщение →";
    health();
  }
});
window.addEventListener("pagehide", () => { token.value = ""; });
health();
