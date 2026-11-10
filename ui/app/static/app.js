// Luna Socia: the chat page. Messages stream from /api/send as server-sent events.
"use strict";

const $ = (id) => document.getElementById(id);
const thread = $("thread");
const list = $("messages");
const text = $("text");
const presence = $("presence");
const state = { oldest: null, sending: false, photo: null, voice: false, recorder: null, lastDay: null };

function stored(key, fallback) {
  try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; }
}
function store(key, value) {
  try { localStorage.setItem(key, value); } catch { /* private window */ }
}

function setPresence(label, busy = false) {
  presence.textContent = label;
  presence.classList.toggle("sc-busy", busy);
}

function nearBottom() {
  return thread.scrollHeight - thread.scrollTop - thread.clientHeight < 160;
}
function toBottom() {
  thread.scrollTop = thread.scrollHeight;
}

function dayLabel(date) {
  const today = new Date();
  const yesterday = new Date(Date.now() - 86400000);
  if (date.toDateString() === today.toDateString()) return "Today";
  if (date.toDateString() === yesterday.toDateString()) return "Yesterday";
  return date.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" });
}

function picture(url) {
  const image = document.createElement("img");
  image.className = "sc-picture";
  image.src = url;
  image.alt = "Picture";
  image.loading = "lazy";
  image.addEventListener("click", () => {
    const box = document.createElement("div");
    box.className = "sc-lightbox";
    box.append(Object.assign(document.createElement("img"), { src: url, alt: "Picture" }));
    box.addEventListener("click", () => box.remove());
    document.body.append(box);
  });
  return image;
}

// One message's element; Luna's can be filled in while she types.
function messageElement(message) {
  const element = document.createElement("div");
  element.className = `sc-message sc-${message.role}`;
  if (message.image_url) element.append(picture(message.image_url));
  if (message.content) {
    const bubble = document.createElement("div");
    bubble.className = "sc-bubble";
    bubble.textContent = message.content;
    element.append(bubble);
  }
  element.append(meta(message));
  return element;
}

function meta(message) {
  const line = document.createElement("div");
  line.className = "sc-meta";
  const time = new Date(message.created_at);
  line.append(time.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" }));
  if (message.role === "luna" && message.content && message.id) {
    const play = document.createElement("button");
    play.type = "button";
    play.className = "sc-play";
    play.textContent = "play";
    play.addEventListener("click", () => speak(message.id));
    line.append(play);
  }
  return line;
}

function daySeparator(date) {
  const element = document.createElement("div");
  element.className = "sc-day";
  element.textContent = dayLabel(date);
  return element;
}

function append(message) {
  const day = new Date(message.created_at).toDateString();
  if (day !== state.lastDay) {
    list.append(daySeparator(new Date(message.created_at)));
    state.lastDay = day;
  }
  const element = messageElement(message);
  list.append(element);
  return element;
}

async function loadMessages(before) {
  const reply = await fetch(`/api/messages${before ? `?before=${before}` : ""}`);
  if (!reply.ok) return;
  const { messages } = await reply.json();
  if (!messages.length) return;
  if (before) {
    const height = thread.scrollHeight;
    const fragment = document.createDocumentFragment();
    let day = null;
    for (const message of messages) {
      const thisDay = new Date(message.created_at).toDateString();
      if (thisDay !== day) { fragment.append(daySeparator(new Date(message.created_at))); day = thisDay; }
      fragment.append(messageElement(message));
    }
    list.prepend(fragment);
    thread.scrollTop = thread.scrollHeight - height;
  } else {
    messages.forEach(append);
    toBottom();
  }
  state.oldest = messages[0].id;
  $("older").hidden = messages.length < 50;
}

// Luna's reply as it arrives.
function replyElement() {
  const element = document.createElement("div");
  element.className = "sc-message sc-luna";
  const typing = document.createElement("div");
  typing.className = "sc-bubble sc-typing";
  typing.innerHTML = "<i></i><i></i><i></i>";
  element.append(typing);
  list.append(element);
  toBottom();
  return { element, typing, bubble: null, developing: null };
}

function handle(event, reply) {
  const follow = nearBottom();
  switch (event.type) {
    case "user":
      append(event.message);
      list.append(reply.element);
      break;
    case "status":
      setPresence(event.text.toLowerCase() + "…", true);
      break;
    case "text":
      if (!reply.bubble) {
        reply.bubble = document.createElement("div");
        reply.bubble.className = "sc-bubble";
        reply.typing.replaceWith(reply.bubble);
        setPresence("typing…", true);
      }
      reply.bubble.textContent += event.text;
      break;
    case "image_started":
      reply.developing = document.createElement("div");
      reply.developing.className = "sc-developing";
      reply.developing.textContent = "taking a picture…";
      reply.element.prepend(reply.developing);
      setPresence("taking a picture…", true);
      break;
    case "image":
      reply.developing?.replaceWith(picture(event.url));
      reply.developing = null;
      break;
    case "error": {
      const error = document.createElement("div");
      error.className = "sc-error";
      error.textContent = event.text;
      reply.element.append(error);
      reply.developing?.remove();
      break;
    }
    case "done": {
      const finished = messageElement(event.message);
      reply.element.querySelectorAll(".sc-error").forEach((error) => finished.append(error));
      reply.element.replaceWith(finished);
      if (state.voice && event.message.content) speak(event.message.id);
      break;
    }
  }
  if (follow) toBottom();
}

async function send(words) {
  const message = (words ?? text.value).trim();
  if (state.sending || (!message && !state.photo)) return;
  state.sending = true;
  $("send").disabled = true;
  const form = new FormData();
  form.append("text", message);
  form.append("timezone", Intl.DateTimeFormat().resolvedOptions().timeZone || "");
  if (state.photo) form.append("photo", state.photo);
  text.value = "";
  resize();
  clearPhoto();
  const reply = replyElement();
  reply.element.remove();
  try {
    const response = await fetch("/api/send", { method: "POST", body: form });
    if (!response.ok) {
      const { error } = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
      throw new Error(error);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let end;
      while ((end = buffer.indexOf("\n\n")) >= 0) {
        const chunk = buffer.slice(0, end);
        buffer = buffer.slice(end + 2);
        if (chunk.startsWith("data: ")) handle(JSON.parse(chunk.slice(6)), reply);
      }
    }
  } catch (error) {
    if (!reply.element.isConnected) list.append(reply.element);
    handle({ type: "error", text: error.message }, reply);
    reply.typing.remove();
  } finally {
    state.sending = false;
    $("send").disabled = false;
    setPresence("online");
    text.focus();
  }
}

// Voice: Luna's replies aloud, and voice messages through the server's Whisper.
let playing = null;
async function speak(id) {
  playing?.pause();
  playing = new Audio(`/api/speech/${id}`);
  playing.play().catch(() => {});
}

async function toggleRecording() {
  const mic = $("mic");
  if (state.recorder) {
    state.recorder.stop();
    return;
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    setPresence("microphone not allowed");
    return;
  }
  const chunks = [];
  const recorder = new MediaRecorder(stream);
  state.recorder = recorder;
  mic.classList.add("sc-recording");
  setPresence("listening… tap the mic to send", true);
  recorder.ondataavailable = (event) => chunks.push(event.data);
  recorder.onstop = async () => {
    stream.getTracks().forEach((track) => track.stop());
    state.recorder = null;
    mic.classList.remove("sc-recording");
    setPresence("hearing you…", true);
    const reply = await fetch("/api/transcribe", { method: "POST", body: new Blob(chunks, { type: recorder.mimeType }) });
    const answer = await reply.json().catch(() => ({}));
    setPresence("online");
    if (answer.text) send(answer.text);
    else setPresence(answer.error || "didn't catch that");
  };
  recorder.start();
}

// Photo to send.
function clearPhoto() {
  state.photo = null;
  $("photo").value = "";
  $("attachment").hidden = true;
}

function resize() {
  text.style.height = "auto";
  text.style.height = `${text.scrollHeight}px`;
}

async function loadProfile() {
  const reply = await fetch("/api/profile", { method: "HEAD" }).catch(() => null);
  const has = reply?.ok;
  const url = `/api/profile?t=${Date.now()}`;
  if (has) {
    $("avatar").src = url;
    $("profile-large").src = url;
  }
  $("profile-large").hidden = !has;
  $("profile-note").textContent = has
    ? "Her profile picture is her face in every picture she takes."
    : "Luna has no profile picture yet. Add one: it becomes her face in every picture she takes.";
}

async function loadStatus() {
  const reply = await fetch("/api/status").catch(() => null);
  if (!reply?.ok) return;
  const status = await reply.json();
  $("model-note").textContent = `Chat: ${status.chat_model} · Pictures: ${status.images ? "ready" : "image model not running"}`;
}

function wire() {
  $("composer").addEventListener("submit", (event) => { event.preventDefault(); send(); });
  text.addEventListener("input", resize);
  text.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); send(); }
  });
  $("photo").addEventListener("change", () => {
    const file = $("photo").files[0];
    if (!file) return;
    state.photo = file;
    $("attachment-preview").src = URL.createObjectURL(file);
    $("attachment").hidden = false;
    text.focus();
  });
  $("attachment-remove").addEventListener("click", clearPhoto);
  $("mic").addEventListener("click", toggleRecording);
  $("older").addEventListener("click", () => loadMessages(state.oldest));

  state.voice = stored("socia.voice", "false") === "true";
  $("voice-toggle").setAttribute("aria-pressed", String(state.voice));
  $("voice-toggle").addEventListener("click", () => {
    state.voice = !state.voice;
    store("socia.voice", String(state.voice));
    $("voice-toggle").setAttribute("aria-pressed", String(state.voice));
  });

  const menu = $("menu");
  const openMenu = () => { loadStatus(); menu.showModal(); };
  $("menu-open").addEventListener("click", openMenu);
  $("profile-open").addEventListener("click", openMenu);
  $("profile-file").addEventListener("change", async () => {
    const file = $("profile-file").files[0];
    if (!file) return;
    const form = new FormData();
    form.append("photo", file);
    const reply = await fetch("/api/profile", { method: "POST", body: form });
    $("profile-note").textContent = reply.ok ? "Saved." : "That picture could not be saved.";
    if (reply.ok) loadProfile();
  });
  $("clear").addEventListener("click", async () => {
    if (!confirm("Forget the whole conversation, with its photos and pictures?")) return;
    await fetch("/api/messages", { method: "DELETE" });
    list.replaceChildren();
    state.lastDay = null;
    menu.close();
  });
}

wire();
loadProfile();
loadMessages();
