// Luna Socia: choose a girlfriend or create one, then chat with her. Messages stream from
// /api/send as server-sent events. Views follow the address: "#" lists them, "#new" creates
// one, "#her=3&edit" changes her, and "#her=3&chat=5" is a chat with her.
"use strict";

const $ = (id) => document.getElementById(id);
const thread = $("thread");
const list = $("messages");
const text = $("text");
const presence = $("presence");
const state = {
  her: null, chat: null, oldest: null, sending: false, photo: null, voice: false, recorder: null, lastDay: null,
  options: null, companions: [],
};
const LABELS = {
  length: { short: "Short", medium: "Medium", long: "Long" },
  emoji: { none: "None", some: "Some", lots: "Lots" },
  language: { match: "Same as yours", english: "English", hinglish: "Hinglish", hindi: "Hindi (Devanagari)" },
};

function stored(key, fallback) {
  try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; }
}
function store(key, value) {
  try { localStorage.setItem(key, value); } catch { /* private window */ }
}

function herApi(id = state.her?.id) {
  return `/api/companions/${id}`;
}

async function json(url, init) {
  const reply = await fetch(url, init).catch(() => null);
  if (!reply) throw new Error("Socia could not be reached.");
  const body = await reply.json().catch(() => ({}));
  if (!reply.ok) throw new Error(body.error || `HTTP ${reply.status}`);
  return body;
}

function element(tag, props = {}, ...children) {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children.filter((child) => child !== null && child !== undefined));
  return node;
}

function timeAgo(iso) {
  if (!iso) return "";
  const minutes = Math.round((Date.now() - new Date(iso)) / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} min ago`;
  if (minutes < 1440) return `${Math.round(minutes / 60)} h ago`;
  return dayLabel(new Date(iso));
}

function styleLabel(her) {
  return state.options?.styles.find((style) => style.id === her.persona.style)?.label ?? "";
}
function voiceLabel(id) {
  const voice = state.options?.voices.find((item) => item.id === id);
  return voice ? `${voice.label} · ${voice.accent}` : id;
}
function memoryLines(memory) {
  return (memory || "").split("\n").filter((line) => line.trim()).length;
}
function memorySummary(her) {
  const progress = her.memory_progress;
  if (progress.reading) return `reading your chats · ${progress.read} of ${progress.sources}`;
  const lines = memoryLines(her.memory);
  return lines ? `${lines} things about you` : "nothing yet";
}

// Views

function show(view) {
  for (const id of ["home", "editor", "chat"]) $(id).hidden = id !== view;
  if (view !== "chat") closeSidebar();
}

async function loadOptions() {
  if (!state.options) state.options = await json("/api/options");
  return state.options;
}

async function route() {
  const params = new URLSearchParams(location.hash.slice(1));
  try {
    await loadOptions();
  } catch (error) {
    show("home");
    $("companions").replaceChildren(element("li", { className: "sc-error", textContent: error.message }));
    return;
  }
  if (params.has("new")) return openEditor(null);
  const id = Number(params.get("her"));
  if (!id) return openHome();
  let her;
  try {
    her = (await json(herApi(id))).companion;
  } catch {
    location.hash = "";
    return;
  }
  if (params.has("edit")) return openEditor(her);
  return openHer(her, Number(params.get("chat")) || null);
}

function go(hash) {
  if (location.hash === hash || (hash === "" && !location.hash)) route();
  else location.hash = hash;
}

// Landing: every girlfriend, and a card to create one.

let homePoll = null;
async function openHome() {
  show("home");
  document.title = "Luna Socia";
  state.her = null;
  await loadCompanions();
}

async function loadCompanions() {
  clearTimeout(homePoll);
  let companions;
  try {
    ({ companions } = await json("/api/companions"));
  } catch (error) {
    $("companions").replaceChildren(element("li", { className: "sc-error", textContent: error.message }));
    return;
  }
  state.companions = companions;
  const cards = companions.map((her) => {
    const card = element("button", { type: "button", className: "sc-card" });
    card.setAttribute("aria-label", `Chat with ${her.name}`);
    if (her.has_profile) {
      card.append(element("img", { src: `${herApi(her.id)}/profile?t=${Date.now()}`, alt: "", loading: "lazy" }));
    } else {
      card.append(element("span", { className: "sc-card-initial", textContent: her.name.slice(0, 1).toUpperCase() }));
    }
    if (her.memory_progress.reading) {
      card.append(element("span", { className: "sc-card-badge",
        textContent: `learning about you · ${her.memory_progress.read}/${her.memory_progress.sources}` }));
    }
    const chats = her.chats ? `${her.chats} chat${her.chats === 1 ? "" : "s"} · ${timeAgo(her.last_active)}` : "no chats yet";
    card.append(element("span", { className: "sc-card-text" },
      element("strong", { textContent: her.name }),
      element("span", { textContent: styleLabel(her) }),
      element("span", { textContent: chats })));
    card.addEventListener("click", () => go(`#her=${her.id}`));
    return element("li", {}, card);
  });
  const create = element("button", { type: "button", className: "sc-card sc-card-new" },
    element("b", { textContent: "+" }),
    element("strong", { textContent: "Create a girlfriend" }),
    element("span", { textContent: "Her looks, personality, voice, and memories from your ChatGPT or Claude chats" }));
  create.addEventListener("click", () => go("#new"));
  $("companions").replaceChildren(...cards, element("li", {}, create));
  $("companions").setAttribute("aria-busy", "false");
  if (companions.some((her) => her.memory_progress.reading) && !$("home").hidden) {
    homePoll = setTimeout(loadCompanions, 5000);
  }
}

// Create or edit a girlfriend.

const draft = { her: null, persona: null, voice: null, picture: null, exportFile: null, memory: "" };
let memoryPoll = null;
let sample = null;

function radio(container, items, current, pick, render) {
  container.replaceChildren(...items.map((item) => {
    const button = render(item);
    button.setAttribute("role", "radio");
    button.setAttribute("aria-checked", String(item.id === current));
    button.addEventListener("click", () => {
      pick(item.id);
      container.querySelectorAll("[role=radio]").forEach((other) => other.setAttribute("aria-checked", String(other === button)));
      drawPreview();
    });
    return button;
  }));
}

function segmented(container, values, key) {
  radio(container, values.map((id) => ({ id })), draft.persona[key], (id) => { draft.persona[key] = id; },
    (item) => element("button", { type: "button", textContent: LABELS[key][item.id] ?? item.id }));
}

function drawVoices() {
  const container = $("her-voices");
  container.replaceChildren(...state.options.voices.map((voice) => {
    const row = element("div", { className: "sc-voice" });
    row.setAttribute("role", "radio");
    row.setAttribute("aria-checked", String(voice.id === draft.voice));
    const pick = element("button", { type: "button", className: "sc-voice-pick" },
      element("strong", { textContent: voice.label }), element("span", { textContent: voice.accent }));
    pick.addEventListener("click", () => {
      draft.voice = voice.id;
      container.querySelectorAll(".sc-voice").forEach((other) => other.setAttribute("aria-checked", String(other === row)));
      drawPreview();
    });
    const play = element("button", { type: "button", className: "sc-icon", title: `Hear ${voice.label}` });
    play.setAttribute("aria-label", `Hear ${voice.label}`);
    play.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M7 4v16l13-8Z"/></svg>';
    play.addEventListener("click", () => playSample(voice, play));
    row.append(pick, play);
    return row;
  }));
}

function playSample(voice, button) {
  sample?.audio.pause();
  sample?.button.classList.remove("sc-playing");
  if (sample?.button === button) { sample = null; return; }
  const audio = new Audio(`/api/voices/${voice.id}/sample`);
  sample = { audio, button };
  button.classList.add("sc-playing");
  const done = () => { button.classList.remove("sc-playing"); if (sample?.audio === audio) sample = null; };
  audio.addEventListener("ended", done);
  audio.addEventListener("error", done);
  audio.play().catch(done);
}

function drawPreview() {
  const name = $("her-name").value.trim();
  $("preview-name").textContent = name || "Her name";
  const initial = $("preview-initial");
  initial.textContent = (name || "?").slice(0, 1).toUpperCase();
  const style = state.options.styles.find((item) => item.id === draft.persona.style);
  $("preview-style").textContent = [style?.label, LABELS.length[draft.persona.length] + " replies"].join(" · ");
  let example = style?.example ?? "";
  if (draft.persona.emoji === "none") example = example.replace(/[\u{1F000}-\u{1FAFF}\u{2600}-\u{27BF}\u{FE0F}\u{200D}]/gu, "").replace(/\s+/g, " ").trim();
  if (draft.persona.emoji === "lots" && !/✨/.test(example)) example += " ✨💕";
  $("preview-bubble").textContent = example;
  $("preview-voice").textContent = `Voice: ${voiceLabel(draft.voice)}`;
  const lines = memoryLines($("her-memory").value);
  $("preview-memory").textContent = draft.exportFile
    ? `Memory: will read ${draft.exportFile.name}`
    : `Memory: ${lines ? `${lines} things about you` : "nothing yet"}`;
}

function setPicture(url) {
  for (const [img, empty] of [[$("her-picture-preview"), $("her-picture-empty")], [$("preview-picture"), $("preview-initial")]]) {
    img.hidden = !url;
    empty.hidden = Boolean(url);
    if (url) img.src = url;
  }
}

function setExport(file) {
  draft.exportFile = file;
  $("her-export-label").textContent = file
    ? `${file.name} · ${(file.size / 1048576).toFixed(1)} MB, imported when you save`
    : "Choose an export (.zip or conversations.json)";
  drawPreview();
}

function editorError(message) {
  $("editor-error").textContent = message || "";
  $("editor-error").hidden = !message;
}

async function openEditor(her) {
  show("editor");
  const options = state.options;
  Object.assign(draft, { her, persona: { ...(her?.persona ?? options.default_persona) },
    voice: her?.voice ?? options.default_voice, picture: null, exportFile: null });
  document.title = her ? `Edit ${her.name} · Luna Socia` : "Create a girlfriend · Luna Socia";
  $("editor-title").textContent = her ? `Edit ${her.name}` : "Create a girlfriend";
  $("editor-subtitle").textContent = her
    ? "Change how she texts, her voice and what she remembers"
    : "Her looks, how she texts, her voice and what she knows about you";
  $("editor-save").textContent = her ? "Save" : "Create her";
  $("her-delete").hidden = !her;
  $("her-name").value = her?.name ?? "";
  $("her-name").maxLength = options.limits.name;
  $("her-about").value = her?.persona.about ?? "";
  $("her-about").maxLength = options.limits.about;
  $("her-nude").checked = her?.persona.nude_pictures === "on";
  $("her-memory").value = her?.memory ?? "";
  $("her-memory").maxLength = options.limits.memory;
  $("memory-field").hidden = !her;
  $("her-picture").value = "";
  $("her-export").value = "";
  $("her-picture-label").textContent = her?.has_profile ? "Change her profile picture" : "Choose her profile picture";
  setPicture(her?.has_profile ? `${herApi(her.id)}/profile?t=${Date.now()}` : null);
  setExport(null);
  editorError("");

  radio($("her-styles"), options.styles, draft.persona.style, (id) => { draft.persona.style = id; }, (style) =>
    element("button", { type: "button", className: "sc-style" },
      element("strong", { textContent: style.label }), element("span", { textContent: `“${style.example}”` })));
  segmented($("her-length"), options.lengths, "length");
  segmented($("her-emoji"), options.emoji, "emoji");
  $("her-language").replaceChildren(...options.languages.map((id) =>
    element("option", { value: id, textContent: LABELS.language[id] ?? id, selected: id === draft.persona.language })));
  drawVoices();
  drawMemory(her);
  drawPreview();
  if (!her) $("her-name").focus();
}

function drawMemory(her) {
  clearTimeout(memoryPoll);
  const progress = her?.memory_progress;
  $("memory-forget").hidden = !progress?.sources;
  $("memory-state").hidden = !progress?.sources;
  if (!progress?.sources) return;
  const share = progress.sources ? progress.read / progress.sources : 0;
  $("memory-bar").querySelector("i").style.width = `${Math.round(share * 100)}%`;
  $("memory-bar").setAttribute("aria-valuenow", String(Math.round(share * 100)));
  $("memory-note").textContent = progress.error
    ? `Reading stopped: ${progress.error}. It tries again by itself.`
    : progress.reading
      ? `Reading your chats: ${progress.read} of ${progress.sources}. She learns while you talk; replies come first.`
      : `She has read all ${progress.sources} of your imported chats.`;
  if (progress.reading && !$("editor").hidden && draft.her?.id === her.id) {
    memoryPoll = setTimeout(async () => {
      try {
        const answer = await json(`${herApi(her.id)}/memory`);
        const fresh = { ...her, memory_progress: answer.progress };
        // What she remembers grows while she reads; the box follows unless it was edited.
        if ($("her-memory").value === (draft.her.memory ?? "")) $("her-memory").value = answer.memory;
        draft.her = { ...draft.her, memory: answer.memory, memory_progress: answer.progress };
        drawMemory(fresh);
        drawPreview();
      } catch { /* tried again on the next visit */ }
    }, 3000);
  }
}

// Sends a file with an upload progress note: an export can be hundreds of MB.
function upload(url, form, onProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", url);
    request.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    });
    request.addEventListener("load", () => {
      let body = {};
      try { body = JSON.parse(request.responseText); } catch { /* not JSON */ }
      if (request.status >= 200 && request.status < 300) resolve(body);
      else reject(new Error(body.error || `HTTP ${request.status}`));
    });
    request.addEventListener("error", () => reject(new Error("The upload failed.")));
    request.send(form);
  });
}

async function saveEditor(event) {
  event.preventDefault();
  const name = $("her-name").value.trim();
  if (!name) { editorError("Give her a name."); $("her-name").focus(); return; }
  editorError("");
  const save = $("editor-save");
  save.disabled = true;
  const label = save.textContent;
  const nude = $("her-nude").checked ? "on" : "off";
  const persona = { ...draft.persona, about: $("her-about").value, language: $("her-language").value, nude_pictures: nude };
  const body = { name, persona, voice: draft.voice };
  if (draft.her && $("her-memory").value !== (draft.her.memory ?? "")) body.memory = $("her-memory").value;
  let her = draft.her;
  try {
    save.textContent = "Saving…";
    her = draft.her
      ? (await json(herApi(her.id), { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).companion
      : (await json("/api/companions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).companion;
    draft.her = her;
    if (draft.picture) {
      save.textContent = "Saving her picture…";
      const form = new FormData();
      form.append("photo", draft.picture);
      await json(`${herApi(her.id)}/profile`, { method: "POST", body: form });
      // A new picture turns nude pictures off; ticked in this same save, the owner confirmed the new face.
      if (nude === "on") her = (await json(herApi(her.id), { method: "PATCH", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ persona: { nude_pictures: "on" } }) })).companion;
    }
    if (draft.exportFile) {
      const form = new FormData();
      form.append("export", draft.exportFile);
      const answer = await upload(`${herApi(her.id)}/memory/imports`, form, (share) => {
        save.textContent = share < 1 ? `Uploading your chats… ${Math.round(share * 100)}%` : "Reading the export…";
      });
      setExport(null);
      if (!answer.added) {
        editorError(answer.conversations
          ? "She already has every chat in that export."
          : "That export has no chats with messages from you.");
        her = (await json(herApi(her.id))).companion;
        openEditor(her);
        return;
      }
    }
    go(`#her=${her.id}`);
  } catch (error) {
    editorError(error.message);
    // Created already: what follows is an edit of her.
    if (her && !draft.her?.memory_progress) draft.her = her;
    if (her) { $("editor-title").textContent = `Edit ${her.name}`; $("her-delete").hidden = false; }
  } finally {
    save.disabled = false;
    save.textContent = draft.her ? "Save" : label;
  }
}

async function deleteHer(her) {
  if (!confirm(`Delete ${her.name}, with all her chats, pictures and memories? This cannot be undone.`)) return;
  try {
    await json(herApi(her.id), { method: "DELETE" });
    go("");
  } catch (error) {
    editorError(error.message);
  }
}

function wireEditor() {
  $("editor-form").addEventListener("submit", saveEditor);
  $("her-name").addEventListener("input", drawPreview);
  $("her-memory").addEventListener("input", drawPreview);
  $("her-language").addEventListener("change", () => { draft.persona.language = $("her-language").value; });
  const back = () => go(draft.her ? `#her=${draft.her.id}` : "");
  $("editor-back").addEventListener("click", back);
  $("editor-cancel").addEventListener("click", back);
  $("her-delete").addEventListener("click", () => draft.her && deleteHer(draft.her));
  $("her-picture").addEventListener("change", () => {
    const file = $("her-picture").files[0];
    if (!file) return;
    draft.picture = file;
    $("her-nude").checked = false;
    setPicture(URL.createObjectURL(file));
    $("her-picture-label").textContent = "Choose another picture";
  });
  $("her-export").addEventListener("change", () => setExport($("her-export").files[0] ?? null));
  const drop = $("her-export-drop");
  drop.addEventListener("dragover", (event) => { event.preventDefault(); drop.classList.add("sc-over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("sc-over"));
  drop.addEventListener("drop", (event) => {
    event.preventDefault();
    drop.classList.remove("sc-over");
    const file = event.dataTransfer.files[0];
    if (file) setExport(file);
  });
  $("memory-forget").addEventListener("click", async () => {
    if (!draft.her || !confirm("Delete the chats you imported? What she already remembers stays until you edit it.")) return;
    try {
      await json(`${herApi(draft.her.id)}/memory/imports`, { method: "DELETE" });
      openEditor((await json(herApi(draft.her.id))).companion);
    } catch (error) {
      editorError(error.message);
    }
  });
}

// One girlfriend: her chats, with a list beside them.

async function openHer(her, chatId) {
  show("chat");
  state.her = her;
  $("her-title").textContent = her.name;
  $("menu-title").textContent = her.name;
  text.placeholder = `Message ${her.name}`;
  $("profile-open").setAttribute("aria-label", `${her.name}'s profile picture`);
  drawAbout(her);
  loadProfile();
  drawSwitcher();
  const chats = await loadChats();
  openChat(chats.find((chat) => chat.id === chatId) ?? chats[0] ?? null);
}

function drawAbout(her) {
  $("about-name").textContent = her.name;
  $("about-style").textContent = [styleLabel(her), `${LABELS.length[her.persona.length]} replies`, LABELS.language[her.persona.language]].join(" · ");
  $("about-text").textContent = her.persona.about;
  $("about-voice").textContent = voiceLabel(her.voice);
  $("about-memory").textContent = memorySummary(her);
  $("about-chats").textContent = String(her.chats);
}

async function drawSwitcher() {
  if (!state.companions.length) {
    try { ({ companions: state.companions } = await json("/api/companions")); } catch { return; }
  }
  $("switcher").hidden = state.companions.length < 2;
  $("switcher").replaceChildren(...state.companions.map((her) => {
    const button = element("button", { type: "button", className: "sc-switch", title: her.name });
    button.setAttribute("aria-current", String(her.id === state.her?.id));
    button.append(her.has_profile
      ? element("img", { src: `${herApi(her.id)}/avatar`, alt: "" })
      : element("i", { textContent: her.name.slice(0, 1).toUpperCase() }));
    button.append(element("span", { textContent: her.name }));
    button.addEventListener("click", () => { if (her.id !== state.her?.id) go(`#her=${her.id}`); });
    return button;
  }));
}

function openSidebar() {
  $("sidebar").classList.add("sc-open");
  $("sidebar-backdrop").hidden = false;
  $("chats-open").setAttribute("aria-expanded", "true");
  loadChats();
}
function closeSidebar() {
  $("sidebar").classList.remove("sc-open");
  $("sidebar-backdrop").hidden = true;
  $("chats-open").setAttribute("aria-expanded", "false");
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
  image.addEventListener("click", () => lightbox(url));
  return image;
}

function lightbox(url) {
  const box = document.createElement("div");
  box.className = "sc-lightbox";
  box.append(Object.assign(document.createElement("img"), { src: url, alt: "Picture" }));
  const save = Object.assign(document.createElement("a"), {
    className: "sc-lightbox-save", href: `${url}?download=1`, textContent: "Download" });
  save.addEventListener("click", (event) => event.stopPropagation());
  box.append(save);
  box.addEventListener("click", () => box.remove());
  document.body.append(box);
}

// One message's element; hers can be filled in while she types.
function messageElement(message) {
  const element = document.createElement("div");
  element.className = `sc-message sc-${message.role}`;
  if (message.image_url) {
    // The picture with its download button in the corner; saved with the date in its name.
    const figure = document.createElement("div");
    figure.className = "sc-figure";
    const save = metaAction("a", "download", "Download this picture");
    save.classList.add("sc-picture-save");
    save.href = `${message.image_url}?download=1`;
    figure.append(picture(message.image_url), save);
    element.append(figure);
  }
  if (message.content) {
    const row = document.createElement("div");
    row.className = "sc-bubble-row";
    const bubble = document.createElement("div");
    bubble.className = "sc-bubble";
    bubble.textContent = message.content;
    row.append(bubble);
    if (message.role === "luna" && message.id) {
      // Her voice, right beside what she said.
      const play = metaAction("button", "play", "Play her voice");
      play.addEventListener("click", () => speak(message.id));
      row.append(play);
    }
    element.append(row);
  }
  element.append(meta(message));
  return element;
}

// A message's small icon buttons (voice, download, delete), in the page's own line style.
const META_ICONS = {
  play: '<path d="M11 5 6 9H2v6h4l5 4V5Z"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/>',
  download: '<path d="M12 4v11"/><path d="m7 10 5 5 5-5"/><path d="M5 20h14"/>',
  delete: '<path d="M4 7h16"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M6 7l1 13h10l1-13"/><path d="M9 7V4h6v3"/>',
};

function metaAction(tag, icon, label) {
  const action = document.createElement(tag);
  if (tag === "button") action.type = "button";
  action.className = `sc-meta-action sc-${icon}`;
  action.title = label;
  action.setAttribute("aria-label", label);
  action.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" `
    + `stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${META_ICONS[icon]}</svg>`;
  return action;
}

function meta(message) {
  const line = document.createElement("div");
  line.className = "sc-meta";
  const time = new Date(message.created_at);
  line.append(time.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" }));
  if (message.id) {
    const remove = metaAction("button", "delete", "Delete this message");
    remove.addEventListener("click", (event) => deleteMessage(message.id, event.currentTarget.closest(".sc-message")));
    line.append(remove);
  }
  return line;
}

// Forget one message, with its photo or picture. She stops seeing it too.
async function deleteMessage(id, element) {
  if (!confirm(`Delete this message? ${state.her?.name ?? "She"} forgets it too.`)) return;
  const reply = await fetch(`/api/messages/${id}`, { method: "DELETE" }).catch(() => null);
  if (!reply?.ok) {
    const { error } = (await reply?.json().catch(() => null)) || {};
    alert(error || "That message could not be deleted.");
    return;
  }
  const day = element?.previousElementSibling;
  element?.remove();
  // A day with no messages left loses its heading too.
  if (day?.classList.contains("sc-day") && (!day.nextElementSibling || day.nextElementSibling.classList.contains("sc-day"))) {
    day.remove();
  }
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
  $("empty").hidden = true;
  return element;
}

async function loadMessages(before) {
  const chat = state.chat;
  if (!chat) return;
  const reply = await fetch(`/api/chats/${chat.id}/messages${before ? `?before=${before}` : ""}`);
  if (!reply.ok || state.chat !== chat) return;
  const { messages, page } = await reply.json();
  if (!messages.length) { $("empty").hidden = Boolean(before) || list.childElementCount > 0; return; }
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
  $("older").hidden = messages.length < page;
}

// Chats: each is its own conversation with her, and every one is kept.
function startedLabel(iso) {
  const date = new Date(iso);
  const sameYear = date.getFullYear() === new Date().getFullYear();
  return `Started ${date.toLocaleDateString(undefined, { day: "numeric", month: "short", ...(sameYear ? {} : { year: "numeric" }) })}`;
}

function showChat(chat) {
  state.chat = chat;
  $("chat-title").textContent = chat ? `${chat.title} · ${startedLabel(chat.created_at)}` : "";
  document.title = chat ? `${chat.title} · ${state.her.name}` : `${state.her.name} · Luna Socia`;
  $("delete-chat").hidden = !chat;
  try {
    history.replaceState(null, "", `#her=${state.her.id}${chat ? `&chat=${chat.id}` : ""}`);
  } catch { /* sandboxed */ }
  document.querySelectorAll(".sc-chat-item").forEach((item) =>
    item.setAttribute("aria-current", String(Number(item.dataset.chat) === chat?.id)));
}

function openChat(chat) {
  if (state.sending) { setPresence("wait for her reply first"); return; }
  showChat(chat);
  list.replaceChildren();
  state.lastDay = null;
  state.oldest = null;
  $("older").hidden = true;
  $("empty").textContent = `A new chat with ${state.her.name}. Say hi.`;
  $("empty").hidden = Boolean(chat);
  if (chat) loadMessages();
  closeSidebar();
  text.focus();
}

function newChat() {
  clearPhoto();
  openChat(null);
}

function chatTime(iso) {
  const date = new Date(iso);
  return date.toDateString() === new Date().toDateString()
    ? date.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" })
    : dayLabel(date);
}

async function loadChats() {
  const her = state.her;
  if (!her) return [];
  const reply = await fetch(`${herApi(her.id)}/chats`).catch(() => null);
  if (!reply?.ok || state.her !== her) return [];
  const { chats } = await reply.json();
  const items = chats.map((chat) => {
    const item = document.createElement("li");
    item.className = "sc-chat-item";
    item.dataset.chat = String(chat.id);
    item.setAttribute("aria-current", String(chat.id === state.chat?.id));
    const open = document.createElement("button");
    open.type = "button";
    open.className = "sc-chat-open";
    const title = document.createElement("strong");
    title.textContent = chat.title;
    const line = document.createElement("span");
    line.textContent = [chatTime(chat.updated_at), chat.last].filter(Boolean).join(" · ");
    const started = document.createElement("small");
    started.textContent = startedLabel(chat.created_at);
    open.append(title, started, line);
    open.addEventListener("click", () => openChat(chat));
    const rename = document.createElement("button");
    rename.type = "button";
    rename.className = "sc-icon";
    rename.title = "Rename";
    rename.setAttribute("aria-label", `Rename ${chat.title}`);
    rename.textContent = "✎";
    rename.addEventListener("click", () => renameChat(chat));
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "sc-icon";
    remove.title = "Delete";
    remove.setAttribute("aria-label", `Delete ${chat.title}`);
    remove.textContent = "×";
    remove.addEventListener("click", () => deleteChat(chat));
    item.append(open, rename, remove);
    return item;
  });
  $("chat-list").replaceChildren(...items);
  $("chats-note").textContent = chats.length
    ? `Every chat is kept, so ${her.name} can remember it.`
    : "No chats yet. Your first message starts one.";
  $("about-chats").textContent = String(chats.length);
  return chats;
}

async function renameChat(chat) {
  const title = prompt("Name this chat", chat.title);
  if (!title?.trim()) return;
  const reply = await fetch(`/api/chats/${chat.id}`, {
    method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title }),
  });
  if (!reply.ok) return;
  const { chat: renamed } = await reply.json();
  if (state.chat?.id === renamed.id) showChat(renamed);
  loadChats();
}

async function deleteChat(chat) {
  if (!confirm(`Delete "${chat.title}", with its photos and pictures? ${state.her.name} forgets it.`)) return;
  const reply = await fetch(`/api/chats/${chat.id}`, { method: "DELETE" });
  if (!reply.ok) {
    const { error } = await reply.json().catch(() => ({ error: `HTTP ${reply.status}` }));
    setPresence(error);
    return;
  }
  if (state.chat?.id === chat.id) {
    const chats = await loadChats();
    openChat(chats[0] ?? null);
  } else {
    loadChats();
  }
}

// Her reply as it arrives.
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
    case "chat":
      showChat(event.chat);
      break;
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
  if (state.sending || !state.her || (!message && !state.photo)) return;
  state.sending = true;
  $("send").disabled = true;
  const form = new FormData();
  form.append("text", message);
  form.append("companion_id", state.her.id);
  if (state.chat) form.append("chat_id", state.chat.id);
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
    loadChats();
  }
}

// Voice: her replies aloud, and voice messages through the server's Whisper.
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

// The message box grows with what you type, borders included, so it never shows a
// scrollbar until it reaches its tallest (9rem).
function resize() {
  text.style.height = "auto";
  const borders = text.offsetHeight - text.clientHeight;
  text.style.height = `${text.scrollHeight + borders}px`;
  text.style.overflowY = text.scrollHeight > text.clientHeight ? "auto" : "hidden";
}

async function loadProfile() {
  const her = state.her;
  const reply = await fetch(`${herApi()}/profile`, { method: "HEAD" }).catch(() => null);
  if (state.her !== her) return;
  const has = reply?.ok;
  const stamp = Date.now();
  $("avatar").src = has ? `${herApi()}/avatar?t=${stamp}` : "/static/logo.png";
  if (has) {
    $("profile-large").src = `${herApi()}/profile?t=${stamp}`;
    $("about-photo").src = `${herApi()}/profile?t=${stamp}`;
  }
  $("profile-large").hidden = !has;
  $("about-photo").hidden = !has;
  $("crop-open").hidden = !has;
  if (!has) $("crop").hidden = true;
  $("profile-note").textContent = has
    ? "Her profile picture is her face in every picture she takes."
    : `${her.name} has no profile picture yet. Add one: it becomes her face in every picture she takes.`;
}

// Her avatar's crop: the square's centre (x, y as fractions of the picture) and zoom
// (1 = the largest square the picture holds). The server cuts the avatar from it.
const crop = { width: 1, height: 1, x: 0.5, y: 0.5, zoom: 1, drag: null };
const CROP_MAX_ZOOM = 6;

function cropSide() {
  return Math.min(crop.width, crop.height) / crop.zoom;
}

function clampCrop() {
  crop.zoom = Math.min(Math.max(crop.zoom, 1), CROP_MAX_ZOOM);
  const side = cropSide();
  const halfX = side / 2 / crop.width;
  const halfY = side / 2 / crop.height;
  crop.x = Math.min(Math.max(crop.x, halfX), 1 - halfX);
  crop.y = Math.min(Math.max(crop.y, halfY), 1 - halfY);
}

// Place the picture so the crop's square fills the stage.
function drawCrop() {
  const stage = $("crop-stage");
  const image = $("crop-image");
  const side = cropSide();
  const scale = stage.clientWidth / side;
  image.style.width = `${crop.width * scale}px`;
  image.style.height = `${crop.height * scale}px`;
  image.style.left = `${-(crop.x * crop.width - side / 2) * scale}px`;
  image.style.top = `${-(crop.y * crop.height - side / 2) * scale}px`;
  $("crop-zoom").value = String(crop.zoom);
}

function takeCrop(answer) {
  Object.assign(crop, { width: answer.width, height: answer.height, x: answer.x, y: answer.y, zoom: answer.zoom });
  clampCrop();
  drawCrop();
}

async function openCrop() {
  const reply = await fetch(`${herApi()}/avatar/crop`).catch(() => null);
  if (!reply?.ok) return;
  const answer = await reply.json();
  const image = $("crop-image");
  image.onload = () => takeCrop(answer);
  image.src = `${herApi()}/profile?t=${Date.now()}`;
  $("crop").hidden = false;
  $("crop-open").hidden = true;
  $("crop-stage").focus();
}

function closeCrop() {
  $("crop").hidden = true;
  $("crop-open").hidden = false;
}

async function cropRequest(method, body) {
  const reply = await fetch(`${herApi()}/avatar/crop`, {
    method, headers: body ? { "Content-Type": "application/json" } : {}, body: body ? JSON.stringify(body) : undefined,
  }).catch(() => null);
  if (!reply?.ok) {
    $("profile-note").textContent = "The crop could not be saved.";
    return null;
  }
  $("avatar").src = `${herApi()}/avatar?t=${Date.now()}`;
  return reply.json();
}

function wireCrop() {
  const stage = $("crop-stage");
  stage.addEventListener("pointerdown", (event) => {
    stage.setPointerCapture(event.pointerId);
    crop.drag = { pointer: event.pointerId, startX: event.clientX, startY: event.clientY, x: crop.x, y: crop.y };
  });
  stage.addEventListener("pointermove", (event) => {
    if (!crop.drag || crop.drag.pointer !== event.pointerId) return;
    const scale = stage.clientWidth / cropSide();
    crop.x = crop.drag.x - (event.clientX - crop.drag.startX) / scale / crop.width;
    crop.y = crop.drag.y - (event.clientY - crop.drag.startY) / scale / crop.height;
    clampCrop();
    drawCrop();
  });
  const endDrag = () => { crop.drag = null; };
  stage.addEventListener("pointerup", endDrag);
  stage.addEventListener("pointercancel", endDrag);
  stage.addEventListener("wheel", (event) => {
    event.preventDefault();
    crop.zoom *= Math.exp(-event.deltaY * 0.0015);
    clampCrop();
    drawCrop();
  }, { passive: false });
  stage.addEventListener("keydown", (event) => {
    const step = 0.02 / crop.zoom;
    const moves = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
    if (event.key === "+" || event.key === "=") crop.zoom *= 1.1;
    else if (event.key === "-") crop.zoom /= 1.1;
    else if (moves[event.key]) { crop.x += moves[event.key][0]; crop.y += moves[event.key][1]; }
    else return;
    event.preventDefault();
    clampCrop();
    drawCrop();
  });
  $("crop-zoom").addEventListener("input", () => {
    crop.zoom = Number($("crop-zoom").value);
    clampCrop();
    drawCrop();
  });
  $("crop-open").addEventListener("click", openCrop);
  $("crop-cancel").addEventListener("click", closeCrop);
  $("crop-save").addEventListener("click", async () => {
    if (await cropRequest("PUT", { x: crop.x, y: crop.y, zoom: crop.zoom })) {
      $("profile-note").textContent = "Her avatar is updated.";
      closeCrop();
    }
  });
  $("crop-reset").addEventListener("click", async () => {
    const answer = await cropRequest("DELETE");
    if (answer) takeCrop(answer);
  });
  window.addEventListener("resize", () => { if (!$("crop").hidden) drawCrop(); });
}

// Her other pictures, sent with her profile picture so her face stays the same.
async function loadReferences(note) {
  const reply = await fetch(`${herApi()}/references`).catch(() => null);
  if (!reply?.ok) return;
  const { references, max } = await reply.json();
  $("references").replaceChildren(...references.map((reference) => {
    const item = document.createElement("li");
    const image = picture(reference.url);
    image.className = "";
    image.alt = `Picture of ${state.her.name}`;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "×";
    remove.setAttribute("aria-label", "Remove this picture");
    remove.addEventListener("click", async () => {
      if (!confirm("Stop using this picture of her?")) return;
      await fetch(reference.url, { method: "DELETE" });
      loadReferences();
    });
    item.append(image, remove);
    return item;
  }));
  $("references-add").hidden = references.length >= max;
  $("references-note").textContent = note || (references.length
    ? `${references.length} of ${max}. Use pictures where her face is clear; each one makes a picture a little slower.`
    : "More pictures of her keep her face the same from other sides. Each one makes a picture a little slower.");
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
  const openMenu = () => { loadStatus(); loadReferences(); menu.showModal(); };
  $("references-file").addEventListener("change", async () => {
    const files = [...$("references-file").files];
    if (!files.length) return;
    const form = new FormData();
    files.forEach((file) => form.append("photos", file));
    $("references-note").textContent = "Saving…";
    const reply = await fetch(`${herApi()}/references`, { method: "POST", body: form });
    const answer = await reply.json().catch(() => ({}));
    $("references-file").value = "";
    loadReferences(reply.ok ? "" : answer.error || "Those pictures could not be saved.");
  });
  $("menu-open").addEventListener("click", openMenu);
  $("profile-open").addEventListener("click", openMenu);
  $("about-settings").addEventListener("click", openMenu);
  $("about-photo").addEventListener("click", () => lightbox($("about-photo").src));
  const edit = () => { menu.close(); go(`#her=${state.her.id}&edit`); };
  $("about-edit").addEventListener("click", edit);
  $("menu-edit").addEventListener("click", edit);
  wireCrop();
  $("profile-file").addEventListener("change", async () => {
    const file = $("profile-file").files[0];
    if (!file) return;
    const form = new FormData();
    form.append("photo", file);
    const reply = await fetch(`${herApi()}/profile`, { method: "POST", body: form });
    $("profile-note").textContent = reply.ok ? "Saved." : "That picture could not be saved.";
    if (reply.ok) {
      loadProfile();
      state.companions = [];
      drawSwitcher();
    }
  });
  $("delete-chat").addEventListener("click", async () => {
    menu.close();
    if (state.chat) await deleteChat(state.chat);
  });

  $("chats-open").addEventListener("click", openSidebar);
  $("chats-close").addEventListener("click", closeSidebar);
  $("sidebar-backdrop").addEventListener("click", closeSidebar);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && $("sidebar").classList.contains("sc-open")) closeSidebar();
  });
  $("to-home").addEventListener("click", () => go(""));
  $("new-chat").addEventListener("click", newChat);
  $("drawer-new-chat").addEventListener("click", newChat);

  wireEditor();
  // A fresh list of them whenever the view changes, so cards and the switcher stay current.
  window.addEventListener("hashchange", () => { state.companions = []; route(); });
}

wire();
route();
