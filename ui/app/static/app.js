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
  onSpeaker(audio).then(() => audio.play()).catch(done);
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
  // Her bio as her memory of you shows: a "- " line each (a bio from before is shown that way too).
  $("her-about").value = (her?.persona.about ?? "").split("\n").map((line) => line.trim().replace(/^[-•*]\s*/, ""))
    .filter(Boolean).map((line) => `- ${line}`).join("\n");
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
  $("memory-refresh").addEventListener("click", async () => {
    // What she remembers changes while you chat (her remember tool): load it again.
    if (!draft.her) return;
    const box = $("her-memory");
    if (box.value !== (draft.her.memory ?? "") && !confirm("Replace your unsaved edits with what she remembers now?")) return;
    const button = $("memory-refresh");
    button.disabled = true;
    try {
      const answer = await json(`${herApi(draft.her.id)}/memory`);
      draft.her = { ...draft.her, memory: answer.memory, memory_progress: answer.progress };
      box.value = answer.memory ?? "";
      drawMemory(draft.her);
      drawPreview();
      editorError("");
    } catch (error) {
      editorError(error.message);
    } finally {
      button.disabled = false;
    }
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
  // Her bio a line a bullet, without any dash or bullet the line already starts with.
  $("about-text").replaceChildren(...(her.persona.about || "").split("\n")
    .map((line) => line.trim().replace(/^[-•*]\s*/, "")).filter(Boolean)
    .map((line) => element("li", { textContent: line })));
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

// The on-screen keyboard: browsers that ignore the viewport's interactive-widget hint (Safari, some
// Android browsers) only shrink the visual viewport and scroll the page to show the focused box. The
// chat is fixed to the visual viewport, its height and its offset, so the composer sits right on the keys.
if (window.visualViewport) {
  const viewport = window.visualViewport;
  const root = document.documentElement.style;
  const fit = () => {
    const stick = nearBottom();
    root.setProperty("--sc-viewport", `${viewport.height}px`);
    root.setProperty("--sc-viewport-top", `${viewport.offsetTop}px`);
    if (stick) toBottom();
  };
  viewport.addEventListener("resize", fit);
  viewport.addEventListener("scroll", fit);
  // The keyboard opens a moment after focus, and some browsers report its size late.
  for (const event of ["focus", "blur"]) {
    text.addEventListener(event, () => { fit(); setTimeout(fit, 300); });
  }
  fit();
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
  image.addEventListener("click", () => lightbox(url, picturesBeside(image)));
  return image;
}

// A picture over everything, dialogs too (a modal dialog is on the top layer). With others beside it (her other
// pictures, your photos, or the chat's pictures), ← and → or the arrow keys go through them. Esc or a click closes.
function lightbox(url, urls = [url]) {
  let at = Math.max(urls.indexOf(url), 0);
  const box = element("dialog", { className: "sc-lightbox", autofocus: true });
  const image = element("img", { alt: "Picture" });
  const save = element("a", { className: "sc-lightbox-save", textContent: "Download" });
  const count = element("span", { className: "sc-lightbox-count" });
  const step = (by) => (event) => { event?.stopPropagation(); at = (at + by + urls.length) % urls.length; show(); };
  const previous = element("button", { type: "button", className: "sc-lightbox-nav sc-lightbox-prev", textContent: "‹" });
  const next = element("button", { type: "button", className: "sc-lightbox-nav sc-lightbox-next", textContent: "›" });
  previous.setAttribute("aria-label", "Previous picture");
  next.setAttribute("aria-label", "Next picture");
  const show = () => {
    image.src = urls[at];
    save.href = `${urls[at]}${urls[at].includes("?") ? "&" : "?"}download=1`;
    count.textContent = `${at + 1} / ${urls.length}`;
  };
  previous.addEventListener("click", step(-1));
  next.addEventListener("click", step(1));
  save.addEventListener("click", (event) => event.stopPropagation());
  box.addEventListener("click", () => box.close());
  box.addEventListener("keydown", (event) => {
    if (urls.length < 2) return;
    if (event.key === "ArrowLeft") { event.preventDefault(); step(-1)(); }
    if (event.key === "ArrowRight") { event.preventDefault(); step(1)(); }
  });
  box.addEventListener("close", () => box.remove());
  box.append(image, save);
  if (urls.length > 1) box.append(previous, next, count);
  show();
  document.body.append(box);
  box.showModal();
}

// The pictures shown beside one: the others in its list of pictures, or the chat's pictures.
function picturesBeside(image) {
  const group = image.closest(".sc-references");
  const found = group ? [...group.querySelectorAll("img")] : [...list.querySelectorAll("img.sc-picture")];
  const urls = found.map((img) => img.getAttribute("src"));
  return urls.includes(image.getAttribute("src")) ? urls : [image.getAttribute("src")];
}

// One message's element; hers can be filled in while she types.
function messageElement(message) {
  const element = document.createElement("div");
  element.className = `sc-message sc-${message.role}`;
  if (message.id) element.dataset.id = message.id;
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
  if (message.role === "luna" && message.id) {
    // A double tap on what she said or sent puts a heart on it, or takes it off, as on a phone.
    element.addEventListener("dblclick", (event) => {
      if (event.target.closest(".sc-meta, .sc-meta-action, .sc-reactions")) return;
      react(message.id, element.dataset.reaction === REACTIONS()[0] ? null : REACTIONS()[0]);
    });
  }
  if (message.apps?.length) element.append(usedApps(message.apps));
  if (message.remembered?.length) element.append(rememberedLine(message));
  element.append(meta(message));
  showReaction(element, message.reaction);
  return element;
}

// What she saved with this reply, for every chat: each fact with an × to take it back if she got it wrong.
function rememberedLine(message) {
  const line = element("p", { className: "sc-remembered" }, "Remembered: ");
  message.remembered.forEach((item, position) => {
    if (position) line.append(" · ");
    const where = item.about === "me" ? "her bio" : "her memory of you";
    const forget = element("button", { type: "button", className: "sc-forget", textContent: "×",
                                       title: `Wrong? Take it out of ${where}` });
    forget.setAttribute("aria-label", `Forget: ${item.fact}`);
    forget.addEventListener("click", () => forgetFact(message.id, item, line));
    line.append(element("span", { className: "sc-fact" }, item.fact, forget));
  });
  return line;
}

async function forgetFact(id, item, line) {
  const where = item.about === "me" ? "her bio" : "her memory of you";
  if (!confirm(`Forget "${item.fact}"? It comes out of ${where}.`)) return;
  const reply = await fetch(`/api/messages/${id}/remembered/${item.index}`, { method: "DELETE" }).catch(() => null);
  if (!reply?.ok) {
    const { error } = (await reply?.json().catch(() => null)) || {};
    alert(error || "She could not forget that just now.");
    return;
  }
  const { message, found } = await reply.json();
  if (message.remembered.length) line.replaceWith(rememberedLine(message));
  else line.remove();
  if (!found) alert(`It was no longer in ${where} (changed there since), so nothing else changed.`);
}

// Each Luna app's logo (copied from its page into static/apps), for her replies and the settings list.
const appLogo = (key, className) => element("img", { className, src: `/static/apps/${key}.png`, alt: "", loading: "lazy" });
const APP_NAMES = { fiora: "Luna Fiora", lexia: "Luna Lexia", lumia: "Luna Lumia", vibia: "Luna Vibia",
                    visia: "Luna Visia", vita: "Luna Vita", sentia: "Luna Sentia" };

// What she did in your Luna apps for this reply.
function usedApps(apps) {
  const list = document.createElement("div");
  list.className = "sc-used-apps";
  for (const use of apps) {
    const item = element("span", { className: "sc-used-app" },
      appLogo(use.app, "sc-used-app-logo"),
      element("span", { textContent: `${APP_NAMES[use.app] || use.app}${use.tool ? ` · ${use.tool}` : ""}` }));
    list.append(item);
  }
  return list;
}

// Reactions: the emoji either of them can put on the other's message.
function REACTIONS() {
  return state.options?.reactions ?? ["❤️", "😂", "😘", "😮", "😢", "🔥", "👍"];
}

function showReaction(element, reaction) {
  if (!element) return;
  element.querySelector(".sc-reaction")?.remove();
  if (reaction) element.dataset.reaction = reaction;
  else delete element.dataset.reaction;
  if (!reaction) return;
  const badge = document.createElement("span");
  badge.className = "sc-reaction";
  badge.textContent = reaction;
  const mine = element.classList.contains("sc-luna");
  badge.title = state.her ? `${state.her.name}'s reaction` : "Her reaction";
  if (mine) {
    badge.setAttribute("role", "button");
    badge.tabIndex = 0;
    badge.title = "Your reaction: tap to take it off";
    badge.addEventListener("click", () => react(Number(element.dataset.id), null));
  }
  element.querySelector(".sc-meta").before(badge);
}

async function react(id, reaction) {
  const element = list.querySelector(`.sc-message[data-id="${id}"]`);
  const before = element?.dataset.reaction ?? null;
  showReaction(element, reaction);
  closeReactions();
  const reply = await fetch(`/api/messages/${id}/reaction`, {
    method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ reaction }),
  }).catch(() => null);
  if (!reply?.ok) {
    showReaction(element, before);
    const { error } = (await reply?.json().catch(() => null)) || {};
    alert(error || "That reaction did not stick.");
  }
}

// The row of reactions that opens above one of her messages.
function openReactions(id, anchor) {
  const open = document.querySelector(".sc-reactions");
  closeReactions();
  if (open?.dataset.for === String(id)) return;
  const element = list.querySelector(`.sc-message[data-id="${id}"]`);
  const current = element?.dataset.reaction ?? null;
  const picker = document.createElement("div");
  picker.className = "sc-reactions";
  picker.dataset.for = id;
  picker.setAttribute("role", "menu");
  for (const emoji of REACTIONS()) {
    const choice = document.createElement("button");
    choice.type = "button";
    choice.textContent = emoji;
    choice.setAttribute("role", "menuitem");
    choice.setAttribute("aria-label", emoji === current ? `Take off ${emoji}` : `React ${emoji}`);
    if (emoji === current) choice.setAttribute("aria-pressed", "true");
    choice.addEventListener("click", () => react(id, emoji === current ? null : emoji));
    picker.append(choice);
  }
  anchor.closest(".sc-meta").before(picker);
  picker.querySelector("button")?.focus();
}

function closeReactions() {
  document.querySelector(".sc-reactions")?.remove();
}

// A message's small icon buttons (voice, download, delete), in the page's own line style.
const META_ICONS = {
  play: '<path d="M11 5 6 9H2v6h4l5 4V5Z"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/>',
  download: '<path d="M12 4v11"/><path d="m7 10 5 5 5-5"/><path d="M5 20h14"/>',
  react: '<circle cx="12" cy="12" r="9"/><path d="M8.5 14.5a4.5 4.5 0 0 0 7 0"/><path d="M9 9.5h.01"/><path d="M15 9.5h.01"/>',
  edit: '<path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/>',
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
  if (message.edited) line.append(element("span", { className: "sc-edited", textContent: "edited" }));
  if (message.id && message.role === "luna") {
    const add = metaAction("button", "react", "React to this message");
    add.addEventListener("click", (event) => openReactions(message.id, event.currentTarget));
    line.append(add);
  }
  if (message.id && message.role === "user" && message.content) {
    const edit = metaAction("button", "edit", "Edit this message");
    edit.addEventListener("click", (event) => editMessage(message, event.currentTarget.closest(".sc-message")));
    line.append(edit);
  }
  if (message.id) {
    const remove = metaAction("button", "delete", "Delete this message");
    remove.addEventListener("click", (event) => deleteMessage(message.id, event.currentTarget.closest(".sc-message")));
    line.append(remove);
  }
  return line;
}

// New words for one of your messages, typed in place of it: Enter saves, Esc or Cancel leaves it as it was.
// Saving sends it again: what came after it goes, and she answers the new words.
function editMessage(message, item) {
  const row = item?.querySelector(".sc-bubble-row");
  if (!row || item.querySelector(".sc-edit-form")) return;
  const box = element("textarea", { className: "sc-edit-box", value: message.content, rows: 1 });
  box.setAttribute("aria-label", "Edit your message");
  const save = element("button", { type: "button", className: "sc-edit-save", textContent: "Save" });
  const cancel = element("button", { type: "button", className: "sc-edit-cancel", textContent: "Cancel" });
  const form = element("div", { className: "sc-edit-form" }, box,
    element("div", { className: "sc-edit-actions" }, cancel, save));
  const fit = () => { box.style.height = "auto"; box.style.height = `${box.scrollHeight}px`; };
  const close = () => { form.replaceWith(row); };
  const done = async () => {
    const content = box.value.trim();
    if (!content) { alert("Say something, or delete the message."); return; }
    if (content === message.content) { close(); return; }
    if (state.sending) { alert(`Wait for ${state.her?.name ?? "her"} to finish answering, then edit it.`); return; }
    const later = [];
    for (let next = item.nextElementSibling; next; next = next.nextElementSibling) later.push(next);
    const yours = later.filter((next) => next.classList.contains("sc-user")).length;
    if (yours && !confirm(`Saving sends it again, and the ${later.filter((n) => n.classList.contains("sc-message")).length} messages after it are deleted, ${yours} of them yours. Go on?`)) return;
    save.disabled = true;
    const reply = await fetch(`/api/messages/${message.id}`, {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ content }),
    }).catch(() => null);
    save.disabled = false;
    if (!reply?.ok) {
      const { error } = (await reply?.json().catch(() => null)) || {};
      alert(error || "That edit did not stick.");
      return;
    }
    item.replaceWith(messageElement((await reply.json()).message));
    later.forEach((next) => next.remove());
    resend(message.id);
  };
  box.addEventListener("input", fit);
  box.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); done(); }
    if (event.key === "Escape") { event.preventDefault(); close(); }
  });
  save.addEventListener("click", done);
  cancel.addEventListener("click", close);
  row.replaceWith(form);
  fit();
  box.focus();
  box.setSelectionRange(box.value.length, box.value.length);
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
      // The server has your message: it takes the place of the one still uploading.
      reply.pending?.remove();
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
    case "reaction":
      // She reacted to what you just sent.
      showReaction(list.querySelector(`.sc-message[data-id="${event.message_id}"]`), event.reaction);
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
    case "remembered":
      // What she saved for every chat: shown under her reply when it is done, which carries it.
      break;
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
  if (state.sending || !state.her || (!message && !state.photos.length)) return;
  state.sending = true;
  $("send").disabled = true;
  const form = new FormData();
  form.append("text", message);
  form.append("companion_id", state.her.id);
  if (state.chat) form.append("chat_id", state.chat.id);
  form.append("timezone", Intl.DateTimeFormat().resolvedOptions().timeZone || "");
  const photos = [...state.photos];
  photos.forEach((photo) => form.append("photo", photo));
  text.value = "";
  resize();
  clearPhoto();
  closeEmoji();
  const reply = replyElement();
  reply.element.remove();
  // A photo shows straight away, with how much of it has gone up.
  reply.pending = photos.length ? pendingPhoto(photos, message) : null;
  const read = eventReader(reply);
  try {
    const answer = await postStream("/api/send", form, (sent) => reply.pending?.progress(sent), read);
    if (answer.status !== 200) {
      let error = `HTTP ${answer.status}`;
      try { error = JSON.parse(answer.responseText).error || error; } catch {}
      throw new Error(error);
    }
  } catch (error) {
    if (reply.pending?.isConnected) {
      // It never arrived: the photo goes back into the composer to send again.
      reply.pending.remove();
      restorePhoto(photos, message);
    }
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

// Her answer's events as they stream in, each handled as it completes.
function eventReader(reply) {
  let buffer = "";
  return (piece) => {
    buffer += piece;
    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const chunk = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      if (chunk.startsWith("data: ")) handle(JSON.parse(chunk.slice(6)), reply);
    }
  };
}

// One of your messages sent again after you edited it: she answers its new words.
async function resend(id) {
  if (state.sending || !state.her || !state.chat) return;
  state.sending = true;
  $("send").disabled = true;
  const form = new FormData();
  form.append("companion_id", state.her.id);
  form.append("chat_id", state.chat.id);
  form.append("resend", id);
  form.append("timezone", Intl.DateTimeFormat().resolvedOptions().timeZone || "");
  const reply = replyElement();
  try {
    const answer = await postStream("/api/send", form, () => {}, eventReader(reply));
    if (answer.status !== 200) {
      let error = `HTTP ${answer.status}`;
      try { error = JSON.parse(answer.responseText).error || error; } catch {}
      throw new Error(error);
    }
  } catch (error) {
    handle({ type: "error", text: error.message }, reply);
    reply.typing.remove();
  } finally {
    state.sending = false;
    $("send").disabled = false;
    setPresence("online");
    loadChats();
  }
}

// POST a form and read the streamed answer as it comes. XMLHttpRequest, not fetch, because only it
// reports how much of an upload has gone; it resolves with the request once the answer is complete.
function postStream(url, form, onUpload, onText) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    let seen = 0;
    const more = () => {
      if (request.status !== 200) return;
      const all = request.responseText;
      if (all.length > seen) { onText(all.slice(seen)); seen = all.length; }
    };
    request.open("POST", url);
    request.upload.addEventListener("progress", (event) => { if (event.lengthComputable) onUpload(event.loaded / event.total); });
    request.upload.addEventListener("load", () => onUpload(1));
    request.addEventListener("progress", more);
    request.addEventListener("load", () => { more(); resolve(request); });
    request.addEventListener("error", () => reject(new Error("The connection dropped; your message was not sent.")));
    request.addEventListener("abort", () => reject(new Error("Sending was stopped.")));
    request.send(form);
  });
}

// Your photo in the chat while it uploads: a bar and how far it has got, then "saving" until the server has it.
function pendingPhoto(files, words) {
  const urls = files.map((file) => URL.createObjectURL(file));
  const total = files.reduce((sum, file) => sum + file.size, 0);
  const bar = element("i");
  const label = element("span", { textContent: "Uploading…" });
  const upload = element("div", { className: "sc-upload" }, element("div", { className: "sc-upload-bar" }, bar), label);
  upload.setAttribute("role", "progressbar");
  upload.setAttribute("aria-valuemin", "0");
  upload.setAttribute("aria-valuemax", "100");
  const figure = element("div", { className: "sc-figure" },
    ...urls.map((url) => element("img", { className: "sc-picture", src: url, alt: "Your photo, sending" })), upload);
  const node = element("div", { className: "sc-message sc-user sc-pending" }, figure,
    words ? element("div", { className: "sc-bubble-row" }, element("div", { className: "sc-bubble", textContent: words })) : null);
  list.append(node);
  $("empty").hidden = true;
  toBottom();
  node.progress = (fraction) => {
    const percent = Math.round(fraction * 100);
    bar.style.width = `${percent}%`;
    upload.setAttribute("aria-valuenow", String(percent));
    label.textContent = percent < 100 ? `Uploading ${percent}% of ${size(total)}` : "Saving…";
  };
  const remove = node.remove.bind(node);
  node.remove = () => { urls.forEach((url) => URL.revokeObjectURL(url)); remove(); };
  return node;
}

function size(bytes) {
  return bytes >= 1048576 ? `${(bytes / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

function restorePhoto(files, words) {
  state.photos = files;
  drawPhotos();
  if (words && !text.value) { text.value = words; resize(); }
}

// Emoji, on a desktop: a phone's keyboard has its own.
const EMOJI = ["😀", "😂", "🤣", "😊", "😍", "🥰", "😘", "😉", "😜", "😎", "🤗", "🤔", "🙈", "😏", "😌", "😴",
  "🥺", "😢", "😭", "😤", "😡", "😮", "😳", "🤭", "❤️", "💕", "💖", "💔", "🔥", "✨", "🌹", "🌙",
  "👍", "👎", "👏", "🙏", "💪", "👋", "🤞", "👀", "🎉", "🎂", "☕", "🍕", "🍫", "🍷", "🎵", "🌧️",
  "☀️", "🌸", "🐱", "🐶", "💋", "🫶", "😇", "🤤", "😋", "🥲", "🙃", "💤", "😅", "🤩", "😚", "💯"];

function closeEmoji() {
  $("emoji-panel").hidden = true;
  $("emoji-toggle").setAttribute("aria-expanded", "false");
}

function wireEmoji() {
  const panel = $("emoji-panel");
  panel.replaceChildren(...EMOJI.map((emoji) => {
    const button = element("button", { type: "button", textContent: emoji, title: emoji });
    button.addEventListener("click", () => {
      // At the cursor, replacing a selection, and the cursor after it.
      const start = text.selectionStart ?? text.value.length;
      const end = text.selectionEnd ?? start;
      text.setRangeText(emoji, start, end, "end");
      resize();
      text.focus();
    });
    return button;
  }));
  $("emoji-toggle").addEventListener("click", () => {
    const opening = panel.hidden;
    panel.hidden = !opening;
    $("emoji-toggle").setAttribute("aria-expanded", String(opening));
    if (opening) panel.querySelector("button")?.focus({ preventScroll: true });
  });
  document.addEventListener("click", (event) => {
    if (!panel.hidden && !event.target.closest("#emoji-panel, #emoji-toggle")) closeEmoji();
  });
  panel.addEventListener("keydown", (event) => { if (event.key === "Escape") { closeEmoji(); text.focus(); } });
}

// Voice: her replies aloud, and voice messages through the server's Whisper.
let playing = null;
async function speak(id) {
  playing?.pause();
  const audio = new Audio(`/api/speech/${id}`);
  playing = audio;
  await onSpeaker(audio);
  if (playing === audio) audio.play().catch(() => {});
}

// Speaker: where her voice plays, chosen in this browser when it finds more than one. Chrome and Edge can play on
// a chosen speaker (setSinkId); other browsers play on the default. Browsers name their speakers only once the page
// may use the microphone, so until then "Find speakers" asks for it.
const CAN_PICK_SPEAKER = "setSinkId" in HTMLMediaElement.prototype && !!navigator.mediaDevices?.enumerateDevices;
const NOT_A_SPEAKER = new Set(["", "default", "communications"]);

async function onSpeaker(audio) {
  const id = stored("socia.speaker", "");
  if (!CAN_PICK_SPEAKER || !id) return;
  try { await audio.setSinkId(id); } catch { /* not connected now: the default speaker plays it */ }
}

async function findSpeakers() {
  const devices = await navigator.mediaDevices.enumerateDevices().catch(() => []);
  const outputs = devices.filter((device) => device.kind === "audiooutput");
  return {
    named: outputs.some((device) => device.label),
    speakers: outputs.filter((device) => !NOT_A_SPEAKER.has(device.deviceId)),
  };
}

async function drawSpeakers() {
  const box = $("speaker-box");
  if (!CAN_PICK_SPEAKER) { box.hidden = true; return; }
  const { named, speakers } = await findSpeakers();
  const select = $("speaker");
  const note = $("speaker-note");
  if (!named) {
    box.hidden = false;
    select.parentElement.hidden = true;
    $("speaker-find").hidden = false;
    note.textContent = "Your browser names its speakers once Socia may use the microphone.";
    return;
  }
  $("speaker-find").hidden = true;
  if (speakers.length < 2) { box.hidden = true; return; }
  const chosen = stored("socia.speaker", "");
  select.replaceChildren(
    element("option", { value: "", textContent: "The default speaker" }),
    ...speakers.map((speaker) => element("option", { value: speaker.deviceId, textContent: speaker.label })),
  );
  const connected = speakers.some((speaker) => speaker.deviceId === chosen);
  select.value = connected ? chosen : "";
  select.parentElement.hidden = false;
  box.hidden = false;
  note.textContent = chosen && !connected
    ? "The speaker you chose is not connected, so her voice plays on the default."
    : "Her replies and voice samples play here, in this browser.";
}

function wireSpeakers() {
  if (!CAN_PICK_SPEAKER) return;
  $("speaker").addEventListener("change", () => {
    const id = $("speaker").value;
    store("socia.speaker", id);
    // What is playing moves over at once.
    for (const audio of [playing, sample?.audio]) if (audio) audio.setSinkId(id).catch(() => {});
    drawSpeakers();
  });
  $("speaker-find").addEventListener("click", async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach((track) => track.stop());
    } catch {
      $("speaker-note").textContent = "Without the microphone, your browser keeps its speakers' names hidden.";
      return;
    }
    drawSpeakers();
  });
  navigator.mediaDevices.addEventListener?.("devicechange", () => { if ($("me").open) drawSpeakers(); });
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
// Photos to send: up to as many as she sees at once, each shown with a button to take it out.
const PHOTOS_AT_ONCE = 3;
state.photos = [];

function drawPhotos() {
  const box = $("attachment");
  [...box.querySelectorAll("img")].forEach((image) => URL.revokeObjectURL(image.src));
  box.replaceChildren(...state.photos.map((file, index) => {
    const remove = element("button", { type: "button", textContent: "×" });
    remove.setAttribute("aria-label", "Remove this photo");
    remove.addEventListener("click", () => { state.photos.splice(index, 1); drawPhotos(); });
    return element("div", { className: "sc-attachment" },
      element("img", { src: URL.createObjectURL(file), alt: "Photo to send" }), remove);
  }));
  box.hidden = !state.photos.length;
}

function addPhotos(files) {
  const room = PHOTOS_AT_ONCE - state.photos.length;
  state.photos.push(...files.slice(0, Math.max(room, 0)));
  if (files.length > room) setPresence(`${PHOTOS_AT_ONCE} photos at most at once`);
  drawPhotos();
}

function clearPhoto() {
  state.photos = [];
  $("photo").value = "";
  drawPhotos();
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
    ? "Her face in every picture she takes."
    : "None yet. It becomes her face in every picture she takes.";
}

// An avatar's crop: the square's centre (x, y as fractions of the picture) and zoom (1 = the largest
// square the picture holds). The server cuts the avatar from it. One cropper each, for her and for you:
// `prefix` names its elements (crop, crop-stage, crop-image, crop-zoom, crop-open, crop-cancel, crop-save,
// crop-reset), `api` gives its crop's address and `picture` the picture's.
const CROP_MAX_ZOOM = 6;

function cropper(prefix, { api, picture, note, saved, changed }) {
  const crop = { width: 1, height: 1, x: 0.5, y: 0.5, zoom: 1, drag: null };
  const el = (name) => $(name ? `${prefix}-${name}` : prefix);
  const side = () => Math.min(crop.width, crop.height) / crop.zoom;

  function clamp() {
    crop.zoom = Math.min(Math.max(crop.zoom, 1), CROP_MAX_ZOOM);
    const halfX = side() / 2 / crop.width;
    const halfY = side() / 2 / crop.height;
    crop.x = Math.min(Math.max(crop.x, halfX), 1 - halfX);
    crop.y = Math.min(Math.max(crop.y, halfY), 1 - halfY);
  }

  // Place the picture so the crop's square fills the stage.
  function draw() {
    const image = el("image");
    const scale = el("stage").clientWidth / side();
    image.style.width = `${crop.width * scale}px`;
    image.style.height = `${crop.height * scale}px`;
    image.style.left = `${-(crop.x * crop.width - side() / 2) * scale}px`;
    image.style.top = `${-(crop.y * crop.height - side() / 2) * scale}px`;
    el("zoom").value = String(crop.zoom);
  }

  function take(answer) {
    Object.assign(crop, { width: answer.width, height: answer.height, x: answer.x, y: answer.y, zoom: answer.zoom });
    clamp();
    draw();
  }

  async function open() {
    const reply = await fetch(api()).catch(() => null);
    if (!reply?.ok) return;
    const answer = await reply.json();
    el("image").onload = () => take(answer);
    el("image").src = `${picture()}?t=${Date.now()}`;
    el().hidden = false;
    el("open").hidden = true;
    el("stage").focus();
  }

  function close() {
    el().hidden = true;
    el("open").hidden = false;
  }

  async function request(method, body) {
    const reply = await fetch(api(), {
      method, headers: body ? { "Content-Type": "application/json" } : {}, body: body ? JSON.stringify(body) : undefined,
    }).catch(() => null);
    if (!reply?.ok) {
      $(note).textContent = "The crop could not be saved.";
      return null;
    }
    changed();
    return reply.json();
  }

  function wire() {
    const stage = el("stage");
    stage.addEventListener("pointerdown", (event) => {
      stage.setPointerCapture(event.pointerId);
      crop.drag = { pointer: event.pointerId, startX: event.clientX, startY: event.clientY, x: crop.x, y: crop.y };
    });
    stage.addEventListener("pointermove", (event) => {
      if (!crop.drag || crop.drag.pointer !== event.pointerId) return;
      const scale = stage.clientWidth / side();
      crop.x = crop.drag.x - (event.clientX - crop.drag.startX) / scale / crop.width;
      crop.y = crop.drag.y - (event.clientY - crop.drag.startY) / scale / crop.height;
      clamp();
      draw();
    });
    const endDrag = () => { crop.drag = null; };
    stage.addEventListener("pointerup", endDrag);
    stage.addEventListener("pointercancel", endDrag);
    stage.addEventListener("wheel", (event) => {
      event.preventDefault();
      crop.zoom *= Math.exp(-event.deltaY * 0.0015);
      clamp();
      draw();
    }, { passive: false });
    stage.addEventListener("keydown", (event) => {
      const step = 0.02 / crop.zoom;
      const moves = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
      if (event.key === "+" || event.key === "=") crop.zoom *= 1.1;
      else if (event.key === "-") crop.zoom /= 1.1;
      else if (moves[event.key]) { crop.x += moves[event.key][0]; crop.y += moves[event.key][1]; }
      else return;
      event.preventDefault();
      clamp();
      draw();
    });
    el("zoom").addEventListener("input", () => {
      crop.zoom = Number(el("zoom").value);
      clamp();
      draw();
    });
    el("open").addEventListener("click", open);
    el("cancel").addEventListener("click", close);
    el("save").addEventListener("click", async () => {
      if (await request("PUT", { x: crop.x, y: crop.y, zoom: crop.zoom })) {
        $(note).textContent = saved;
        close();
      }
    });
    el("reset").addEventListener("click", async () => {
      const answer = await request("DELETE");
      if (answer) take(answer);
    });
    window.addEventListener("resize", () => { if (!el().hidden) draw(); });
  }

  return { wire, close };
}

const herCrop = cropper("crop", {
  api: () => `${herApi()}/avatar/crop`, picture: () => `${herApi()}/profile`, note: "profile-note",
  saved: "Her avatar is updated.", changed: () => { $("avatar").src = `${herApi()}/avatar?t=${Date.now()}`; },
});
const myCrop = cropper("me-crop", {
  api: () => "/api/owner/avatar/crop", picture: () => "/api/owner/picture", note: "me-note",
  saved: "Your avatar is updated.", changed: () => loadMe(),
});

function wireCrop() {
  herCrop.wire();
  myCrop.wire();
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
    const promote = document.createElement("button");
    promote.type = "button";
    promote.className = "sc-promote";
    promote.textContent = "★";
    promote.title = "Make this her profile picture";
    promote.setAttribute("aria-label", "Make this her profile picture");
    promote.addEventListener("click", async () => {
      if (!confirm("Make this her profile picture? Her current one becomes one of her other pictures. "
        + "Nude pictures go off until you confirm her new face again.")) return;
      const reply = await fetch(`${reference.url}/profile`, { method: "POST" }).catch(() => null);
      $("profile-note").textContent = reply?.ok ? "Her profile picture is changed." : "That did not work.";
      if (!reply?.ok) return;
      loadProfile();
      loadReferences();
      state.companions = [];
      drawSwitcher();
    });
    item.append(image, promote, remove);
    return item;
  }));
  $("references-add").hidden = references.length >= max;
  $("references-count").textContent = `${references.length} / ${max}`;
  $("references-note").textContent = note || "Her face clear, in different poses. Each picture uses the ones used least lately; ones in pictures you ❤️ come back sooner.";
}

// Photos of the owner's face, for face recognition in the photos they send her.
async function loadOwnerPhotos(note) {
  const reply = await fetch("/api/owner/photos").catch(() => null);
  if (!reply?.ok) return;
  const { photos, max } = await reply.json();
  $("owner-photos").replaceChildren(...photos.map((photo) => {
    const item = document.createElement("li");
    const image = picture(photo.url);
    image.className = "";
    image.alt = "A photo of you";
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "×";
    remove.setAttribute("aria-label", "Remove this photo");
    remove.addEventListener("click", async () => {
      if (!confirm("Stop using this photo of you?")) return;
      await fetch(photo.url, { method: "DELETE" });
      loadOwnerPhotos();
    });
    const promote = document.createElement("button");
    promote.type = "button";
    promote.className = "sc-promote";
    promote.textContent = "★";
    promote.title = "Make this your picture";
    promote.setAttribute("aria-label", "Make this your picture");
    promote.addEventListener("click", async () => {
      if (!confirm("Make this your picture? Your current one becomes one of these photos.")) return;
      const reply = await fetch(`${photo.url}/profile`, { method: "POST" }).catch(() => null);
      $("me-picture-note").textContent = reply?.ok ? "Your picture is changed." : "That did not work.";
      if (!reply?.ok) return;
      loadMe();
      loadOwnerPhotos();
    });
    item.append(image, promote, remove);
    return item;
  }));
  $("owner-photos-add").hidden = photos.length >= max;
  $("owner-photos-count").textContent = `${photos.length} / ${max}`;
  $("owner-photos-note").textContent = note || (photos.length >= 3
    ? `${photos.length} of ${max}. She knows your face in the photos you send.`
    : "So she knows you in the photos you send: 3 to 5 clear ones, from the front and a little to each side.");
}

// You: your picture at the top right, the name she calls you, what you wrote about yourself, photos of you.
async function loadMe(note) {
  const reply = await fetch("/api/owner").catch(() => null);
  if (!reply?.ok) return;
  const me = await reply.json();
  const stamp = Date.now();
  document.querySelectorAll("[data-me]").forEach((button) => {
    const image = button.querySelector("img");
    image.hidden = !me.has_picture;
    if (me.has_picture) image.src = `/api/owner/avatar?t=${stamp}`;
    button.querySelector(".sc-me-initial").textContent = me.has_picture ? "" : (me.called || "You").slice(0, 1).toUpperCase();
  });
  $("me-picture").hidden = !me.has_picture;
  if (me.has_picture) $("me-picture").src = `/api/owner/picture?t=${stamp}`;
  $("me-picture-remove").hidden = !me.has_picture;
  if (!me.has_picture) myCrop.close();
  $("me-crop-open").hidden = !me.has_picture || !$("me-crop").hidden;
  $("me-picture-label").textContent = me.has_picture ? "Change picture" : "Add your picture";
  $("me-name").value = me.name;
  $("me-name").placeholder = me.signed_in_as || "Your name";
  $("me-name").maxLength = me.limits.name;
  $("me-about").value = me.about;
  $("me-about").maxLength = me.limits.about;
  if (note !== undefined) $("me-note").textContent = note;
}

function wireMe() {
  const dialog = $("me");
  document.querySelectorAll("[data-me]").forEach((button) => button.addEventListener("click", () => {
    $("me-note").textContent = "";
    loadMe();
    loadOwnerPhotos();
    // The settings for the whole app are yours: the speaker, your Luna apps, and the models.
    loadStatus();
    showApps();
    drawSpeakers();
    dialog.showModal();
  }));
  $("me-save").addEventListener("click", async () => {
    const reply = await fetch("/api/owner", {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: $("me-name").value, about: $("me-about").value }),
    }).catch(() => null);
    loadMe(reply?.ok ? "Saved. She reads it from her next reply." : "That could not be saved.");
  });
  $("me-picture-file").addEventListener("change", async () => {
    const file = $("me-picture-file").files[0];
    if (!file) return;
    const form = new FormData();
    form.append("photo", file);
    $("me-note").textContent = "Saving…";
    myCrop.close();
    const reply = await fetch("/api/owner/picture", { method: "POST", body: form }).catch(() => null);
    const answer = await reply?.json().catch(() => ({})) || {};
    $("me-picture-file").value = "";
    loadMe(reply?.ok ? answer.note || "Saved." : answer.error || "That picture could not be saved.");
  });
  $("me-picture-remove").addEventListener("click", async () => {
    if (!confirm("Remove your picture?")) return;
    await fetch("/api/owner/picture", { method: "DELETE" }).catch(() => null);
    loadMe("Removed.");
  });
}

async function loadStatus() {
  const reply = await fetch("/api/status").catch(() => null);
  if (!reply?.ok) return;
  const status = await reply.json();
  // Every model she runs on: what it does for her, its name, and where it runs.
  $("models").replaceChildren(...(status.models || []).map((model) => element("div", {},
    element("dt", { textContent: model.role }),
    element("dd", {}, element("span", { textContent: model.name }),
      element("small", { className: model.ok === false ? "sc-off" : "", textContent: model.where })))));
}

function wire() {
  $("composer").addEventListener("submit", (event) => { event.preventDefault(); send(); });
  text.addEventListener("input", resize);
  text.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); send(); }
  });
  $("photo").addEventListener("change", () => {
    const files = [...$("photo").files];
    $("photo").value = "";
    if (!files.length) return;
    addPhotos(files);
    text.focus();
  });
  // A picture pasted from the clipboard (a screenshot, an image copied from a page) joins the photos to
  // send, like one chosen with the button. Text pasted with it still goes into the message box.
  document.addEventListener("paste", (event) => {
    if ($("chat").hidden || document.querySelector("dialog[open]")) return;
    const pictures = [...(event.clipboardData?.items ?? [])]
      .filter((item) => item.kind === "file" && item.type.startsWith("image/"))
      .map((item) => item.getAsFile())
      .filter(Boolean)
      // Every clipboard picture is called image.png: give each its own name.
      .map((file, i) => new File([file], `pasted-${Date.now()}-${i}.${file.type.split("/")[1] || "png"}`, { type: file.type }));
    if (!pictures.length) return;
    if (!event.clipboardData.getData("text/plain").trim()) event.preventDefault();
    addPhotos(pictures);
    text.focus();
  });
  wireEmoji();
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
  const openMenu = () => { loadReferences(); menu.showModal(); };
  wireSpeakers();
  $("owner-photos-file").addEventListener("change", async () => {
    const files = [...$("owner-photos-file").files];
    if (!files.length) return;
    const form = new FormData();
    files.forEach((file) => form.append("photos", file));
    $("owner-photos-note").textContent = "Looking for your face…";
    const reply = await fetch("/api/owner/photos", { method: "POST", body: form });
    const answer = await reply.json().catch(() => ({}));
    $("owner-photos-file").value = "";
    loadOwnerPhotos(reply.ok ? answer.note || "" : answer.error || "Those photos could not be saved.");
  });
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
    if (event.key === "Escape") closeReactions();
  });
  // The reactions row closes on a tap anywhere else.
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".sc-reactions, .sc-react")) closeReactions();
  });
  $("to-home").addEventListener("click", () => go(""));
  $("new-chat").addEventListener("click", newChat);
  $("drawer-new-chat").addEventListener("click", newChat);

  wireEditor();
  // A fresh list of them whenever the view changes, so cards and the switcher stay current.
  window.addEventListener("hashchange", () => { state.companions = []; route(); });
}

/* Her phone: the owner's Luna apps, which she can use once they are connected. */

async function showApps() {
  const button = $("apps");
  const list = $("apps-list");
  if (!list.children.length) list.replaceChildren(element("li", { className: "sc-muted", textContent: "Checking…" }));
  let status;
  try { status = await json("/api/apps"); } catch { return; }
  const running = status.apps.filter((app) => app.running);
  // The home page's button.
  button.textContent = status.connected ? `Luna apps: ${running.length} running` : "Connect Luna apps";
  button.title = status.connected
    ? `She can use ${running.map((app) => app.name.replace("Luna ", "")).join(", ") || "none of them right now"}. Select to disconnect.`
    : "Let her use your Luna apps: music, lights, photos, meals, money, documents";
  button.dataset.connected = status.connected ? "yes" : "";
  button.hidden = false;
  // The settings menu's list: every app, and whether she can use it now.
  $("apps-note").textContent = status.connected
    ? "She can use the running ones for you, as you, when you ask."
    : "Not connected: connect them, then allow Luna Socia on Luna Sentia's sign-in page, once.";
  list.replaceChildren(...status.apps.map((app) => {
    const item = element("li", { className: `sc-luna-app${status.connected && app.running ? " sc-luna-app-on" : ""}` });
    const label = !status.connected ? "not connected" : app.running ? "running" : "not running";
    item.append(
      appLogo(app.key, "sc-luna-app-logo"),
      element("span", { className: "sc-luna-app-text" },
        element("strong", { textContent: app.name }),
        element("small", { textContent: app.purpose })),
      element("span", { className: "sc-luna-app-state", textContent: label }));
    return item;
  }));
  $("apps-menu-toggle").textContent = status.connected ? "Disconnect Luna apps" : "Connect Luna apps";
  $("apps-menu-toggle").dataset.connected = button.dataset.connected;
}

async function toggleApps(event) {
  if (!event.currentTarget.dataset.connected) {
    location.href = "/apps/connect";
    return;
  }
  if (!confirm("Disconnect your Luna apps? She will no longer be able to use them.")) return;
  await fetch("/api/apps", { method: "DELETE" });
  showApps();
}

$("apps").addEventListener("click", toggleApps);
$("apps-menu-toggle").addEventListener("click", toggleApps);

wire();
wireMe();
route();
showApps();
loadMe();
