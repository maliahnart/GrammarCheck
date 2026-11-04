import {applyEdits, endpoint, request} from "./core.mjs";

const $ = (id) => document.getElementById(id);
let mode = "document", result = null, selected = new Set(), target = null, busy = false;
const message = (text, error = false) => { $("status").textContent = text; $("status").className = error ? "error" : ""; };
const text = (tag, value, parent) => { const node = document.createElement(tag); node.textContent = value; parent.append(node); return node; };
const options = () => ({model: $("model").value, preserve_style: $("preserve").checked,
  normalize_abbrev: $("abbrev").checked, protected_terms: $("terms").value.split("\n").map(s => s.trim()).filter(Boolean)});
const call = (path, body, multipart = false) => request($("endpoint").value, path, $("api-key").value, body, multipart);

async function task(action) {
  if (busy) return;
  busy = true;
  const controls = [...document.querySelectorAll("button, input, select, textarea:not([readonly]), summary")];
  controls.forEach(node => { node.disabled = true; });
  message("Đang xử lý…");
  try { await action(); } catch (error) { message(error.message, true); }
  finally { busy = false; controls.forEach(node => { node.disabled = false; }); }
}

function reset() { result = null; $("results").hidden = true; $("replace").hidden = true; $("count").textContent = `${Array.from($("source").value).length.toLocaleString("vi")} / 30.000 ký tự`; }
$("source").addEventListener("input", () => { reset(); target = null; });
for (const node of [$("model"), $("preserve"), $("abbrev"), $("terms")]) node.addEventListener("change", reset);
for (const button of document.querySelectorAll("[data-mode]")) button.onclick = () => {
  if (busy) return;
  mode = button.dataset.mode;
  for (const tab of document.querySelectorAll("[data-mode]")) tab.setAttribute("aria-pressed", String(tab === button));
  $("editor").hidden = mode === "dataset"; $("dataset").hidden = mode !== "dataset"; $("learning").hidden = mode !== "learn";
  $("intro").textContent = {document: "Rà soát văn bản, giữ cách diễn đạt của bạn.", learn: "Tìm hiểu chính tả, từ vựng, ngữ pháp và cách giao tiếp.", dataset: "Chuẩn hóa có kiểm soát, luôn giữ bản gốc."}[mode];
  reset(); message("");
};

$("save").onclick = () => task(async () => {
  const base = endpoint($("endpoint").value);
  const url = new URL(base);
  const granted = await chrome.permissions.request({origins: [`${url.protocol}//${url.hostname}/*`]});
  if (!granted) throw new Error("Chưa được cấp quyền kết nối đến API này.");
  await request(base, "/health", $("api-key").value);
  await chrome.storage.local.set({endpoint: base, options: options()});
  await chrome.storage.session.set({apiKey: $("api-key").value});
  message("Đã lưu. Kết nối API hoạt động.");
});

function renderResult() {
  $("output").value = applyEdits(result.input, result.edits, selected);
  $("replace").hidden = !target || target.source !== result.input;
}

$("check").onclick = () => task(async () => {
  if (!$("source").value.trim()) throw new Error("Vui lòng nhập văn bản.");
  reset();
  result = await call("/v1/check", {text: $("source").value, mode, ...options()});
  selected = new Set(result.edits.filter(edit => edit.accepted).map(edit => edit.id));
  $("edits").replaceChildren();
  for (const edit of result.edits) {
    const card = document.createElement("article"); card.className = "edit"; $("edits").append(card);
    const label = document.createElement("label"), cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = selected.has(edit.id);
    label.append(cb); text("span", `${edit.source || "∅"} → ${edit.replacement || "∅"}`, label); card.append(label);
    text("p", edit.explanation + (edit.reason ? ` · ${edit.reason}` : ""), card);
    if (mode === "learn" && edit.lesson_id) {
      const button = text("button", "Xem bài học liên quan", card); button.className = "quiet";
      button.onclick = () => task(async () => { await loadLessons(edit.lesson_id); message("Đã mở bài học liên quan."); });
    }
    cb.onchange = () => { cb.checked ? selected.add(edit.id) : selected.delete(edit.id); renderResult(); };
  }
  $("results").hidden = false; renderResult();
  message(`${result.edits.length} đề xuất · ${result.chunks} đoạn · ${result.seconds.toFixed(1)} giây. Bạn có thể duyệt từng thay đổi.`);
});

$("copy").onclick = () => task(async () => { await navigator.clipboard.writeText($("output").value); message("Đã sao chép bản sửa đang chọn."); });
$("expand").onclick = () => chrome.tabs.create({url: chrome.runtime.getURL("index.html")});
$("selection").onclick = () => task(async () => {
  const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
  const [response] = await chrome.scripting.executeScript({target: {tabId: tab.id}, func: () => {
    const element = document.activeElement;
    if (element instanceof HTMLInputElement && !["text", "search"].includes(element.type)) return {text: "", editable: false};
    if ((element instanceof HTMLTextAreaElement || element instanceof HTMLInputElement) && element.selectionStart !== null) {
      const start = element.selectionStart, end = element.selectionEnd, original = element.value.slice(start, end);
      window.__grammarcheckTarget = {element, start, end, original, value: element.value};
      return {text: original, editable: !element.readOnly && !element.disabled};
    }
    const selection = window.getSelection(), original = selection?.toString() || "";
    const range = selection?.rangeCount ? selection.getRangeAt(0).cloneRange() : null;
    const parent = range?.commonAncestorContainer.nodeType === Node.ELEMENT_NODE ? range.commonAncestorContainer : range?.commonAncestorContainer.parentElement;
    const editable = Boolean(parent?.closest('[contenteditable="true"]'));
    window.__grammarcheckTarget = {range, original};
    return {text: original, editable};
  }});
  const captured = response.result;
  if (!captured?.text) throw new Error("Hãy chọn một đoạn văn bản trên trang trước khi mở extension.");
  if (Array.from(captured.text).length > 30000) throw new Error("Đoạn đã chọn vượt quá 30.000 ký tự.");
  $("source").value = captured.text; reset();
  target = captured.editable ? {tabId: tab.id, source: captured.text} : null;
  message(target ? "Đã lấy đoạn chọn. Sau khi kiểm tra, bạn có thể duyệt rồi thay thế." : "Đã lấy văn bản. Có thể sao chép kết quả sau khi kiểm tra.");
});

$("replace").onclick = () => task(async () => {
  if (!target || result.input !== target.source || $("source").value !== target.source) throw new Error("Văn bản đã thay đổi; hãy lấy lại đoạn chọn.");
  const [response] = await chrome.scripting.executeScript({target: {tabId: target.tabId}, args: [$("output").value, target.source], func: (replacement, expected) => {
    const t = window.__grammarcheckTarget;
    if (!t || t.original !== expected) return false;
    if (t.element) {
      if (!t.element.isConnected || t.element.value !== t.value || t.element.readOnly || t.element.disabled) return false;
      t.element.setRangeText(replacement, t.start, t.end, "select");
      t.element.dispatchEvent(new InputEvent("input", {bubbles: true, inputType: "insertReplacementText", data: replacement}));
    } else {
      if (!t.range || !t.range.startContainer.isConnected || t.range.toString() !== expected) return false;
      const container = t.range.commonAncestorContainer.nodeType === Node.ELEMENT_NODE ? t.range.commonAncestorContainer : t.range.commonAncestorContainer.parentElement;
      const editor = container?.closest('[contenteditable="true"]');
      if (!editor) return false;
      t.range.deleteContents(); t.range.insertNode(document.createTextNode(replacement));
      editor.dispatchEvent(new InputEvent("input", {bubbles: true, inputType: "insertReplacementText", data: replacement}));
    }
    delete window.__grammarcheckTarget; return true;
  }});
  if (!response.result) throw new Error("Nội dung trên trang đã thay đổi. Hãy chọn lại để tránh ghi đè.");
  target = null; $("replace").hidden = true; message("Đã thay đoạn văn bản bạn chọn.");
});

$("text-file").onchange = () => task(async () => {
  const file = $("text-file").files[0]; if (!file) return;
  if (file.size > 200000) throw new Error("Tệp TXT quá lớn. Giới hạn văn bản là 30.000 ký tự.");
  const value = await file.text(); if (Array.from(value).length > 30000) throw new Error("Văn bản vượt quá 30.000 ký tự.");
  $("source").value = value; target = null; reset(); message("Đã mở TXT. Bấm kiểm tra để gửi đến API.");
});

$("clean").onclick = () => task(async () => {
  const file = $("data-file").files[0]; if (!file) throw new Error("Chọn tệp CSV hoặc JSONL.");
  if (file.size > 2 * 1024 * 1024) throw new Error("Tệp vượt quá 2 MB.");
  const form = new FormData(); form.append("file", file); form.append("column", $("column").value);
  for (const [key, value] of Object.entries(options())) form.append(key, Array.isArray(value) ? JSON.stringify(value) : String(value));
  const data = await call("/v1/datasets/clean", form, true);
  const url = URL.createObjectURL(new Blob([data.content], {type: data.content_type + ";charset=utf-8"}));
  const anchor = document.createElement("a"); anchor.href = url; anchor.download = data.filename; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  message(`Đã xử lý ${data.rows} bản ghi; ${data.changed_rows} bản ghi thay đổi. Bản gốc được giữ trong tệp kết quả.`);
});

async function loadLessons(focus) {
  const data = await call("/v1/lessons"); $("lessons").replaceChildren();
  for (const lesson of data.lessons) {
    const card = document.createElement("article"); card.className = "lesson"; card.id = `lesson-${lesson.id}`; $("lessons").append(card);
    text("small", lesson.category, card); text("h2", lesson.title, card); text("p", lesson.explanation, card);
    for (const example of lesson.examples) text("p", example, card);
    text("strong", lesson.question, card); const feedback = document.createElement("p"); feedback.setAttribute("role", "status");
    lesson.choices.forEach((choice, index) => { const button = text("button", choice, card); button.onclick = () => { feedback.textContent = (index === lesson.answer ? "Đúng. " : "Chưa đúng. ") + lesson.feedback; }; });
    card.append(feedback);
  }
  if (focus) $("lesson-" + focus)?.scrollIntoView({behavior: "smooth"});
}
$("load-lessons").onclick = () => task(async () => { await loadLessons(); message("Đã mở thư viện học tiếng Việt."); });

try {
  const [saved, session] = await Promise.all([chrome.storage.local.get(["endpoint", "options"]), chrome.storage.session.get(["apiKey", "pendingText"])]);
  if (saved.endpoint) $("endpoint").value = saved.endpoint;
  $("api-key").value = session.apiKey || "";
  if (saved.options) { $("model").value = saved.options.model; $("preserve").checked = saved.options.preserve_style; $("abbrev").checked = saved.options.normalize_abbrev; $("terms").value = saved.options.protected_terms.join("\n"); }
  if (session.pendingText) { $("source").value = session.pendingText; reset(); await chrome.storage.session.remove("pendingText"); }
} catch { message("Chưa đọc được thiết lập; bạn có thể nhập kết nối mới.", true); }
