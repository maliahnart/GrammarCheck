import test from "node:test";
import assert from "node:assert/strict";
import {applyEdits, endpoint, request} from "./core.mjs";

test("Unicode offsets, rejected suggestions, and untouched paragraphs", () => {
  const source = "🙂 Toi\n\nđi hoc.";
  const edits = [{id: "a", start: 2, end: 5, source: "Toi", replacement: "Tôi"},
    {id: "b", start: 10, end: 13, source: "hoc", replacement: "học"}];
  assert.equal(applyEdits(source, edits, new Set(["a", "b"])), "🙂 Tôi\n\nđi học.");
  assert.equal(applyEdits(source, edits, new Set()), source);
});
test("stale edits and overlapping spans are rejected", () => {
  assert.throws(() => applyEdits("abc", [{id: "a", start: 0, end: 2, source: "ab", replacement: "x"},
    {id: "b", start: 1, end: 3, source: "bc", replacement: "y"}], new Set(["a", "b"])));
});
test("connection settings reject credentials and URL paths", () => {
  assert.equal(endpoint("http://127.0.0.1:8000/"), "http://127.0.0.1:8000");
  for (const url of ["file:///tmp/x", "https://user:pass@host", "https://host/path"]) assert.throws(() => endpoint(url));
});
test("HTTP errors surface server explanations", async () => {
  const previous = globalThis.fetch;
  globalThis.fetch = async () => ({ok: false, json: async () => ({detail: "Chưa có checkpoint."})});
  try { await assert.rejects(request("http://localhost:8000", "/v1/check", "", {text: "x"}), /checkpoint/); }
  finally { globalThis.fetch = previous; }
});
