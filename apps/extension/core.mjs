export function applyEdits(text, edits, selected) {
  const points = Array.from(text);
  let cursor = 0;
  const output = [];
  for (const edit of [...edits].sort((a, b) => a.start - b.start || a.end - b.end)) {
    if (!selected.has(edit.id)) continue;
    if (edit.start < cursor || points.slice(edit.start, edit.end).join("") !== edit.source) {
      throw new Error("Phép sửa không khớp văn bản gốc. Hãy kiểm tra lại.");
    }
    output.push(points.slice(cursor, edit.start).join(""), edit.replacement);
    cursor = edit.end;
  }
  return output.join("") + points.slice(cursor).join("");
}

export function endpoint(value) {
  const url = new URL(value.trim());
  if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.search || url.hash || !["", "/"].includes(url.pathname)) {
    throw new Error("Địa chỉ API cần có dạng https://host hoặc http://IP:8000.");
  }
  return url.origin;
}

export async function request(base, path, key, body, multipart = false) {
  const headers = {};
  if (key) headers["X-API-Key"] = key;
  if (body && !multipart) headers["Content-Type"] = "application/json";
  let response;
  try {
    response = await fetch(endpoint(base) + path, {
      method: body ? "POST" : "GET", headers,
      body: body ? (multipart ? body : JSON.stringify(body)) : undefined,
      signal: AbortSignal.timeout(300000),
    });
  } catch (error) {
    throw new Error(error.name === "TimeoutError" ? "Yêu cầu hết thời gian chờ. Hãy chia văn bản nhỏ hơn." : "Không kết nối được API. Kiểm tra địa chỉ, quyền kết nối và máy chủ.");
  }
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Dữ liệu hoặc cấu hình không hợp lệ.");
  return data;
}
