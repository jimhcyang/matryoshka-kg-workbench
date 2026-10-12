/** Matryoshka visitor-registration endpoint for a spreadsheet-bound script. */

const VISITOR_SHEET_NAME = "Visitor Events";
const VISITOR_HEADERS = [
  "recorded_at_utc",
  "event_type",
  "action",
  "email",
  "page_url",
  "route",
  "target_type",
  "target_id",
  "detail",
  "referrer",
  "timezone",
  "language",
  "visitor_id",
  "session_id",
  "registration_version",
  "request_id",
];
const ALLOWED_PAGE_PREFIXES = [
  "https://jimhcyang.github.io/matryoshka-kg-workbench/",
  "http://127.0.0.1:",
  "http://localhost:",
];

function setupVisitorLog() {
  const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  if (!spreadsheet) {
    throw new Error("Open this script from Extensions > Apps Script in the destination Google Sheet.");
  }
  PropertiesService.getScriptProperties().setProperty("VISITOR_SPREADSHEET_ID", spreadsheet.getId());
  const sheet = visitorSheet_(spreadsheet);
  sheet.setFrozenRows(1);
  sheet.getRange(1, 1, 1, VISITOR_HEADERS.length).setFontWeight("bold").setBackground("#13242a").setFontColor("#fffaf0");
  sheet.autoResizeColumns(1, VISITOR_HEADERS.length);
  return "Visitor log ready: " + spreadsheet.getUrl();
}

function doGet() {
  return jsonResponse_({ok: true, service: "matryoshka_visitor_registration", version: "2"});
}

function doPost(event) {
  let requestId = "";
  try {
    const payload = parsePayload_(event);
    requestId = payload.request_id;
    if (payload.website) return registrationResponse_({ok: true, request_id: requestId});
    validatePayload_(payload);

    const cache = CacheService.getScriptCache();
    const cacheKey = "request:" + sha256_(payload.request_id);
    if (cache.get(cacheKey)) return registrationResponse_({ok: true, duplicate: true, request_id: requestId});

    const lock = LockService.getScriptLock();
    if (!lock.tryLock(10000)) throw new Error("Visitor log is busy; retry shortly.");
    try {
      const spreadsheetId = PropertiesService.getScriptProperties().getProperty("VISITOR_SPREADSHEET_ID");
      if (!spreadsheetId) throw new Error("Run setupVisitorLog before deploying the web app.");
      const sheet = visitorSheet_(SpreadsheetApp.openById(spreadsheetId));
      sheet.appendRow([
        new Date().toISOString(),
        safeCell_(payload.event_type, 40),
        safeCell_(payload.action, 80),
        safeCell_(payload.email, 254),
        safeCell_(payload.page_url, 1000),
        safeCell_(payload.route, 160),
        safeCell_(payload.target_type, 80),
        safeCell_(payload.target_id, 240),
        safeCell_(payload.detail, 1200),
        safeCell_(payload.referrer, 1000),
        safeCell_(payload.timezone, 100),
        safeCell_(payload.language, 50),
        safeCell_(payload.visitor_id, 120),
        safeCell_(payload.session_id, 120),
        safeCell_(payload.registration_version, 100),
        safeCell_(payload.request_id, 120),
      ]);
      cache.put(cacheKey, "1", 21600);
    } finally {
      lock.releaseLock();
    }
    return registrationResponse_({ok: true, request_id: requestId});
  } catch (error) {
    console.error(error);
    return registrationResponse_({ok: false, error: String(error && error.message || error), request_id: requestId});
  }
}

function parsePayload_(event) {
  const raw = event && event.parameter && event.parameter.payload || event && event.postData && event.postData.contents;
  if (!raw) throw new Error("Request body is empty.");
  const payload = JSON.parse(raw);
  const eventType = String(payload.event_type || "");
  const visitorId = String(payload.visitor_id || "").trim();
  const detail = typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail || {});
  return {
    event_type: eventType,
    action: String(payload.action || (eventType === "visitor_registration" ? "email_submitted" : "")).trim(),
    email: String(payload.email || "").trim().toLowerCase(),
    website: String(payload.website || "").trim(),
    page_url: String(payload.page_url || "").trim(),
    route: String(payload.route || "home").trim(),
    target_type: String(payload.target_type || "").trim(),
    target_id: String(payload.target_id || "").trim(),
    detail: detail.trim(),
    referrer: String(payload.referrer || "").trim(),
    timezone: String(payload.timezone || "").trim(),
    language: String(payload.language || "").trim(),
    visitor_id: visitorId,
    session_id: String(payload.session_id || (visitorId ? "legacy_" + visitorId : "")).trim(),
    registration_version: String(payload.registration_version || "").trim(),
    request_id: String(payload.request_id || "").trim(),
  };
}

function validatePayload_(payload) {
  if (["visitor_registration", "session_start", "page_view", "interaction"].indexOf(payload.event_type) < 0) throw new Error("Unsupported event type.");
  if (!/^[a-z0-9_:-]{2,80}$/.test(payload.action)) throw new Error("A valid action is required.");
  if (payload.email.length > 254 || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(payload.email)) throw new Error("A valid email is required.");
  if (!/^[A-Za-z0-9_-]{8,120}$/.test(payload.visitor_id)) throw new Error("A valid visitor ID is required.");
  if (!/^[A-Za-z0-9_-]{8,120}$/.test(payload.session_id)) throw new Error("A valid session ID is required.");
  if (!/^[A-Za-z0-9_-]{8,120}$/.test(payload.request_id)) throw new Error("A valid request ID is required.");
  if (payload.route.length > 160 || payload.target_type.length > 80 || payload.target_id.length > 240 || payload.detail.length > 1200) {
    throw new Error("Event metadata is too long.");
  }
  if (!ALLOWED_PAGE_PREFIXES.some(function(prefix) { return payload.page_url.indexOf(prefix) === 0; })) {
    throw new Error("Page URL is not permitted.");
  }
}

function visitorSheet_(spreadsheet) {
  let sheet = spreadsheet.getSheetByName(VISITOR_SHEET_NAME);
  if (!sheet) sheet = spreadsheet.insertSheet(VISITOR_SHEET_NAME);
  const existing = sheet.getRange(1, 1, 1, VISITOR_HEADERS.length).getValues()[0];
  if (existing.join("|") !== VISITOR_HEADERS.join("|")) {
    sheet.getRange(1, 1, 1, VISITOR_HEADERS.length).setValues([VISITOR_HEADERS]);
  }
  return sheet;
}

function safeCell_(value, maximumLength) {
  let text = String(value || "").slice(0, maximumLength);
  if (/^[=+\-@]/.test(text)) text = "'" + text;
  return text;
}

function sha256_(value) {
  const bytes = Utilities.computeDigest(Utilities.DigestAlgorithm.SHA_256, value, Utilities.Charset.UTF_8);
  return bytes.map(function(byte) { return (byte + 256).toString(16).slice(-2); }).join("");
}

function jsonResponse_(payload) {
  return ContentService.createTextOutput(JSON.stringify(payload)).setMimeType(ContentService.MimeType.JSON);
}

function registrationResponse_(payload) {
  const message = {
    source: "matryoshka_visitor_registration",
    ok: Boolean(payload.ok),
    request_id: String(payload.request_id || ""),
    error: String(payload.error || ""),
  };
  const serialized = JSON.stringify(message)
    .replace(/</g, "\\u003c")
    .replace(/\u2028/g, "\\u2028")
    .replace(/\u2029/g, "\\u2029");
  return HtmlService.createHtmlOutput(
    "<!doctype html><meta charset=\"utf-8\"><script>window.top.postMessage(" + serialized + ", '*');<\/script>"
  );
}
