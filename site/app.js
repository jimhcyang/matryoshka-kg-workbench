"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"})[char]);
const compact = (value, limit = 150) => {
  const text = String(value ?? "").replace(/\s+/g, " ").trim();
  return text.length <= limit ? text : `${text.slice(0, limit - 3).trim()}...`;
};
const formatNumber = (value) => new Intl.NumberFormat("en-US").format(Number(value || 0));
const titleCase = (value) => String(value ?? "").replace(/_/g, " ").toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());
const statusClass = (value) => String(value || "not assessed").toLowerCase().replace(/[^a-z]+/g, "_");
const regexEscape = (value) => String(value ?? "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const NEIGHBOR_STEP = 5;
const MAX_NEIGHBORS = 10;
const PATH_DISPLAY_OPTIONS = [8, 16, 32];
const DEFAULT_PATH_DISPLAY_LIMIT = 16;
const GUIDE_REPLAY_SLOWDOWN = 3;
const GUIDE_REPLAY_PHASE_MS = 450;
const REQUIRED_API_PROTOCOL = "matryoshka_iran_demo_api_v2";
const RESTART_COMMAND = ".venv/bin/python src/demo/iran_site_visit/serve.py --open";
const STATIC_MANIFEST_PATH = "data/static_manifest.json";
const VISITOR_REGISTRATION = Object.freeze({
  enabled: false,
  endpoint: "",
  registrationVersion: "2026-10-12-events-v2",
  sessionStorageKey: "matryoshka_visitor_session_v2",
  visitorStorageKey: "matryoshka_visitor_id_v1",
  ...(window.MATRYOSHKA_VISITOR_REGISTRATION || {}),
});
const visitorTelemetry = {
  email: "",
  visitorId: "",
  sessionId: "",
  recentEvents: new Map(),
};
const RECORDED_DEMO_CASES = {
  reconstruction: [
    "iran_reconstruction_regency_illegality_03",
    "iran_demo_reconstruction_parallel_authority_06",
    "iran_demo_reconstruction_takeover_cost_01",
  ],
  hypothesis: [
    "iran_demo_hypothesis_unfinished_khomeini_18",
    "iran_demo_hypothesis_shah_return_08",
    "iran_demo_hypothesis_leftist_takeover_03",
  ],
  redaction: [
    "iran_demo_redaction_haeri_16",
    "iran_demo_redaction_entezam_mosaic_12",
    "iran_demo_redaction_minatchi_05",
  ],
};
const LIVE_DEMO_PROFILES = {
  reconstruction: "iran_demo_reconstruction_defensive_posture_04",
  hypothesis: "iran_demo_hypothesis_takeover_us_cost_01",
  redaction: "iran_demo_redaction_gharabaghi_mosaic_11",
};

class ApiContractError extends Error {
  constructor(message) {
    super(message);
    this.name = "ApiContractError";
  }
}

async function decodeJsonResponse(response, context) {
  const contentType = String(response.headers.get("content-type") || "").toLowerCase();
  const body = await response.text();
  if (!contentType.includes("application/json")) {
    const htmlResponse = /^\s*(?:<!doctype\s+html|<html)/i.test(body);
    const detail = htmlResponse
      ? "The browser reached an older or incorrect local server that returned an HTML page instead of API data."
      : `The server returned ${contentType || "an unknown content type"} instead of JSON.`;
    throw new ApiContractError(`${context} failed (HTTP ${response.status}). ${detail}`);
  }
  let payload;
  try {
    payload = body ? JSON.parse(body) : {};
  } catch (error) {
    throw new ApiContractError(`${context} returned malformed JSON (HTTP ${response.status}).`);
  }
  if (!response.ok) {
    const message = payload.detail || payload.error || `${context} failed with HTTP ${response.status}.`;
    const failure = new Error(message);
    failure.status = response.status;
    failure.payload = payload;
    throw failure;
  }
  return payload;
}

async function requestJson(path, options = {}, context = "API request") {
  const response = await fetch(path, {cache: "no-store", ...options});
  return decodeJsonResponse(response, context);
}

async function staticResource(name) {
  if (!state.staticMode) throw new Error("Static resources are only available in hosted replay mode.");
  if (state.staticCache.has(name)) return state.staticCache.get(name);
  const path = state.staticManifest?.[`${name}_url`];
  if (!path) throw new Error(`Hosted replay does not declare a ${name} resource.`);
  const response = await fetch(path, {cache: "no-store"});
  const payload = await decodeJsonResponse(response, `Hosted ${name} resource`);
  state.staticCache.set(name, payload);
  return payload;
}

function staticScopeMatch(item, scopes) {
  if (!scopes?.length || scopes.includes("all")) return true;
  return searchResultMatchesScopes(item, scopes);
}

async function staticSearch(query, selectedScopes) {
  const records = await staticResource("search");
  const prefix = query.match(/^([A-Za-z_|]+)\s*:\s*(.+)$/);
  const scopeAliases = {entity:"entity",entities:"entity",person:"person",people:"person",organization:"organization",organizations:"organization",org:"organization",location:"location",locations:"location",place:"location",event:"event",events:"event",claim:"claim",claims:"claim",relation:"relation",relations:"relation",narrative:"narrative",narratives:"narrative",community:"community",communities:"community",document:"document",documents:"document",doc:"document"};
  let effectiveScopes = selectedScopes?.length ? selectedScopes : ["all"];
  let effectiveQuery = query.trim();
  let prefixApplied = false;
  if (prefix) {
    const parsed = prefix[1].split("|").map((value) => scopeAliases[value.toLowerCase()]).filter(Boolean);
    if (parsed.length) { effectiveScopes = [...new Set(parsed)]; effectiveQuery = prefix[2].trim(); prefixApplied = true; }
  }
  const normalized = effectiveQuery.toLowerCase().replace(/[^a-z0-9]+/g," ").trim();
  const tokens = normalized.split(/\s+/).filter(Boolean);
  const results = records
    .filter((item) => staticScopeMatch(item, effectiveScopes))
    .map((item) => {
      const surface = String(item.search_text || `${item.label} ${item.id} ${item.summary || ""}`).toLowerCase().replace(/[^a-z0-9]+/g," ").trim();
      const label = String(item.label || "").toLowerCase().replace(/[^a-z0-9]+/g," ").trim();
      const identifier = String(item.id || "").toLowerCase().replace(/[^a-z0-9]+/g," ").trim();
      const tokenHits = tokens.filter((token) => surface.includes(token)).length;
      let score = tokens.length ? tokenHits / tokens.length * .72 : 0;
      if (label === normalized || identifier === normalized) score = 1;
      else if (label.startsWith(normalized) || identifier.startsWith(normalized)) score = Math.max(score,.93);
      else if (surface.includes(normalized)) score = Math.max(score,.84);
      return {...item, score};
    })
    .filter((item) => item.score > 0)
    .sort((left,right) => right.score - left.score || String(left.label).localeCompare(String(right.label)))
    .slice(0,16);
  return {query:effectiveQuery, prefix_applied:prefixApplied, effective_scopes:effectiveScopes, results, search_contract:"Hosted replay uses exact IDs, labels, retained aliases, and deterministic token coverage over the shipped hierarchy/search index."};
}

function restartHintMarkup() {
  return `<small class="restart-hint">Restart the local Python server, then reload this page.<code>${esc(RESTART_COMMAND)}</code></small>`;
}

function apiFailureMarkup(error, title = "Request did not complete.") {
  const restart = error instanceof ApiContractError || error instanceof TypeError;
  return `<div class="live-error"><b>${esc(title)}</b><p>${esc(error.message || String(error))}</p>${restart ? restartHintMarkup() : ""}</div>`;
}

function highlighted(value, terms = []) {
  const source = String(value ?? "");
  const seen = new Set();
  const uniqueTerms = terms
    .map((term) => String(term || "").trim())
    .filter((term) => {
      const key = term.toLocaleLowerCase();
      if (!term || seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  const candidates = [];
  uniqueTerms.forEach((term, priority) => {
    const pattern = new RegExp(regexEscape(term), "ig");
    for (const match of source.matchAll(pattern)) {
      candidates.push({
        start: match.index,
        end: match.index + match[0].length,
        priority,
      });
    }
  });
  candidates.sort((left, right) =>
    (right.end - right.start) - (left.end - left.start) ||
    left.priority - right.priority ||
    left.start - right.start
  );
  const accepted = [];
  candidates.forEach((candidate) => {
    const overlaps = accepted.some((prior) =>
      candidate.start < prior.end && prior.start < candidate.end
    );
    if (!overlaps) accepted.push(candidate);
  });
  accepted.sort((left, right) => left.start - right.start);
  if (!accepted.length) return esc(source);
  let cursor = 0;
  let output = "";
  accepted.forEach((match) => {
    output += esc(source.slice(cursor, match.start));
    output += `<mark class="${match.priority ? "secondary" : ""}">${esc(source.slice(match.start, match.end))}</mark>`;
    cursor = match.end;
  });
  return output + esc(source.slice(cursor));
}

const state = {
  bundle: null,
  mode: "explorer",
  level: "narratives",
  narrativeId: null,
  communityId: null,
  selected: null,
  hovered: null,
  searchFocus: null,
  searchPulseUntil: 0,
  showLabels: true,
  connectedOnly: false,
  transform: {x: 0, y: 0, scale: 1},
  pointer: {x: 0, y: 0},
  dragging: null,
  suppressClick: false,
  graph: null,
  neighborLimit: NEIGHBOR_STEP,
  searchScopes: ["all"],
  taskFilter: "reconstruction",
  workbenchMode: "recorded",
  caseId: null,
  livePayload: null,
  liveDraft: null,
  liveTask: "reconstruction",
  communityGraphs: new Map(),
  communityGraphLoading: null,
  staticMode: false,
  staticManifest: null,
  staticCache: new Map(),
  layoutOverrides: new Map(),
  edgePage: 0,
  longPressTimer: null,
  longPressTriggered: false,
  activeGuide: null,
  pathDisplayLimit: DEFAULT_PATH_DISPLAY_LIMIT,
  fitGeneration: 0,
  transitionTimer: null,
  guideTransitionTimers: [],
  guideNavigationToken: 0,
  guideHighlights: [],
  guidePulseUntil: 0,
  guideTransitionPhase: "",
  guideReplayDurationScale: GUIDE_REPLAY_SLOWDOWN,
  playedGuideBridges: new Set(),
  guideBridgeDirections: new Map(),
  guidePanelLayout: {left:null, top:null, width:null, height:null},
};

const indexes = {
  narrative: new Map(),
  community: new Map(),
  node: new Map(),
};

const colors = {
  narrative: "#b84d36",
  community: "#1d5367",
  Entity: "#47785c",
  Event: "#d29b2e",
  Claim: "#71566f",
  RelationEvidence: "#b84d36",
  Segment: "#708086",
  Unknown: "#708086",
};

async function init() {
  try {
    let staticManifest = null;
    try {
      const response = await fetch(STATIC_MANIFEST_PATH, {cache: "no-store"});
      if (response.ok && String(response.headers.get("content-type") || "").includes("application/json")) staticManifest = await response.json();
    } catch (_error) {
      staticManifest = null;
    }
    state.staticMode = Boolean(staticManifest?.static_replay);
    state.staticManifest = staticManifest;
    const [bundleResponse, health] = state.staticMode
      ? await Promise.all([
          fetch(staticManifest.bundle_url || "data/demo_bundle.json", {cache: "no-store"}),
          Promise.resolve({api_protocol:REQUIRED_API_PROTOCOL, schema:staticManifest.bundle_schema}),
        ])
      : await Promise.all([
          fetch("data/demo_bundle.json", {cache: "no-store"}),
          requestJson("/api/health", {}, "Demo server compatibility check"),
        ]);
    state.bundle = await decodeJsonResponse(bundleResponse, "Demo bundle request");
    if (health.api_protocol !== REQUIRED_API_PROTOCOL) {
      throw new ApiContractError(
        `Frontend/backend version mismatch: this interface requires ${REQUIRED_API_PROTOCOL}, ` +
        `but the running server reported ${health.api_protocol || "no API protocol"}.`
      );
    }
    if (health.schema !== state.bundle.schema_version) {
      throw new ApiContractError(
        `Server/bundle schema mismatch: server ${health.schema || "unknown"}, ` +
        `bundle ${state.bundle.schema_version || "unknown"}.`
      );
    }
    buildIndexes();
    renderHomeStats();
    bindEvents();
    renderCaseFilters();
    initializeVisitorRegistration();
    const firstCase = state.bundle.workbench.cases.find((item) => item.task === state.taskFilter);
    state.caseId = firstCase?.id || state.bundle.workbench.cases[0]?.id;
    if (state.staticMode) $("#corpus-chip").insertAdjacentHTML("beforeend", "<span class=\"hosted-badge\">Hosted replay</span>");
    $("#boot").style.opacity = "0";
    setTimeout(() => {
      $("#boot").hidden = true;
      openInitialRoute();
    }, 430);
  } catch (error) {
    $("#boot").innerHTML = `<div class="boot-error"><b>Demo could not start.</b><p>${esc(error.message)}</p>${restartHintMarkup()}</div>`;
  }
}

function validRegistrationEndpoint(value) {
  try {
    const url = new URL(String(value || ""));
    return url.protocol === "https:" && url.hostname === "script.google.com" && /^\/macros\/s\/[A-Za-z0-9_-]+\/exec$/.test(url.pathname);
  } catch (_error) {
    return false;
  }
}

function visitorIdentifier(prefix) {
  const random = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  return `${prefix}_${random}`;
}

function persistentVisitorId() {
  try {
    const existing = localStorage.getItem(VISITOR_REGISTRATION.visitorStorageKey);
    if (existing) return existing;
    const created = visitorIdentifier("visitor");
    localStorage.setItem(VISITOR_REGISTRATION.visitorStorageKey, created);
    return created;
  } catch (_error) {
    return visitorIdentifier("visitor");
  }
}

function storedVisitorSession() {
  try {
    return JSON.parse(sessionStorage.getItem(VISITOR_REGISTRATION.sessionStorageKey) || "null");
  } catch (_error) {
    return null;
  }
}

function storeVisitorSession(profile) {
  try {
    sessionStorage.setItem(VISITOR_REGISTRATION.sessionStorageKey, JSON.stringify(profile));
  } catch (_error) {
    // The in-memory session remains usable when browser storage is unavailable.
  }
}

function activateVisitorSession(profile) {
  visitorTelemetry.email = String(profile.email || "").trim().toLowerCase();
  visitorTelemetry.visitorId = String(profile.visitorId || persistentVisitorId());
  visitorTelemetry.sessionId = String(profile.sessionId || visitorIdentifier("session"));
}

function activityRoute() {
  if (!$("#home")?.hidden) return "home";
  if (state.mode === "explorer") return `explorer:${state.level}`;
  const task = state.workbenchMode === "live" ? state.liveTask : state.taskFilter;
  return `workbench:${state.workbenchMode}:${task}`;
}

function submitVisitorEvent(payload) {
  const requestId = visitorIdentifier("event");
  return fetch(VISITOR_REGISTRATION.endpoint, {
    method: "POST",
    mode: "no-cors",
    cache: "no-store",
    credentials: "omit",
    keepalive: true,
    body: JSON.stringify({...payload, request_id: requestId}),
  });
}

function visitorEventPayload(eventType, action, targetType = "", targetId = "", detail = {}) {
  return {
    event_type: eventType,
    action,
    target_type: String(targetType || ""),
    target_id: compact(targetId, 240),
    detail: compact(JSON.stringify(detail || {}), 1200),
    email: visitorTelemetry.email,
    website: "",
    page_url: window.location.href,
    route: activityRoute(),
    referrer: document.referrer,
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || "",
    language: navigator.language || "",
    visitor_id: visitorTelemetry.visitorId,
    session_id: visitorTelemetry.sessionId,
    registration_version: VISITOR_REGISTRATION.registrationVersion,
  };
}

function trackVisitorEvent(eventType, action, targetType = "", targetId = "", detail = {}, dedupeMs = 800) {
  if (!VISITOR_REGISTRATION.enabled || !visitorTelemetry.email || !validRegistrationEndpoint(VISITOR_REGISTRATION.endpoint)) return;
  const key = `${eventType}:${action}:${targetType}:${targetId}`;
  const now = Date.now();
  if (dedupeMs && now - Number(visitorTelemetry.recentEvents.get(key) || 0) < dedupeMs) return;
  visitorTelemetry.recentEvents.set(key, now);
  submitVisitorEvent(visitorEventPayload(eventType, action, targetType, targetId, detail)).catch(() => {
    // Analytics must never interrupt the demonstration.
  });
}

function trackInteraction(action, targetType = "", targetId = "", detail = {}, dedupeMs = 800) {
  trackVisitorEvent("interaction", action, targetType, targetId, detail, dedupeMs);
}

function releaseVisitorGate() {
  const gate = $("#visitor-registration");
  if (!gate) return;
  gate.classList.add("leaving");
  document.body.classList.remove("registration-required");
  setTimeout(() => { gate.hidden = true; gate.classList.remove("leaving"); }, 330);
}

function initializeVisitorRegistration() {
  const gate = $("#visitor-registration");
  const form = $("#visitor-registration-form");
  if (!gate || !form || !VISITOR_REGISTRATION.enabled) return;
  if (!validRegistrationEndpoint(VISITOR_REGISTRATION.endpoint)) {
    console.warn("Visitor registration is enabled without a valid Google Apps Script /exec endpoint; the gate was not shown.");
    return;
  }
  const params = new URLSearchParams(window.location.search);
  if (params.get("registration") === "reset") {
    try { sessionStorage.removeItem(VISITOR_REGISTRATION.sessionStorageKey); } catch (_error) { /* no-op */ }
  }
  try { localStorage.removeItem("matryoshka_visitor_registration_v1"); } catch (_error) { /* remove the retired persistent gate marker */ }
  const stored = storedVisitorSession();
  if (stored?.version === VISITOR_REGISTRATION.registrationVersion && stored.email) {
    activateVisitorSession(stored);
    setTimeout(() => trackVisitorEvent("page_view", "load", "page", activityRoute(), {}, 0), 700);
    return;
  }

  document.body.classList.add("registration-required");
  gate.hidden = false;
  setTimeout(() => $("#visitor-email")?.focus(), 50);

  $("#visitor-continue")?.addEventListener("click", () => releaseVisitorGate(), {once:true});
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const emailInput = $("#visitor-email");
    const submit = $("#visitor-submit");
    const status = $("#visitor-registration-status");
    const fallback = $("#visitor-continue");
    const email = String(emailInput?.value || "").trim().toLowerCase();
    if (!emailInput?.checkValidity() || email.length > 254) {
      emailInput?.reportValidity();
      return;
    }

    submit.disabled = true;
    emailInput.disabled = true;
    status.textContent = "Recording your visit...";
    fallback.hidden = true;
    const payload = {
      event_type: "session_start",
      action: "email_submitted",
      target_type: "page",
      target_id: activityRoute(),
      detail: "{}",
      email,
      website: String($("#visitor-website")?.value || ""),
      page_url: window.location.href,
      route: activityRoute(),
      referrer: document.referrer,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || "",
      language: navigator.language || "",
      visitor_id: persistentVisitorId(),
      session_id: visitorIdentifier("session"),
      registration_version: VISITOR_REGISTRATION.registrationVersion,
    };

    try {
      await submitVisitorEvent(payload);
      const profile = {
        email,
        visitorId: payload.visitor_id,
        sessionId: payload.session_id,
        version: VISITOR_REGISTRATION.registrationVersion,
        startedAt: new Date().toISOString(),
      };
      activateVisitorSession(profile);
      storeVisitorSession(profile);
      setTimeout(() => trackVisitorEvent("page_view", "load", "page", activityRoute(), {}, 0), 500);
      status.textContent = "Registration submitted. Opening Matryoshka...";
      setTimeout(() => releaseVisitorGate(), 260);
    } catch (_error) {
      submit.disabled = false;
      emailInput.disabled = false;
      status.textContent = "The visitor log could not be reached. You may retry or continue to the demonstration.";
      fallback.hidden = false;
    }
  });
}

function openInitialRoute() {
  const params = new URLSearchParams(window.location.search);
  const mode = params.get("mode");
  if (!mode) { $("#home").hidden = false; return; }
  $("#home").hidden = true; $("#shell").hidden = false;
  if (mode === "workbench") {
    if (params.get("view") === "live") {
      const requestedTask = params.get("task");
      if (liveTaskMeta[requestedTask]) state.liveTask = requestedTask;
      const profile = liveProfiles(state.liveTask).find((row) => row.profileId === params.get("profile"));
      if (profile) state.liveDraft = {...profile};
      state.workbenchMode = "live";
      $$('[data-workbench-mode]').forEach((button) => button.classList.toggle("active", button.dataset.workbenchMode === "live"));
      renderCaseFilters();
      setMode("workbench");
      return;
    }
    const requested = params.get("case");
    const item = state.bundle.workbench.cases.find((row) => row.id === requested);
    if (item) { state.caseId = item.id; state.taskFilter = item.task; renderCaseFilters(); }
    setMode("workbench");
    return;
  }
  setMode("explorer");
  const communityId = params.get("community");
  const narrativeId = params.get("narrative") || indexes.community.get(communityId)?.narrative_id;
  state.narrativeId = indexes.narrative.has(narrativeId) ? narrativeId : null;
  state.communityId = indexes.community.has(communityId) ? communityId : null;
  const level = state.communityId ? "nodes" : state.narrativeId ? "communities" : "narratives";
  setTimeout(() => { resizeCanvas(); setGraphLevel(level); }, 20);
}

function buildIndexes() {
  const hierarchy = state.bundle.hierarchy;
  hierarchy.narratives.forEach((item) => indexes.narrative.set(item.id, item));
  hierarchy.communities.forEach((item) => indexes.community.set(item.id, item));
  hierarchy.nodes.forEach((item) => indexes.node.set(item.id, item));
}

function renderHomeStats() {
  const counts = state.bundle.hierarchy.manifest.counts;
  $("#home-stats").innerHTML = [
    [formatNumber(counts.document_count), "documents"],
    [formatNumber(counts.node_count), "graph nodes"],
    [formatNumber(counts.edge_count), "relations"],
  ].map(([value, label]) => `<div class="home-stat"><b>${value}</b><span>${label}</span></div>`).join("");
  const dateStart = String(state.bundle.hierarchy.manifest.date_start || "").slice(0, 4);
  const dateEnd = String(state.bundle.hierarchy.manifest.date_end || "").slice(0, 4);
  $("#corpus-chip").innerHTML = `<b>Iran Foreign Policy Cables</b>${formatNumber(counts.document_count)} documents / ${dateStart}-${dateEnd}`;
}

function bindEvents() {
  $$('[data-launch]').forEach((button) => button.addEventListener("click", () => launch(button.dataset.launch)));
  $("#home-button").addEventListener("click", showHome);
  $$('[data-mode]').forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
  $$('[data-workbench-mode]').forEach((button) => button.addEventListener("click", () => {
    state.workbenchMode = button.dataset.workbenchMode;
    $$('[data-workbench-mode]').forEach((item) => item.classList.toggle("active", item.dataset.workbenchMode === state.workbenchMode));
    renderCaseFilters(); renderWorkbench();
  }));
  $("#graph-back").addEventListener("click", graphBack);
  $("#graph-fit").addEventListener("click", fitGraph);
  $("#graph-load-complete").addEventListener("click", () => toggleCompleteCommunity(state.communityId));
  $("#labels-toggle").addEventListener("change", (event) => { state.showLabels = event.target.checked; drawGraph(); });
  $("#connected-toggle").addEventListener("change", (event) => {
    state.connectedOnly = event.target.checked;
    state.selected = null;
    fitGraph();
    renderLevelCaption();
    renderInspector();
  });
  $("#source-close").addEventListener("click", closeSource);
  $("#drawer-scrim").addEventListener("click", closeSource);
  document.addEventListener("keydown", (event) => {
    if (event.key === "/" && document.activeElement?.tagName !== "INPUT") {
      event.preventDefault(); $("#global-search").focus();
    }
    if (event.key === "Escape") { closeSource(); closeSearch(); }
  });
  bindSearch();
  bindCanvas();
  bindActivityTracking();
  new ResizeObserver(() => {
    resizeCanvas();
    constrainGraphGuidePanel($("#graph-guide"));
  }).observe($("#graph-stage"));
}

function bindActivityTracking() {
  const dataRules = [
    ["launch", "launch_mode", "mode"],
    ["mode", "switch_mode", "mode"],
    ["workbenchMode", "switch_workbench_mode", "mode"],
    ["task", "select_recorded_task", "task"],
    ["liveTask", "select_live_task", "task"],
    ["case", "select_recorded_case", "case"],
    ["liveExample", "select_live_profile", "profile"],
    ["routePath", "open_recorded_route", "route"],
    ["liveRouteIndex", "open_live_route", "route"],
    ["recordedGuidePath", "guide_recorded_route", "route"],
    ["atomIndex", "open_recorded_atom", "atom"],
    ["liveAtomIndex", "open_live_atom", "atom"],
    ["liveAtomId", "open_live_atom", "atom"],
    ["selectNode", "select_related_node", "graph_node"],
    ["selectEdge", "select_related_edge", "graph_edge"],
    ["openRecorded", "open_audited_companion", "case"],
    ["graphGuideStep", "select_guide_step", "guide_step"],
    ["guidedStep", "select_route_step", "guide_step"],
  ];
  const idRules = {
    "home-button": ["return_home", "page", "home"],
    "graph-back": ["graph_back", "graph_level", "parent"],
    "graph-fit": ["graph_fit", "graph_level", "current"],
    "graph-load-complete": ["toggle_complete_community", "community", "current"],
    "neighbors-more": ["expand_neighbors", "graph_node", "selected"],
    "neighbors-less": ["collapse_neighbors", "graph_node", "selected"],
    "recorded-guide-map": ["open_recorded_route_guide", "route", "selected"],
    "guided-map": ["open_route_guide", "route", "selected"],
    "graph-guide-prev": ["guide_previous", "guide_step", "previous"],
    "graph-guide-next": ["guide_next", "guide_step", "next"],
    "graph-guide-replay": ["guide_replay", "route", "active"],
    "graph-guide-workbench": ["guide_return_workbench", "mode", "workbench"],
    "candidate-live": ["load_candidate_live", "case", "selected"],
  };
  document.addEventListener("click", (event) => {
    if (event.target.closest("#visitor-registration")) return;
    const control = event.target.closest("button, a");
    if (!control || control.dataset.sourceDoc) return;
    for (const [datasetKey, action, targetType] of dataRules) {
      if (control.dataset[datasetKey] != null) {
        trackInteraction(action, targetType, control.dataset[datasetKey]);
        return;
      }
    }
    const rule = idRules[control.id];
    if (rule) trackInteraction(rule[0], rule[1], rule[2]);
  });
  document.addEventListener("change", (event) => {
    if (event.target.closest("#visitor-registration")) return;
    const control = event.target;
    if (control.id === "labels-toggle" || control.id === "connected-toggle") {
      trackInteraction("graph_display_toggle", "control", control.id, {checked:Boolean(control.checked)});
    } else if (control.dataset.pathDisplayLimit != null || control.id === "live-limit") {
      trackInteraction("change_path_limit", "control", control.id || "path_display_limit", {value:String(control.value)});
    }
  });
}

function launch(mode) {
  $("#home").hidden = true;
  $("#shell").hidden = false;
  setMode(mode);
  if (mode === "explorer") setTimeout(() => { resizeCanvas(); setGraphLevel("narratives"); }, 20);
}

function showHome() {
  $("#shell").hidden = true;
  $("#home").hidden = false;
  closeSource();
}

function setMode(mode) {
  state.mode = mode;
  $$('[data-mode]').forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
  $("#explorer-screen").hidden = mode !== "explorer";
  $("#workbench-screen").hidden = mode !== "workbench";
  if (mode === "explorer") setTimeout(() => { resizeCanvas(); drawGraph(); renderGraphGuide(); }, 20);
  if (mode === "workbench") renderWorkbench();
  renderGraphGuide();
}

function bindSearch() {
  const input = $("#global-search");
  const menu = $("#search-filter-menu");
  const filterButton = $("#search-filter-button");
  let timer = null;
  input.addEventListener("input", () => {
    clearTimeout(timer);
    const query = input.value.trim();
    if (query.length < 2) { closeSearch(); return; }
    timer = setTimeout(() => runSearch(query), 180);
  });
  filterButton.addEventListener("click", (event) => {
    event.stopPropagation();
    menu.hidden = !menu.hidden;
    filterButton.setAttribute("aria-expanded", String(!menu.hidden));
  });
  $$('input[type="checkbox"]', menu).forEach((checkbox) => checkbox.addEventListener("change", () => {
    const checked = $$('input[type="checkbox"]:checked', menu).map((item) => item.value);
    state.searchScopes = checked.length ? checked : ["all"];
    updateSearchFilterLabel();
    const query = input.value.trim();
    if (query.length >= 2) runSearch(query);
  }));
  $("[data-search-any]", menu).addEventListener("click", () => {
    $$('input[type="checkbox"]', menu).forEach((item) => { item.checked = false; });
    state.searchScopes = ["all"]; updateSearchFilterLabel();
    const query = input.value.trim();
    if (query.length >= 2) runSearch(query);
  });
  input.addEventListener("focus", () => { if ($("#search-results").children.length) $("#search-results").classList.add("open"); });
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".global-search")) { closeSearch(); menu.hidden = true; filterButton.setAttribute("aria-expanded", "false"); }
  });
  updateSearchFilterLabel();
}

function updateSearchFilterLabel() {
  const labels = {person:"People",organization:"Organizations",location:"Places",event:"Events",claim:"Claims",relation:"Relations",narrative:"Narratives",community:"Communities",document:"Documents"};
  const selected = state.searchScopes;
  $("#search-filter-button").textContent = selected.includes("all") ? "All types" : selected.length <= 2 ? selected.map((item) => labels[item] || titleCase(item)).join(" + ") : `${selected.length} types`;
}

async function runSearch(query) {
  const panel = $("#search-results");
  panel.innerHTML = `<div class="search-result"><span class="search-kind">Searching</span><div><strong>Scanning the semantic index</strong></div></div>`;
  panel.classList.add("open");
  try {
    const scopes = state.searchScopes.join(",");
    const payload = state.staticMode
      ? await staticSearch(query, state.searchScopes)
      : await requestJson(
          `/api/search?q=${encodeURIComponent(query)}&scopes=${encodeURIComponent(scopes)}&limit=16`,
          {},
          "Corpus search",
        );
    const effectiveScopes = payload.effective_scopes || state.searchScopes;
    const results = payload.results.filter((item) => searchResultMatchesScopes(item, effectiveScopes));
    if (!results.length) {
      trackInteraction("search", "search_scope", effectiveScopes.join("+"), {query_length:query.length, result_count:0, prefix_applied:Boolean(payload.prefix_applied)}, 2000);
      panel.innerHTML = `<div class="search-result"><span class="search-kind">No match</span><div><strong>Try another spelling or broaden the result type.</strong><small>Prefix examples: PERSON: Shah / DOCUMENT: 1979TEHRAN01121.</small></div></div>`;
      return;
    }
    trackInteraction("search", "search_scope", effectiveScopes.join("+"), {query_length:query.length, result_count:results.length, prefix_applied:Boolean(payload.prefix_applied)}, 2000);
    panel.innerHTML = `<div class="search-contract"><b>${payload.prefix_applied ? "Typed prefix applied" : "Strict type filter"}</b><span>${payload.prefix_applied ? `${effectiveScopes.map(titleCase).join(" + ")} only / query: ${payload.query}` : `${effectiveScopes.includes("all") ? "All result types" : `${effectiveScopes.map(titleCase).join(" + ")} only`} / exact alias or ID -> exact label -> semantic node match -> hierarchy/document keyword`}</span></div>` + results.map((item, index) => `
      <button class="search-result" data-index="${index}">
        <span class="search-kind">${esc(item.entity_type || item.node_type || item.kind)}</span>
        <span><strong>${esc(compact(item.label, 86))}</strong><small>${esc(compact(item.summary, 130))}</small></span>
        <span class="search-score" title="Relative local-search match">${Math.min(100, Math.round(item.score * 100))}%</span>
      </button>`).join("");
    $$(".search-result[data-index]", panel).forEach((button) => button.addEventListener("click", () => focusSearchResult(results[Number(button.dataset.index)])));
  } catch (error) {
    trackInteraction("search_error", "search_scope", state.searchScopes.join("+"), {query_length:query.length, error_type:error?.name || "Error"}, 2000);
    panel.innerHTML = `<div class="search-result"><span class="search-kind">Offline</span><div><strong>${esc(error.message)}</strong></div></div>`;
  }
}

function searchResultMatchesScopes(item, scopes) {
  if (!scopes?.length || scopes.includes("all")) return true;
  const entityType = String(item.entity_type || "").toUpperCase();
  const nodeType = String(item.node_type || "");
  return scopes.some((scope) => ({
    entity: item.kind === "node" && nodeType === "Entity",
    person: item.kind === "node" && nodeType === "Entity" && entityType === "PERSON",
    organization: item.kind === "node" && nodeType === "Entity" && entityType === "ORGANIZATION",
    location: item.kind === "node" && nodeType === "Entity" && ["LOCATION", "COUNTRY", "CITY", "REGION", "FACILITY"].includes(entityType),
    event: item.kind === "node" && nodeType === "Event",
    claim: item.kind === "node" && nodeType === "Claim",
    relation: item.kind === "node" && nodeType === "RelationEvidence",
    narrative: item.kind === "narrative",
    community: item.kind === "community",
    document: item.kind === "document",
  })[scope] === true);
}

function closeSearch() { $("#search-results").classList.remove("open"); }

function focusSearchResult(result) {
  closeSearch();
  $("#global-search").value = result.label;
  trackInteraction("select_search_result", result.kind || "search_result", result.id || result.doc_id || "unidentified_result", {
    result_type:result.entity_type || result.node_type || result.kind,
    scopes:state.searchScopes,
  });
  if (result.kind === "document") {
    openSource(result.doc_id || result.id, "", result.label);
    return;
  }
  pulseGraphNode(result, false);
  state.neighborLimit = NEIGHBOR_STEP;
  if (result.kind === "narrative") {
    state.narrativeId = null;
    state.communityId = null;
    transitionGraph("narratives", result.id);
  } else if (result.kind === "community") {
    state.narrativeId = result.narrative_id;
    state.communityId = null;
    transitionGraph("communities", result.id);
  } else {
    state.narrativeId = result.narrative_id;
    state.communityId = result.community_id;
    if (!indexes.node.has(result.id)) {
      indexes.node.set(result.id, {...result, kind: "node", type: result.node_type, subtype: result.entity_type, weight: 1, injected: true});
    }
    transitionGraph("nodes", result.id);
  }
  animateSearchPulse();
}

function pulseGraphNode(node, animate = true) {
  state.searchFocus = {...node};
  state.searchPulseUntil = Date.now() + 3200;
  if (animate) animateSearchPulse();
}

function animateSearchPulse() {
  if (Date.now() >= state.searchPulseUntil) { drawGraph(); return; }
  drawGraph();
  requestAnimationFrame(animateSearchPulse);
}

function setGuideHighlights(items, duration = 1800) {
  state.guideHighlights = (items || []).filter((item) => item?.id);
  state.guidePulseUntil = Date.now() + duration;
  animateGuideHighlights();
}

function animateGuideHighlights() {
  drawGraph();
  if (Date.now() < state.guidePulseUntil) requestAnimationFrame(animateGuideHighlights);
}

function getGraphData() {
  const hierarchy = state.bundle.hierarchy;
  let data;
  if (state.level === "narratives") {
    data = {nodes: hierarchy.narratives, edges: hierarchy.narrative_edges, layout: hierarchy.narrative_layout};
  } else if (state.level === "communities") {
    const nodes = hierarchy.communities.filter((item) => item.narrative_id === state.narrativeId);
    data = {nodes, edges: hierarchy.community_edges[state.narrativeId] || [], layout: hierarchy.community_layouts[state.narrativeId] || {}};
  } else {
    const complete = state.communityGraphs.get(state.communityId);
    const previewIds = [...(hierarchy.nodes_by_community[state.communityId] || [])];
    const ids = complete
      ? complete.nodes.map((item) => item.id)
      : previewIds;
    if (complete) complete.nodes.forEach((item) => indexes.node.set(item.id, item));
    if (state.searchFocus?.kind === "node" && state.searchFocus.community_id === state.communityId && !ids.includes(state.searchFocus.id)) ids.unshift(state.searchFocus.id);
    const nodes = ids.map((id) => indexes.node.get(id)).filter(Boolean);
    const layout = complete
      ? complete.layout
      : {...(hierarchy.semantic_layouts[state.communityId] || {})};
    if (state.searchFocus?.id && !layout[state.searchFocus.id]) layout[state.searchFocus.id] = [0, 0];
    data = {
      nodes,
      edges: complete?.edges || hierarchy.semantic_edges[state.communityId] || [],
      layout,
      complete: Boolean(complete),
      assignedNodeCount: indexes.community.get(state.communityId)?.member_count || nodes.length,
      previewNodeCount: previewIds.length,
    };
  }
  data.nodes.forEach((node) => {
    const override = state.layoutOverrides.get(layoutOverrideKey(node.id));
    if (override) data.layout[node.id] = override;
  });
  const connectedIds = new Set();
  data.edges.forEach((edge) => { connectedIds.add(edge.source); connectedIds.add(edge.target); });
  const fullNodeCount = data.nodes.length;
  if (state.connectedOnly) {
    data = {
      ...data,
      nodes: data.nodes.filter((node) => connectedIds.has(node.id) || node.id === state.searchFocus?.id),
    };
    const visible = new Set(data.nodes.map((node) => node.id));
    data.edges = data.edges.filter((edge) => visible.has(edge.source) && visible.has(edge.target));
  }
  return {...data, fullNodeCount, connectedNodeCount: connectedIds.size};
}

function layoutOverrideKey(nodeId) {
  return [state.level, state.narrativeId || "-", state.communityId || "-", nodeId].join("::");
}

function nodeMass(node) {
  if (node.kind === "narrative") return Number(node.member_count || 1);
  if (node.kind === "community") return Number(node.member_count || 1);
  return Math.max(1, Number(node.weight || node.doc_count || 1));
}

function radiusFor(node, nodes) {
  const masses = nodes.map(nodeMass);
  const min = Math.log1p(Math.min(...masses));
  const max = Math.log1p(Math.max(...masses));
  const scaled = (Math.log1p(nodeMass(node)) - min) / Math.max(.001, max - min);
  const range = state.level === "nodes" ? [7, 23] : state.level === "communities" ? [13, 36] : [7, 32];
  return range[0] + scaled * (range[1] - range[0]);
}

function colorFor(node) {
  if (node.kind === "narrative") return colors.narrative;
  if (node.kind === "community") return colors.community;
  return colors[node.type] || colors.Unknown;
}

function resizeCanvas() {
  const canvas = $("#graph-canvas");
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * ratio));
  canvas.height = Math.max(1, Math.round(rect.height * ratio));
  canvas.dataset.ratio = ratio;
  fitGraph();
}

function fitGraph() {
  const canvas = $("#graph-canvas");
  const rect = canvas.getBoundingClientRect();
  const graph = getGraphData();
  const points = graph.nodes.map((node) => graph.layout[node.id]).filter(Boolean);
  if (!points.length || rect.width < 20 || rect.height < 20) {
    state.transform = {x: 0, y: 0, scale: .78};
    drawGraph();
    return;
  }
  const xs = points.map((point) => Number(point[0]) || 0);
  const ys = points.map((point) => Number(point[1]) || 0);
  const minX = Math.min(...xs); const maxX = Math.max(...xs);
  const minY = Math.min(...ys); const maxY = Math.max(...ys);
  const base = Math.max(1, Math.min(rect.width - 140, rect.height - 140) * .47);
  const horizontalPadding = Math.min(150, Math.max(72, rect.width * .12));
  const verticalPadding = Math.min(125, Math.max(76, rect.height * .15));
  const widthScale = (rect.width - horizontalPadding * 2) / Math.max(base * (maxX - minX), base * .45);
  const heightScale = (rect.height - verticalPadding * 2) / Math.max(base * (maxY - minY), base * .45);
  const maximum = graph.nodes.length > 500 ? .56 : graph.nodes.length > 120 ? .66 : .74;
  const scale = Math.max(.32, Math.min(maximum, widthScale, heightScale));
  state.transform = {
    x: -((minX + maxX) / 2) * base,
    y: -((minY + maxY) / 2) * base,
    scale,
  };
  drawGraph();
  const generation = ++state.fitGeneration;
  requestAnimationFrame(() => settleGraphFit(generation));
}

function settleGraphFit(generation) {
  if (generation !== state.fitGeneration || !state.graph?.positions?.size) return;
  const {positions, graph, rect} = state.graph;
  const bounds = [...positions.entries()].map(([id, point]) => {
    const node = state.graph.nodeMap.get(id);
    const radius = node ? radiusFor(node, graph.nodes) * Math.sqrt(state.transform.scale) : 0;
    return {left:point.x - radius, right:point.x + radius, top:point.y - radius, bottom:point.y + radius};
  });
  const minX = Math.min(...bounds.map((item) => item.left));
  const maxX = Math.max(...bounds.map((item) => item.right));
  const minY = Math.min(...bounds.map((item) => item.top));
  const maxY = Math.max(...bounds.map((item) => item.bottom));
  const availableWidth = Math.max(120, rect.width - 150);
  const availableHeight = Math.max(120, rect.height - 150);
  const shrink = Math.min(1, availableWidth / Math.max(1, maxX - minX), availableHeight / Math.max(1, maxY - minY));
  if (shrink < .985) {
    state.transform.scale = Math.max(.28, state.transform.scale * shrink * .97);
    drawGraph();
  }
  const current = state.graph;
  if (!current?.positions?.size) return;
  const adjusted = [...current.positions.entries()].map(([id, point]) => {
    const node = current.nodeMap.get(id);
    const radius = node ? radiusFor(node, current.graph.nodes) * Math.sqrt(state.transform.scale) : 0;
    return {left:point.x - radius, right:point.x + radius, top:point.y - radius, bottom:point.y + radius};
  });
  const centerX = (Math.min(...adjusted.map((item) => item.left)) + Math.max(...adjusted.map((item) => item.right))) / 2;
  const centerY = (Math.min(...adjusted.map((item) => item.top)) + Math.max(...adjusted.map((item) => item.bottom))) / 2;
  state.transform.x += (current.rect.width / 2 - centerX) / state.transform.scale;
  state.transform.y += (current.rect.height / 2 - centerY) / state.transform.scale;
  drawGraph();
}

function setGraphLevel(level, focusId = null) {
  if (level === "narratives") {
    state.narrativeId = null;
    state.communityId = null;
  } else if (level === "communities") {
    state.communityId = null;
  }
  state.level = level;
  state.selected = null;
  state.hovered = null;
  state.edgePage = 0;
  fitGraph();
  renderBreadcrumbs();
  renderLevelCaption();
  renderInspector();
  if (focusId) setTimeout(() => focusNode(focusId), 30);
}

function transitionGraph(level, focusId = null, options = {}) {
  const canvas = $("#graph-canvas");
  const viaNodeId = options.viaNodeId || "";
  const viaPoint = viaNodeId ? state.graph?.positions?.get(viaNodeId) : null;
  const direction = options.direction || (viaPoint ? "in" : "crossfade");
  const replayActive = $("#graph-stage")?.classList.contains("guide-replay-active");
  const durationScale = Math.max(1, Number(options.durationScale || (replayActive ? state.guideReplayDurationScale : 1)));
  clearTimeout(state.transitionTimer);
  const begin = () => {
    if (viaPoint) {
      canvas.style.setProperty("--zoom-origin-x", `${viaPoint.x}px`);
      canvas.style.setProperty("--zoom-origin-y", `${viaPoint.y}px`);
    } else {
      canvas.style.removeProperty("--zoom-origin-x");
      canvas.style.removeProperty("--zoom-origin-y");
    }
    canvas.classList.remove("arriving", "zooming-out", "zooming-in");
    canvas.classList.add(direction === "in" ? "zooming-in" : direction === "out" ? "zooming-out" : "transitioning");
    state.transitionTimer = setTimeout(() => {
      canvas.classList.remove("transitioning", "zooming-in", "zooming-out");
      canvas.classList.add("arriving");
      if (options.arrivalNode) {
        pulseGraphNode(options.arrivalNode, false);
        animateSearchPulse();
      }
      setGraphLevel(level, focusId);
      requestAnimationFrame(() => requestAnimationFrame(() => canvas.classList.remove("arriving")));
    }, (direction === "crossfade" ? 150 : 230) * durationScale);
  };
  if (viaPoint) {
    const viaNode = state.graph?.nodeMap?.get(viaNodeId);
    if (viaNode) pulseGraphNode(viaNode, true);
    state.transitionTimer = setTimeout(begin, 175 * durationScale);
  } else {
    begin();
  }
}

function graphBack() {
  if (state.level === "nodes") {
    const parent = state.communityId;
    pulseGraphNode({kind: "community", id: parent, narrative_id: state.narrativeId}, false);
    transitionGraph("communities", parent, {direction:"out", arrivalNode:{kind:"community", id:parent, narrative_id:state.narrativeId}});
  } else if (state.level === "communities") {
    const parent = state.narrativeId;
    pulseGraphNode({kind: "narrative", id: parent}, false);
    transitionGraph("narratives", parent, {direction:"out", arrivalNode:{kind:"narrative", id:parent}});
  }
  setTimeout(animateSearchPulse, 260);
}

function renderBreadcrumbs() {
  const crumbs = [{level: "narratives", label: "Iran corpus"}];
  if (state.narrativeId) crumbs.push({level: "communities", label: indexes.narrative.get(state.narrativeId)?.label || "Narrative"});
  if (state.communityId && state.level === "nodes") crumbs.push({level: "nodes", label: indexes.community.get(state.communityId)?.label || "Community"});
  $("#breadcrumbs").innerHTML = crumbs.map((item, index) => `<button class="crumb ${index === crumbs.length - 1 ? "active" : ""}" data-level="${item.level}">${esc(item.label)}</button>`).join("");
  $$(".crumb", $("#breadcrumbs")).forEach((button) => button.addEventListener("click", () => {
    if (button.dataset.level === "narratives") transitionGraph("narratives", state.narrativeId);
    else if (button.dataset.level === "communities") transitionGraph("communities", state.communityId);
  }));
  renderPathPeek(crumbs);
}

function renderPathPeek(crumbs) {
  const fullPath = crumbs.map((item) => item.label).join(" / ");
  const current = crumbs[crumbs.length - 1]?.label || "Iran corpus";
  $("#path-peek").innerHTML = `<button type="button" aria-label="Current hierarchy path"><span>Path</span><b>${esc(compact(current, 22))}</b></button><div class="path-popover"><small>Current hierarchy path</small><b>${esc(fullPath)}</b><span>Back returns to the parent and pulses the region you came from.</span></div>`;
}

function renderLevelCaption() {
  const descriptions = {
    narratives: ["Narrative map", "Size = semantic membership; line width = aggregated cross-narrative evidence. Hover previews, click inspects, double-click or hold opens."],
    communities: ["Communities inside the selected narrative", "Size = semantic membership. Hover previews, click inspects, double-click or hold opens the semantic graph."],
    nodes: ["Community semantic graph", state.bundle.hierarchy.manifest.preview_contract],
  };
  const graph = getGraphData();
  $("#level-name").textContent = descriptions[state.level][0];
  const filtered = state.connectedOnly ? ` Showing ${formatNumber(graph.nodes.length)} connected nodes; ${formatNumber(graph.fullNodeCount - graph.nodes.length)} inactive singletons hidden.` : "";
  $("#level-description").textContent = descriptions[state.level][1] + filtered;
  $("#graph-back").disabled = state.level === "narratives";
  const loadButton = $("#graph-load-complete");
  const canToggle = state.level === "nodes" && graph.assignedNodeCount > graph.previewNodeCount;
  loadButton.hidden = !canToggle;
  loadButton.disabled = state.communityGraphLoading === state.communityId;
  loadButton.textContent = state.communityGraphLoading === state.communityId
    ? "Loading..."
    : graph.complete
      ? `Undo load all (${formatNumber(graph.previewNodeCount)})`
      : `Load all ${formatNumber(graph.assignedNodeCount || 0)}`;
  loadButton.title = graph.complete
    ? `Return to the ${formatNumber(graph.previewNodeCount)}-node instant view`
    : "Load every node assigned to this community";
  renderGraphLegend();
}

function toggleCompleteCommunity(communityId) {
  if (!communityId) return;
  if (!state.communityGraphs.has(communityId)) {
    loadCompleteCommunity(communityId);
    return;
  }
  state.communityGraphs.delete(communityId);
  state.selected = null;
  state.hovered = null;
  fitGraph();
  renderLevelCaption();
  renderInspector();
}

async function loadCompleteCommunity(communityId) {
  if (!communityId || state.communityGraphs.has(communityId) || state.communityGraphLoading) return;
  if (state.staticMode) {
    openDetailDrawer(
      "Hosted replay boundary",
      "Complete community loading requires the local server",
      [communityId, "startup preview remains available"],
      `<div class="live-error"><b>The hosted site intentionally ships a bounded graph payload.</b><p>Search, recorded routes, and the ${formatNumber(indexes.community.get(communityId)?.displayed_node_count || 0)}-node instant view remain available. Run the local Python server to materialize every assigned node and edge on demand.</p><code>${esc(RESTART_COMMAND)}</code></div>`,
    );
    return;
  }
  state.communityGraphLoading = communityId;
  renderLevelCaption();
  try {
    const payload = await requestJson(
      `/api/community-graph?community_id=${encodeURIComponent(communityId)}`,
      {},
      "Complete community graph",
    );
    payload.nodes.forEach((item) => indexes.node.set(item.id, item));
    state.communityGraphs.set(communityId, payload);
    state.selected = null;
    fitGraph();
    renderInspector();
  } catch (error) {
    openDetailDrawer(
      "Community graph",
      "Complete graph could not be loaded",
      [communityId],
      apiFailureMarkup(error, "Complete community load did not finish."),
    );
  } finally {
    state.communityGraphLoading = null;
    renderLevelCaption();
  }
}

function renderGraphLegend() {
  const legend = $("#graph-legend");
  if (state.level === "nodes") {
    legend.innerHTML = `<span><i class="dot entity"></i>Entity</span><span><i class="dot event"></i>Event</span><span><i class="dot claim"></i>Claim</span><span><i class="dot relation"></i>Relation evidence</span><span class="legend-note">Size = weighted degree / width = relation weight</span>`;
  } else {
    const label = state.level === "narratives" ? "Narrative" : "Community";
    const dot = state.level === "narratives" ? "narrative" : "community";
    legend.innerHTML = `<span><i class="dot ${dot}"></i>${label}</span><span><i class="legend-line"></i>Aggregated evidence link</span><span><i class="legend-halo"></i>Selected or searched</span><span class="legend-note">Node size = members / line width = aggregate strength</span>`;
  }
}

function graphSpreadFactor(nodeCount = state.graph?.graph?.nodes?.length || 0) {
  if (nodeCount > 320) return 1;
  if (state.level === "narratives") return 1.1;
  if (state.level === "communities") return 1.16;
  return 1.28;
}

function graphCoordinates(point, rect, nodeCount = 0) {
  const pad = 70;
  const base = Math.min(rect.width - pad * 2, rect.height - pad * 2) * .47;
  const spread = graphSpreadFactor(nodeCount);
  return {
    x: rect.width / 2 + (point[0] * base + state.transform.x) * state.transform.scale * spread,
    y: rect.height / 2 + (point[1] * base + state.transform.y) * state.transform.scale * spread,
  };
}

function inverseCoordinates(point, rect) {
  const pad = 70;
  const base = Math.min(rect.width - pad * 2, rect.height - pad * 2) * .47;
  const spread = graphSpreadFactor();
  return [
    ((point.x - rect.width / 2) / (state.transform.scale * spread) - state.transform.x) / base,
    ((point.y - rect.height / 2) / (state.transform.scale * spread) - state.transform.y) / base,
  ];
}

function resolveNodeCollisions(nodes, positions, radii, rect) {
  // The bounded views are presentation layouts, so preserve a readable gap
  // around larger nodes after zoom-to-fit. Explicit full-community loads can
  // contain thousands of nodes and intentionally keep their source layout.
  if (nodes.length > 320) return;
  const passes = state.level === "nodes" ? 18 : 14;
  for (let pass = 0; pass < passes; pass += 1) {
    let moved = false;
    for (let leftIndex = 0; leftIndex < nodes.length; leftIndex += 1) {
      const left = positions.get(nodes[leftIndex].id);
      for (let rightIndex = leftIndex + 1; rightIndex < nodes.length; rightIndex += 1) {
        const right = positions.get(nodes[rightIndex].id);
        let dx = right.x - left.x; let dy = right.y - left.y;
        let distance = Math.hypot(dx, dy);
        const leftRadius = radii.get(nodes[leftIndex].id);
        const rightRadius = radii.get(nodes[rightIndex].id);
        const visualGap = 5 + Math.min(12, Math.sqrt(leftRadius * rightRadius) * .35);
        const minimum = leftRadius + rightRadius + visualGap;
        if (distance >= minimum) continue;
        if (distance < .001) { dx = .001 * (rightIndex % 2 ? 1 : -1); dy = .001; distance = Math.hypot(dx, dy); }
        const shift = (minimum - distance) * .52;
        const ux = dx / distance; const uy = dy / distance;
        left.x -= ux * shift; left.y -= uy * shift;
        right.x += ux * shift; right.y += uy * shift;
        moved = true;
      }
    }
    positions.forEach((point, nodeId) => {
      const radius = radii.get(nodeId) + 9;
      point.x = Math.max(radius, Math.min(rect.width - radius, point.x));
      point.y = Math.max(radius, Math.min(rect.height - radius, point.y));
    });
    if (!moved) break;
  }
}

function drawGraph() {
  const canvas = $("#graph-canvas");
  if (!canvas.width || !state.bundle || $("#explorer-screen").hidden) return;
  const ratio = Number(canvas.dataset.ratio || 1);
  const ctx = canvas.getContext("2d");
  const rect = {width: canvas.width / ratio, height: canvas.height / ratio};
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  const graph = getGraphData();
  const nodeMap = new Map(graph.nodes.map((node) => [node.id, node]));
  const positions = new Map();
  const radii = new Map();
  graph.nodes.forEach((node) => {
    const raw = graph.layout[node.id] || [0, 0];
    positions.set(node.id, graphCoordinates(raw, rect, graph.nodes.length));
    radii.set(node.id, radiusFor(node, graph.nodes) * Math.sqrt(state.transform.scale));
  });
  resolveNodeCollisions(graph.nodes, positions, radii, rect);
  state.graph = {graph, nodeMap, positions, rect};

  const hoveredEdge = state.hovered?.kind === "edge" ? state.hovered.data : null;
  const selectedNodeId = state.selected?.kind === "node" ? state.selected.data.id : null;
  graph.edges.forEach((edge) => {
    const source = positions.get(edge.source); const target = positions.get(edge.target);
    if (!source || !target) return;
    const active = hoveredEdge?.id === edge.id || state.selected?.kind === "edge" && state.selected.data.id === edge.id;
    const incident = selectedNodeId && (edge.source === selectedNodeId || edge.target === selectedNodeId);
    ctx.beginPath(); ctx.moveTo(source.x, source.y); ctx.lineTo(target.x, target.y);
    ctx.strokeStyle = active ? "rgba(184,77,54,.96)" : incident ? "rgba(40,121,148,.78)" : "rgba(54,75,81,.21)";
    ctx.lineWidth = active ? 3.2 : incident ? 2.2 : Math.max(.55, Math.min(3.1, .45 + Math.log1p(Number(edge.weight || 0)) * .38));
    ctx.stroke();
  });

  const topLabels = new Set([...graph.nodes].sort((a,b) => nodeMass(b) - nodeMass(a)).slice(0, state.level === "narratives" ? 18 : 12).map((node) => node.id));
  const labelBoxes = [];
  const nodeObstacles = graph.nodes.map((node) => {
    const point = positions.get(node.id);
    return {id: node.id, x: point?.x || 0, y: point?.y || 0, radius: radii.get(node.id)};
  });
  graph.nodes.forEach((node) => {
    const point = positions.get(node.id); if (!point) return;
    const radius = radii.get(node.id);
    const endpointActive = hoveredEdge && (hoveredEdge.source === node.id || hoveredEdge.target === node.id);
    const searchActive = state.searchFocus?.id === node.id && Date.now() < state.searchPulseUntil;
    const guideHighlight = state.guideHighlights.find((item) => item.id === node.id);
    const guideActive = Boolean(guideHighlight) && Date.now() < state.guidePulseUntil;
    const nodeActive = state.hovered?.kind === "node" && state.hovered.data.id === node.id || state.selected?.kind === "node" && state.selected.data.id === node.id || searchActive || guideActive;
    ctx.beginPath(); ctx.arc(point.x, point.y, radius + (nodeActive ? 4 : 0), 0, Math.PI * 2);
    ctx.fillStyle = endpointActive ? (hoveredEdge.source === node.id ? "#b84d36" : "#287994") : colorFor(node);
    ctx.globalAlpha = node.injected ? .72 : .94; ctx.fill(); ctx.globalAlpha = 1;
    ctx.lineWidth = nodeActive ? 3 : 1;
    ctx.strokeStyle = nodeActive ? "#fffdf8" : "rgba(255,255,255,.72)"; ctx.stroke();
    if (nodeActive) { ctx.beginPath(); ctx.arc(point.x, point.y, radius + 7, 0, Math.PI * 2); ctx.strokeStyle = "#13242a"; ctx.lineWidth = 1.5; ctx.stroke(); }
    if (searchActive) {
      const remaining = Math.max(0, state.searchPulseUntil - Date.now());
      const phase = (3000 - remaining) / 180;
      ctx.beginPath(); ctx.arc(point.x, point.y, radius + 11 + Math.sin(phase) * 4, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(210,155,46,${.35 + .45 * remaining / 3000})`; ctx.lineWidth = 4; ctx.stroke();
    }
    if (guideActive) {
      const remaining = Math.max(0, state.guidePulseUntil - Date.now());
      const phase = (1800 - Math.min(1800, remaining)) / 150;
      ctx.beginPath(); ctx.arc(point.x, point.y, radius + 13 + Math.sin(phase) * 3, 0, Math.PI * 2);
      ctx.strokeStyle = guideHighlight.role === "from" ? "rgba(184,77,54,.88)" : "rgba(210,155,46,.96)";
      ctx.lineWidth = guideHighlight.role === "from" ? 3 : 5;
      ctx.stroke();
    }
    const shouldLabel = state.showLabels && (topLabels.has(node.id) || nodeActive || endpointActive || state.level === "communities" && graph.nodes.length <= 18);
    if (shouldLabel) drawNodeLabel(ctx, node, point, radius, rect, labelBoxes, nodeObstacles, nodeActive || endpointActive);
  });
}

function drawNodeLabel(ctx, node, point, radius, rect, occupied, nodeObstacles, force = false) {
  const source = compact(node.label, state.level === "nodes" ? 38 : 64);
  const words = source.split(/\s+/); const lines = [""];
  for (const word of words) {
    const index = lines.length - 1; const candidate = `${lines[index]} ${word}`.trim();
    if (candidate.length <= 31 || !lines[index]) { lines[index] = candidate; continue; }
    if (lines.length < 2) { lines.push(word); continue; }
    lines[index] = compact(candidate, 31); break;
  }
  ctx.font = `700 ${state.level === "nodes" ? 10 : 11}px "Avenir Next", sans-serif`;
  const width = Math.min(225, Math.max(...lines.map((line) => ctx.measureText(line).width)) + 14);
  const height = lines.length * 14 + 7;
  const candidates = [
    [point.x + radius + 6, point.y - height / 2],
    [point.x - radius - width - 6, point.y - height / 2],
    [point.x + radius + 6, point.y + radius + 4],
    [point.x - radius - width - 6, point.y + radius + 4],
  ];
  let box = null;
  for (const [rawX, rawY] of candidates) {
    const candidate = {x: Math.max(5, Math.min(rawX, rect.width - width - 5)), y: Math.max(5, Math.min(rawY, rect.height - height - 5)), width, height};
    const overlapsLabel = occupied.some((other) => candidate.x < other.x + other.width + 5 && candidate.x + candidate.width + 5 > other.x && candidate.y < other.y + other.height + 4 && candidate.y + candidate.height + 4 > other.y);
    const overlapsNode = nodeObstacles.some((other) => {
      if (other.id === node.id) return false;
      const closestX = Math.max(candidate.x, Math.min(other.x, candidate.x + candidate.width));
      const closestY = Math.max(candidate.y, Math.min(other.y, candidate.y + candidate.height));
      return Math.hypot(other.x - closestX, other.y - closestY) < other.radius + 4;
    });
    if ((!overlapsLabel && !overlapsNode) || force) { box = candidate; break; }
  }
  if (!box) return;
  occupied.push(box);
  ctx.fillStyle = "rgba(255,253,248,.92)"; ctx.fillRect(box.x, box.y, box.width, box.height);
  ctx.fillStyle = "#13242a";
  lines.forEach((line, index) => ctx.fillText(line, box.x + 7, box.y + 13 + index * 14));
}

function bindCanvas() {
  const canvas = $("#graph-canvas");
  canvas.addEventListener("pointermove", (event) => {
    const rect = canvas.getBoundingClientRect(); const point = {x: event.clientX - rect.left, y: event.clientY - rect.top};
    state.pointer = point;
    if (state.dragging?.kind === "pan") {
      state.transform.x = state.dragging.origin.x + (point.x - state.dragging.pointer.x) / state.transform.scale;
      state.transform.y = state.dragging.origin.y + (point.y - state.dragging.pointer.y) / state.transform.scale;
      drawGraph(); return;
    }
    if (state.dragging?.kind === "node") {
      if (Math.hypot(point.x - state.dragging.start.x, point.y - state.dragging.start.y) > 6) {
        clearTimeout(state.longPressTimer); state.longPressTimer = null;
      }
      const raw = inverseCoordinates(point, state.graph.rect);
      state.graph.graph.layout[state.dragging.id] = raw;
      state.layoutOverrides.set(layoutOverrideKey(state.dragging.id), raw);
      drawGraph(); return;
    }
    const hit = hitTest(point);
    canvas.classList.toggle("node-target", hit?.kind === "node");
    canvas.classList.toggle("edge-target", hit?.kind === "edge");
    const previous = state.hovered?.data?.id;
    state.hovered = hit;
    if (hit?.data?.id !== previous) drawGraph();
    renderTooltip(hit, event.clientX - rect.left, event.clientY - rect.top);
  });
  canvas.addEventListener("pointerleave", () => {
    canvas.classList.remove("node-target", "edge-target");
    if (!state.dragging) { state.hovered = null; $("#graph-tooltip").hidden = true; drawGraph(); }
  });
  canvas.addEventListener("pointerdown", (event) => {
    try { canvas.setPointerCapture(event.pointerId); } catch (_error) { /* Capture is optional. */ }
    canvas.classList.add("dragging");
    const rect = canvas.getBoundingClientRect(); const point = {x: event.clientX - rect.left, y: event.clientY - rect.top};
    const hit = hitTest(point);
    state.longPressTriggered = false;
    state.dragging = hit?.kind === "node"
      ? {kind: "node", id: hit.data.id, moved: false, start: point}
      : hit?.kind === "edge"
        ? {kind: "edge", id: hit.data.id, start: point}
        : {kind: "pan", pointer: point, start: point, origin: {...state.transform}};
    if (hit?.kind === "node" && state.level !== "nodes") {
      clearTimeout(state.longPressTimer);
      state.longPressTimer = setTimeout(() => {
        state.longPressTriggered = true; state.suppressClick = true;
        drillNode(hit.data);
      }, 620);
    }
  });
  canvas.addEventListener("pointerup", (event) => {
    clearTimeout(state.longPressTimer); state.longPressTimer = null;
    try { canvas.releasePointerCapture(event.pointerId); } catch (_error) { /* Already released. */ }
    canvas.classList.remove("dragging");
    const drag = state.dragging; state.dragging = null;
    if (!drag) return;
    const rect = canvas.getBoundingClientRect(); const point = {x: event.clientX - rect.left, y: event.clientY - rect.top};
    state.suppressClick = state.longPressTriggered || Math.hypot(point.x - drag.start.x, point.y - drag.start.y) > 5;
  });
  canvas.addEventListener("click", (event) => {
    if (state.dragging) return;
    if (state.suppressClick) { state.suppressClick = false; return; }
    const rect = canvas.getBoundingClientRect();
    handleGraphClick(hitTest({x: event.clientX - rect.left, y: event.clientY - rect.top}));
  });
  canvas.addEventListener("dblclick", (event) => {
    const rect = canvas.getBoundingClientRect();
    const hit = hitTest({x: event.clientX - rect.left, y: event.clientY - rect.top});
    if (hit?.kind === "node") { event.preventDefault(); drillNode(hit.data); }
  });
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    state.transform.scale = Math.max(.45, Math.min(3.5, state.transform.scale * (event.deltaY > 0 ? .9 : 1.1)));
    drawGraph();
  }, {passive: false});
}

function hitTest(point) {
  if (!state.graph) return null;
  const {graph, positions} = state.graph;
  for (let index = graph.nodes.length - 1; index >= 0; index -= 1) {
    const node = graph.nodes[index]; const at = positions.get(node.id); if (!at) continue;
    const radius = radiusFor(node, graph.nodes) * Math.sqrt(state.transform.scale) + 7;
    if (Math.hypot(point.x - at.x, point.y - at.y) <= radius) return {kind: "node", data: node};
  }
  // Lines are visually narrow, but their analyst-facing hit target should not be.
  // Higher hierarchy levels are sparse enough to support a wider target.
  let closest = null; let best = state.level === "nodes" ? 11 : 15;
  graph.edges.forEach((edge) => {
    const source = positions.get(edge.source); const target = positions.get(edge.target); if (!source || !target) return;
    const distance = pointToSegment(point, source, target);
    if (distance < best) { best = distance; closest = {kind: "edge", data: edge}; }
  });
  return closest;
}

function pointToSegment(point, start, end) {
  const dx = end.x - start.x; const dy = end.y - start.y;
  if (!dx && !dy) return Math.hypot(point.x - start.x, point.y - start.y);
  const t = Math.max(0, Math.min(1, ((point.x - start.x) * dx + (point.y - start.y) * dy) / (dx * dx + dy * dy)));
  return Math.hypot(point.x - (start.x + t * dx), point.y - (start.y + t * dy));
}

function renderTooltip(hit, x, y) {
  const tooltip = $("#graph-tooltip");
  if (!hit) { tooltip.hidden = true; return; }
  if (hit.kind === "node") {
    tooltip.innerHTML = `<b>${esc(hit.data.label)}</b>${esc(nodeTypeLabel(hit.data))}<br>${esc(nodeStatLine(hit.data))}`;
  } else {
    const source = state.graph.nodeMap.get(hit.data.source); const target = state.graph.nodeMap.get(hit.data.target);
    const relation = Object.keys(hit.data.relation_counts || {})[0] || hit.data.relation || "CONNECTED";
    const snippet = hit.data.evidence?.[0]?.text || "Select this connection to inspect its evidence provenance.";
    tooltip.innerHTML = `<b>${esc(source?.label || hit.data.source)} - ${esc(target?.label || hit.data.target)}</b>${esc(titleCase(relation))}<br>${esc(compact(snippet, 220))}`;
  }
  tooltip.style.left = `${Math.max(12, Math.min(x + 16, state.graph.rect.width - 356))}px`;
  tooltip.style.top = `${Math.max(12, Math.min(y + 14, state.graph.rect.height - 150))}px`;
  tooltip.hidden = false;
}

function handleGraphClick(hit) {
  if (!hit) { state.selected = null; renderInspector(); drawGraph(); return; }
  state.selected = hit; state.edgePage = 0;
  state.neighborLimit = NEIGHBOR_STEP;
  trackInteraction(
    hit.kind === "edge" ? "select_graph_edge" : "select_graph_node",
    hit.kind === "edge" ? "graph_edge" : nodeTypeLabel(hit.data),
    hit.data.id,
    hit.kind === "edge" ? {source:hit.data.source, target:hit.data.target, level:state.level} : {level:state.level},
  );
  renderInspector(); drawGraph();
}

function drillNode(node) {
  if (!node) return;
  trackInteraction("drill_graph_node", node.kind || node.type || "graph_node", node.id, {from_level:state.level});
  if (state.level === "narratives" || node.kind === "narrative") {
    state.narrativeId = node.id;
    state.communityId = null;
    transitionGraph("communities", null, {viaNodeId:node.id, direction:"in"});
  } else if (state.level === "communities" || node.kind === "community") {
    state.communityId = node.id;
    transitionGraph("nodes", null, {viaNodeId:node.id, direction:"in"});
  }
}

function focusNode(nodeId) {
  const point = state.graph?.positions.get(nodeId); if (!point) return;
  state.selected = {kind: "node", data: state.graph.nodeMap.get(nodeId) || indexes.node.get(nodeId)};
  state.neighborLimit = NEIGHBOR_STEP;
  renderInspector(); drawGraph();
}

function nodeTypeLabel(node) { return node.kind === "node" ? [node.type, node.subtype].filter(Boolean).map(titleCase).join(" / ") : titleCase(node.kind); }
function nodeStatLine(node) {
  if (node.kind === "narrative") return `${formatNumber(node.community_count)} communities / ${formatNumber(node.member_count)} semantic nodes`;
  if (node.kind === "community") return `${formatNumber(node.member_count)} assigned semantic nodes / ${formatNumber(node.displayed_node_count)} instant preview`;
  return `${formatNumber(node.doc_count)} documents / weighted degree ${Number(node.weight || 0).toFixed(2)}`;
}

function renderInspector() {
  const panel = $("#graph-inspector");
  if (!state.selected) {
    const graph = getGraphData();
    const nodeView = state.level === "nodes";
    const heading = nodeView
      ? (graph.complete ? `${formatNumber(graph.nodes.length)} semantic nodes loaded` : `${formatNumber(graph.nodes.length)}-node instant view`)
      : `${formatNumber(graph.nodes.length)} visible regions`;
    const firstStat = nodeView
      ? (graph.complete ? "complete community graph" : `${formatNumber(graph.assignedNodeCount)} nodes assigned; use Load all for the complete view`)
      : `complete ${state.level} level`;
    panel.innerHTML = `<div class="inspector-empty"><div class="nest-mark">M</div><p class="eyebrow">${esc(titleCase(state.level))}</p><h2>${heading}</h2><p><b>Hover</b> previews. <b>Click</b> pins a node or connection here. <b>Double-click or hold</b> a narrative/community to move inward. Drag any node to rearrange the view.</p><div class="stats-row"><div class="stat-box"><b>${formatNumber(graph.nodes.length)}</b><small>${esc(firstStat)}</small></div><div class="stat-box"><b>${formatNumber(graph.edges.length)}</b><small>${graph.complete ? "complete internal links" : "links in the instant view"}</small></div></div><p class="technical-id">${esc(state.bundle.hierarchy.manifest.preview_contract)}</p></div>`;
    panel.scrollTop = 0;
    return;
  }
  if (state.selected.kind === "edge") renderEdgeInspector(state.selected.data);
  else renderNodeInspector(state.selected.data);
}

function renderNodeInspector(node) {
  const panel = $("#graph-inspector");
  const connections = graphConnections(node.id);
  const displayableConnections = connections.slice(0, MAX_NEIGHBORS);
  const visibleConnections = displayableConnections.slice(0, state.neighborLimit);
  const manifest = state.bundle.hierarchy.manifest;
  const corpusDocs = Number(manifest.counts.document_count || 3270);
  const corpusNodes = Number(manifest.counts.node_count || 1);
  const visibleMaxWeight = Math.max(1, ...state.graph.graph.nodes.map((item) => Number(item.weight || item.member_count || 0)));
  const metrics = node.kind === "narrative"
    ? [[node.document_count, corpusDocs, "documents represented", `${formatNumber(node.document_count)} of ${formatNumber(corpusDocs)} corpus documents`], [node.member_count, corpusNodes, "semantic membership", `${formatNumber(node.member_count)} of ${formatNumber(corpusNodes)} nodes`], [node.community_count, state.bundle.hierarchy.communities.length, "contained communities", `${formatNumber(node.community_count)} communities`]]
    : node.kind === "community"
      ? [[node.document_count, corpusDocs, "documents represented", `${formatNumber(node.document_count)} of ${formatNumber(corpusDocs)} corpus documents`], [node.member_count, corpusNodes, "semantic membership", `${formatNumber(node.member_count)} corpus nodes`], [node.displayed_node_count, Math.max(1, node.member_count), "instant startup preview", `${formatNumber(node.displayed_node_count)} nodes preloaded; all ${formatNumber(node.member_count)} remain searchable and can be rendered on demand`]]
      : [[Math.max(Number(node.doc_count || 0), node.doc_id ? 1 : 0), corpusDocs, "source document reach", `${formatNumber(Math.max(Number(node.doc_count || 0), node.doc_id ? 1 : 0))} corpus documents`], [Number(node.weight || 0), visibleMaxWeight, "relative weighted degree", `${Number(node.weight || 0).toFixed(2)} in this graph`], [connections.length, Math.max(1, state.graph.graph.nodes.length - 1), "visible neighbors", `${formatNumber(connections.length)} connected nodes`]];
  const profileValues = [...(node.top_entities || []), ...(node.top_events || []), ...(node.top_claims || [])].slice(0, 8);
  const snippets = connections.flatMap((item) => (item.edge.evidence || []).map((evidence) => ({...evidence, neighbor: item.node}))).slice(0, 4);
  panel.innerHTML = `<div class="inspector-title"><div class="type-line">${esc(nodeTypeLabel(node))}${node.injected ? " / search-injected" : ""}</div><h2>${esc(node.label)}</h2></div>
    ${node.summary || node.semantic_text ? `<p class="inspector-summary">${esc(compact(node.summary || node.semantic_text, node.kind === "node" ? 220 : 420))}</p>` : ""}
    ${inspectorLegendMarkup()}
    <section class="inspector-section"><h3>Scale in context</h3><div class="metric-stack">${metrics.map(([value,total,label,detail]) => metricBar(value,total,label,detail)).join("")}</div></section>
    ${node.kind === "node" ? hierarchyPlacementMarkup(node) : ""}
    ${connections.length ? `<section class="inspector-section"><h3>Local neighborhood</h3>${localNetworkSvg(node, visibleConnections, connections.length)}<div class="neighbor-list">${visibleConnections.map((item) => `<div class="neighbor-row"><button class="neighbor-link" data-select-node="${esc(item.node.id)}"><span style="--neighbor:${colorFor(item.node)}"></span><b>${esc(item.node.label)}</b><small>${formatNumber(item.edge.total_count || 1)} relation${Number(item.edge.total_count || 1) === 1 ? "" : "s"}</small></button><button class="neighbor-edge" data-select-edge="${esc(item.edge.id)}" aria-label="Inspect connection to ${esc(item.node.label)}">Inspect link</button></div>`).join("")}</div>${neighborhoodControls(visibleConnections.length, displayableConnections.length, connections.length)}</section>` : ""}
    ${node.kind !== "node" && connections.length ? regionBridgeMarkup(connections) : ""}
    ${profileValues.length ? `<section class="inspector-section"><h3>Explore related concepts</h3><div class="profile-chips">${profileValues.map((value) => `<button data-profile-query="${esc(value)}">${esc(compact(value, 72))}</button>`).join("")}</div></section>` : ""}
    ${node.kind === "node" && snippets.length ? `<section class="inspector-section"><h3>Where this connection appears</h3>${snippets.map((item) => evidenceCard(item, node, item.neighbor)).join("")}</section>` : ""}
    ${node.kind === "narrative" ? `<button class="primary-action" id="inspector-drill">Open ${formatNumber(node.community_count)} communities</button><p class="interaction-hint">Double-click or hold the node to do the same from the map.</p>` : ""}
    ${node.kind === "community" ? `<button class="primary-action" id="inspector-drill">Open ${formatNumber(node.displayed_node_count)}-node instant view</button>${Number(node.member_count) > Number(node.displayed_node_count) ? `<button class="secondary-action" id="inspector-load-complete">Load and open all ${formatNumber(node.member_count)} assigned nodes</button>` : ""}<p class="interaction-hint">The preview limits browser drawing only. Search and retrieval use the complete graph index.</p>` : ""}
    ${node.kind === "node" && node.doc_id ? `<button class="secondary-action" data-source-doc="${esc(node.doc_id)}" data-source-paragraph="${esc(node.paragraph_id)}" data-highlight="${esc((node.highlight_terms?.length ? node.highlight_terms : [node.label]).join("||"))}">View highlighted source cable</button>` : ""}
    <details class="technical-details"><summary>Technical identity</summary><div class="technical-id">${esc(node.id)}</div></details>`;
  panel.scrollTop = 0;
  $("#inspector-drill", panel)?.addEventListener("click", () => {
    if (node.kind === "narrative") {
      state.narrativeId = node.id;
      transitionGraph("communities", null, {viaNodeId:node.id, direction:"in"});
    }
    if (node.kind === "community") {
      state.communityId = node.id;
      transitionGraph("nodes", null, {viaNodeId:node.id, direction:"in"});
    }
  });
  $("#inspector-load-complete", panel)?.addEventListener("click", async () => {
    state.communityId = node.id;
    transitionGraph("nodes", null, {viaNodeId:node.id, direction:"in"});
    await loadCompleteCommunity(node.id);
  });
  $("#neighbors-more", panel)?.addEventListener("click", () => { state.neighborLimit = Math.min(MAX_NEIGHBORS, state.neighborLimit + NEIGHBOR_STEP); renderNodeInspector(node); });
  $("#neighbors-less", panel)?.addEventListener("click", () => { state.neighborLimit = NEIGHBOR_STEP; renderNodeInspector(node); });
  $$('[data-parent-level]', panel).forEach((button) => button.addEventListener("click", () => {
    state.narrativeId = node.narrative_id;
    state.communityId = button.dataset.parentLevel === "nodes" ? node.community_id : null;
    transitionGraph(button.dataset.parentLevel, button.dataset.parentFocus || null);
  }));
  bindInspectorActions(panel);
}

function inspectorLegendMarkup() {
  return `<details class="inspector-key"><summary>Color and hierarchy key</summary><div><span><i class="dot narrative"></i>Narrative</span><span><i class="dot community"></i>Community</span><span><i class="dot entity"></i>Entity</span><span><i class="dot event"></i>Event</span><span><i class="dot claim"></i>Claim</span><span><i class="dot relation"></i>Relation evidence</span></div><p>Node color encodes level or semantic type. Size encodes membership at hierarchy levels and weighted degree in a community. Lines aggregate lower-level relations; select a line for its bridge citations.</p></details>`;
}

function hierarchyPlacementMarkup(node) {
  const community = indexes.community.get(node.community_id);
  const narrative = indexes.narrative.get(node.narrative_id);
  if (!community && !narrative) return "";
  return `<section class="inspector-section hierarchy-placement"><h3>Primary hierarchy placement</h3><p>Each canonical semantic node has one primary community and narrative assignment for navigation. Its cited mentions and graph links may span many documents and cross into other regions.</p><div>${narrative ? `<button data-parent-level="narratives" data-parent-focus="${esc(narrative.id)}"><small>Narrative</small><b>${esc(narrative.label)}</b></button>` : ""}${community ? `<button data-parent-level="communities" data-parent-focus="${esc(community.id)}"><small>Community</small><b>${esc(community.label)}</b></button>` : ""}</div></section>`;
}

function neighborhoodControls(shown, displayable, total) {
  if (total <= NEIGHBOR_STEP) return `<p class="neighborhood-count">All ${formatNumber(total)} visible neighbors shown.</p>`;
  const scope = total > MAX_NEIGHBORS ? `${formatNumber(displayable)} strongest of ${formatNumber(total)} connected` : `${formatNumber(total)} connected`;
  return `<div class="neighborhood-controls"><span>Showing ${formatNumber(shown)} / ${scope}</span><div>${shown < displayable ? `<button id="neighbors-more">+5</button>` : ""}${shown > NEIGHBOR_STEP ? `<button id="neighbors-less">Collapse</button>` : ""}</div></div>`;
}

function regionBridgeMarkup(connections) {
  const rows = connections.slice(0, Math.min(state.neighborLimit, 6));
  return `<section class="inspector-section region-bridges"><h3>Why these regions connect</h3><p class="inspector-summary">A hierarchy edge is not a free-text similarity claim. It aggregates semantic-node relations crossing the two regions. Open the connection to inspect its scale, retained relation types, lower-level endpoints, and cable provenance.</p>${rows.map(({edge,node}) => `<article class="bridge-card"><button class="bridge-neighbor" data-select-node="${esc(node.id)}"><span>Connected ${esc(nodeTypeLabel(node))}</span><b>${esc(node.label)}</b><small>${formatNumber(edge.total_count || 0)} underlying relations / ${Number(edge.weight || 0).toFixed(2)} aggregate weight</small></button><button class="bridge-edge" data-select-edge="${esc(edge.id)}">Inspect this connection</button>${(edge.evidence || []).slice(0,2).map((item) => evidenceCard(item, null, null)).join("") || `<p>No text-bearing relation was retained in this bounded preview. Open the connection for its aggregate structure.</p>`}</article>`).join("")}</section>`;
}

function metricBar(value, total, label, detail) {
  const ratio = Math.max(.01, Math.min(1, Number(value || 0) / Math.max(1, Number(total || 1))));
  return `<div class="metric-row"><div><b>${esc(label)}</b><span>${esc(detail)}</span></div><div class="metric-track"><i style="width:${ratio * 100}%"></i></div></div>`;
}

function graphConnections(nodeId) {
  if (!state.graph) return [];
  return state.graph.graph.edges
    .filter((edge) => edge.source === nodeId || edge.target === nodeId)
    .map((edge) => ({edge, node: state.graph.nodeMap.get(edge.source === nodeId ? edge.target : edge.source)}))
    .filter((item) => item.node)
    .sort((left, right) => Number(right.edge.weight || 0) - Number(left.edge.weight || 0));
}

function localNetworkSvg(node, connections, total) {
  const shown = connections.slice(0, MAX_NEIGHBORS); const centerX = 130; const centerY = 92; const orbit = shown.length > 9 ? 70 : 66;
  const lines = shown.map((item, index) => { const angle = -Math.PI / 2 + index * Math.PI * 2 / shown.length; const x = centerX + Math.cos(angle) * orbit; const y = centerY + Math.sin(angle) * orbit; return `<line x1="${centerX}" y1="${centerY}" x2="${x}" y2="${y}"/><circle cx="${x}" cy="${y}" r="7" style="fill:${colorFor(item.node)}"/><title>${esc(item.node.label)}</title>`; }).join("");
  return `<svg class="local-network" viewBox="0 0 260 184" role="img" aria-label="Selected node and strongest visible neighbors"><g>${lines}</g><circle class="network-center" cx="${centerX}" cy="${centerY}" r="13"/><text x="${centerX}" y="${centerY + 31}" text-anchor="middle">${esc(compact(node.label, 34))}</text><text class="network-note" x="252" y="176" text-anchor="end">${shown.length} strongest of ${total}</text></svg>`;
}

function bindInspectorActions(root) {
  bindSourceButtons(root);
  $$('[data-select-node]', root).forEach((button) => button.addEventListener("click", () => {
    const node = state.graph?.nodeMap.get(button.dataset.selectNode);
    if (!node) return;
    state.selected = {kind: "node", data: node}; state.neighborLimit = NEIGHBOR_STEP; renderInspector(); drawGraph();
  }));
  $$('[data-select-edge]', root).forEach((button) => button.addEventListener("click", () => {
    const edge = state.graph?.graph.edges.find((item) => item.id === button.dataset.selectEdge);
    if (!edge) return;
    state.selected = {kind: "edge", data: edge}; state.edgePage = 0; renderInspector(); drawGraph();
  }));
  $$('[data-profile-query]', root).forEach((button) => button.addEventListener("click", () => {
    $("#global-search").value = button.dataset.profileQuery; state.searchScopes = ["all"]; updateSearchFilterLabel(); runSearch(button.dataset.profileQuery);
  }));
}

function renderEdgeInspector(edge) {
  const panel = $("#graph-inspector");
  const source = state.graph.nodeMap.get(edge.source) || indexes.node.get(edge.source) || indexes.community.get(edge.source) || indexes.narrative.get(edge.source);
  const target = state.graph.nodeMap.get(edge.target) || indexes.node.get(edge.target) || indexes.community.get(edge.target) || indexes.narrative.get(edge.target);
  const evidence = edge.evidence || [];
  const pageSize = 5; const pageCount = Math.max(1, Math.ceil(evidence.length / pageSize)); state.edgePage = Math.min(state.edgePage, pageCount - 1);
  const shown = evidence.slice(state.edgePage * pageSize, (state.edgePage + 1) * pageSize);
  const relations = Object.entries(edge.relation_counts || (edge.relation ? {[edge.relation]: 1} : {}));
  const visibleWeights = state.graph.graph.edges.map((item) => Number(item.weight || 0));
  const maxWeight = Math.max(1, ...visibleWeights);
  const maxCount = Math.max(1, ...state.graph.graph.edges.map((item) => Number(item.total_count || 1)));
  const scope = edgeScopeDetails(edge, source, target, evidence);
  const direct = state.level === "nodes";
  const countLabel = direct ? "stored edge instances" : "underlying relations";
  const countDetail = direct
    ? `${formatNumber(edge.total_count || 1)} direct graph edge`
    : `${formatNumber(edge.total_count || evidence.length)} aggregated lower-level links`;
  const weightLabel = direct ? "relation weight" : "aggregate strength";
  panel.innerHTML = `<div class="inspector-title"><div class="type-line">${esc(scope.label)}</div><h2>${esc(scope.heading)}</h2></div>
    <p class="edge-scope-explanation">${esc(scope.explanation)}</p>
    <div class="endpoint-pair"><button class="endpoint" data-select-node="${esc(source?.id || edge.source)}"><small>${esc(nodeTypeLabel(source || {kind:state.level === "nodes" ? "node" : state.level.slice(0,-1)}))} A</small><b>${esc(source?.label || edge.source)}</b><span>${esc(endpointScale(source))}</span></button><div class="endpoint-arrow">${direct ? "-&gt;" : "&harr;"}</div><button class="endpoint" data-select-node="${esc(target?.id || edge.target)}"><small>${esc(nodeTypeLabel(target || {kind:state.level === "nodes" ? "node" : state.level.slice(0,-1)}))} B</small><b>${esc(target?.label || edge.target)}</b><span>${esc(endpointScale(target))}</span></button></div>
    <section class="inspector-section"><h3>What this line summarizes</h3><div class="connection-facts">${scope.facts.map(([value,label]) => `<article><b>${esc(value)}</b><span>${esc(label)}</span></article>`).join("")}</div></section>
    <section class="inspector-section"><h3>Strength in this level</h3><div class="metric-stack">${metricBar(edge.total_count || evidence.length, maxCount, countLabel, countDetail)}${metricBar(edge.weight || 0, maxWeight, weightLabel, `${Number(edge.weight || 0).toFixed(2)} relative edge weight`)}</div></section>
    <section class="inspector-section"><h3>Retained relation examples</h3><p class="inspector-summary">${direct ? "This is the typed relation stored for the selected semantic edge." : "These labels summarize the strongest provenance examples retained for display, not every relation counted in the aggregate."}</p><div class="relation-tags">${relations.map(([name,count]) => `<span class="relation-tag">${esc(titleCase(name))} ${count > 1 ? `x${count}` : ""}</span>`).join("") || `<span class="relation-tag">Corpus co-connection</span>`}</div></section>
    <section class="inspector-section"><h3>${state.level === "nodes" ? "Direct source provenance" : "Strongest lower-level bridges"}</h3>${shown.length ? shown.map((item) => evidenceCard(item, source, target)).join("") : `<p class="inspector-summary">This connection has no text-bearing example in the bounded display payload. Its aggregate count still comes from stored lower-level graph links.</p>`}</section>
    ${evidence.length > pageSize ? `<div class="pager"><button id="edge-prev" ${state.edgePage === 0 ? "disabled" : ""}>Previous</button><span>${state.edgePage * pageSize + 1}-${Math.min((state.edgePage + 1) * pageSize, evidence.length)} of ${formatNumber(edge.total_count || evidence.length)}${edge.total_count > evidence.length ? ` / top ${evidence.length} retained` : ""}</span><button id="edge-next" ${state.edgePage >= pageCount - 1 ? "disabled" : ""}>Next</button></div>` : ""}
    <details class="technical-details"><summary>Technical identity</summary><div class="technical-id">${esc(edge.id)}</div></details>`;
  panel.scrollTop = 0;
  $("#edge-prev", panel)?.addEventListener("click", () => { state.edgePage -= 1; renderEdgeInspector(edge); });
  $("#edge-next", panel)?.addEventListener("click", () => { state.edgePage += 1; renderEdgeInspector(edge); });
  bindInspectorActions(panel);
}

function endpointScale(node) {
  if (!node) return "Open endpoint";
  if (node.kind === "narrative") return `${formatNumber(node.community_count)} communities / ${formatNumber(node.document_count)} documents`;
  if (node.kind === "community") return `${formatNumber(node.member_count)} semantic nodes / ${formatNumber(node.document_count)} documents`;
  return nodeTypeLabel(node) || "Semantic node";
}

function edgeScopeDetails(edge, source, target, evidence) {
  const documents = new Set(evidence.map((item) => item.doc_id).filter(Boolean)).size;
  if (state.level === "narratives") {
    return {
      label:"Narrative bridge",
      heading:"Cross-narrative connection",
      explanation:"This line aggregates graph relations whose semantic endpoints belong to communities assigned to different narratives. The narrative labels are organizational summaries; the cited lower-level relations are what create the bridge.",
      facts:[
        [formatNumber(edge.community_link_count || 0), "community-pair links"],
        [formatNumber(edge.total_count || 0), "cross-narrative semantic relations"],
        [formatNumber(documents), "documents in retained examples"],
      ],
    };
  }
  if (state.level === "communities") {
    const narrative = indexes.narrative.get(source?.narrative_id || target?.narrative_id || state.narrativeId);
    return {
      label:"Community bridge",
      heading:"Cross-community connection",
      explanation:`This line aggregates graph relations crossing two communities inside ${narrative?.label || "the selected narrative"}. Community membership organizes the corpus; it does not prevent relations from crossing the partition.`,
      facts:[
        [formatNumber(edge.total_count || 0), "cross-community semantic relations"],
        [formatNumber(evidence.length), "strongest examples retained"],
        [formatNumber(documents), "documents in retained examples"],
      ],
    };
  }
  return {
    label:"Direct semantic relation",
    heading:titleCase(edge.relation || relationsLabel(edge)),
    explanation:"At the semantic-node level, this line is a stored graph relation rather than a hierarchy aggregate. Direction follows endpoint A to endpoint B; the source passage below is the provenance for that relation.",
    facts:[
      [titleCase(edge.relation || relationsLabel(edge)), "stored relation type"],
      [Number(edge.weight || 0).toFixed(2), "relation weight"],
      [formatNumber(documents), "provenance documents"],
    ],
  };
}

function relationsLabel(edge) {
  return Object.keys(edge.relation_counts || {})[0] || "connected";
}

function evidenceCard(item, source, target) {
  const sourceTerm = item.source_label || source?.label;
  const targetTerm = item.target_label || target?.label;
  const rawText = item.text || `${item.source_label} ${titleCase(item.relation)} ${item.target_label}`;
  const text = highlighted(rawText, [sourceTerm, targetTerm].filter((value) => value && value.length < 70));
  return `<article class="evidence-card"><header><b>${esc(titleCase(item.relation || "relation"))}</b><span>${Number(item.weight || 0).toFixed(2)}</span></header><div class="evidence-pair"><span>${esc(sourceTerm || "source node")}</span><i>-&gt;</i><span>${esc(targetTerm || "target node")}</span></div><p>${text || "No text-bearing excerpt was stored for this edge."}</p>${item.doc_id ? `<button class="source-link" data-source-doc="${esc(item.doc_id)}" data-source-paragraph="${esc(item.paragraph_id)}" data-highlight="${esc([sourceTerm, targetTerm].filter(Boolean).join("||"))}">${esc(item.doc_id)} / ${esc(item.paragraph_id || "source")}</button>` : ""}</article>`;
}

function bindSourceButtons(root = document) {
  $$('[data-source-doc]', root).forEach((button) => button.addEventListener("click", () => openSource(button.dataset.sourceDoc, button.dataset.sourceParagraph || "", button.dataset.highlight || "")));
}

async function openSource(docId, paragraphId = "", highlight = "") {
  trackInteraction("open_source", "document", docId, {paragraph_id:paragraphId || ""});
  const drawer = $("#source-drawer");
  drawer.classList.remove("guide-destination-document");
  $("#drawer-kicker").textContent = "Source cable";
  $("#source-title").textContent = docId;
  $("#source-meta").innerHTML = `<span>Loading source</span>`;
  $("#source-body").innerHTML = "";
  drawer.classList.add("open"); drawer.setAttribute("aria-hidden", "false"); $("#drawer-scrim").hidden = false;
  try {
    const source = state.staticMode
      ? (await staticResource("sources"))[docId]
      : await requestJson(
          `/api/source?doc_id=${encodeURIComponent(docId)}&paragraph_id=${encodeURIComponent(paragraphId)}`,
          {},
          "Source lookup",
        );
    if (!source) throw new Error(`Source ${docId} is not included in this hosted replay. Use the local server for corpus-wide source access.`);
    source.segments = (source.segments || []).map((segment) => ({...segment, active: String(segment.paragraph_id) === String(paragraphId)}));
    $("#source-title").textContent = source.subject && source.subject !== source.doc_id
      ? `${source.subject} / ${source.doc_id}`
      : source.doc_id;
    $("#source-meta").innerHTML = [source.doc_id, source.date, source.classification, source.handling].filter(Boolean).map((value) => `<span>${esc(value)}</span>`).join("");
    const terms = String(highlight || "").split("||").map((item) => item.trim()).filter(Boolean);
    $("#source-body").innerHTML = source.segments.map((segment) => `<section class="source-segment ${segment.active ? "active" : ""}" data-segment="${esc(segment.paragraph_id)}">${segment.heading ? `<b>${esc(segment.heading)}</b><br>` : ""}${highlighted(segment.text, terms)}</section>`).join("");
    setTimeout(() => $(".source-segment.active")?.scrollIntoView({block: "center", behavior: "smooth"}), 150);
  } catch (error) {
    $("#source-body").innerHTML = `<p>${esc(error.message)}</p>`;
  }
}

function closeSource() {
  $("#source-drawer").classList.remove("open", "guide-destination-document"); $("#source-drawer").setAttribute("aria-hidden", "true"); $("#drawer-scrim").hidden = true;
}

function openDetailDrawer(kicker, title, meta, body) {
  $("#drawer-kicker").textContent = kicker;
  $("#source-title").textContent = title;
  $("#source-meta").innerHTML = meta.map((value) => `<span>${esc(value)}</span>`).join("");
  $("#source-body").innerHTML = body;
  $("#source-drawer").classList.add("open");
  $("#source-drawer").setAttribute("aria-hidden", "false");
  $("#drawer-scrim").hidden = false;
  bindSourceButtons($("#source-body"));
}

function segmentIdentity(step) {
  const match = String(step.id || "").match(/^SEGMENT:([^:]+)::(.+)$/);
  return {
    docId: step.doc_id || match?.[1] || "",
    paragraphId: step.paragraph_id || match?.[2] || "",
  };
}

function isTerminalTargetSegment(path, step, sourceIndex) {
  const routeSteps = path.steps || [];
  if (sourceIndex !== routeSteps.length - 1 || (step.type !== "Segment" && !String(step.id || "").startsWith("SEGMENT:"))) return false;
  const segment = segmentIdentity(step);
  if (!segment.docId || segment.docId !== path.doc_id) return false;
  return !path.paragraph_id || !segment.paragraphId || segment.paragraphId === path.paragraph_id;
}

function terminalTargetSegment(path) {
  const routeSteps = path.steps || [];
  const sourceIndex = routeSteps.length - 1;
  const step = routeSteps[sourceIndex];
  return step && isTerminalTargetSegment(path, step, sourceIndex) ? {step, sourceIndex} : null;
}

function visibleRouteStepEntries(path) {
  return (path.steps || [])
    .map((step, sourceIndex) => ({step, sourceIndex}))
    .filter(({step, sourceIndex}) => !isTerminalTargetSegment(path, step, sourceIndex));
}

function edgeTraversalExplanation(edgeType, fromLabel, toLabel) {
  const raw = String(edgeType || "TRAVERSE").toUpperCase();
  const reverse = raw.endsWith(":REV");
  const base = raw.replace(/:REV$/, "");
  const from = fromLabel || "the prior node";
  const to = toLabel || "the next node";
  const directions = {
    RELATION_SOURCE: reverse
      ? `${from} is the stored source or subject of ${to}; retrieval followed that edge from the entity into the relation node.`
      : `${from} is a relation node whose stored source or subject is ${to}.`,
    RELATION_TARGET: reverse
      ? `${from} is the stored target or object of ${to}; retrieval followed that edge from the entity into the relation node.`
      : `${from} is a relation node whose stored target or object is ${to}.`,
    SEGMENT_HAS_ENTITY: reverse
      ? `${from} is mentioned in ${to}; retrieval followed the segment-membership edge back to its source passage.`
      : `${from} is a source segment that contains the entity ${to}.`,
    CLAIM_CONTEXT_ENTITY: reverse
      ? `${from} is a contextual entity attached to the claim ${to}.`
      : `${from} names ${to} as a contextual entity.`,
    CLAIM_SPEAKER: reverse
      ? `${from} is recorded as the speaker attached to the claim ${to}.`
      : `${from} attributes the statement to ${to}.`,
    CLAIM_TARGET: reverse
      ? `${from} is recorded as the target attached to the claim ${to}.`
      : `${from} directs the claim toward ${to}.`,
    EVENT_PARTICIPANT: reverse
      ? `${from} participates in the event node ${to}.`
      : `${from} records ${to} as an event participant.`,
    EVENT_LOCATION: reverse
      ? `${from} is the location attached to the event node ${to}.`
      : `${from} records ${to} as the event location.`,
    MENTIONS: reverse
      ? `${from} is mentioned by ${to}.`
      : `${from} mentions ${to}.`,
  };
  const detail = directions[base] || `${from} and ${to} are connected by the stored ${titleCase(base)} edge.`;
  return {
    base,
    reverse,
    label:titleCase(raw),
    detail:`${detail}${reverse ? " REV marks traversal opposite the stored edge direction; it does not reverse the factual claim or signal contradiction." : ""}`,
  };
}

function guidePlacement(step) {
  return {
    narrativeId:step.narrative_id || step.narrativeId || "",
    communityId:step.community_id || step.communityId || "",
  };
}

function sameGuidePlacement(left, right) {
  return Boolean(left && right) && left.narrativeId === right.narrativeId && left.communityId === right.communityId;
}

function recordedRouteGuide(item, path) {
  const semanticSteps = path.steps || [];
  const steps = [{
    kind:"atom",
    label:path.route_text,
    detail:`${path.route_id || "Atomic target"} starts this archived retrieval route. Each following edge is a discovery path, not by itself proof.`,
  }];
  let activePlacement = null;
  let lastPlacedNode = null;
  let previousSemantic = null;
  let terminalDocumentCaptured = false;
  semanticSteps.forEach((step, sourceStepIndex) => {
    const placement = guidePlacement(step);
    const hasPlacement = Boolean(placement.narrativeId || placement.communityId);
    const edge = edgeTraversalExplanation(
      step.incoming_edge,
      previousSemantic?.label,
      step.label,
    );
    if (!hasPlacement) {
      const segment = segmentIdentity(step);
      const targetSegment = isTerminalTargetSegment(path, step, sourceStepIndex);
      steps.push({
        kind:targetSegment ? "document" : "provenance",
        id:targetSegment ? path.doc_id : step.id,
        label:targetSegment ? path.doc_id : [segment.docId, segment.paragraphId].filter(Boolean).join(" / ") || step.label || step.id,
        sourceStepIndex,
        docId:segment.docId,
        paragraphId:path.paragraph_id || segment.paragraphId,
        edgeType:step.incoming_edge || "",
        edgeDocId:step.incoming_edge_doc_id || segment.docId,
        edgeParagraphId:step.incoming_edge_paragraph_id || segment.paragraphId,
        edgeExcerpt:step.incoming_edge_excerpt || step.semantic_text || "",
        edgeHighlights:step.incoming_edge_highlight_terms || step.highlight_terms || [],
        detail:targetSegment
          ? `${edge.detail} This source segment is the terminal candidate passage, so the segment provenance and parent cable are presented once.`
          : step.incoming_edge
          ? `${edge.detail} This source-segment node carries document provenance and intentionally has no narrative or community assignment.`
          : "This source-segment seed carries exact cable provenance. Segment nodes are not assigned to narratives or communities, so the viewer does not invent a hierarchy placement for it.",
      });
      if (targetSegment) terminalDocumentCaptured = true;
      previousSemantic = step;
      return;
    }

    if (!activePlacement) {
      if (placement.narrativeId) steps.push({
        kind:"narrative",
        id:placement.narrativeId,
        label:indexes.narrative.get(placement.narrativeId)?.label || placement.narrativeId,
        detail:"Narrative containing the first hierarchy-assigned semantic node. This is an organizational container, not evidence.",
      });
      if (placement.communityId) steps.push({
        kind:"community",
        id:placement.communityId,
        narrativeId:placement.narrativeId,
        label:indexes.community.get(placement.communityId)?.label || placement.communityId,
        detail:"Community containing the first hierarchy-assigned semantic node. Graph edges may legitimately cross this clustering boundary.",
      });
    }

    let bridge = null;
    if (activePlacement && !sameGuidePlacement(activePlacement, placement)) {
      const narrativeChanged = activePlacement.narrativeId !== placement.narrativeId;
      const previousPlacement = guidePlacement(previousSemantic || {});
      const directHierarchyEdge = Boolean(previousPlacement.narrativeId || previousPlacement.communityId);
      const fromNarrativeLabel = indexes.narrative.get(activePlacement.narrativeId)?.label || activePlacement.narrativeId || "unplaced provenance";
      const toNarrativeLabel = indexes.narrative.get(placement.narrativeId)?.label || placement.narrativeId || "unplaced provenance";
      const fromCommunityLabel = indexes.community.get(activePlacement.communityId)?.label || activePlacement.communityId || "unplaced provenance";
      const toCommunityLabel = indexes.community.get(placement.communityId)?.label || placement.communityId || "unplaced provenance";
      bridge = {
        kind:"bridge",
        id:`bridge:${sourceStepIndex}:${step.id}`,
        label:narrativeChanged
          ? `${fromNarrativeLabel} -> ${toNarrativeLabel}`
          : `${fromCommunityLabel} -> ${toCommunityLabel}`,
        detail:`${edge.detail} ${directHierarchyEdge ? `Clustering places the connected semantic nodes in different ${narrativeChanged ? "narratives" : "communities"}` : "The intervening source segment has no hierarchy placement, and the route re-enters through a different hierarchy region"}; hierarchy membership does not block a graph route.`,
        scope:narrativeChanged ? "narrative" : "community",
        edgeType:step.incoming_edge || "",
        edgeLabel:edge.label,
        edgeDocId:step.incoming_edge_doc_id || "",
        edgeParagraphId:step.incoming_edge_paragraph_id || "",
        edgeExcerpt:step.incoming_edge_excerpt || "",
        edgeHighlights:step.incoming_edge_highlight_terms || [],
        edgeWeight:Number(step.incoming_edge_weight || 0),
        fromNodeId:lastPlacedNode?.id || "",
        fromNodeLabel:previousSemantic?.label || lastPlacedNode?.label || "prior node",
        fromNarrativeId:activePlacement.narrativeId,
        fromNarrativeLabel,
        fromCommunityId:activePlacement.communityId,
        fromCommunityLabel,
        toNodeId:step.id,
        toNodeLabel:step.label || step.id,
        toNarrativeId:placement.narrativeId,
        toNarrativeLabel,
        toCommunityId:placement.communityId,
        toCommunityLabel,
      };
      steps.push(bridge);
    }

    const node = {
      id:step.id,
      kind:"node",
      type:step.type || "Unknown",
      label:step.label || step.id,
      semantic_text:step.semantic_text || step.label,
      doc_id:step.doc_id || "",
      paragraph_id:step.paragraph_id || "",
      highlight_terms:step.highlight_terms || [],
      community_id:placement.communityId,
      narrative_id:placement.narrativeId,
      weight:1,
      injected:true,
    };
    if (!indexes.node.has(node.id)) indexes.node.set(node.id, node);
    else if (node.highlight_terms.length) indexes.node.get(node.id).highlight_terms = node.highlight_terms;
    steps.push({
      kind:"node",
      id:node.id,
      narrativeId:node.narrative_id,
      communityId:node.community_id,
      label:node.label,
      sourceStepIndex,
      docId:node.doc_id,
      paragraphId:node.paragraph_id,
      edgeType:step.incoming_edge || "",
      edgeDocId:step.incoming_edge_doc_id || "",
      edgeParagraphId:step.incoming_edge_paragraph_id || "",
      edgeExcerpt:step.incoming_edge_excerpt || "",
      edgeHighlights:step.incoming_edge_highlight_terms || [],
      transition:bridge,
      detail:sourceStepIndex === 0
        ? `${node.type} semantic seed selected for this atom.`
        : edge.detail,
    });
    activePlacement = placement;
    lastPlacedNode = node;
    previousSemantic = step;
  });
  const lastSourceStep = [...semanticSteps].reverse().find((step) => step.doc_id === path.doc_id && step.paragraph_id);
  if (!terminalDocumentCaptured) {
    steps.push({
      kind:"document",
      id:path.doc_id,
      paragraphId:path.paragraph_id || lastSourceStep?.paragraph_id || "",
      label:path.doc_id || "Unbound candidate",
      detail:path.citation_quarantined
        ? `Filtered candidate cable reached at rank ${path.rank || "-"}, but its raw passage did not align with the atom strongly enough to certify evidence. It remains retrieval context only.`
        : `Filtered candidate cable reached at rank ${path.rank || "-"}. Retrieval discovery is not proof.`,
    });
  }
  return {
    title:`${item.title} / ${path.route_id || "route"} / ${path.doc_id || "unbound"}`,
    subtitle:path.route_text,
    task:item.task,
    relation:(path.edge_types || []).join(" -> ") || path.path_kind,
    highlight:(path.highlight_terms?.length ? path.highlight_terms : salientTerms(path.route_text)).join("||"),
    returnCaseId:item.id,
    steps,
  };
}

function openRouteDetail(item, pathIndex, stepIndex) {
  const path = item.graph_paths[pathIndex]; if (!path) return;
  const guide = recordedRouteGuide(item, path);
  const rawSteps = path.steps || [];
  const stepEntries = visibleRouteStepEntries(path);
  const targetSegment = terminalTargetSegment(path);
  const routeHighlights = path.highlight_terms?.length ? path.highlight_terms : salientTerms(path.route_text);
  const cards = stepEntries.map(({step, sourceIndex}, displayIndex) => {
    const stepHighlights = step.highlight_terms?.length ? step.highlight_terms : routeHighlights;
    const edge = step.incoming_edge ? edgeTraversalExplanation(step.incoming_edge, rawSteps[sourceIndex - 1]?.label, step.label) : null;
    const edgeTerms = step.incoming_edge_highlight_terms || stepHighlights;
    const edgeSource = step.incoming_edge_doc_id
      ? `<div class="route-edge-source"><b>Why this edge exists</b>${step.incoming_edge_excerpt ? `<blockquote>${highlighted(step.incoming_edge_excerpt, edgeTerms)}</blockquote>` : ""}<button class="source-link" data-source-doc="${esc(step.incoming_edge_doc_id)}" data-source-paragraph="${esc(step.incoming_edge_paragraph_id || "")}" data-highlight="${esc(edgeTerms.join("||"))}">Open edge source / ${esc(step.incoming_edge_doc_id)}${step.incoming_edge_paragraph_id ? ` / ${esc(step.incoming_edge_paragraph_id)}` : ""}</button></div>`
      : "";
    return `<article class="trail-step ${sourceIndex === stepIndex ? "selected" : ""}"><span class="trail-index">${String(displayIndex + 1).padStart(2,"0")}</span><div><small>${esc(step.type)}${step.incoming_edge ? ` / via ${esc(titleCase(step.incoming_edge))}` : " / semantic seed"}</small><h3>${esc(step.label)}</h3>${edge ? `<p class="route-edge-meaning">${esc(edge.detail)}</p>` : ""}${step.semantic_text && step.semantic_text !== step.label ? `<p>${highlighted(step.semantic_text, stepHighlights)}</p>` : ""}${edgeSource}${step.doc_id ? `<button class="source-link" data-source-doc="${esc(step.doc_id)}" data-source-paragraph="${esc(step.paragraph_id)}" data-highlight="${esc(stepHighlights.join("||"))}">Open this node's exact source passage</button>` : !step.incoming_edge_doc_id ? `<p class="trail-note">This structural node has no single source passage; provenance continues through its adjacent text-bearing node.</p>` : ""}</div></article>`;
  }).join("");
  const destinationStatus = path.citation_quarantined
    ? `<p class="citation-rejected"><b>Context only.</b> The archived matcher promoted this document, but the raw citation does not preserve enough of this atom to certify support or opposition.</p>`
    : `<p>The traversal promoted this document to final rank ${path.rank || "-"}. Promotion is discovery, not proof.</p>`;
  const terminalEdge = targetSegment?.step.incoming_edge
    ? edgeTraversalExplanation(targetSegment.step.incoming_edge, rawSteps[targetSegment.sourceIndex - 1]?.label, path.doc_id)
    : null;
  const terminalHop = terminalEdge
    ? `<p class="route-edge-meaning"><b>Final graph hop:</b> ${esc(terminalEdge.detail)} The terminal segment and its parent cable are consolidated in this destination card.</p>`
    : "";
  const destination = `<article class="trail-step destination ${path.citation_quarantined ? "quarantined" : ""}"><span class="trail-index">D</span><div><small>Ranked candidate document / ${esc(path.match_basis || "retrieval provenance")}</small><h3>${esc(path.doc_id || "Unbound candidate")}</h3>${destinationStatus}${terminalHop}${path.source_excerpt ? `<blockquote>${highlighted(path.source_excerpt, routeHighlights)}</blockquote>` : ""}${path.citation_alignment_shared_terms?.length ? `<p class="trail-note">Atom-aligned surface terms: ${esc(path.citation_alignment_shared_terms.join(", "))}</p>` : ""}${path.doc_id ? `<button class="source-link" data-source-doc="${esc(path.doc_id)}" data-source-paragraph="${esc(path.paragraph_id || "")}" data-highlight="${esc(routeHighlights.join("||"))}">Open exact candidate passage</button>` : ""}</div></article>`;
  openDetailDrawer("Graph retrieval lineage", path.route_text, [titleCase(path.coverage_channel), `${path.hop_count} hops`, `rank ${path.rank || "-"}`, titleCase(path.path_kind)], `<div class="guided-route-intro recorded"><b>Walk this archived route in the corpus map</b><span>The controller stays on the graph while you move from atom to hierarchy placement, semantic nodes, and the candidate cable.</span><div><button id="recorded-guide-map">Open route in corpus map</button></div></div><div class="drawer-query"><small>Atomic route</small><b>${esc(path.route_text)}</b></div><div class="trail-stack">${cards || `<article class="trail-step selected"><span class="trail-index">0</span><div><small>Direct semantic seed</small><h3>No graph hop was needed</h3><p>The document entered through direct semantic or alias matching.</p></div></article>`}${destination}</div>`);
  const guideIndex = stepIndex < 0 ? 0 : Math.max(0, guide.steps.findIndex((step) => step.sourceStepIndex === stepIndex));
  $("#recorded-guide-map")?.addEventListener("click", () => activateGraphGuide(guide, guideIndex));
}

function openAtomDetail(item, atomIndex) {
  const atom = item.atoms[atomIndex]; if (!atom) return;
  const relatedEvidence = item.evidence.filter((evidence) => (evidence.atom_assessments || []).some((row) => row.atom_text === atom.text) || (atom.evidence_ids || []).includes(evidence.evidence_id));
  const evidenceLinks = relatedEvidence.length ? relatedEvidence.map((evidence) => `<article class="atom-citation"><span class="atom-status ${statusClass(atom.coverage_status)}">${esc(titleCase(atom.coverage_status))}</span><b>${esc(evidence.doc_id || evidence.evidence_id)}</b><p>${esc((evidence.atom_assessments || []).find((row) => row.atom_text === atom.text)?.reason || atom.coverage_reason || "Recorded as relevant to this atom.")}</p>${evidence.doc_id ? `<button class="source-link" data-source-doc="${esc(evidence.doc_id)}" data-source-paragraph="${esc(evidence.paragraph_id)}" data-highlight="${esc((evidence.highlight_terms?.length ? evidence.highlight_terms : salientTerms(atom.text)).join("||"))}">Open cited passage</button>` : ""}</article>`).join("") : `<p class="trail-note">No citation was promoted for this atom. Related retrieval routes remain available in stage 04.</p>`;
  openDetailDrawer("Atomic evidence question", `${atom.display_id} / ${atom.text}`, [titleCase(atom.coverage_status), atom.constraint_role ? titleCase(atom.constraint_role) : "required target", `${Math.round(Number(atom.confidence || 0) * 100)}% parser confidence`], `<div class="atom-detail"><dl><dt>Subject</dt><dd>${esc(atom.subject || "not resolved")}</dd><dt>Predicate</dt><dd>${esc(atom.predicate || "not resolved")}</dd><dt>Object</dt><dd>${esc(atom.object || "not resolved")}</dd></dl><h3>Recorded coverage rationale</h3><p>${esc(atom.coverage_reason || "No separate citation rationale was recorded.")}</p><h3>Cited documents</h3>${evidenceLinks}</div>`);
}

function renderCaseFilters() {
  const labels = {reconstruction: "Reconstruct", hypothesis: "Hypothesis", redaction: "Redaction"};
  if (state.workbenchMode === "live") {
    $("#task-filters").hidden = false;
    $("#task-filters").innerHTML = state.bundle.workbench.task_order.map((task) => `<button class="task-filter ${task === state.liveTask ? "active" : ""}" data-live-task="${task}">${labels[task]}</button>`).join("");
    $$('[data-live-task]', $("#task-filters")).forEach((button) => button.addEventListener("click", () => {
      state.liveTask = button.dataset.liveTask;
      state.liveDraft = null;
      state.livePayload = null;
      renderCaseFilters();
      renderWorkbench();
    }));
    renderLiveExamples();
    return;
  }
  $("#task-filters").hidden = false;
  $("#task-filters").innerHTML = state.bundle.workbench.task_order.map((task) => `<button class="task-filter ${task === state.taskFilter ? "active" : ""}" data-task="${task}">${labels[task]}</button>`).join("");
  $$(".task-filter").forEach((button) => button.addEventListener("click", () => {
    state.taskFilter = button.dataset.task;
    state.caseId = RECORDED_DEMO_CASES[state.taskFilter]?.[0] || state.bundle.workbench.cases.find((item) => item.task === state.taskFilter)?.id;
    renderCaseFilters(); renderWorkbench();
  }));
}

function renderWorkbench() {
  if (state.workbenchMode === "live") { renderLiveWorkbench(); return; }
  renderCaseList();
  const entries = [...state.bundle.workbench.cases, ...(state.bundle.workbench.candidates || [])];
  const current = entries.find((item) => item.id === state.caseId) || state.bundle.workbench.cases[0];
  if (!current) return;
  const candidate = current.audit_status === "source_verified_candidate_not_benchmarked";
  $("#case-workspace").innerHTML = candidate ? candidateMarkup(current) : caseMarkup(current);
  if (candidate) bindCandidateWorkspace(current); else bindCaseWorkspace(current);
  $("#case-workspace").scrollTop = 0;
}

const liveTaskMeta = {
  reconstruction: {label: "Reconstruction", input: "Passage or claim to reconstruct", stage: "Fact coverage review"},
  hypothesis: {label: "Hypothesis", input: "Hypothesis to test", stage: "Evidence-direction review"},
  redaction: {label: "Redaction", input: "Masked passage to recover", stage: "Shared-candidate review"},
};

function normalizedDisplayLimit(value) {
  const numeric = Number(value);
  return PATH_DISPLAY_OPTIONS.includes(numeric) ? numeric : DEFAULT_PATH_DISPLAY_LIMIT;
}

function displayLimitOptions(value) {
  const selected = normalizedDisplayLimit(value);
  return PATH_DISPLAY_OPTIONS.map((option) => `<option value="${option}" ${option === selected ? "selected" : ""}>${option}</option>`).join("");
}

function pathDisplayControl(total) {
  return `<label class="path-limit-control"><span>Paths shown</span><select data-path-display-limit aria-label="Number of graph paths shown">${displayLimitOptions(state.pathDisplayLimit)}</select><small>${formatNumber(Math.min(state.pathDisplayLimit, total))} of ${formatNumber(total)} retained</small></label>`;
}

function bindPathDisplayControl(root, rerender) {
  $$('[data-path-display-limit]', root).forEach((select) => select.addEventListener("change", () => {
    state.pathDisplayLimit = normalizedDisplayLimit(select.value);
    rerender();
  }));
}

function liveProfiles(task = state.liveTask) {
  const recorded = state.bundle.workbench.cases.map((item) => ({...item, profileKind: "recorded"}));
  const candidates = (state.bundle.workbench.candidates || []).map((item) => ({...item, profileKind: "candidate"}));
  const hostedIds = new Set(state.staticManifest?.precomputed_profile_ids || []);
  return [...recorded, ...candidates]
    .filter((item) => item.task === task)
    .filter((item) => item.id === LIVE_DEMO_PROFILES[task])
    .filter((item) => !state.staticMode || hostedIds.has(item.id))
    .map((item) => ({
      profileId: item.id,
      profileKind: item.profileKind,
      title: item.title,
      task: item.task,
      date: item.source?.date || "",
      sourceDocId: item.source?.doc_id || "",
      query: item.query,
      atoms: (item.atoms || []).map((atom) => ({
        text: atom.text,
        role: atom.constraint_role || liveTaskMeta[item.task].stage,
        required: atom.required_for_resolution !== false,
      })),
      summary: item.profileKind === "recorded"
        ? `Recorded ${item.result?.display_outcome || "outcome"}; live retrieval reruns independently.`
        : "SECRET-source rehearsal candidate; no benchmark outcome asserted.",
      recordedCaseId: item.profileKind === "recorded" ? item.id : "",
    }));
}

function selectLiveProfile(profile) {
  state.liveTask = profile.task;
  state.liveDraft = {...profile};
  state.livePayload = null;
  renderCaseFilters();
  renderWorkbench();
}

function renderLiveExamples() {
  const profiles = liveProfiles();
  $("#case-list").innerHTML = `<p class="live-example-heading">Prepared ${esc(liveTaskMeta[state.liveTask].label)} targets / ${profiles.length}</p>${profiles.map((item) => `<button class="case-button live-example ${item.profileId === state.liveDraft?.profileId ? "active" : ""}" data-live-example="${esc(item.profileId)}" data-role="${item.profileKind === "recorded" ? "diagnostic" : "candidate"}"><strong>${esc(item.title)}</strong><small>${item.profileKind === "recorded" ? "audited source case / fresh retrieval" : "source-verified rehearsal"} / before ${esc(item.date)}</small></button>`).join("")}`;
  $$('[data-live-example]', $("#case-list")).forEach((button) => button.addEventListener("click", () => {
    const profile = profiles.find((item) => item.profileId === button.dataset.liveExample);
    if (profile) selectLiveProfile(profile);
  }));
}

function preparedAtomsMarkup(draft) {
  if (!draft?.atoms?.length) return `<p class="live-atom-note">Custom text is atomized locally at deterministic sentence and clause boundaries. The demo makes no OpenAI API request.</p>`;
  const required = draft.atoms.filter((atom) => atom.required !== false).length;
  const corroborating = draft.atoms.length - required;
  const summary = `${required} required${corroborating ? ` + ${corroborating} corroborating` : ""} / no live API call`;
  return `<div class="prepared-atom-preview"><header><b>Reviewed, preloaded retrieval targets</b><span>${esc(summary)}</span></header>${draft.atoms.map((atom,index) => `<div><span>A${index + 1}</span><p>${esc(atom.text)}<small>${esc(atom.required === false ? "Corroborating context" : atom.role || "Required retrieval target")}</small></p></div>`).join("")}</div>`;
}

function renderLiveWorkbench() {
  const profiles = liveProfiles();
  if (!state.liveDraft || state.liveDraft.task !== state.liveTask) state.liveDraft = {...profiles[0]};
  renderLiveExamples();
  const prior = state.livePayload?.task === state.liveTask ? state.livePayload : null;
  const draft = state.liveDraft || profiles[0] || {query: "", date: "", atoms: []};
  const task = liveTaskMeta[state.liveTask];
  const rankedLimit = normalizedDisplayLimit(prior?.retrieval_limits?.ranked_documents_per_method);
  const provenance = draft.sourceDocId
    ? `Prepared from held-out ${draft.sourceDocId}; that cable is independently excluded and the cutoff is strict-before.`
    : "Custom analyst target; no source document or known ground truth is assumed.";
  $("#case-workspace").innerHTML = `<header class="case-hero live-hero"><p class="eyebrow">Live ${esc(task.label)} / atom-first retrieval</p><h1>${esc(task.input)}</h1><p>${esc(provenance)}</p><div class="case-status-row"><span class="status-pill outcome">${esc(task.label)} task</span><span class="status-pill">Filtered evidence only</span><span class="status-pill">Retrieval is not proof</span></div></header><div class="workspace-body"><section class="live-query-card"><label for="live-query">${esc(task.input)}</label><textarea id="live-query" rows="4" placeholder="Enter a ${esc(task.label.toLowerCase())} target">${esc(prior?.query || draft.query || "")}</textarea>${preparedAtomsMarkup(draft)}<div class="live-query-options"><label>Task <select id="live-task">${Object.entries(liveTaskMeta).map(([key,value]) => `<option value="${key}" ${key === state.liveTask ? "selected" : ""}>${esc(value.label)}</option>`).join("")}</select></label><label>Evidence strictly before <input id="live-cutoff" type="date" value="${esc(prior?.eligibility?.cutoff || draft.date || "")}"></label><label>Ranked cables per method <select id="live-limit">${displayLimitOptions(rankedLimit)}</select></label><button id="run-live-query" class="primary-action">Run ${esc(task.label.toLowerCase())}</button></div><p>Each atom starts an independent document and graph route. Rankings are bounded display lists; passage extraction balances atoms across the union of both candidate lists.</p></section><div id="live-results">${prior ? liveResultsMarkup(prior) : liveEmptyMarkup()}</div></div>`;
  $("#run-live-query").addEventListener("click", runLiveQuery);
  $("#live-task").addEventListener("change", (event) => {
    state.liveTask = event.target.value;
    state.liveDraft = null;
    state.livePayload = null;
    renderCaseFilters();
    renderWorkbench();
  });
  bindLiveResults($("#case-workspace"), prior);
}

function liveEmptyMarkup() {
  const task = liveTaskMeta[state.liveTask];
  return `<section class="live-empty"><b>Ready for ${esc(task.label.toLowerCase())}</b><p>The result will expose each atom, compare document-only and graph-assisted candidates, retain complete route provenance, and show verbatim policy-filtered cable passages.</p></section>`;
}

async function runLiveQuery() {
  const button = $("#run-live-query");
  const query = $("#live-query").value.trim();
  const cutoff = $("#live-cutoff").value;
  const task = $("#live-task").value;
  if (query.length < 3) { $("#live-results").innerHTML = `<div class="live-error">Enter a longer ${esc(liveTaskMeta[task].label.toLowerCase())} target.</div>`; return; }
  const prepared = state.liveDraft?.query === query && state.liveDraft?.task === task && state.liveDraft?.date === cutoff;
  const suppliedAtoms = prepared ? state.liveDraft.atoms || [] : [];
  const sourceDocId = prepared ? state.liveDraft.sourceDocId || "" : "";
  const profileId = prepared ? state.liveDraft.profileId || "" : "";
  state.liveTask = task;
  trackInteraction("run_live_query", "task", task, {
    prepared,
    profile_id:profileId,
    cutoff:cutoff || "",
    query_length:query.length,
    supplied_atom_count:suppliedAtoms.length,
    ranked_document_limit:Number($("#live-limit").value),
  }, 0);
  button.disabled = true; button.textContent = "Retrieving...";
  $("#live-results").innerHTML = `<section class="live-loading"><i></i><b>Filtering cables, ranking each atom, and traversing graph neighbors</b></section>`;
  try {
    let payload;
    if (state.staticMode) {
      if (!prepared || !profileId) throw new Error("The hosted site can replay prepared targets only. Start the local Python server to execute a new free-form query.");
      const profiles = await staticResource("profiles");
      payload = profiles[profileId];
      if (!payload) throw new Error("This prepared target was not precomputed for the hosted replay. Use the local server to execute it.");
    } else {
      payload = await requestJson(
        "/api/analyst-query",
        {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({query, cutoff, task, source_doc_id: sourceDocId, limit: Number($("#live-limit").value), atoms: suppliedAtoms}),
        },
        "Live retrieval",
      );
    }
    payload.profile_id = profileId;
    state.livePayload = payload;
    state.liveDraft = prepared ? state.liveDraft : {query, date: cutoff, task, atoms: [], sourceDocId: "", profileId: ""};
    $("#live-results").innerHTML = liveResultsMarkup(payload);
    bindLiveResults($("#live-results"), payload);
    trackInteraction("live_query_complete", "task", task, {
      prepared,
      profile_id:profileId,
      atom_count:(payload.atoms || []).length,
      document_candidate_count:(payload.document_only || []).length,
      graph_candidate_count:(payload.graph_assisted || []).length,
      passage_count:(payload.passages || []).length,
      coverage_status:payload.provisional_assessment?.coverage_status || "",
    }, 0);
  } catch (error) {
    trackInteraction("live_query_error", "task", task, {prepared, profile_id:profileId, error_type:error?.name || "Error"}, 0);
    $("#live-results").innerHTML = apiFailureMarkup(error, "Retrieval did not complete.");
  } finally {
    button.disabled = false; button.textContent = `Run ${liveTaskMeta[task].label.toLowerCase()}`;
  }
}

function liveResultsMarkup(payload) {
  const eligibility = payload.eligibility;
  const task = liveTaskMeta[payload.task] || liveTaskMeta.reconstruction;
  const prepared = liveProfiles(payload.task).find((item) => item.profileId === payload.profile_id);
  const companion = prepared?.recordedCaseId
    ? `<div class="recorded-companion"><div><b>Audited companion available</b><span>This fresh retrieval is separate from the saved outcome. Open it to compare citations and final aggregation.</span></div><button data-open-recorded="${esc(prepared.recordedCaseId)}">Open audited result</button></div>`
    : "";
  const sourceRule = eligibility.source_guard_applied
    ? `${eligibility.source_document_id} independently guarded by ID; it was already classification-ineligible.`
    : "No held-out source ID applies to this custom analyst target.";
  const assessmentByAtom = new Map((payload.provisional_assessment?.atom_assessments || []).map((row) => [row.question_id, row]));
  const atomBlock = `<div class="query-block">${esc(payload.query)}</div><p class="task-contract">${esc(payload.task_contract)}</p><div class="atom-table live-atoms">${payload.atoms.map((atom,index) => {
    const assessment = assessmentByAtom.get(atom.id);
    const label = assessment ? `${titleCase(assessment.band)} ${Number(assessment.max_passage_score || 0).toFixed(2)}` : "Trace routes";
    return `<button class="atom-row" data-live-atom-index="${index}"><span class="atom-id">${esc(atom.id)}</span><span class="atom-text"><b>${esc(atom.text)}</b><small>${esc(atom.role)} / ${esc(atom.method)}</small></span><span class="atom-status ${esc(assessment?.band || "retrieval_constraint")}">${esc(label)}</span></button>`;
  }).join("")}</div>`;
  return `<div class="live-contract"><b>${esc(task.label)} retrieval packet / provisional corpus assessment</b><span>No hidden answer was used. ${esc(payload.contract.proof)} ${esc(sourceRule)}</span></div>${companion}
    ${workflowStep("01", `${task.label} target and atomic retrieval plan`, payload.contract.parser, atomBlock)}
    ${workflowStep("02", "Filtered evidence universe", "CFPF metadata policy", `<div class="eligibility-summary"><b>${formatNumber(eligibility.eligible_documents)} cables remained</b><span>${esc(eligibility.date_rule)}; ${esc(eligibility.classification_rule)}; ${esc(eligibility.handling_rule)}.</span></div><div class="filter-accounting"><span><b>${formatNumber(eligibility.total_documents)}</b> corpus</span><span><b>${formatNumber(eligibility.excluded_classification)}</b> class-excluded</span><span><b>${formatNumber(eligibility.excluded_noforn)}</b> NOFORN</span><span><b>${formatNumber(eligibility.excluded_date_after_prior_gates)}</b> date-excluded</span></div>`)}
    ${workflowStep("03", "Candidate cables: document-only versus graph-assisted", "Parallel deterministic retrieval", liveRankingMarkup(payload))}
    ${workflowStep("04", "Atom-specific graph retrieval paths", "Semantic seeds plus one hop", liveGraphPathsMarkup(payload))}
    ${workflowStep("05", "Candidate source passages", "Verbatim corpus text", livePassagesMarkup(payload))}
    ${workflowStep("06", "Provisional corpus assessment", "Deterministic retrieval leads / no ground truth", liveDispositionMarkup(payload))}`;
}

function liveDispositionMarkup(payload) {
  const assessment = payload.provisional_assessment;
  if (!assessment) return `<div class="live-disposition"><b>No assessment was stored in this older packet.</b><p>Re-run locally or rebuild the hosted replay with the current API.</p></div>`;
  const ratio = Number(assessment.retrieval_coverage_ratio || 0);
  const percent = Math.round(ratio * 100);
  const atomCards = (assessment.atom_assessments || []).map((atom) => {
    const citation = atom.citations?.[0];
    return `<article class="assessment-atom ${esc(atom.band)}"><header><b>${esc(atom.question_id)}</b><span>${esc(titleCase(atom.band))}</span></header><p>${esc(atom.text)}</p><small>Best filtered passage overlap ${Number(atom.max_passage_score || 0).toFixed(3)}</small>${citation ? `<button class="source-link" data-source-doc="${esc(citation.doc_id)}" data-source-paragraph="${esc(citation.paragraph_id)}" data-highlight="${esc(salientTerms(atom.text).join("||"))}">Inspect strongest lead / ${esc(citation.doc_id)}</button>` : ""}</article>`;
  }).join("");
  let taskSpecific = "";
  if (payload.task === "hypothesis") {
    const directional = assessment.directional_claim_leads || [];
    taskSpecific = `<section class="assessment-specific"><h3>Potential directional claims</h3><p>${esc(assessment.direction_caveat || "Direction still requires citation review.")}</p><div class="directional-leads">${directional.map((row) => `<article class="${esc(row.extraction_stance)}"><span>${esc(titleCase(row.extraction_stance))} lead / ${esc(row.question_id)}</span><b>${esc(row.text)}</b><small>Extraction target: ${esc(row.target || "unspecified")} / match ${Number(row.best_atom_score || 0).toFixed(3)}</small><button class="source-link" data-source-doc="${esc(row.doc_id)}" data-source-paragraph="${esc(row.paragraph_id)}" data-highlight="${esc(salientTerms(row.text).join("||"))}">Open cited claim</button></article>`).join("") || `<p>No support- or oppose-labeled extracted claim cleared the usable-lead threshold.</p>`}</div></section>`;
  } else if (payload.task === "redaction") {
    const candidates = assessment.candidate_guesses || [];
    taskSpecific = `<section class="assessment-specific"><h3>Corpus-derived candidate guesses</h3><p>${esc(assessment.candidate_rule || "Candidate phrases are ranked from structured extraction in the retained policy-filtered passages.")} They are leads, not certified fills.</p><div class="candidate-guess-list">${candidates.map((row,index) => {
      const citation = row.citations?.[0] || {doc_id:row.doc_ids?.[0], paragraph_id:row.paragraph_id};
      return `<article><span>${index + 1}</span><div><b>${esc(row.candidate)}</b><small>${formatNumber(row.covered_atom_count)} of ${formatNumber(assessment.required_atom_count)} constraints with usable passage leads / aggregate ${Number(row.score || 0).toFixed(3)}</small></div>${citation?.doc_id ? `<button class="source-link" data-source-doc="${esc(citation.doc_id)}" data-source-paragraph="${esc(citation.paragraph_id || "")}" data-highlight="${esc(salientTerms(row.candidate).join("||"))}">Source</button>` : ""}</article>`;
    }).join("") || `<p>No structured candidate was found in the retained policy-filtered passages.</p>`}</div></section>`;
  }
  const headline = payload.task === "reconstruction"
    ? `${assessment.located_required_atom_count} of ${assessment.required_atom_count} required atoms have usable retrieval leads (${percent}%).`
    : `${assessment.located_required_atom_count} of ${assessment.required_atom_count} required retrieval targets have usable leads (${percent}%).`;
  return `<div class="live-assessment"><header><div><small>${esc(titleCase(assessment.coverage_status))}</small><b>${esc(headline)}</b></div><div class="coverage-gauge" style="--coverage:${percent}%"><span>${percent}%</span></div></header><div class="assessment-atoms">${atomCards}</div>${taskSpecific}<div class="assessment-caveat"><b>What this does and does not mean</b><p>${esc(assessment.interpretation)} Usable = cosine overlap at least ${Number(assessment.thresholds.usable_lead).toFixed(2)}; strong = at least ${Number(assessment.thresholds.strong_lead).toFixed(2)}. Ground truth was not consulted.</p></div></div>`;
}

function liveRankingMarkup(payload) {
  const table = (rows, mode) => `<div class="ranking-column"><header><b>${mode === "document" ? "Document only" : "Graph assisted"}</b><span>${mode === "document" ? "full-cable TF-IDF" : "document score + graph contribution"}</span></header>${rows.map((row,index) => {
    const lift = row.rank_change == null ? "new" : `${row.rank_change > 0 ? "+" : ""}${row.rank_change}`;
    const detail = mode === "graph"
      ? `${compact(row.subject,58)} / direct ${Number(row.document_score || 0).toFixed(2)} / graph ${Number(row.graph_contribution || 0).toFixed(2)}`
      : `${compact(row.subject,64)} / score ${Number(row.ranking_score || row.document_score || 0).toFixed(2)}`;
    const atomScores = (row.atom_scores || []).map((atom) => `<i class="${Number(atom.score) >= .08 ? "hit" : ""}">${esc(atom.question_id)} ${Number(atom.score).toFixed(2)}</i>`).join("");
    const highlights = row.highlight_terms?.length ? row.highlight_terms : salientTerms(payload.query);
    return `<button class="ranking-row source-link" data-source-doc="${esc(row.doc_id)}" data-source-paragraph="${esc(row.paragraph_id || "")}" data-highlight="${esc(highlights.join("||"))}" title="${esc(mode === "graph" ? `Combined score ${Number(row.ranking_score || 0).toFixed(3)}; strongest graph-text overlap ${Number(row.graph_evidence_alignment || 0).toFixed(3)}` : `Document score ${Number(row.document_score || 0).toFixed(3)}`)}"><span>${index + 1}</span><div><b>${esc(row.doc_id)}</b><small>${esc(detail)}</small><span class="atom-score-strip">${atomScores}</span></div><em>${mode === "graph" ? `${lift} rank` : Number(row.document_score || 0).toFixed(2)}</em></button>`;
  }).join("") || `<p>No filtered document produced a positive match.</p>`}</div>`;
  const limits = payload.retrieval_limits || {};
  return `<div class="ranking-method-note"><b>What is ranked</b><p>These are candidate cables, not accepted evidence. Document-only score = 62% strongest atom overlap + 38% mean atom coverage. Graph-assisted score = 72% document score + 28% normalized graph contribution. Atom badges expose the cable's direct score for every retrieval target.</p></div><div class="ranking-compare">${table(payload.document_only,"document")}${table(payload.graph_assisted,"graph")}</div><p class="inspector-summary">Each panel shows at most ${formatNumber(limits.ranked_documents_per_method || payload.document_only.length)} ranked cables. The interleaved passage pool contains ${formatNumber(limits.unique_passage_candidate_documents || 0)} unique cables. A positive graph rank change means graph-linked semantic evidence promoted a cable; it does not certify proof.</p>`;
}

function liveGraphPathsMarkup(payload) {
  if (!payload.graph_paths.length) return `<p>No policy-filtered graph expansion path contributed to this query.</p>`;
  const paths = payload.graph_paths.slice(0, state.pathDisplayLimit);
  return `<div class="route-display-toolbar"><p class="route-limit-note">Showing ${formatNumber(paths.length)} of ${formatNumber(payload.graph_paths.length)} retained graph paths, balanced across atoms before score fill. This controls presentation only; retrieval has already used the full policy-filtered graph index. Select <b>Guide path</b> to walk from atom through hierarchy placement, semantic seed, graph edge, and candidate cable.</p>${pathDisplayControl(payload.graph_paths.length)}</div><div class="live-paths">${paths.map((path,index) => { const highlights = path.highlight_terms?.length ? path.highlight_terms : salientTerms(path.question_text || payload.query); return `<article><button class="path-atom" data-live-atom-id="${esc(path.question_id)}"><span>${esc(path.question_id)} / retrieval target</span><b>${esc(path.question_text)}</b></button><div><span>${esc(path.seed_type)} seed</span><b>${esc(path.seed_label)}</b></div><i>-&gt;<small>${esc(titleCase(path.relation))}</small></i><div><span>${path.neighbor_label ? `Graph neighbor${path.evidence_alignment != null ? ` / edge-text overlap ${Number(path.evidence_alignment).toFixed(2)}` : ""}` : "Node provenance"}</span><b>${esc(path.neighbor_label || path.doc_id)}</b></div><div class="live-path-actions"><button data-live-route-index="${index}">Guide path</button><button class="source-link" data-source-doc="${esc(path.doc_id)}" data-source-paragraph="${esc(path.paragraph_id || "")}" data-highlight="${esc(highlights.join("||"))}">${esc(path.doc_id)}</button></div></article>`; }).join("")}</div>`;
}

function livePassagesMarkup(payload) {
  if (!payload.passages.length) return `<p>No positive-scoring filtered passage was returned.</p>`;
  return `<div class="evidence-grid live-passages">${payload.passages.slice(0,12).map((row) => { const atom = payload.atoms.find((item) => item.id === row.best_question_id); const highlights = row.highlight_terms?.length ? row.highlight_terms : salientTerms(atom?.text || payload.query); return `<article class="workbench-evidence context"><span class="set-label">${esc(row.best_question_id)} / ${esc(row.retrieval_channels.join(" + "))}</span><h3>${esc(row.doc_id)}</h3><div class="doc-meta">${esc([row.date,row.classification,row.paragraph_id].filter(Boolean).join(" / "))}</div><blockquote>${highlighted(row.text,highlights)}</blockquote><button class="source-link" data-source-doc="${esc(row.doc_id)}" data-source-paragraph="${esc(row.paragraph_id)}" data-highlight="${esc(highlights.join("||"))}">Open exact cable segment</button></article>`; }).join("")}</div>`;
}

function bindLiveResults(root, payload) {
  bindSourceButtons(root);
  if (!payload) return;
  bindPathDisplayControl(root, () => {
    const workspace = $("#case-workspace");
    const scrollTop = workspace.scrollTop;
    const results = $("#live-results");
    results.innerHTML = liveResultsMarkup(payload);
    bindLiveResults(results, payload);
    requestAnimationFrame(() => { workspace.scrollTop = scrollTop; });
  });
  $$('[data-live-atom-index]', root).forEach((button) => button.addEventListener("click", () => openLiveAtomDetail(payload, Number(button.dataset.liveAtomIndex))));
  $$('[data-live-atom-id]', root).forEach((button) => button.addEventListener("click", () => {
    const index = payload.atoms.findIndex((atom) => atom.id === button.dataset.liveAtomId);
    if (index >= 0) openLiveAtomDetail(payload, index);
  }));
  $$('[data-live-route-index]', root).forEach((button) => button.addEventListener("click", () => openLiveRouteDetail(payload, Number(button.dataset.liveRouteIndex))));
  $$('[data-open-recorded]', root).forEach((button) => button.addEventListener("click", () => {
    const item = state.bundle.workbench.cases.find((row) => row.id === button.dataset.openRecorded);
    if (!item) return;
    state.workbenchMode = "recorded";
    state.taskFilter = item.task;
    state.caseId = item.id;
    $$('[data-workbench-mode]').forEach((modeButton) => modeButton.classList.toggle("active", modeButton.dataset.workbenchMode === "recorded"));
    renderCaseFilters();
    renderWorkbench();
  }));
}

function liveHierarchy(path, prefix = "seed") {
  const direct = prefix === "seed" && !path.seed_narrative_id;
  return {
    narrativeId: direct ? path.narrative_id : path[`${prefix}_narrative_id`],
    narrativeLabel: direct ? path.narrative_label : path[`${prefix}_narrative_label`],
    communityId: direct ? path.community_id : path[`${prefix}_community_id`],
    communityLabel: direct ? path.community_label : path[`${prefix}_community_label`],
  };
}

function openLiveAtomDetail(payload, atomIndex) {
  const atom = payload.atoms[atomIndex]; if (!atom) return;
  const seeds = (payload.graph_seeds || []).filter((row) => (row.matched_question_ids || [row.question_id]).includes(atom.id));
  const paths = (payload.graph_paths || []).filter((row) => row.question_id === atom.id);
  const passages = (payload.passages || []).filter((row) => (row.question_scores || []).some((score) => score.question_id === atom.id));
  const placements = seeds.slice(0,5).map((seed) => {
    const narrative = indexes.narrative.get(seed.narrative_id);
    const community = indexes.community.get(seed.community_id);
    return `<article class="atom-citation"><span class="atom-status retrieval_constraint">Seed ${Number(seed.score || 0).toFixed(2)}</span><b>${esc(seed.label)}</b><p>${esc([narrative?.label, community?.label].filter(Boolean).join(" / ") || "No hierarchy label retained")}</p>${seed.doc_id ? `<button class="source-link" data-source-doc="${esc(seed.doc_id)}" data-source-paragraph="${esc(seed.paragraph_id || "")}" data-highlight="${esc(salientTerms(atom.text).join("||"))}">Open seed provenance</button>` : ""}</article>`;
  }).join("");
  const passageCards = passages.sort((left,right) => Number((right.question_scores.find((score) => score.question_id === atom.id) || {}).score || 0) - Number((left.question_scores.find((score) => score.question_id === atom.id) || {}).score || 0)).slice(0,5).map((row) => {
    const score = (row.question_scores.find((item) => item.question_id === atom.id) || {}).score || 0;
    return `<article class="atom-citation"><span class="atom-status retrieval_constraint">Passage ${Number(score).toFixed(2)}</span><b>${esc(row.doc_id)} / ${esc(row.paragraph_id)}</b><p>${highlighted(row.text, salientTerms(atom.text))}</p><button class="source-link" data-source-doc="${esc(row.doc_id)}" data-source-paragraph="${esc(row.paragraph_id)}" data-highlight="${esc(salientTerms(atom.text).join("||"))}">Open exact passage</button></article>`;
  }).join("");
  openDetailDrawer("Live atomic retrieval", `${atom.id} / ${atom.text}`, [titleCase(payload.task), atom.role, `${paths.length} retained paths`, `${passages.length} candidate passages`], `<div class="atom-detail"><h3>How this atom entered the graph</h3>${placements || `<p>No displayed semantic seed was retained for this atom.</p>`}<h3>Candidate passages for this atom</h3>${passageCards || `<p>No positive-scoring passage was retained for this atom.</p>`}<p class="trail-note">Scores are retrieval similarities, not calibrated confidence or citation judgments.</p></div>`);
}

function liveRouteGuide(payload, path) {
  const seedHierarchy = liveHierarchy(path, "seed");
  const neighborHierarchy = liveHierarchy(path, "neighbor");
  if (path.seed_id && !indexes.node.has(path.seed_id)) {
    indexes.node.set(path.seed_id, {id:path.seed_id, kind:"node", type:path.seed_type || "Unknown", label:path.seed_label || path.seed_id, community_id:seedHierarchy.communityId, narrative_id:seedHierarchy.narrativeId, highlight_terms:path.highlight_terms || [], weight:1, injected:true});
  }
  if (path.neighbor_id && !indexes.node.has(path.neighbor_id)) {
    indexes.node.set(path.neighbor_id, {id:path.neighbor_id, kind:"node", type:"Unknown", label:path.neighbor_label || path.neighbor_id, community_id:neighborHierarchy.communityId, narrative_id:neighborHierarchy.narrativeId, highlight_terms:path.highlight_terms || [], weight:1, injected:true});
  }
  const seedStep = {kind:"node", id:path.seed_id, narrativeId:seedHierarchy.narrativeId, communityId:seedHierarchy.communityId, label:path.seed_label, detail:`${path.seed_type} semantic seed matched to this atom.`};
  const steps = [
    {kind:"atom", label:path.question_text, detail:`${path.question_id} starts an independent retrieval route.`},
    seedHierarchy.narrativeId ? {kind:"narrative", id:seedHierarchy.narrativeId, label:seedHierarchy.narrativeLabel || seedHierarchy.narrativeId, detail:"Primary narrative containing the semantic seed."} : null,
    seedHierarchy.communityId ? {kind:"community", id:seedHierarchy.communityId, narrativeId:seedHierarchy.narrativeId, label:seedHierarchy.communityLabel || seedHierarchy.communityId, detail:"Primary community containing the semantic seed."} : null,
    seedStep,
  ].filter(Boolean);
  if (path.neighbor_id) {
    const placementChanged = seedHierarchy.narrativeId !== neighborHierarchy.narrativeId || seedHierarchy.communityId !== neighborHierarchy.communityId;
    let bridge = null;
    if (placementChanged) {
      const edge = edgeTraversalExplanation(path.relation, path.seed_label, path.neighbor_label);
      const narrativeChanged = seedHierarchy.narrativeId !== neighborHierarchy.narrativeId;
      bridge = {
        kind:"bridge",
        id:`live-bridge:${path.question_id}:${path.neighbor_id}`,
        scope:narrativeChanged ? "narrative" : "community",
        label:narrativeChanged
          ? `${seedHierarchy.narrativeLabel || seedHierarchy.narrativeId} -> ${neighborHierarchy.narrativeLabel || neighborHierarchy.narrativeId}`
          : `${seedHierarchy.communityLabel || seedHierarchy.communityId} -> ${neighborHierarchy.communityLabel || neighborHierarchy.communityId}`,
        detail:`${edge.detail} The hierarchy boundary reflects clustering, not a prohibition on cross-cluster graph edges.`,
        edgeType:path.relation,
        edgeWeight:Number(path.contribution || 0),
        edgeDocId:path.doc_id,
        edgeParagraphId:path.paragraph_id,
        edgeExcerpt:path.source_excerpt || "",
        edgeHighlights:path.highlight_terms || [],
        fromNodeId:path.seed_id,
        fromNodeLabel:path.seed_label,
        fromNarrativeId:seedHierarchy.narrativeId,
        fromNarrativeLabel:seedHierarchy.narrativeLabel || seedHierarchy.narrativeId,
        fromCommunityId:seedHierarchy.communityId,
        fromCommunityLabel:seedHierarchy.communityLabel || seedHierarchy.communityId,
        toNodeId:path.neighbor_id,
        toNodeLabel:path.neighbor_label,
        toNarrativeId:neighborHierarchy.narrativeId,
        toNarrativeLabel:neighborHierarchy.narrativeLabel || neighborHierarchy.narrativeId,
        toCommunityId:neighborHierarchy.communityId,
        toCommunityLabel:neighborHierarchy.communityLabel || neighborHierarchy.communityId,
      };
      steps.push(bridge);
    }
    steps.push({kind:"node", id:path.neighbor_id, narrativeId:neighborHierarchy.narrativeId, communityId:neighborHierarchy.communityId, label:path.neighbor_label, detail:`Traversed via ${titleCase(path.relation)}; contribution ${Number(path.contribution || 0).toFixed(3)}.`, edgeType:path.relation, edgeDocId:path.doc_id, edgeParagraphId:path.paragraph_id, edgeExcerpt:path.source_excerpt || "", edgeHighlights:path.highlight_terms || [], transition:bridge});
  }
  steps.push({kind:"document", id:path.doc_id, paragraphId:path.paragraph_id, label:path.doc_id, detail:`Filtered candidate cable reached through ${path.match_basis || "retrieval provenance"}. Discovery is not proof.`});
  return {
    title: `${path.question_id} graph route`,
    subtitle: path.question_text,
    task: payload.task,
    relation: path.relation,
    highlight: (path.highlight_terms?.length ? path.highlight_terms : salientTerms(path.question_text)).join("||"),
    steps,
  };
}

function openLiveRouteDetail(payload, routeIndex) {
  const path = payload.graph_paths[routeIndex]; if (!path) return;
  const guide = liveRouteGuide(payload, path);
  const steps = guide.steps;
  const routeHighlights = path.highlight_terms?.length ? path.highlight_terms : salientTerms(path.question_text);
  const cards = steps.map((step,index) => `<article class="guided-step" data-guided-step="${index}"><span class="trail-index">${String(index + 1).padStart(2,"0")}</span><div><small>${esc(titleCase(step.kind))}</small><h3>${esc(step.label)}</h3><p>${esc(step.detail)}</p>${step.kind === "document" && path.source_excerpt ? `<blockquote>${highlighted(path.source_excerpt, routeHighlights)}</blockquote>` : ""}${["narrative","community","node"].includes(step.kind) ? `<button class="guide-explore" data-guide-kind="${esc(step.kind)}" data-guide-id="${esc(step.id)}" data-guide-narrative="${esc(step.narrativeId || "")}" data-guide-community="${esc(step.communityId || "")}">Show this step in the corpus map</button>` : ""}${step.kind === "document" ? `<button class="source-link" data-source-doc="${esc(step.id)}" data-source-paragraph="${esc(step.paragraphId || "")}" data-highlight="${esc(routeHighlights.join("||"))}">Open exact candidate passage</button>` : ""}</div></article>`).join("");
  openDetailDrawer("Guided graph retrieval path", `${path.question_id} / ${path.question_text}`, [titleCase(payload.task), titleCase(path.relation), path.neighbor_id ? "one graph hop" : "node provenance", path.doc_id], `<div class="guided-route-intro"><b>Follow the highlighted step</b><span>Use this drawer, or keep the whole route on the corpus map while you traverse it.</span><div><button id="guided-next">Start walkthrough</button><button id="guided-map">Open route in corpus map</button></div></div><div class="trail-stack guided-trail">${cards}</div>`);
  let active = -1;
  const update = () => {
    $$('[data-guided-step]', $("#source-body")).forEach((card,index) => card.classList.toggle("active", index === active));
    $("#guided-next").textContent = active < steps.length - 1 ? (active < 0 ? "Start walkthrough" : "Next step") : "Restart";
    if (active >= 0) $$('[data-guided-step]', $("#source-body"))[active]?.scrollIntoView({block:"center", behavior:"smooth"});
  };
  $("#guided-next")?.addEventListener("click", () => { active = active >= steps.length - 1 ? 0 : active + 1; update(); });
  $("#guided-map")?.addEventListener("click", () => activateGraphGuide(guide, 0));
  $$('[data-guided-step]', $("#source-body")).forEach((card,index) => card.addEventListener("click", (event) => { if (event.target.closest("button")) return; active = index; update(); }));
  $$('.guide-explore', $("#source-body")).forEach((button) => button.addEventListener("click", () => {
    const index = steps.findIndex((step) => step.id === button.dataset.guideId && step.kind === button.dataset.guideKind);
    activateGraphGuide(guide, Math.max(0, index));
  }));
}

function activateGraphGuide(guide, startIndex = 0) {
  clearGuideTransitionAnimation();
  state.activeGuide = {...guide, index: Math.max(0, Math.min(startIndex, guide.steps.length - 1))};
  closeSource();
  setMode("explorer");
  renderGraphGuide();
  setTimeout(() => {
    if (!state.activeGuide) return;
    if (state.activeGuide.index === 0 && state.activeGuide.steps[0]?.kind === "atom") {
      transitionGraph("narratives", null, {direction:"out"});
      return;
    }
    navigateGraphGuide(state.activeGuide.index);
  }, 30);
}

function clearGuideTransitionAnimation() {
  state.guideTransitionTimers.forEach((timer) => clearTimeout(timer));
  state.guideTransitionTimers = [];
  state.guideNavigationToken += 1;
  state.guideTransitionPhase = "";
  state.guideHighlights = [];
  state.guidePulseUntil = 0;
  state.guideReplayDurationScale = GUIDE_REPLAY_SLOWDOWN;
  const stage = $("#graph-stage");
  stage?.classList.remove("guide-replay-active");
  stage?.style.removeProperty("--guide-opacity-duration");
  stage?.style.removeProperty("--guide-transform-duration");
  $("#source-drawer")?.classList.remove("guide-destination-document");
  const arrival = $("#guide-document-arrival");
  if (arrival) {
    arrival.classList.remove("visible");
    arrival.hidden = true;
  }
}

function scheduleGuideTransition(token, callback, delay) {
  const timer = setTimeout(() => {
    if (token === state.guideNavigationToken && state.activeGuide) callback();
  }, delay);
  state.guideTransitionTimers.push(timer);
}

function guideReplayDelay(delay) {
  return Math.round(delay * state.guideReplayDurationScale);
}

function beginGuideReplay(speedMultiplier = 1) {
  const multiplier = Math.max(1, Number(speedMultiplier) || 1);
  state.guideReplayDurationScale = GUIDE_REPLAY_SLOWDOWN / multiplier;
  const stage = $("#graph-stage");
  stage?.classList.add("guide-replay-active");
  stage?.style.setProperty("--guide-opacity-duration", `${0.28 * state.guideReplayDurationScale}s`);
  stage?.style.setProperty("--guide-transform-duration", `${0.36 * state.guideReplayDurationScale}s`);
}

function showGuidePhase(token, value) {
  if (token !== state.guideNavigationToken) return;
  state.guideTransitionPhase = value;
  renderGraphGuide();
}

function guideNode(point) {
  return {
    kind:"node",
    id:point.nodeId,
    label:point.nodeLabel,
    narrative_id:point.narrativeId,
    community_id:point.communityId,
  };
}

function showGuideSourceNode(token, point) {
  showGuidePhase(token, `Source semantic node: ${point.nodeLabel || point.nodeId}`);
  const alreadyVisible = state.level === "nodes"
    && state.narrativeId === point.narrativeId
    && state.communityId === point.communityId
    && state.graph?.nodeMap?.has(point.nodeId);
  state.narrativeId = point.narrativeId;
  state.communityId = point.communityId;
  setGuideHighlights([{id:point.nodeId, role:"from"}], guideReplayDelay(600));
  if (alreadyVisible) {
    focusNode(point.nodeId);
    return;
  }
  transitionGraph("nodes", point.nodeId, {
    direction:"crossfade",
    arrivalNode:guideNode(point),
  });
}

function animateGuideDocument(step, guide) {
  const token = state.guideNavigationToken;
  const arrival = $("#guide-document-arrival");
  beginGuideReplay();
  showGuidePhase(token, `Destination document: ${step.id}`);
  if (arrival) {
    $("#guide-document-id").textContent = step.id;
    $("#guide-document-detail").textContent = step.paragraphId
      ? `Opening the cited passage ${step.paragraphId}`
      : "Opening the retrieved source cable";
    arrival.hidden = false;
    requestAnimationFrame(() => arrival.classList.add("visible"));
  }
  scheduleGuideTransition(token, async () => {
    if (arrival) {
      arrival.classList.remove("visible");
      arrival.hidden = true;
    }
    showGuidePhase(token, `Opened destination document: ${step.id}`);
    await openSource(step.id, step.paragraphId || "", guide.highlight || "");
    if (token !== state.guideNavigationToken) return;
    $("#source-drawer").classList.add("guide-destination-document");
    $("#drawer-kicker").textContent = "Traversal destination / cited cable";
  }, guideReplayDelay(GUIDE_REPLAY_PHASE_MS));
}

function guideBridgeEndpoints(step, movingForward = true) {
  return movingForward ? {
    from:{
      nodeId:step.fromNodeId, nodeLabel:step.fromNodeLabel,
      narrativeId:step.fromNarrativeId, narrativeLabel:step.fromNarrativeLabel,
      communityId:step.fromCommunityId, communityLabel:step.fromCommunityLabel,
    },
    to:{
      nodeId:step.toNodeId, nodeLabel:step.toNodeLabel,
      narrativeId:step.toNarrativeId, narrativeLabel:step.toNarrativeLabel,
      communityId:step.toCommunityId, communityLabel:step.toCommunityLabel,
    },
  } : {
    from:{
      nodeId:step.toNodeId, nodeLabel:step.toNodeLabel,
      narrativeId:step.toNarrativeId, narrativeLabel:step.toNarrativeLabel,
      communityId:step.toCommunityId, communityLabel:step.toCommunityLabel,
    },
    to:{
      nodeId:step.fromNodeId, nodeLabel:step.fromNodeLabel,
      narrativeId:step.fromNarrativeId, narrativeLabel:step.fromNarrativeLabel,
      communityId:step.fromCommunityId, communityLabel:step.fromCommunityLabel,
    },
  };
}

function guideBridgeKey(guide, step) {
  return [guide.title, step.id, step.fromNodeId, step.toNodeId].join("::");
}

function settleGuideBridge(step, movingForward = true) {
  const {to} = guideBridgeEndpoints(step, movingForward);
  closeSource();
  state.guideTransitionPhase = "Animation already shown. Replay this crossing at 2x speed if needed.";
  state.narrativeId = to.narrativeId;
  state.communityId = to.communityId;
  setGuideHighlights([{id:to.nodeId, role:"to"}], 1800);
  renderGraphGuide();
  if (to.nodeId && to.communityId) {
    transitionGraph("nodes", to.nodeId, {direction:"crossfade", arrivalNode:guideNode(to)});
  } else if (to.communityId) {
    transitionGraph("communities", to.communityId, {
      direction:"crossfade",
      arrivalNode:{kind:"community", id:to.communityId, narrative_id:to.narrativeId},
    });
  } else {
    transitionGraph("narratives", to.narrativeId, {
      direction:"crossfade",
      arrivalNode:{kind:"narrative", id:to.narrativeId},
    });
  }
}

function animateGuideBridge(step, movingForward = true, options = {}) {
  clearGuideTransitionAnimation();
  const token = state.guideNavigationToken;
  const {from, to} = guideBridgeEndpoints(step, movingForward);
  closeSource();
  beginGuideReplay(options.speedMultiplier || 1);
  showGuideSourceNode(token, from);

  if (step.scope === "narrative" && to.narrativeId) {
    scheduleGuideTransition(token, () => {
      showGuidePhase(token, `Ascending to source community: ${from.communityLabel || from.communityId}`);
      state.narrativeId = from.narrativeId;
      setGuideHighlights([{id:from.communityId, role:"from"}], guideReplayDelay(600));
      transitionGraph("communities", from.communityId, {
        direction:"out",
        arrivalNode:{kind:"community", id:from.communityId, narrative_id:from.narrativeId},
      });
    }, guideReplayDelay(GUIDE_REPLAY_PHASE_MS));
    scheduleGuideTransition(token, () => {
      showGuidePhase(token, `Ascending to source narrative: ${from.narrativeLabel || from.narrativeId}`);
      setGuideHighlights([{id:from.narrativeId, role:"from"}], guideReplayDelay(650));
      transitionGraph("narratives", from.narrativeId, {
        direction:"out",
        arrivalNode:{kind:"narrative", id:from.narrativeId},
      });
    }, guideReplayDelay(GUIDE_REPLAY_PHASE_MS * 2));
    scheduleGuideTransition(token, () => {
      showGuidePhase(token, `Crossing narratives via ${titleCase(step.edgeType || "graph relation")}: ${from.narrativeLabel} -> ${to.narrativeLabel}`);
      setGuideHighlights([
        {id:from.narrativeId, role:"from"},
        {id:to.narrativeId, role:"to"},
      ], guideReplayDelay(750));
      transitionGraph("narratives", to.narrativeId, {
        direction:"crossfade",
        arrivalNode:{kind:"narrative", id:to.narrativeId},
      });
    }, guideReplayDelay(GUIDE_REPLAY_PHASE_MS * 3.2));
    scheduleGuideTransition(token, () => {
      if (!to.communityId) return;
      showGuidePhase(token, `Descending into destination community: ${to.communityLabel || to.communityId}`);
      state.narrativeId = to.narrativeId;
      setGuideHighlights([{id:to.communityId, role:"to"}], guideReplayDelay(650));
      transitionGraph("communities", to.communityId, {
        viaNodeId:to.narrativeId,
        direction:"in",
        arrivalNode:{kind:"community", id:to.communityId, narrative_id:to.narrativeId},
      });
    }, guideReplayDelay(GUIDE_REPLAY_PHASE_MS * 4.5));
    scheduleGuideTransition(token, () => {
      if (!to.communityId || !to.nodeId) return;
      showGuidePhase(token, `Descending to linked semantic node: ${to.nodeLabel || to.nodeId}`);
      state.narrativeId = to.narrativeId;
      state.communityId = to.communityId;
      setGuideHighlights([{id:to.nodeId, role:"to"}], guideReplayDelay(750));
      transitionGraph("nodes", to.nodeId, {
        viaNodeId:to.communityId,
        direction:"in",
        arrivalNode:guideNode(to),
      });
    }, guideReplayDelay(GUIDE_REPLAY_PHASE_MS * 5.8));
  } else if (to.communityId) {
    scheduleGuideTransition(token, () => {
      showGuidePhase(token, `Ascending to source community: ${from.communityLabel || from.communityId}`);
      state.narrativeId = from.narrativeId;
      setGuideHighlights([{id:from.communityId, role:"from"}], guideReplayDelay(600));
      transitionGraph("communities", from.communityId, {
        direction:"out",
        arrivalNode:{kind:"community", id:from.communityId, narrative_id:from.narrativeId},
      });
    }, guideReplayDelay(GUIDE_REPLAY_PHASE_MS));
    scheduleGuideTransition(token, () => {
      showGuidePhase(token, `Crossing communities via ${titleCase(step.edgeType || "graph relation")}: ${from.communityLabel} -> ${to.communityLabel}`);
      state.narrativeId = to.narrativeId;
      setGuideHighlights([
        {id:from.communityId, role:"from"},
        {id:to.communityId, role:"to"},
      ], guideReplayDelay(750));
      transitionGraph("communities", to.communityId, {
        direction:"crossfade",
        arrivalNode:{kind:"community", id:to.communityId, narrative_id:to.narrativeId},
      });
    }, guideReplayDelay(GUIDE_REPLAY_PHASE_MS * 2.2));
    scheduleGuideTransition(token, () => {
      if (!to.nodeId) return;
      showGuidePhase(token, `Descending to linked semantic node: ${to.nodeLabel || to.nodeId}`);
      state.communityId = to.communityId;
      setGuideHighlights([{id:to.nodeId, role:"to"}], guideReplayDelay(750));
      transitionGraph("nodes", to.nodeId, {
        viaNodeId:to.communityId,
        direction:"in",
        arrivalNode:guideNode(to),
      });
    }, guideReplayDelay(GUIDE_REPLAY_PHASE_MS * 3.6));
  }
  scheduleGuideTransition(token, () => {
    state.guideTransitionPhase = "Arrived at the linked semantic node";
    const stage = $("#graph-stage");
    stage?.classList.remove("guide-replay-active");
    stage?.style.removeProperty("--guide-opacity-duration");
    stage?.style.removeProperty("--guide-transform-duration");
    renderGraphGuide();
  }, guideReplayDelay(step.scope === "narrative" ? GUIDE_REPLAY_PHASE_MS * 7.3 : GUIDE_REPLAY_PHASE_MS * 5.1));
}

function runGuideBridge(guide, step, movingForward = true) {
  const key = guideBridgeKey(guide, step);
  state.guideBridgeDirections.set(key, movingForward);
  if (!state.playedGuideBridges.has(key)) {
    state.playedGuideBridges.add(key);
    animateGuideBridge(step, movingForward);
    return;
  }
  settleGuideBridge(step, movingForward);
}

function navigateGraphGuide(index) {
  const guide = state.activeGuide;
  if (!guide) return;
  clearGuideTransitionAnimation();
  const previousIndex = guide.index;
  guide.index = Math.max(0, Math.min(index, guide.steps.length - 1));
  const step = guide.steps[guide.index];
  const movingForward = guide.index >= previousIndex;
  renderGraphGuide();
  if (step.kind === "atom") {
    if (state.level !== "narratives") transitionGraph("narratives", null, {direction:"out"});
    return;
  }
  if (step.kind === "document") {
    animateGuideDocument(step, guide);
    return;
  }
  if (step.kind === "provenance") {
    const docId = step.edgeDocId || step.docId;
    if (docId) openSource(docId, step.edgeParagraphId || step.paragraphId || "", (step.edgeHighlights || []).join("||") || guide.highlight || "");
    return;
  }
  closeSource();
  if (step.kind === "bridge") {
    runGuideBridge(guide, step, movingForward);
    return;
  }
  if (step.kind === "narrative") {
    const arrivalNode = {kind:"narrative", id:step.id};
    transitionGraph("narratives", step.id, {direction:state.level === "narratives" ? "crossfade" : "out", arrivalNode});
  } else if (step.kind === "community") {
    const enteringFromNarrative = movingForward && state.level === "narratives" && state.graph?.nodeMap?.has(step.narrativeId);
    state.narrativeId = step.narrativeId;
    const arrivalNode = {kind:"community", id:step.id, narrative_id:step.narrativeId};
    transitionGraph("communities", step.id, {viaNodeId:enteringFromNarrative ? step.narrativeId : "", direction:enteringFromNarrative ? "in" : state.level === "nodes" ? "out" : "crossfade", arrivalNode});
  } else if (step.kind === "node") {
    const hierarchyMismatch = state.level !== "nodes" || state.narrativeId !== step.narrativeId || state.communityId !== step.communityId;
    if (step.transition && hierarchyMismatch) {
      runGuideBridge(guide, step.transition, true);
      return;
    }
    const enteringFromCommunity = movingForward && state.level === "communities" && state.graph?.nodeMap?.has(step.communityId);
    state.narrativeId = step.narrativeId;
    state.communityId = step.communityId;
    const arrivalNode = {kind:"node", id:step.id, type:indexes.node.get(step.id)?.type || "Unknown", label:step.label, narrative_id:step.narrativeId, community_id:step.communityId};
    transitionGraph("nodes", step.id, {viaNodeId:enteringFromCommunity ? step.communityId : "", direction:enteringFromCommunity ? "in" : "crossfade", arrivalNode});
  }
}

let guidePanelResizeObserver = null;

function constrainGraphGuidePanel(panel) {
  const stage = $("#graph-stage");
  if (!panel || !stage || panel.hidden) return;
  const stageRect = stage.getBoundingClientRect();
  const width = Math.min(panel.offsetWidth, Math.max(300, stageRect.width - 16));
  const height = Math.min(panel.offsetHeight, Math.max(220, stageRect.height - 16));
  const left = Math.max(8, Math.min(Number.parseFloat(panel.style.left || "16") || 16, stageRect.width - width - 8));
  const top = Math.max(8, Math.min(Number.parseFloat(panel.style.top || "16") || 16, stageRect.height - height - 8));
  panel.style.left = `${left}px`;
  panel.style.top = `${top}px`;
  if (panel.offsetWidth > stageRect.width - 16) panel.style.width = `${width}px`;
  if (panel.offsetHeight > stageRect.height - 16) panel.style.height = `${height}px`;
  state.guidePanelLayout = {left, top, width:panel.offsetWidth, height:panel.offsetHeight};
}

function bindGraphGuidePanel(panel) {
  const layout = state.guidePanelLayout;
  if (layout.left != null) panel.style.left = `${layout.left}px`;
  if (layout.top != null) panel.style.top = `${layout.top}px`;
  if (layout.width != null) panel.style.width = `${layout.width}px`;
  if (layout.height != null) panel.style.height = `${layout.height}px`;
  constrainGraphGuidePanel(panel);
  const handle = $("header", panel);
  handle?.addEventListener("pointerdown", (event) => {
    if (event.target.closest("button")) return;
    event.preventDefault();
    const stageRect = $("#graph-stage").getBoundingClientRect();
    const panelRect = panel.getBoundingClientRect();
    const offsetX = event.clientX - panelRect.left;
    const offsetY = event.clientY - panelRect.top;
    panel.classList.add("dragging");
    const move = (moveEvent) => {
      const left = Math.max(8, Math.min(moveEvent.clientX - stageRect.left - offsetX, stageRect.width - panel.offsetWidth - 8));
      const top = Math.max(8, Math.min(moveEvent.clientY - stageRect.top - offsetY, stageRect.height - panel.offsetHeight - 8));
      panel.style.left = `${left}px`;
      panel.style.top = `${top}px`;
      state.guidePanelLayout.left = left;
      state.guidePanelLayout.top = top;
    };
    const finish = () => {
      panel.classList.remove("dragging");
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", finish);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", finish, {once:true});
  });
  handle?.addEventListener("dblclick", (event) => {
    if (event.target.closest("button")) return;
    state.guidePanelLayout = {left:null, top:null, width:null, height:null};
    panel.removeAttribute("style");
  });
  if (!guidePanelResizeObserver && window.ResizeObserver) {
    guidePanelResizeObserver = new ResizeObserver(() => {
      if (!panel.hidden && panel.offsetWidth > 0 && panel.offsetHeight > 0) constrainGraphGuidePanel(panel);
    });
    guidePanelResizeObserver.observe(panel);
  }
}

function graphGuideEvidenceMarkup(current) {
  const docId = current.edgeDocId || (current.kind === "provenance" ? current.docId : "");
  const paragraphId = current.edgeParagraphId || current.paragraphId || "";
  const terms = current.edgeHighlights || [];
  const excerpt = current.edgeExcerpt || "";
  const hierarchy = current.kind === "bridge"
    ? `<div class="graph-guide-hierarchy"><span><small>From</small>${esc(current.fromNarrativeLabel)}<i>${esc(current.fromCommunityLabel)}</i></span><b>-&gt;</b><span><small>To</small>${esc(current.toNarrativeLabel)}<i>${esc(current.toCommunityLabel)}</i></span></div>`
    : "";
  if (!current.edgeType && !docId && !hierarchy) return "";
  return `<section class="graph-guide-hop"><header><b>${current.edgeType ? `Why this hop / ${esc(titleCase(current.edgeType))}` : "Exact provenance"}</b>${current.edgeWeight ? `<span>edge weight ${Number(current.edgeWeight).toFixed(3)}</span>` : ""}</header>${hierarchy}${excerpt ? `<blockquote>${highlighted(excerpt, terms)}</blockquote>` : ""}${docId ? `<button class="source-link" data-source-doc="${esc(docId)}" data-source-paragraph="${esc(paragraphId)}" data-highlight="${esc(terms.join("||"))}">Open supporting cable ${esc(docId)}${paragraphId ? ` / ${esc(paragraphId)}` : ""}</button>` : `<p>No single source cable is attached to this structural step.</p>`}</section>`;
}

function renderGraphGuide() {
  const panel = $("#graph-guide");
  const guide = state.activeGuide;
  if (!panel || !guide || state.mode !== "explorer") {
    if (panel) panel.hidden = true;
    return;
  }
  const current = guide.steps[guide.index];
  const replayKey = current.kind === "bridge" ? guideBridgeKey(guide, current) : "";
  const replayMarkup = replayKey && state.playedGuideBridges.has(replayKey)
    ? `<button id="graph-guide-replay" class="graph-guide-replay"><b>Replay hierarchy crossing</b><span>2x speed</span></button>`
    : "";
  panel.hidden = false;
  panel.innerHTML = `<header title="Drag this panel. Double-click the header to reset its size and position."><div><small>Persistent retrieval walkthrough</small><b>${esc(guide.title)}</b><em>Drag header / resize lower-right</em></div><button id="graph-guide-close" title="Close route">x</button></header><p>${esc(guide.subtitle)}</p>${state.guideTransitionPhase ? `<div class="graph-guide-phase"><i></i>${esc(state.guideTransitionPhase)}</div>` : ""}<div class="graph-guide-steps">${guide.steps.map((step,index) => `<button class="${index === guide.index ? "active" : ""} ${step.kind === "bridge" ? "bridge" : ""} ${step.kind === "document" ? "document" : ""}" data-graph-guide-step="${index}" title="${esc(step.detail)}"><span>${index + 1}</span><b>${esc(titleCase(step.kind === "bridge" ? `${step.scope} bridge` : step.kind))}</b><small>${esc(compact(step.label, 38))}</small></button>`).join("")}</div><div class="graph-guide-current"><small>${esc(titleCase(current.kind === "bridge" ? `${current.scope} bridge` : current.kind))} / step ${guide.index + 1} of ${guide.steps.length}</small><b>${esc(current.label)}</b><span>${esc(current.detail)}</span></div>${graphGuideEvidenceMarkup(current)}${replayMarkup}<footer><button id="graph-guide-prev" ${guide.index === 0 ? "disabled" : ""}>Previous</button><button id="graph-guide-workbench">Return to workbench</button><button id="graph-guide-next" ${guide.index === guide.steps.length - 1 ? "disabled" : ""}>Next</button></footer>`;
  $$('[data-graph-guide-step]', panel).forEach((button) => button.addEventListener("click", () => navigateGraphGuide(Number(button.dataset.graphGuideStep))));
  $("#graph-guide-prev")?.addEventListener("click", () => navigateGraphGuide(guide.index - 1));
  $("#graph-guide-next")?.addEventListener("click", () => navigateGraphGuide(guide.index + 1));
  $("#graph-guide-replay")?.addEventListener("click", () => {
    const direction = state.guideBridgeDirections.get(replayKey) ?? true;
    animateGuideBridge(current, direction, {speedMultiplier:2});
  });
  $("#graph-guide-workbench")?.addEventListener("click", () => { clearGuideTransitionAnimation(); closeSource(); setMode("workbench"); });
  $("#graph-guide-close")?.addEventListener("click", () => { clearGuideTransitionAnimation(); state.activeGuide = null; renderGraphGuide(); drawGraph(); });
  bindSourceButtons(panel);
  bindGraphGuidePanel(panel);
}

function renderCaseList() {
  const selectedIds = RECORDED_DEMO_CASES[state.taskFilter] || [];
  const byId = new Map(state.bundle.workbench.cases.map((item) => [item.id, item]));
  const recorded = selectedIds.map((id) => byId.get(id)).filter(Boolean);
  const roleLabels = {baseline: "Recorded / direct baseline", success: "Recorded / full coverage", partial: "Recorded / partial mosaic", support: "Recorded / support", opposed: "Recorded / opposition", inconclusive: "Recorded / inconclusive", unresolved: "Recorded / insufficient"};
  $("#case-list").innerHTML = `<p class="case-group-label">Presenter shortlist / ${recorded.length}</p>${recorded.map((item) => `<button class="case-button ${item.id === state.caseId ? "active" : ""}" data-case="${item.id}" data-role="${item.role}"><strong>${esc(item.title)}</strong><small>${esc(roleLabels[item.role] || item.role)}</small></button>`).join("")}<p class="shortlist-note">The data bundle retains the broader audit archive; this rail shows only the three reviewed presentation cases for the selected task.</p>`;
  $$(".case-button").forEach((button) => button.addEventListener("click", () => { state.caseId = button.dataset.case; renderWorkbench(); }));
}

function candidateScreeningMarkup(item) {
  const screen = item.screening;
  if (!screen) return `<div class="candidate-screen-note"><b>Screen pending</b><p>Run <code>tools/audit_rehearsal_candidates.py</code>, rebuild the bundle, and inspect the exact passages before rehearsal.</p></div>`;
  const duplicate = screen.full_query_duplicate_screen;
  const retrieval = screen.atom_retrieval_screen;
  const graph = screen.graph_retrieval_screen;
  const bandLabel = {high_overlap:"High retrieval overlap",moderate_overlap:"Moderate retrieval overlap",weak_overlap:"Weak retrieval overlap"};
  return `<div class="candidate-screen-summary"><article><b>${formatNumber(screen.eligible_document_count)}</b><span>filtered earlier cables</span></article><article><b>${duplicate.status === "no_near_duplicate_observed" ? "None observed" : "Review required"}</b><span>full-target near duplicates</span></article><article><b>${esc(titleCase(retrieval.profile))}</b><span>atom retrieval profile</span></article><article><b>${Number(graph.maximum_observed_rank_lift || 0) > 0 ? "+" : ""}${formatNumber(graph.maximum_observed_rank_lift || 0)}</b><span>largest graph rank change</span></article></div>
    <div class="candidate-screen-atoms">${retrieval.atoms.map((atom) => {
      const top = atom.top_matches?.[0];
      return `<article><span class="atom-id">${esc(atom.atom_id)}</span><div><b>${esc(atom.text)}</b><small>${esc(bandLabel[atom.retrieval_band] || titleCase(atom.retrieval_band))} / cosine ${Number(atom.top_score || 0).toFixed(3)}</small>${top ? `<button class="source-link" data-source-doc="${esc(top.doc_id)}" data-source-paragraph="${esc(top.paragraph_id)}" data-highlight="${esc(salientTerms(atom.text).join("||"))}">Inspect top retrieval: ${esc(top.doc_id)} / ${esc(top.paragraph_id)}</button>` : ""}</div></article>`;
    }).join("")}</div><p class="candidate-screen-caveat">Retrieval overlap is not an evidence judgment. A high score can still be a false semantic match; promotion requires exact passage adjudication against every atom.</p>`;
}

function candidateMarkup(item) {
  const sourceTerms = salientTerms(item.query);
  const sourceSegments = item.source.paragraph_ids?.join(", ") || item.source.paragraph_id;
  return `<header class="case-hero candidate-hero"><p class="eyebrow">${esc(titleCase(item.task))} rehearsal candidate / ${esc(item.source.doc_id)}</p><h1>${esc(item.title)}</h1><p>${esc(item.authority)}</p><div class="case-status-row"><span class="status-pill outcome">Source verified</span><span class="status-pill">Benchmark not executed</span><span class="status-pill">No coverage label</span></div></header><div class="workspace-body">
    ${workflowStep("01", "Proposed analyst target", "Exact SECRET-source provenance", `<div class="query-block">${esc(item.query)}</div><div class="source-origin"><span><b>${esc(item.source.doc_id)} / ${esc(sourceSegments)}</b><small>${esc(item.source.date)} / ${esc(item.source.classification)} / ${esc(item.source.subject)}</small></span><button class="source-link" data-source-doc="${esc(item.source.doc_id)}" data-source-paragraph="${esc(item.source.paragraph_id)}" data-highlight="${esc(sourceTerms.join("||"))}">Open source cable and highlight target terms</button></div><div class="source-excerpt">${highlighted(item.source.context, sourceTerms)}</div>`)}
    ${workflowStep("02", "Proposed atomic questions", "Rehearsal design / not adjudicated", `<div class="atom-table">${item.atoms.map((atom) => `<div class="atom-row candidate-atom"><span class="atom-id">${esc(atom.id)}</span><span class="atom-text"><b>${esc(atom.text)}</b><small>Candidate retrieval target</small></span><span class="atom-status not_assessed">Not assessed</span></div>`).join("")}</div>`)}
    ${workflowStep("03", "Pre-benchmark retrieval screen", "Deterministic / descriptive only", candidateScreeningMarkup(item))}
    ${workflowStep("04", "Required validation", "Before promotion to the recorded casebook", `<div class="candidate-gates"><article><b>1 / Leakage screen</b><p>Review the strongest policy-filtered matches and confirm that no passage restates the complete target.</p></article><article><b>2 / Retrieval comparison</b><p>Run document-only and graph-assisted conditions under the same strict date and metadata policy.</p></article><article><b>3 / Citation adjudication</b><p>Inspect exact source passages atom by atom; retrieval relevance alone cannot certify coverage.</p></article><article><b>4 / Outcome lock</b><p>Only an executed, archived run may receive FULL, PARTIAL, opposition, resolved, or insufficient labels.</p></article></div>`)}
    ${workflowStep("05", "Rehearse as a live task", "Local exploratory retrieval", `<div class="candidate-launch"><p>This keeps the selected ${esc(liveTaskMeta[item.task]?.label.toLowerCase() || item.task)} contract and reviewed atoms while withholding the source cable from evidence. The result remains a retrieval rehearsal, not a benchmark verdict.</p><button id="candidate-live" class="primary-action">Load into live task</button>${item.answer ? `<details><summary>Show held-out redaction answer for curator review</summary><b>${esc(item.answer)}</b></details>` : ""}</div>`)}</div>`;
}

function bindCandidateWorkspace(item) {
  bindSourceButtons($("#case-workspace"));
  $("#candidate-live")?.addEventListener("click", () => {
    state.liveTask = item.task;
    state.liveDraft = liveProfiles(item.task).find((profile) => profile.profileId === item.id) || {
      profileId:item.id, profileKind:"candidate", task:item.task, query:item.query,
      date:item.source.date, sourceDocId:item.source.doc_id,
      atoms:item.atoms.map((atom) => ({text:atom.text, required:true})),
    };
    state.livePayload = null;
    state.workbenchMode = "live";
    $$('[data-workbench-mode]').forEach((button) => button.classList.toggle("active", button.dataset.workbenchMode === "live"));
    renderCaseFilters(); renderWorkbench();
  });
}

function caseMarkup(item) {
  const result = item.result;
  const audited = String(result.authority || "").startsWith("audited_");
  return `<header class="case-hero"><p class="eyebrow">${esc(titleCase(item.task))} / ${esc(item.source.doc_id)}</p><h1>${esc(item.title)}</h1><p>${esc(item.stage_note)}</p><div class="case-status-row"><span class="status-pill outcome">${esc(result.display_outcome)}</span><span class="status-pill">${formatNumber(item.eligibility.eligible_document_count)} filtered earlier cables</span><span class="status-pill">${esc(audited ? "citation assessment retained separately" : "native deterministic outcome")}</span></div></header>
    <div class="workspace-body">
      ${workflowStep("01", "Analyst target and provenance", "Input", queryMarkup(item))}
      ${workflowStep("02", "Atomic evidence questions", recordedAtomMethod(item), atomMarkup(item))}
      ${workflowStep("03", "Filtered evidence universe", "Deterministic policy", eligibilityMarkup(item))}
      ${workflowStep("04", "Graph retrieval paths", "Deterministic traversal", routesMarkup(item))}
      ${workflowStep("05", "Retrieved source passages", "Citations and context", workbenchEvidenceMarkup(item))}
      ${workflowStep("06", "Coverage and outcome", "Rule-based aggregation", resultMarkup(item))}
      ${workflowStep("07", "Analyst digest", item.digest_source || "Explanatory only", digestMarkup(item))}
    </div>`;
}

function recordedAtomMethod(item) {
  const sources = new Set((item.atoms || []).map((atom) => String(atom.parser_source || "").toLowerCase()).filter(Boolean));
  if ([...sources].some((source) => source.startsWith("openai"))) return "Archived OpenAI parse / no live API call";
  return "Archived reviewed parse / no live API call";
}

function workflowStep(index, title, method, body) {
  return `<section class="workflow-step"><div class="step-index">${index}</div><div class="step-card"><header class="step-header"><div><p>Pipeline stage ${index}</p><h2>${esc(title)}</h2></div><div class="step-method">${esc(method)}</div></header><div class="step-body">${body}</div></div></section>`;
}

function queryMarkup(item) {
  const terms = salientTerms(item.query);
  return `<div class="query-block">${esc(item.query)}</div><div class="source-origin"><span><b>${esc(item.source.doc_id)} / ${esc(item.source.paragraph_id)}</b><small>${esc(item.source.date)} / held-out ${esc(item.source.classification)} source</small></span><button class="source-link" data-source-doc="${esc(item.source.doc_id)}" data-source-paragraph="${esc(item.source.paragraph_id)}" data-highlight="${esc(terms.join("||"))}">Open source with query terms highlighted</button></div>${item.source.context ? `<div class="source-excerpt">${highlighted(item.source.context, terms)}</div>` : ""}`;
}

function salientTerms(value) {
  const stop = new Set(["the","and","that","with","from","were","was","had","not","for","into","could","would","this","their","there","against","toward"]);
  return [...new Set(String(value || "").match(/[A-Za-z][A-Za-z'-]{3,}/g) || [])].filter((term) => !stop.has(term.toLowerCase())).sort((a,b) => b.length - a.length).slice(0,8);
}

function atomMarkup(item) {
  if (!item.atoms.length) return `<p>No validated atomic parse was recorded.</p>`;
  const contractNote = item.task === "redaction" && item.task_targets?.length
    ? `<p class="atom-contract-note"><b>${item.task_targets.length} hidden ${item.native_task === "sentence_redaction" ? "sentence" : "value"} slot${item.task_targets.length === 1 ? "" : "s"}</b> / ${item.atoms.length} retrieval question${item.atoms.length === 1 ? "" : "s"}. The rows below constrain the same hidden target; they are not separate redactions.</p>`
    : "";
  return `${contractNote}<div class="atom-table">${item.atoms.map((atom, index) => `<button class="atom-row" data-atom-index="${index}"><span class="atom-id">${esc(atom.display_id)}</span><span class="atom-text"><b>${esc(atom.text)}</b><small>${esc([atom.subject, atom.predicate, atom.object].filter(Boolean).join(" / "))}</small>${atom.constraint_role ? `<small class="constraint-role">${esc(titleCase(atom.constraint_role))}${atom.required_for_resolution ? " / required for the fill" : " / corroborating only"}</small>` : ""}</span><span class="atom-status ${statusClass(atom.coverage_status)}">${esc(titleCase(atom.coverage_status))}</span></button>`).join("")}</div><p class="interaction-hint">These are archived parser outputs, replayed without an API call. Select an atom to inspect its cited documents and recorded assessment.</p>`;
}

function eligibilityMarkup(item) {
  return `<div class="eligibility-summary"><b>${formatNumber(item.eligibility.eligible_document_count)} cables remained</b><span>Only documents strictly before ${esc(item.source.date)} with a permitted original classification and no NOFORN marking entered retrieval. The held-out source was independently guarded by document ID.</span></div>`;
}

function routesMarkup(item) {
  if (!item.graph_paths.length) return `<p>No graph route was emitted for this result.</p>`;
  const paths = item.graph_paths.slice(0, state.pathDisplayLimit);
  const rolesByDocument = new Map();
  item.evidence.forEach((row) => {
    if (!row.doc_id) return;
    const role = evidenceRouteRole(row);
    if (!rolesByDocument.has(row.doc_id)) rolesByDocument.set(row.doc_id, new Set());
    rolesByDocument.get(row.doc_id).add(role);
  });
  return `<div class="route-display-toolbar"><p class="route-limit-note">Showing ${formatNumber(paths.length)} of ${formatNumber(item.graph_paths.length)} retained retrieval paths. Multiple paths may reach the same cable; the selected limit controls presentation, not the number of documents searched.</p>${pathDisplayControl(item.graph_paths.length)}</div><div class="route-list">${paths.map((path) => {
    const stepEntries = visibleRouteStepEntries(path);
    const targetSegment = terminalTargetSegment(path);
    const chain = stepEntries.length ? stepEntries.map(({step, sourceIndex}, index) => `${index ? `<span class="route-arrow">-&gt;<small>${esc(titleCase(step.incoming_edge || "traverse"))}</small></span>` : ""}<button class="route-node" data-route-path="${item.graph_paths.indexOf(path)}" data-route-step="${sourceIndex}"><span>${esc(step.type)}</span><b>${esc(step.label)}</b></button>`).join("") : `<button class="route-node direct" data-route-path="${item.graph_paths.indexOf(path)}" data-route-step="-1"><span>Direct seed</span><b>No graph hop required</b></button>`;
    const evidenceRoles = [...(rolesByDocument.get(path.doc_id) || [])];
    const documentRole = evidenceRoles.length ? evidenceRoles.map(titleCase).join(" + ") : "Awaiting evidence validation";
    const pathIndex = item.graph_paths.indexOf(path);
    const routeHighlights = path.highlight_terms?.length ? path.highlight_terms : salientTerms(path.route_text);
    const terminalLabel = targetSegment?.step.incoming_edge ? `${titleCase(targetSegment.step.incoming_edge)} / ranks` : "ranks";
    return `<div class="route"><div class="route-origin"><button class="route-query" data-route-path="${pathIndex}" data-route-step="-1"><span>${esc(titleCase(path.coverage_channel || "query route"))}</span><b>${esc(path.route_text)}</b></button><button class="route-guide-button" data-recorded-guide-path="${pathIndex}">Guide path in corpus map</button></div><div class="route-chain">${chain}<span class="route-arrow">-&gt;<small>${esc(terminalLabel)}</small></span><button class="route-document" data-source-doc="${esc(path.doc_id)}" data-source-paragraph="${esc(path.paragraph_id || "")}" data-highlight="${esc(routeHighlights.join("||"))}" title="${esc(path.match_basis || "retrieval provenance")}"><span>Retrieved candidate / ${esc(documentRole)}</span><b>${esc(path.doc_id || "unbound")}</b><small>${path.hop_count} hop${path.hop_count === 1 ? "" : "s"} / final rank ${path.rank || "-"}</small></button></div></div>`;
  }).join("")}</div><p class="inspector-summary">Each row is a retrieval lineage. Guide path keeps a persistent controller on the corpus map; selecting an individual step opens its detailed provenance. The terminal cable remains only a candidate until a cited passage is validated in the evidence stage.</p>`;
}

function evidenceRouteRole(evidence) {
  const setLabels = {minimal:"minimal set",supplementary:"supplementary",context:"context / not proof",fill:"answer-bearing"};
  const directionLabels = {support:"support",opposition:"opposition",partial:"partial clue",context:"context / not proof",fill:"answer-bearing"};
  const labels = [setLabels[evidence.set_role], directionLabels[evidence.direction_role]].filter(Boolean);
  return [...new Set(labels)].join(" / ") || "validated evidence";
}

function consolidateEvidenceRows(rows) {
  const consolidated = new Map();
  rows.forEach((row) => {
    const textKey = String(row.text || "").replace(/\s+/g, " ").trim();
    const key = [row.doc_id || "", row.paragraph_id || "", textKey].join("::");
    if (!consolidated.has(key)) {
      consolidated.set(key, {
        ...row,
        _recordCount:1,
        _evidenceIds:[row.evidence_id].filter(Boolean),
        atom_assessments:[...(row.atom_assessments || [])],
        highlight_terms:[...(row.highlight_terms || [])],
      });
      return;
    }
    const existing = consolidated.get(key);
    existing._recordCount += 1;
    if (row.evidence_id && !existing._evidenceIds.includes(row.evidence_id)) existing._evidenceIds.push(row.evidence_id);
    const assessmentKeys = new Set(existing.atom_assessments.map((item) => [item.atom_text, item.status, item.reason].join("::")));
    (row.atom_assessments || []).forEach((item) => {
      const assessmentKey = [item.atom_text, item.status, item.reason].join("::");
      if (!assessmentKeys.has(assessmentKey)) {
        assessmentKeys.add(assessmentKey);
        existing.atom_assessments.push(item);
      }
    });
    existing.highlight_terms = [...new Set([...existing.highlight_terms, ...(row.highlight_terms || [])])];
    if (row.citation_valid === false) existing.citation_valid = false;
  });
  return [...consolidated.values()];
}

function evidenceBindingCount(rows) {
  return rows.reduce((total, row) => total + Math.max(1, (row.atom_assessments || []).length), 0);
}

function workbenchEvidenceMarkup(item) {
  if (!item.evidence.length) return `<p>No certifying evidence was emitted. Retrieved context remains visible in the graph route stage, but it is not promoted to proof.</p>`;
  const order = item.task === "hypothesis" ? ["support", "opposition", "partial", "context"] : ["minimal", "supplementary", "partial", "context", "fill", "support"];
  const groupKey = (evidence) => item.task === "hypothesis" ? (evidence.direction_role || "context") : (evidence.set_role || evidence.direction_role || "context");
  const groups = new Map();
  item.evidence.forEach((evidence) => { const key = groupKey(evidence); if (!groups.has(key)) groups.set(key, []); groups.get(key).push(evidence); });
  return `<div class="evidence-groups">${[...groups.entries()].sort((a,b) => order.indexOf(a[0]) - order.indexOf(b[0])).map(([group, evidenceRows]) => {
    const displayRows = consolidateEvidenceRows(evidenceRows);
    const bindingCount = evidenceBindingCount(displayRows);
    const bindingNote = bindingCount > displayRows.length ? ` / ${formatNumber(bindingCount)} atom bindings` : "";
    return `<section class="evidence-group ${esc(group)}"><header><div><b>${esc(evidenceGroupLabel(group, item))}</b><span>${formatNumber(displayRows.length)} unique cited passage${displayRows.length === 1 ? "" : "s"}${bindingNote}</span></div><p>${esc(evidenceGroupDescription(group, item))}</p></header><div class="evidence-grid">${displayRows.map((evidence) => workbenchEvidenceCard(evidence, item)).join("")}</div></section>`;
  }).join("")}</div>`;
}

function usesCandidateConstraintEngine(item) {
  return String(item?.result?.authority || "").includes("candidate_constraint");
}

function evidenceGroupLabel(group, item) {
  if (usesCandidateConstraintEngine(item) && group === "minimal") return "Minimal candidate-binding set";
  return ({support: "Support-bearing citations", opposition: "Opposition-bearing citations", partial: "Relevant but incomplete clues", minimal: "Minimal certifying set", supplementary: "Corroborating citations", fill: "Answer-bearing citations", context: "Retrieved context / not proof"})[group] || titleCase(group);
}

function evidenceGroupDescription(group, item) {
  if (usesCandidateConstraintEngine(item) && group === "minimal") return "These raw passages passed deterministic candidate-binding gates. The hidden slot resolves only if the same canonical candidate clears every required constraint.";
  return ({support: "These passages align with the hypothesis.", opposition: "These passages refute at least one hypothesis atom.", partial: "These passages contribute clues without certifying the full atom.", minimal: "Smallest recorded set used for coverage.", supplementary: "Additional support retained for analyst review.", fill: "Passages used to bind the same candidate across constraints.", context: "Useful background that was not promoted to certifying evidence."})[group] || "Recorded evidence role.";
}

function workbenchEvidenceCard(evidence, item) {
  const atomRows = evidence.atom_assessments || [];
  const assessedTerms = atomRows.flatMap((row) => salientTerms(row.atom_text));
  const highlightTerms = [...new Set([
    ...(evidence.highlight_terms || []),
    ...(evidence.candidate ? [evidence.candidate] : []),
    ...assessedTerms,
  ].filter(Boolean))].slice(0,8);
  const matchedAnchors = (evidence.highlight_terms || []).slice(0,4);
  const citationFlag = evidence.citation_valid === false ? `<span class="citation-rejected">Citation check failed / context only</span>` : "";
  const recordLabel = evidence._recordCount > 1 ? `${evidenceBindingCount([evidence])} atom bindings` : evidence.evidence_id;
  return `<article class="workbench-evidence ${esc(evidence.set_role)} ${esc(evidence.direction_role || "")}"><span class="set-label">${esc(recordLabel)} / ${esc(titleCase(evidence.direction_role || evidence.set_role))}</span>${citationFlag}<h3>${esc(evidence.doc_id || "Unbound evidence")}</h3><div class="doc-meta">${esc([evidence.date, evidence.classification, evidence.path_kind, evidence.hop_count ? `${evidence.hop_count} graph hops` : "direct candidate"].filter(Boolean).join(" / "))}</div>${matchedAnchors.length ? `<div class="match-anchors"><small>Why this passage was selected</small><span>${matchedAnchors.map((term) => `<i>${esc(term)}</i>`).join("")}</span></div>` : ""}<blockquote>${highlighted(evidence.text, highlightTerms)}</blockquote>${atomRows.length ? `<div class="citation-assessment">${atomRows.map((row) => `<div><span class="atom-status ${statusClass(row.status)}">${esc(row.status)}</span><b>${esc(row.atom_text)}</b><p>${esc(row.reason)}</p>${row.citation_valid === false ? `<small>Rejected by citation validation; this passage is retained for audit only.</small>` : ""}</div>`).join("")}</div>` : ""}${evidence.doc_id ? `<button class="source-link" data-source-doc="${esc(evidence.doc_id)}" data-source-paragraph="${esc(evidence.paragraph_id || "")}" data-highlight="${esc(highlightTerms.join("||"))}">Read and highlight this passage in the cable</button>` : ""}</article>`;
}

function resultMarkup(item) {
  const result = item.result; const total = result.total || item.atoms.length || 1;
  const targetTotal = item.task === "redaction" && result.target_unit_count ? result.target_unit_count : total;
  const targetCovered = item.task === "redaction" && result.target_unit_count ? result.resolved_target_count : Number(result.covered || 0);
  const targetPartial = item.task === "redaction" ? 0 : Number(result.partial || 0);
  const fullAngle = Math.max(0, Math.min(360, targetCovered / targetTotal * 360));
  const partialAngle = Math.max(fullAngle, Math.min(360, (targetCovered + targetPartial) / targetTotal * 360));
  let redactionSummary = "";
  if (item.task === "redaction" && item.redaction_constraints.length) {
    const required = item.redaction_constraints.filter((row) => row.required);
    const corroborating = item.redaction_constraints.filter((row) => !row.required);
    const requiredPass = required.filter((row) => row.support_gate_pass).length;
    const corroboratingPass = corroborating.filter((row) => row.support_gate_pass).length;
    const detail = corroborating.length
      ? `${corroboratingPass}/${corroborating.length} corroborating constraints also passed. Corroborating atoms add confidence but do not enlarge the one-slot resolution denominator.`
      : `${targetCovered}/${targetTotal} hidden slot${targetTotal === 1 ? "" : "s"} resolved. Every required constraint must pass before the candidate can fill the slot.`;
    redactionSummary = `<div class="constraint-summary"><b>${requiredPass}/${required.length} required candidate-binding constraints</b><span>${detail}</span></div>`;
  } else if (item.task === "redaction") {
    redactionSummary = `<div class="constraint-summary"><b>${targetCovered}/${targetTotal} hidden slot${targetTotal === 1 ? "" : "s"} resolved</b><span>The matcher separately recorded ${result.covered}/${total} fully cleared retrieval gate${total === 1 ? "" : "s"}${result.partial ? ` and ${result.partial} partial clue${result.partial === 1 ? "" : "s"}` : ""}. Retrieval questions constrain the slot; they do not create additional redactions.</span></div>`;
  }
  const coverageLabel = item.task === "redaction" ? "slots resolved" : "fully certified";
  const coverageKey = item.task === "redaction"
    ? `<span><i class="full"></i>Resolved ${targetCovered}</span><span><i></i>Unresolved ${Math.max(0, targetTotal-targetCovered)}</span>`
    : `<span><i class="full"></i>Certified ${result.covered}</span><span><i class="partial"></i>Partial ${result.partial || 0}</span><span><i></i>Unresolved ${Math.max(0,total-result.covered-(result.partial || 0))}</span>`;
  const candidateLead = !result.selected_answer && result.candidate_lead ? `<div class="selected-answer unresolved"><span>Unresolved audit lead</span><b>${esc(result.candidate_lead)}</b><small>Not accepted as the hidden value</small></div>` : "";
  const auditLabel = usesCandidateConstraintEngine(item) ? "separate LLM citation audit" : "separate citation audit";
  const integrity = usesCandidateConstraintEngine(item) ? `<div class="result-fact"><b>${result.citation_integrity ? "PASS" : "FAILED"}</b><span>citation integrity gate</span></div>` : "";
  return `${taskDecisionVisual(item)}${redactionSummary}<div class="result-grid"><div class="coverage-visual"><div class="coverage-ring" style="--full:${fullAngle}deg;--partial:${partialAngle}deg"><span><b>${targetCovered}/${targetTotal}</b><small>${coverageLabel}</small>${targetPartial ? `<em>+ ${targetPartial} partial</em>` : ""}</span></div><div class="coverage-key">${coverageKey}</div></div><div class="result-summary"><p class="eyebrow">Recorded outcome</p><h3>${esc(result.display_outcome)}</h3><p>${esc(item.stage_note)}</p>${result.selected_answer ? `<div class="selected-answer"><span>Selected answer</span><b>${esc(result.selected_answer)}</b></div>` : ""}${candidateLead}<details class="decision-audit"><summary>Show native and citation-layer audit</summary><div class="result-facts"><div class="result-fact"><b>${esc(result.native_verdict || "N/A")}</b><span>native deterministic matcher</span></div><div class="result-fact"><b>${esc(result.citation_verdict || "N/A")}</b><span>${esc(auditLabel)}</span></div>${integrity}<div class="result-fact"><b>${esc(result.direction || "No certifying direction")}</b><span>evidence direction</span></div><div class="result-fact"><b>${esc(result.evidence_strength || "Not calibrated")}</b><span>recorded strength band</span></div></div></details></div></div><p class="inspector-summary">${esc(result.definition || "Coverage is a completeness ratio, not a calibrated probability.")}</p>`;
}

function taskDecisionVisual(item) {
  const result = item.result;
  if (item.task === "redaction" && item.candidate_chart.length) {
    return `<div class="candidate-chart-wrap"><div class="chart-heading"><b>Candidate fit across shared constraints</b><span>Deterministic score; not a probability</span></div><div class="candidate-chart">${item.candidate_chart.slice(0,8).map((row) => { const selected = String(row.candidate).toUpperCase() === String(result.selected_answer).toUpperCase(); const lead = !selected && String(row.candidate).toUpperCase() === String(result.candidate_lead).toUpperCase(); return `<div class="candidate-bar ${selected ? "selected" : ""} ${lead ? "audit-lead" : ""}" style="--score:${Math.max(.03, Math.min(1, row.score || 0))}" title="${esc(`${row.required_supported}/${row.required_count} required; ${row.corroborating_supported}/${row.corroborating_count} corroborating`)}"><span>${Number(row.score || 0).toFixed(2)}</span><b>${esc(row.candidate)}</b><small>${lead ? "audit lead / unresolved" : `${row.required_supported}/${row.required_count} required`}</small></div>`; }).join("")}</div></div>`;
  }
  if (item.task === "hypothesis") {
    const direction = String(result.direction || "").toUpperCase();
    const active = direction.includes("OPPOSITION") ? "oppose" : direction.includes("SUPPORT") ? "support" : direction.includes("MIXED") ? "mixed" : "insufficient";
    const counts = item.evidence.reduce((output, evidence) => { const key = evidence.direction_role || "context"; output[key] = (output[key] || 0) + 1; return output; }, {});
    return `<div class="direction-wrap"><div class="chart-heading"><b>Evidence direction</b><span>Certified direction and retained context are shown separately</span></div><div class="direction-scale"><div class="direction-zone ${active === "support" ? "active" : ""}"><b>Supports</b><span>${counts.support || 0} cited passage${counts.support === 1 ? "" : "s"}</span></div><div class="direction-zone ${active === "mixed" || active === "insufficient" ? "active" : ""}"><b>${active === "insufficient" ? "Insufficient" : "Mixed"}</b><span>${(counts.partial || 0) + (counts.context || 0)} partial/context passages</span></div><div class="direction-zone oppose ${active === "oppose" ? "active" : ""}"><b>Opposes</b><span>${counts.opposition || 0} cited passage${counts.opposition === 1 ? "" : "s"}</span></div></div></div>`;
  }
  if (item.task === "reconstruction") {
    return `<div class="atom-coverage-wrap"><div class="chart-heading"><b>Piece-by-piece coverage</b><span>Each tile is one required target</span></div><div class="atom-coverage-strip">${item.atoms.map((atom) => `<div class="coverage-tile ${statusClass(atom.coverage_status)}"><b>${esc(atom.display_id)}</b><span>${esc(titleCase(atom.coverage_status))}</span></div>`).join("")}</div></div>`;
  }
  return "";
}

function digestMarkup(item) {
  const explanations = item.atoms.map((atom) => `<article><span class="atom-id">${esc(atom.display_id)}</span><div><b>${esc(atom.text)}</b><p>${esc(atom.coverage_reason || "No separate citation explanation was recorded for this target.")}</p>${atom.document_ids?.length ? `<small>Evidence: ${esc(atom.document_ids.join(", "))}</small>` : ""}</div></article>`).join("");
  const explanationTitle = item.task === "redaction" ? "How each retrieval question constrained the hidden slot" : "Why each target did or did not clear";
  return `<div class="digest-card"><p>${esc(item.digest || item.stage_note)}</p><small>Explanatory synthesis only. It cannot change retrieval, citations, target coverage, or the native deterministic verdict.</small></div><div class="digest-walkthrough"><div class="chart-heading"><b>${explanationTitle}</b><span>Recorded citation assessment</span></div>${explanations}</div>`;
}

function bindCaseWorkspace(item) {
  const root = $("#case-workspace");
  bindSourceButtons(root);
  bindPathDisplayControl(root, () => {
    const scrollTop = root.scrollTop;
    renderWorkbench();
    requestAnimationFrame(() => { $("#case-workspace").scrollTop = scrollTop; });
  });
  $$('[data-route-path]', root).forEach((button) => button.addEventListener("click", () => openRouteDetail(item, Number(button.dataset.routePath), Number(button.dataset.routeStep))));
  $$('[data-recorded-guide-path]', root).forEach((button) => button.addEventListener("click", () => {
    const path = item.graph_paths[Number(button.dataset.recordedGuidePath)];
    if (path) activateGraphGuide(recordedRouteGuide(item, path), 0);
  }));
  $$('[data-atom-index]', root).forEach((button) => button.addEventListener("click", () => openAtomDetail(item, Number(button.dataset.atomIndex))));
}

init();
