"use strict";

const API = ""; 
const POLL_MS = 4000;

const STATE = {
  status:      "idle",
  workers:     [],
  assignment:  null,
  modelMeta:   null,
  generating:  false,
  tokenCount:  0,
  t0:          null,
  currentTab:  "chat",
};

document.addEventListener("DOMContentLoaded", () => {
  refreshStatus();
  refreshWorkers();
  refreshModels();
  setInterval(refreshStatus,  POLL_MS);
  setInterval(refreshWorkers, POLL_MS * 2);
});

async function apiFetch(path, opts = {}) {
  const resp = await fetch(API + path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || resp.statusText);
  }
  return resp.json();
}

function switchTab(name) {
  STATE.currentTab = name;
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
  document.getElementById(`tab${cap(name)}`).classList.add("active");
  document.getElementById(`panel${cap(name)}`).classList.add("active");
}
const cap = s => s.charAt(0).toUpperCase() + s.slice(1);

async function refreshStatus() {
  try {
    const d = await apiFetch("/api/status");
    STATE.status = d.status;
    renderStatusChip(d);
    updateButtons(d);
    if (d.master_ip) {
      document.getElementById("statusText").textContent = statusLabel(d.status);
    }
  } catch (_) {}
}

function statusLabel(s) {
  return {
    idle:          "Idle",
    discovering:   "Scanning…",
    discovered:    "Workers found",
    no_workers:    "No workers",
    loading_model: "Analysing…",
    model_ready:   "Model ready",
    deploying:     "Deploying…",
    deployed:      "Ready",
    ready:         "Ready",
    error:         "Error",
  }[s] || s;
}

function renderStatusChip(d) {
  const chip = document.getElementById("statusChip");
  chip.className = "status-chip";
  if (["discovering", "loading_model", "deploying"].includes(d.status))
    chip.classList.add("active");
  else if (["deployed", "ready", "discovered"].includes(d.status))
    chip.classList.add("ready");
  else if (d.status === "error")
    chip.classList.add("error");
  document.getElementById("statusText").textContent = statusLabel(d.status);
}

function updateButtons(d) {
  const deployed = d.status === "deployed" || d.status === "ready";
  const canDeploy = d.model_loaded && !["deploying", "deployed", "ready"].includes(d.status);
  const btnDeploy = document.getElementById("btnDeploy");
  const hint      = document.getElementById("deployHint");
  const sendBtn   = document.getElementById("sendBtn");
  const caption   = document.getElementById("inputCaption");

  btnDeploy.disabled = !canDeploy;
  hint.textContent = deployed
    ? "✓ Deployed and running"
    : d.model_loaded
    ? "Ready to deploy"
    : "Analyse a model first";

  sendBtn.disabled = !deployed || STATE.generating;
  caption.textContent = deployed
    ? "Press ↵ or click ↑ to send · Shift+↵ for new line"
    : "Deploy a model from the Cluster tab to begin";
}

async function discoverWorkers() {
  const timeout = parseFloat(document.getElementById("discoverTimeout").value) || 15;
  const btn = document.getElementById("btnDiscover");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Scanning…`;

  try {
    await apiFetch("/api/discover", {
      method: "POST",
      body: JSON.stringify({ timeout }),
    });
    let polls = 0;
    const iv = setInterval(async () => {
      polls++;
      await refreshStatus();
      await refreshWorkers();
      if (STATE.status !== "discovering" || polls > 50) {
        clearInterval(iv);
        btn.disabled = false;
        btn.innerHTML = "Scan for Workers";
      }
    }, 400);
  } catch (e) {
    btn.disabled = false;
    btn.innerHTML = "Scan for Workers";
    console.error(e);
  }
}

async function refreshWorkers() {
  try {
    const d = await apiFetch("/api/workers");
    STATE.workers = d.workers || [];
    renderSidebar(STATE.workers);
    renderDeviceTable(STATE.workers);
    const n = STATE.workers.length;
    document.getElementById("sidebarCount").textContent = n;
    document.getElementById("workerCount").textContent = n;

    try {
      const asgn = await apiFetch("/api/layer-assignment");
      STATE.assignment = asgn;
      renderPipeline(asgn);
    } catch (_) {}
  } catch (_) {}
}

function deviceType(id = "", isMaster = false) {
  if (isMaster) return "laptop";
  const s = id.toLowerCase();
  if (/phone|pixel|galaxy|iphone|android|mobile|oneplus|xiaomi|redmi/.test(s)) return "phone";
  if (/tablet|ipad|tab/.test(s))  return "tablet";
  if (/pi|raspberry|rpi/.test(s)) return "pi";
  if (/server|rack|workstation/.test(s)) return "server";
  return "laptop";
}

const DEVICE_ICONS = {
  laptop: `<svg width="20" height="20" viewBox="0 0 20 20" fill="none">
    <rect x="3" y="4" width="14" height="9" rx="1.5" stroke="currentColor" stroke-width="1.4"/>
    <path d="M1 16H19" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/>
    <path d="M7 16L7.5 17.5H12.5L13 16" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>`,

  phone: `<svg width="20" height="20" viewBox="0 0 20 20" fill="none">
    <rect x="5.5" y="2" width="9" height="16" rx="2" stroke="currentColor" stroke-width="1.4"/>
    <circle cx="10" cy="15.5" r="0.75" fill="currentColor"/>
    <path d="M8 4H12" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/>
  </svg>`,

  tablet: `<svg width="20" height="20" viewBox="0 0 20 20" fill="none">
    <rect x="3" y="2" width="14" height="16" rx="2" stroke="currentColor" stroke-width="1.4"/>
    <circle cx="10" cy="15.5" r="0.75" fill="currentColor"/>
    <path d="M7 4H13" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/>
  </svg>`,

  pi: `<svg width="20" height="20" viewBox="0 0 20 20" fill="none">
    <rect x="3" y="5" width="14" height="10" rx="1.5" stroke="currentColor" stroke-width="1.4"/>
    <circle cx="7" cy="10" r="1.2" fill="currentColor" opacity=".5"/>
    <circle cx="10" cy="10" r="1.2" fill="currentColor" opacity=".5"/>
    <circle cx="13" cy="10" r="1.2" fill="currentColor" opacity=".5"/>
    <path d="M6 5V3.5M10 5V3M14 5V3.5" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/>
  </svg>`,

  server: `<svg width="20" height="20" viewBox="0 0 20 20" fill="none">
    <rect x="2" y="4" width="16" height="5" rx="1.5" stroke="currentColor" stroke-width="1.4"/>
    <rect x="2" y="11" width="16" height="5" rx="1.5" stroke="currentColor" stroke-width="1.4"/>
    <circle cx="15.5" cy="6.5" r="1" fill="currentColor" opacity=".5"/>
    <circle cx="15.5" cy="13.5" r="1" fill="currentColor" opacity=".5"/>
    <path d="M5 6.5H11M5 13.5H11" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/>
  </svg>`,
};

function renderSidebar(workers) {
  const list = document.getElementById("deviceList");
  if (!workers.length) {
    list.innerHTML = `<p class="list-empty">No devices found.<br>Run a scan from Cluster.</p>`;
    return;
  }

  list.innerHTML = workers.map(w => {
    const total   = w.total_ram_bytes    || 0;
    const avail   = w.available_ram_bytes || 0;
    const usedPct = total ? Math.round((1 - avail / total) * 100) : 0;
    const freePct = 100 - usedPct;
    const totalGiB = (total / (1024 ** 3)).toFixed(1);
    const availGiB = (avail / (1024 ** 3)).toFixed(1);
    const dtype = deviceType(w.device_id, w.is_master);
    const icon  = DEVICE_ICONS[dtype] || DEVICE_ICONS.laptop;
    const isMaster = !!w.is_master;
    const layers = w.assigned_layers || [];
    const layerPill = layers.length
      ? `<div class="layer-pill">Layers ${layers[0]}–${layers[layers.length-1]}</div>`
      : "";

    return `
      <div class="dev-card">
        <div class="dev-card-top">
          <div class="dev-icon ${isMaster ? "master-icon" : ""}">${icon}</div>
          <div class="dev-info">
            <div class="dev-name">${w.device_id}</div>
            <div class="dev-role">${isMaster ? "Master" : "Worker"} · ${w.host}</div>
          </div>
        </div>
        <div class="ram-section">
          <div class="ram-header">
            <span class="ram-label">RAM</span>
            <span class="ram-value">${availGiB} / ${totalGiB} GiB free</span>
          </div>
          <div class="ram-track">
            <div class="ram-fill" style="width:${freePct}%"></div>
          </div>
        </div>
        ${layerPill}
      </div>`;
  }).join("");
}

function renderDeviceTable(workers) {
  const grid = document.getElementById("deviceGrid");
  if (!workers.length) {
    grid.innerHTML = `<p class="list-empty" style="padding:12px 4px;">No devices yet — run a scan.</p>`;
    return;
  }

  grid.innerHTML = workers.map(w => {
    const total   = w.total_ram_bytes    || 0;
    const avail   = w.available_ram_bytes || 0;
    const usedPct = total ? Math.round((1 - avail / total) * 100) : 0;
    const freePct = 100 - usedPct;
    const totalGiB = (total / (1024 ** 3)).toFixed(1);
    const availGiB = (avail / (1024 ** 3)).toFixed(1);
    const dtype  = deviceType(w.device_id, w.is_master);
    const icon   = DEVICE_ICONS[dtype] || DEVICE_ICONS.laptop;
    const layers = w.assigned_layers || [];

    return `
      <div class="device-row">
        <div class="device-row-icon">${icon}</div>
        <div class="device-row-info">
          <div class="device-row-name">${w.device_id}</div>
          <div class="device-row-ram">${availGiB} GiB free${layers.length ? ` · ${layers.length} layers` : ""}</div>
        </div>
        <div class="device-row-bar">
          <div style="font-size:10px;color:var(--text-3);text-align:right;">${totalGiB}G</div>
          <div class="device-row-track">
            <div class="device-row-fill" style="width:${freePct}%"></div>
          </div>
        </div>
      </div>`;
  }).join("");
}

async function refreshModels() {
  try {
    const d = await apiFetch("/api/models");
    const sel = document.getElementById("modelSelect");
    const prev = sel.value;
    sel.innerHTML = `<option value="">Select a GGUF model…</option>`;
    (d.models || []).forEach(p => {
      const o = document.createElement("option");
      o.value = p;
      o.textContent = p.split("/").pop();
      sel.appendChild(o);
    });
    if (prev) sel.value = prev;
  } catch (_) {}
}

async function loadModel() {
  const path = document.getElementById("modelPathInput").value.trim()
             || document.getElementById("modelSelect").value;
  if (!path) { alert("Please select or enter a model path."); return; }

  const btn = document.getElementById("btnLoad");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Analysing…`;

  try {
    const d = await apiFetch("/api/load-model", {
      method: "POST",
      body: JSON.stringify({ model_path: path }),
    });
    STATE.modelMeta = d;
    renderModelInfo(d);
    updateModelBanner(d);
    await refreshWorkers();
    await refreshStatus();
  } catch (e) {
    alert(`Failed to load model: ${e.message}`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = "Analyse Model";
  }
}

function updateModelBanner(d) {
  const nameEl = document.getElementById("modelNameLabel");
  const metaEl = document.getElementById("modelMetaLabel");
  const dot    = document.getElementById("modelDot");
  const fname  = STATE.modelMeta?.model_path?.split("/").pop() || "Unknown model";
  const tokType = d.tokenizer?.tokenizer_type?.toUpperCase() || "";
  nameEl.textContent = fname;
  metaEl.textContent = d.n_layers
    ? `${d.architecture} · ${d.n_layers} layers · ${tokType}`
    : "";
  dot.classList.add("active");
}

function renderModelInfo(d) {
  const card = document.getElementById("modelInfoCard");
  card.style.display = "";

  const items = [
    { k: "Arch",        v: d.architecture,   u: "" },
    { k: "Layers",      v: d.n_layers,       u: "blocks" },
    { k: "Embed",       v: d.embedding_dim,  u: "dims" },
    { k: "Heads",       v: d.n_heads,        u: `(${d.n_kv_heads} KV)` },
    { k: "FFN dim",     v: d.intermediate_dim, u: "" },
    { k: "Vocab",       v: d.vocab_size,     u: "tokens" },
    { k: "Per layer",   v: `${(d.bytes_per_layer/(1024**2)).toFixed(1)}`, u: "MiB" },
    { k: "Total",       v: `${(d.total_weight_bytes/(1024**3)).toFixed(2)}`, u: "GiB" },
    { k: "Quant",       v: quantLabel(d.dominant_quant_type), u: "" },
  ];

  document.getElementById("modelMetaGrid").innerHTML = items.map(i => `
    <div class="meta-chip">
      <div class="meta-chip-key">${i.k}</div>
      <div class="meta-chip-val">${i.v}</div>
      <div class="meta-chip-unit">${i.u}</div>
    </div>`).join("");

  if (d.tokenizer) {
    const t = d.tokenizer;
    const badge = document.getElementById("tokenizerBadge");
    badge.textContent = `${t.tokenizer_type?.toUpperCase()} · vocab ${t.vocab_size?.toLocaleString()}`;
    badge.style.display = t.tokenizer_type ? "" : "none";
  }

  if (d.unassigned_layers?.length) {
    const wrap = document.getElementById("modelInfoCard").querySelector(".card-body");
    wrap.insertAdjacentHTML("beforeend", `
      <div class="warn-banner">
        ⚠ ${d.unassigned_layers.length} layer(s) unassigned — add more devices or use a smaller model.
      </div>`);
  }
}

function quantLabel(qt) {
  return {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",
    10:"Q2_K",11:"Q3_K_S",12:"Q4_K_S",13:"Q4_K_M",14:"Q5_K_S",
    15:"Q5_K_M",16:"Q6_K",17:"Q8_K"}[qt] || `Q${qt}`;
}

function renderPipeline(asgn) {
  const wrap = document.getElementById("pipelineViz");
  if (!asgn?.pipeline?.length) {
    wrap.innerHTML = `<div class="pipeline-empty">
      <svg width="26" height="26" viewBox="0 0 26 26" fill="none" opacity=".3">
        <rect x="1" y="7" width="7" height="12" rx="2" stroke="#999" stroke-width="1.5"/>
        <rect x="9.5" y="4" width="7" height="18" rx="2" stroke="#999" stroke-width="1.5"/>
        <rect x="18" y="9" width="7" height="9" rx="2" stroke="#999" stroke-width="1.5"/>
      </svg>
      <span>Appears after model analysis</span>
    </div>`;
    return;
  }

  const total = asgn.total_layers;
  const nodes = asgn.pipeline.map((n, i) => {
    const pct = total ? Math.round(n.n_layers / total * 100) : 0;
    const memMiB = (n.mem_used_bytes / (1024**2)).toFixed(0);
    return `
      <div class="pipe-node">
        <div class="pipe-node-name">${n.device_id}</div>
        <div class="pipe-node-range">Layers ${n.layer_start ?? "—"}–${n.layer_end ?? "—"}</div>
        <div class="pipe-node-mem">${n.n_layers} blocks · ${memMiB} MiB</div>
        <div class="pipe-node-bar">
          <div class="pipe-node-bar-fill" style="width:${pct}%"></div>
        </div>
      </div>`;
  });

  const parts = [];
  nodes.forEach((n, i) => {
    parts.push(n);
    if (i < nodes.length - 1) parts.push(`<div class="pipe-arrow">→</div>`);
  });

  wrap.innerHTML = `<div class="pipeline-track">${parts.join("")}</div>`;

  if (asgn.unassigned_layers?.length) {
    wrap.insertAdjacentHTML("beforeend", `
      <div class="warn-banner" style="margin-top:10px;">
        ⚠ ${asgn.unassigned_layers.length} unassigned layers
      </div>`);
  }
}

async function deployModel() {
  const btn = document.getElementById("btnDeploy");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Deploying…`;

  try {
    await apiFetch("/api/deploy", { method: "POST" });
    let polls = 0;
    const iv = setInterval(async () => {
      polls++;
      await refreshStatus();
      if (["deployed","ready"].includes(STATE.status)) {
        clearInterval(iv);
        btn.innerHTML = "Deploy to Cluster";
        document.getElementById("modelDot").classList.add("active");
        document.getElementById("modelNameLabel").textContent =
          STATE.modelMeta?.model_path?.split("/").pop() || "Model active";
      } else if (STATE.status === "error" || polls > 120) {
        clearInterval(iv);
        btn.disabled = false;
        btn.innerHTML = "Deploy to Cluster";
      }
    }, 1000);
  } catch (e) {
    btn.disabled = false;
    btn.innerHTML = "Deploy to Cluster";
    alert(`Deploy failed: ${e.message}`);
  }
}

async function resetCluster() {
  try {
    await apiFetch("/api/reset", { method: "POST" });
  } catch (e) { console.error(e); }
}

function handleKey(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!document.getElementById("sendBtn").disabled) generate();
  }
}

function autoResize(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 160) + "px";
}

async function generate() {
  if (STATE.generating) return;
  const input = document.getElementById("chatInput");
  const prompt = input.value.trim();
  if (!prompt) return;

  const temperature = parseFloat(document.getElementById("temperature").value) || 0.7;
  const topK        = parseInt(document.getElementById("topK").value) || 40;
  const maxTokens   = parseInt(document.getElementById("maxTokens").value) || 256;

  const welcome = document.querySelector(".welcome-screen");
  if (welcome) welcome.remove();

  appendMessage("user", prompt);
  input.value = "";
  input.style.height = "auto";

  const typingId = addTypingIndicator();

  STATE.generating = true;
  STATE.tokenCount = 0;
  STATE.t0 = performance.now();
  document.getElementById("sendBtn").disabled = true;

  let aiText = "";

  try {
    const resp = await fetch(API + "/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, temperature, top_k: topK, max_new_tokens: maxTokens }),
    });

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let aiBubble = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        let evt;
        try { evt = JSON.parse(line.slice(6)); } catch { continue; }

        if (evt.error) {
          removeTypingIndicator(typingId);
          appendMessage("ai", `Error: ${evt.error}`);
          break;
        }

        if (evt.token_text) {
          STATE.tokenCount++;
          aiText += evt.token_text;

          if (!aiBubble) {
            removeTypingIndicator(typingId);
            aiBubble = createAiBubble();
          }
          aiBubble.textContent = aiText;
          scrollMessages();
        }

        if (evt.done) {
          const elapsed = ((performance.now() - STATE.t0) / 1000).toFixed(1);
          const tps = (STATE.tokenCount / parseFloat(elapsed)).toFixed(1);
          if (aiBubble) {
            const stats = document.createElement("div");
            stats.style.cssText = "font-size:11px;color:var(--text-3);margin-top:4px;";
            stats.textContent = `${STATE.tokenCount} tokens · ${elapsed}s · ${tps} tok/s`;
            aiBubble.parentElement.appendChild(stats);
          }
        }
      }
    }
  } catch (e) {
    removeTypingIndicator(typingId);
    if (!aiText) appendMessage("ai", `Network error: ${e.message}`);
  } finally {
    STATE.generating = false;
    document.getElementById("sendBtn").disabled = false;
    scrollMessages();
  }
}

function appendMessage(role, text) {
  const msgs = document.getElementById("messages");
  const row  = document.createElement("div");
  row.className = `msg-row ${role}`;

  const avatar = document.createElement("div");
  avatar.className = `msg-avatar ${role === "ai" ? "ai-avatar" : "user-avatar"}`;
  avatar.textContent = role === "ai" ? "NC" : "You";

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.textContent = text;

  if (role === "ai") {
    row.appendChild(avatar);
    row.appendChild(bubble);
  } else {
    row.appendChild(bubble);
    row.appendChild(avatar);
  }

  msgs.appendChild(row);
  scrollMessages();
  return bubble;
}

function addTypingIndicator() {
  const id = "typing-" + Date.now();
  const msgs = document.getElementById("messages");
  const row  = document.createElement("div");
  row.className = "msg-row ai";
  row.id = id;

  const avatar = document.createElement("div");
  avatar.className = "msg-avatar ai-avatar";
  avatar.textContent = "NC";

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.innerHTML = `<div class="typing-indicator">
    <div class="typing-dot"></div>
    <div class="typing-dot"></div>
    <div class="typing-dot"></div>
  </div>`;

  row.appendChild(avatar);
  row.appendChild(bubble);
  msgs.appendChild(row);
  scrollMessages();
  return id;
}

function removeTypingIndicator(id) {
  document.getElementById(id)?.remove();
}

function createAiBubble() {
  const msgs   = document.getElementById("messages");
  const row    = document.createElement("div");
  row.className = "msg-row ai";

  const avatar = document.createElement("div");
  avatar.className = "msg-avatar ai-avatar";
  avatar.textContent = "NC";

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";

  row.appendChild(avatar);
  row.appendChild(bubble);
  msgs.appendChild(row);
  return bubble;
}

function scrollMessages() {
  const msgs = document.getElementById("messages");
  msgs.scrollTop = msgs.scrollHeight;
}