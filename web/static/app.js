const API = "/api/v1";

const state = {
  tasks: [],
  activeTaskId: null,
  activeTab: "dataset",
  pollTimer: null,
  // Tracks whether a training run was active on the previous poll, so we can
  // do a final panel refresh when it transitions to completed/failed (the
  // UI would otherwise stay stuck on the last in-progress percentage).
  wasTrainingActive: false,
  // The user's chosen base model for fine-tuning, keyed by task ID. This
  // lets the dropdown survive the 1.5s progress polls without resetting
  // the user's selection.
  selectedBaseModel: {},
  // API keys are only ever returned once, at deploy time — kept in memory
  // for this browser session only (never persisted), keyed by deployment ID.
  apiKeys: {},
};

// --- API helpers ---
async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(API + path, opts);
  if (res.status === 204) return null;
  const isJSON = (res.headers.get("content-type") || "").includes("application/json");
  const data = isJSON ? await res.json() : null;
  if (!res.ok) {
    throw new Error((data && data.error) || `Request failed (${res.status})`);
  }
  return data;
}

// apiRaw sends a non-JSON body (e.g. CSV text) and returns the raw Response,
// for endpoints that accept file uploads or return file downloads.
async function apiRaw(method, path, body, headers) {
  return fetch(API + path, { method, headers: headers || {}, body });
}

async function apiRawOrThrow(method, path, body, headers) {
  const res = await apiRaw(method, path, body, headers);
  if (!res.ok) {
    const isJSON = (res.headers.get("content-type") || "").includes("application/json");
    const data = isJSON ? await res.json().catch(() => null) : null;
    throw new Error((data && data.error) || `Request failed (${res.status})`);
  }
  return res;
}

function toast(msg, isError) {
  const el = document.createElement("div");
  el.className = "toast" + (isError ? " error" : "");
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3500);
}

// --- Task list / sidebar ---
async function loadTasks() {
  state.tasks = (await api("GET", "/tasks")) || [];
  renderTaskList();
}

function renderTaskList() {
  const list = document.getElementById("task-list");
  list.innerHTML = "";
  if (state.tasks.length === 0) {
    const p = document.createElement("div");
    p.className = "muted";
    p.textContent = "No tasks yet.";
    list.appendChild(p);
  }
  state.tasks
    .slice()
    .sort((a, b) => (a.created_at < b.created_at ? 1 : -1))
    .forEach((t) => {
      const card = document.createElement("div");
      card.className = "task-card" + (t.id === state.activeTaskId ? " active" : "");
      card.innerHTML = `<div class="name">${escapeHtml(t.name)}</div><div class="type">${t.type}</div>`;
      card.onclick = () => selectTask(t.id);
      list.appendChild(card);
    });
}

function selectTask(id) {
  state.activeTaskId = id;
  state.activeTab = "dataset";
  renderTaskList();
  renderTaskDetail();
}

// --- New task modal ---
function showNewTaskModal() {
  const root = document.getElementById("modal-root");
  root.innerHTML = `
    <div class="modal-backdrop" id="modal-backdrop">
      <div class="modal">
        <h3>New Task</h3>
        <label>Task name</label>
        <input type="text" id="nt-name" placeholder="e.g. Support Ticket Router" />
        <label>Description</label>
        <textarea id="nt-desc" rows="3" placeholder="Classify incoming support tickets into billing, technical, or account categories."></textarea>
        <label>Task type</label>
        <select id="nt-type" style="width:100%;background:var(--charcoal);color:var(--paper);border:1px solid var(--charcoal-3);border-radius:3px;padding:9px 11px;font-family:var(--font-mono);font-size:13px;">
          <option value="classification">Classification</option>
          <option value="extraction">Extraction</option>
          <option value="generation">Generation</option>
        </select>
        <label>Model Architecture / Kind</label>
        <select id="nt-kind" style="width:100%;background:var(--charcoal);color:var(--paper);border:1px solid var(--charcoal-3);border-radius:3px;padding:9px 11px;font-family:var(--font-mono);font-size:13px;">
          <option value="causal_lm">Causal LM (Generative / Instruction / Structured Extraction)</option>
          <option value="token_classifier">Token Classifier (NER / Span Extraction)</option>
          <option value="seq_classifier">Sequence Classifier (Sentiment / Intent)</option>
          <option value="embedding">Embedding (Search / Similarity)</option>
          <option value="reranker">Reranker (Cross-Encoder)</option>
          <option value="vision_lm">Vision-Language (Document AI / Image Extraction)</option>
          <option value="asr">Speech-to-Text (Whisper Fine-Tuning)</option>
        </select>
        <div class="modal-actions">
          <button class="btn ghost" id="nt-cancel">Cancel</button>
          <button class="btn primary" id="nt-create">Create Task</button>
        </div>
      </div>
    </div>`;
  document.getElementById("nt-cancel").onclick = closeModal;
  document.getElementById("modal-backdrop").onclick = (e) => {
    if (e.target.id === "modal-backdrop") closeModal();
  };
  document.getElementById("nt-create").onclick = async () => {
    const name = document.getElementById("nt-name").value.trim();
    const description = document.getElementById("nt-desc").value.trim();
    const type = document.getElementById("nt-type").value;
    const kind = document.getElementById("nt-kind").value;
    if (!name) {
      toast("Name your task before creating it.", true);
      return;
    }
    try {
      const t = await api("POST", "/tasks", { name, description, type, kind });
      closeModal();
      await loadTasks();
      selectTask(t.id);
      toast(`Task "${t.name}" created.`);
    } catch (e) {
      toast(e.message, true);
    }
  };
}

function closeModal() {
  document.getElementById("modal-root").innerHTML = "";
}

// --- Task detail ---
async function renderTaskDetail() {
  clearInterval(state.pollTimer);
  state.wasTrainingActive = false;
  const main = document.getElementById("main");
  const task = state.tasks.find((t) => t.id === state.activeTaskId);
  if (!task) {
    main.innerHTML = `<div class="empty-state"><h2>Task not found</h2></div>`;
    return;
  }

  main.innerHTML = `
    <div class="task-header">
      <div>
        <h1>${escapeHtml(task.name)}</h1>
        <p class="task-desc">${escapeHtml(task.description || "No description provided.")}</p>
      </div>
      <div>
        <span class="badge">${task.type}</span>
        <button class="btn danger small" id="delete-task-btn" style="margin-left:10px;">Delete task</button>
      </div>
    </div>
    <div class="tabs">
      <div class="tab" data-tab="dataset">1. Dataset</div>
      <div class="tab" data-tab="training">2. Fine-Tuning</div>
      <div class="tab" data-tab="deploy">3. Deploy & Test</div>
      <div class="tab" data-tab="feedback">4. Feedback Loop</div>
      <div class="tab" data-tab="docs">5. API Docs</div>
    </div>
    <div id="panel-dataset" class="panel"></div>
    <div id="panel-training" class="panel"></div>
    <div id="panel-deploy" class="panel"></div>
    <div id="panel-feedback" class="panel"></div>
    <div id="panel-docs" class="panel"></div>
  `;

  document.getElementById("delete-task-btn").onclick = async () => {
    if (!confirm(`Delete task "${task.name}" and all its data?`)) return;
    await api("DELETE", `/tasks/${task.id}`);
    state.activeTaskId = null;
    await loadTasks();
    document.getElementById("main").innerHTML = `<div class="empty-state"><h2>Task deleted</h2></div>`;
  };

  main.querySelectorAll(".tab").forEach((tabEl) => {
    tabEl.onclick = () => {
      state.activeTab = tabEl.dataset.tab;
      applyActiveTab();
    };
  });

  await Promise.all([
    renderDatasetPanel(task),
    renderTrainingPanel(task),
    renderDeployPanel(task),
    renderFeedbackPanel(task),
    renderDocsPanel(task),
  ]);
  applyActiveTab();

  state.pollTimer = setInterval(async () => {
    if (state.activeTaskId !== task.id) return;
    // Only re-render the training panel when a training run is actively
    // progressing — otherwise the 1.5s refresh would reset the model
    // dropdown while the user is still choosing one.
    try {
      const jobs = await api("GET", `/tasks/${task.id}/training`);
      const hasActive = (jobs || []).some((j) => j.status === "queued" || j.status === "running");
      // When a run was active on the previous poll but is now terminal
      // (completed/failed), render once more so the UI shows the final state
      // (100% / metrics) instead of staying stuck on the last tick (e.g. 95%).
      const justFinished = state.wasTrainingActive && !hasActive;
      state.wasTrainingActive = hasActive;
      if (hasActive || justFinished) renderTrainingPanel(task);
    } catch (e) {
      // Transient network errors are safe to ignore during polling.
    }
  }, 1500);
}

function applyActiveTab() {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === state.activeTab));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === `panel-${state.activeTab}`));
}

// --- Dataset panel ---
async function renderDatasetPanel(task) {
  const panel = document.getElementById("panel-dataset");
  if (!panel) return;
  let stats, examples;
  try {
    [stats, examples] = await Promise.all([
      api("GET", `/tasks/${task.id}/dataset/stats`).catch(() => null),
      api("GET", `/tasks/${task.id}/examples`),
    ]);
  } catch (e) {
    panel.innerHTML = `<div class="card">Failed to load dataset: ${escapeHtml(e.message)}</div>`;
    return;
  }

  panel.innerHTML = `
    ${task.kind === "vision_lm" ? renderVisionAddCards() : task.kind === "asr" ? renderASRAddCards() : `
    <div class="card">
      <h3>Add example input/output pairs</h3>
      <p class="hint">One pair per line, input and output separated by " -&gt; ". Example:<br/>
      <span class="source-tag">My invoice charged me twice -> billing</span></p>
      <textarea id="ex-textarea" rows="6" placeholder="How do I reset my password? -> technical
I was charged twice this month -> billing
Can I change my email on file? -> account"></textarea>
      <div class="modal-actions" style="justify-content:flex-start; margin-top:12px;">
        <button class="btn primary" id="add-examples-btn">Add examples</button>
      </div>
    </div>

    ${task.kind === "token_classifier" ? `
    <div class="card">
      <h3>Import Named Entity Recognition (NER) Spans</h3>
      <p class="hint">Upload annotations in <strong>JSONL spans</strong>, <strong>CoNLL</strong> (token-per-line BIO), or <strong>CSV</strong> format.</p>
      <div class="field" style="margin-bottom:12px;">
        <label>JSONL Spans (<code>{"text": "...", "entities": [{"start":0,"end":4,"label":"ORG"}]}</code>)</label>
        <input type="file" id="ner-jsonl-file-input" accept=".jsonl,text/plain" />
      </div>
      <div class="field" style="margin-bottom:12px;">
        <label>CoNLL (Token + BIO tag per line)</label>
        <input type="file" id="conll-file-input" accept=".conll,.txt,text/plain" />
      </div>
      <div class="field">
        <label>CSV with span columns</label>
        <input type="file" id="ner-csv-file-input" accept=".csv,text/csv" />
      </div>
      <div class="hint" id="ner-file-status"></div>
    </div>
    ` : `
    <div class="card">
      <h3>Import from CSV</h3>
      <p class="hint">Upload a CSV with <span class="source-tag">input</span> and
      <span class="source-tag">output</span> columns (headers optional — first two
      columns are used if no header row matches).</p>
      <input type="file" id="csv-file-input" accept=".csv,text/csv" />
      <div class="hint" id="csv-file-status"></div>
    </div>

    <div class="card">
      <h3>Import from JSONL (Alpaca / chat)</h3>
      <p class="hint">Upload a fine-tuning JSONL dataset. Supports
      <span class="source-tag">Alpaca</span> style
      (<code>{"instruction","input","output"}</code>) and
      <span class="source-tag">chat</span> style
      (<code>{"messages":[{"role","content"},...]}</code>). Format is
      auto-detected from the first record.</p>
      <div class="inline-form" style="align-items:center;">
        <label>Format</label>
        <select id="jsonl-format" style="width:auto;background:var(--charcoal);color:var(--paper);border:1px solid var(--charcoal-3);border-radius:3px;padding:8px 10px;font-family:var(--font-mono);font-size:13px;">
          <option value="">Auto-detect</option>
          <option value="alpaca">Alpaca</option>
          <option value="chat">Chat</option>
        </select>
        <input type="file" id="jsonl-file-input" accept=".jsonl,.jsonl.gz,application/jsonl,application/x-ndjson" />
      </div>
      <div class="hint" id="jsonl-file-status"></div>
    </div>
    `}
    `}

    ${task.kind === "causal_lm" ? `
    <div class="card">
      <h3>Preference pairs (for DPO / ORPO tuning)</h3>
      <p class="hint">Add or import <span class="source-tag">{prompt, chosen, rejected}</span> triples -
      pairs where one answer is better than the other. These feed a preference-tuning run in the
      Training tab, separate from the ordinary input/output dataset above.</p>
      <div class="field" style="margin-bottom:10px;">
        <label>Prompt</label>
        <textarea id="pref-prompt" rows="2" placeholder="What's your return policy?"></textarea>
      </div>
      <div class="field" style="margin-bottom:10px;">
        <label>Chosen (better answer)</label>
        <textarea id="pref-chosen" rows="2" placeholder="You can return any item within 30 days for a full refund."></textarea>
      </div>
      <div class="field" style="margin-bottom:10px;">
        <label>Rejected (worse answer)</label>
        <textarea id="pref-rejected" rows="2" placeholder="idk, check the website"></textarea>
      </div>
      <div class="modal-actions" style="justify-content:flex-start; margin-top:4px; margin-bottom:14px;">
        <button class="btn primary" id="add-preference-btn">Add preference pair</button>
      </div>
      <div class="field">
        <label>Import JSONL (<code>{"prompt","chosen","rejected"}</code> per line)</label>
        <input type="file" id="preference-jsonl-file-input" accept=".jsonl,text/plain" />
      </div>
      <div class="hint" id="preference-file-status"></div>
      <div id="preference-stats" style="margin-top:14px;"></div>
    </div>
    ` : ""}

    ${task.type === "extraction" ? `
    <div class="card">
      <h3>Track B: JSON Schema for Extraction</h3>
      <p class="hint">Constrain LLM output to valid JSON matching this schema (generates GBNF grammar for llama.cpp). Validated on import.</p>
      <textarea id="task-schema-input" rows="4" placeholder='{"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}'>${escapeHtml(task.json_schema || "")}</textarea>
      <div class="modal-actions" style="justify-content:flex-start; margin-top:8px;">
        <button class="btn small" id="save-schema-btn">Save JSON Schema</button>
      </div>
    </div>
    ` : ""}

    ${task.kind === "vision_lm" || task.kind === "asr" ? "" : `
    <div class="card">
      <h3>Bootstrap synthetic examples</h3>
      <p class="hint">Generate additional training pairs from your existing examples to help
      the model generalize (simulates calling a frontier model to expand a handful of seeds).</p>
      <div class="inline-form">
        <div class="field">
          <label>How many to generate</label>
          <input type="number" id="synth-count" value="20" min="1" max="200" />
        </div>
        <button class="btn" id="synth-btn">Generate synthetic examples</button>
      </div>
    </div>
    `}

    <div class="card">
      <h3>Dataset health</h3>
      ${stats ? renderStats(stats) : '<p class="muted">No dataset yet — add some examples above.</p>'}
    </div>

    ${task.kind === "asr" ? `<div class="card"><h3>Audio dataset stats</h3><div id="asr-stats-body" class="hint">Loading...</div></div>` : ""}

    <div class="card">
      <h3>Examples (${examples.length})</h3>
      ${task.kind === "vision_lm" ? renderVisionGallery(examples, task.id) : task.kind === "asr" ? renderASRGallery(examples, task.id) : renderExamplesTable(examples)}
    </div>
  `;

  const addExamplesBtn = document.getElementById("add-examples-btn");
  if (addExamplesBtn) {
    addExamplesBtn.onclick = async () => {
      const raw = document.getElementById("ex-textarea").value;
      const pairs = raw
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean)
        .map((line) => {
          const idx = line.indexOf("->");
          if (idx === -1) return null;
          return { input: line.slice(0, idx).trim(), output: line.slice(idx + 2).trim() };
        })
        .filter(Boolean);
      if (pairs.length === 0) {
        toast('Use the format "input -> output", one pair per line.', true);
        return;
      }
      try {
        await api("POST", `/tasks/${task.id}/examples`, { pairs });
        toast(`Added ${pairs.length} example(s).`);
        renderDatasetPanel(task);
      } catch (e) {
        toast(e.message, true);
      }
    };
  }

  const synthBtn = document.getElementById("synth-btn");
  if (synthBtn) {
    synthBtn.onclick = async () => {
      const count = parseInt(document.getElementById("synth-count").value, 10) || 20;
      try {
        await api("POST", `/tasks/${task.id}/examples/synthetic`, { count });
        toast(`Generated ${count} synthetic example(s).`);
        renderDatasetPanel(task);
      } catch (e) {
        toast(e.message, true);
      }
    };
  }

  const csvInput = document.getElementById("csv-file-input");
  if (csvInput) {
    csvInput.onchange = async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const statusEl = document.getElementById("csv-file-status");
      statusEl.textContent = "Importing…";
      try {
        const text = await file.text();
        const res = await apiRawOrThrow("POST", `/tasks/${task.id}/examples/import`, text, { "Content-Type": "text/csv" });
        const stats = await res.json();
        toast(`Imported CSV — dataset now has ${stats.total} example(s).`);
        renderDatasetPanel(task);
      } catch (err) {
        statusEl.textContent = "";
        toast(err.message, true);
      }
    };
  }

  const jsonlInput = document.getElementById("jsonl-file-input");
  if (jsonlInput) {
    jsonlInput.onchange = async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const statusEl = document.getElementById("jsonl-file-status");
      statusEl.textContent = "Importing…";
      const format = (document.getElementById("jsonl-format") || {}).value;
      try {
        const text = await file.text();
        const url = `/tasks/${task.id}/examples/import-jsonl` + (format ? `?format=${encodeURIComponent(format)}` : "");
        const res = await apiRawOrThrow("POST", url, text, { "Content-Type": "application/x-ndjson" });
        const stats = await res.json();
        toast(`Imported JSONL — dataset now has ${stats.total} example(s).`);
        renderDatasetPanel(task);
      } catch (err) {
        statusEl.textContent = "";
        toast(err.message, true);
      }
    };
  }

  // --- NER importers ---
  const nerJsonlInput = document.getElementById("ner-jsonl-file-input");
  if (nerJsonlInput) {
    nerJsonlInput.onchange = async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const statusEl = document.getElementById("ner-file-status");
      statusEl.textContent = "Importing NER JSONL…";
      try {
        const text = await file.text();
        const res = await apiRawOrThrow("POST", `/tasks/${task.id}/examples/import-ner-jsonl`, text, { "Content-Type": "application/x-ndjson" });
        const stats = await res.json();
        toast(`Imported NER spans — dataset now has ${stats.total} example(s).`);
        renderDatasetPanel(task);
      } catch (err) {
        statusEl.textContent = "";
        toast(err.message, true);
      }
    };
  }

  const conllInput = document.getElementById("conll-file-input");
  if (conllInput) {
    conllInput.onchange = async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const statusEl = document.getElementById("ner-file-status");
      statusEl.textContent = "Importing CoNLL…";
      try {
        const text = await file.text();
        const res = await apiRawOrThrow("POST", `/tasks/${task.id}/examples/import-conll`, text, { "Content-Type": "text/plain" });
        const stats = await res.json();
        toast(`Imported CoNLL sentences — dataset now has ${stats.total} example(s).`);
        renderDatasetPanel(task);
      } catch (err) {
        statusEl.textContent = "";
        toast(err.message, true);
      }
    };
  }

  const nerCsvInput = document.getElementById("ner-csv-file-input");
  if (nerCsvInput) {
    nerCsvInput.onchange = async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const statusEl = document.getElementById("ner-file-status");
      statusEl.textContent = "Importing NER CSV…";
      try {
        const text = await file.text();
        const res = await apiRawOrThrow("POST", `/tasks/${task.id}/examples/import-ner-csv`, text, { "Content-Type": "text/csv" });
        const stats = await res.json();
        toast(`Imported NER CSV — dataset now has ${stats.total} example(s).`);
        renderDatasetPanel(task);
      } catch (err) {
        statusEl.textContent = "";
        toast(err.message, true);
      }
    };
  }

  // --- Preference pairs (DPO / ORPO) ---
  const addPreferenceBtn = document.getElementById("add-preference-btn");
  if (addPreferenceBtn) {
    addPreferenceBtn.onclick = async () => {
      const prompt = document.getElementById("pref-prompt").value.trim();
      const chosen = document.getElementById("pref-chosen").value.trim();
      const rejected = document.getElementById("pref-rejected").value.trim();
      if (!prompt || !chosen || !rejected) {
        toast("Prompt, chosen, and rejected are all required.", true);
        return;
      }
      try {
        await api("POST", `/tasks/${task.id}/preferences`, { prompt, chosen, rejected });
        toast("Preference pair added.");
        document.getElementById("pref-prompt").value = "";
        document.getElementById("pref-chosen").value = "";
        document.getElementById("pref-rejected").value = "";
        renderPreferenceStats(task);
      } catch (e) {
        toast(e.message, true);
      }
    };
  }

  const preferenceJsonlInput = document.getElementById("preference-jsonl-file-input");
  if (preferenceJsonlInput) {
    preferenceJsonlInput.onchange = async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const statusEl = document.getElementById("preference-file-status");
      statusEl.textContent = "Importing…";
      try {
        const text = await file.text();
        const res = await apiRawOrThrow("POST", `/tasks/${task.id}/preferences/import`, text, { "Content-Type": "application/x-ndjson" });
        const stats = await res.json();
        statusEl.textContent = "";
        toast(`Imported preference pairs - dataset now has ${stats.total} pair(s).`);
        renderPreferenceStats(task, stats);
      } catch (err) {
        statusEl.textContent = "";
        toast(err.message, true);
      }
    };
  }

  if (document.getElementById("preference-stats")) {
    renderPreferenceStats(task);
  }

  // --- Track B Schema save ---
  const saveSchemaBtn = document.getElementById("save-schema-btn");
  if (saveSchemaBtn) {
    saveSchemaBtn.onclick = async () => {
      const schemaText = document.getElementById("task-schema-input").value.trim();
      try {
        await api("PATCH", `/tasks/${task.id}`, { json_schema: schemaText });
        task.json_schema = schemaText;
        toast("JSON Schema saved.");
        renderDatasetPanel(task);
      } catch (err) {
        toast(err.message, true);
      }
    };
  }

  panel.querySelectorAll("[data-edit-id]").forEach((btn) => {
    btn.onclick = () => showEditExampleModal(task, btn.dataset.editId, btn.dataset.editInput, btn.dataset.editOutput);
  });
  panel.querySelectorAll("[data-delete-id]").forEach((btn) => {
    btn.onclick = async () => {
      if (!confirm("Delete this example?")) return;
      try {
        await api("DELETE", `/tasks/${task.id}/examples/${btn.dataset.deleteId}`);
        toast("Example deleted.");
        renderDatasetPanel(task);
      } catch (e) {
        toast(e.message, true);
      }
    };
  });

  // --- Vision-language (document AI) dataset controls ---
  const addVisionBtn = document.getElementById("add-vision-btn");
  if (addVisionBtn) {
    addVisionBtn.onclick = async () => {
      const fileInput = document.getElementById("vision-image-input");
      const file = fileInput.files[0];
      const prompt = document.getElementById("vision-prompt").value.trim();
      const answer = document.getElementById("vision-answer").value.trim();
      if (!file) {
        toast("Choose an image to add.", true);
        return;
      }
      if (!prompt) {
        toast("A prompt is required.", true);
        return;
      }
      try {
        const imageBase64 = await fileToDataURL(file);
        await api("POST", `/tasks/${task.id}/examples/vision`, { image_base64: imageBase64, prompt, answer });
        toast("Example added.");
        fileInput.value = "";
        document.getElementById("vision-prompt").value = "";
        document.getElementById("vision-answer").value = "";
        renderDatasetPanel(task);
      } catch (e) {
        toast(e.message, true);
      }
    };
  }

  const visionZipInput = document.getElementById("vision-zip-file-input");
  if (visionZipInput) {
    visionZipInput.onchange = async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const statusEl = document.getElementById("vision-zip-status");
      statusEl.textContent = "Importing ZIP — this can take a moment for large archives…";
      try {
        const buf = await file.arrayBuffer();
        const res = await apiRawOrThrow("POST", `/tasks/${task.id}/examples/import-vision-zip`, buf, { "Content-Type": "application/zip" });
        const stats = await res.json();
        statusEl.textContent = "";
        toast(`Imported ZIP — dataset now has ${stats.total} example(s).`);
        renderDatasetPanel(task);
      } catch (err) {
        statusEl.textContent = "";
        toast(err.message, true);
      }
    };
  }

  const visionPdfInput = document.getElementById("vision-pdf-file-input");
  if (visionPdfInput) {
    visionPdfInput.onchange = async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const prompt = document.getElementById("vision-pdf-prompt").value.trim();
      const dpi = parseInt(document.getElementById("vision-pdf-dpi").value, 10) || 150;
      const statusEl = document.getElementById("vision-pdf-status");
      statusEl.textContent = "Rasterizing PDF pages…";
      try {
        const buf = await file.arrayBuffer();
        const url = `/tasks/${task.id}/examples/import-vision-pdf?dpi=${dpi}` + (prompt ? `&prompt=${encodeURIComponent(prompt)}` : "");
        const res = await apiRawOrThrow("POST", url, buf, { "Content-Type": "application/pdf" });
        const stats = await res.json();
        statusEl.textContent = "";
        toast(`Imported PDF pages — dataset now has ${stats.total} example(s).`);
        renderDatasetPanel(task);
      } catch (err) {
        statusEl.textContent = "";
        toast(err.message, true);
      }
    };
  }

  // --- ASR (speech-to-text) dataset controls ---
  const addASRBtn = document.getElementById("add-asr-btn");
  if (addASRBtn) {
    addASRBtn.onclick = async () => {
      const fileInput = document.getElementById("asr-audio-input");
      const file = fileInput.files[0];
      const text = document.getElementById("asr-text").value.trim();
      const speaker = document.getElementById("asr-speaker").value.trim();
      if (!file) {
        toast("Choose an audio file to add.", true);
        return;
      }
      if (!text) {
        toast("A transcript is required.", true);
        return;
      }
      try {
        const audioBase64 = await fileToDataURL(file);
        await api("POST", `/tasks/${task.id}/examples/asr`, { audio_base64: audioBase64, filename: file.name, text, speaker });
        toast("Example added.");
        fileInput.value = "";
        document.getElementById("asr-text").value = "";
        document.getElementById("asr-speaker").value = "";
        renderDatasetPanel(task);
      } catch (e) {
        toast(e.message, true);
      }
    };
  }

  const asrZipInput = document.getElementById("asr-zip-file-input");
  if (asrZipInput) {
    asrZipInput.onchange = async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const statusEl = document.getElementById("asr-zip-status");
      statusEl.textContent = "Importing ZIP (this can take a moment for large archives)";
      try {
        const buf = await file.arrayBuffer();
        const res = await apiRawOrThrow("POST", `/tasks/${task.id}/examples/import-asr-zip`, buf, { "Content-Type": "application/zip" });
        const stats = await res.json();
        statusEl.textContent = "";
        toast(`Imported ZIP (dataset now has ${stats.clips ?? stats.total} clip(s)).`);
        renderDatasetPanel(task);
      } catch (err) {
        statusEl.textContent = "";
        toast(err.message, true);
      }
    };
  }

  if (task.kind === "asr") {
    api("GET", `/tasks/${task.id}/examples/asr-stats`)
      .then((s) => {
        const body = document.getElementById("asr-stats-body");
        if (!body) return;
        const hours = (s.total_seconds / 3600).toFixed(2);
        const avg = s.average_seconds ? s.average_seconds.toFixed(1) + "s avg" : "durations unknown";
        body.innerHTML = `
          <div class="stat-grid">
            <div class="stat"><div class="value">${s.clips}</div><div class="label">Clips</div></div>
            <div class="stat"><div class="value">${hours}</div><div class="label">Hours (known)</div></div>
            <div class="stat"><div class="value">${avg}</div><div class="label">Avg length</div></div>
            <div class="stat"><div class="value">${s.empty_transcripts}</div><div class="label">Empty transcripts</div></div>
          </div>
          <div class="hint">${s.known_duration_clips}/${s.clips} clips carry a duration hint; the trainer recomputes true durations during its feature pass.</div>`;
      })
      .catch(() => {
        const body = document.getElementById("asr-stats-body");
        if (body) body.textContent = "Audio stats unavailable.";
      });
  }
}

// fileToDataURL reads a File/Blob into a base64 data URL
// ("data:image/png;base64,...") for the vision example-add endpoint, which
// accepts a data-URL-prefixed or bare base64 string.
function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Failed to read the image file."));
    reader.readAsDataURL(file);
  });
}

// renderVisionAddCards renders the vision_lm dataset tab's add/import cards:
// a single image+prompt+answer form, a ZIP-of-images bulk import, and a
// PDF-to-page-images import — replacing the generic text pair/CSV/JSONL
// cards, which don't apply to an image dataset.
function renderVisionAddCards() {
  return `
    <div class="card">
      <h3>Add a document/image example</h3>
      <p class="hint">Upload one page image, the instruction/question to ask about it, and
      (optionally) the expected answer — free text, or JSON text when your task uses a JSON
      Schema (see the schema card below, if this task's type is Extraction).</p>
      <div class="field" style="margin-bottom:10px;">
        <label>Image (PNG / JPEG)</label>
        <input type="file" id="vision-image-input" accept=".png,.jpg,.jpeg,.gif,image/png,image/jpeg,image/gif" />
      </div>
      <div class="field" style="margin-bottom:10px;">
        <label>Prompt</label>
        <textarea id="vision-prompt" rows="2" placeholder="Extract vendor, total and date as JSON."></textarea>
      </div>
      <div class="field" style="margin-bottom:10px;">
        <label>Answer (leave blank to add unanswered, for later human correction)</label>
        <textarea id="vision-answer" rows="2" placeholder='{"vendor":"Acme","total":4200.00,"date":"2026-05-03"}'></textarea>
      </div>
      <div class="modal-actions" style="justify-content:flex-start; margin-top:4px;">
        <button class="btn primary" id="add-vision-btn">Add example</button>
      </div>
    </div>

    <div class="card">
      <h3>Import a dataset ZIP</h3>
      <p class="hint">Upload a ZIP archive containing page images plus a <span class="source-tag">data.jsonl</span>
      manifest (one line per example: <code>{"image":"images/inv_0042.png","prompt":"...","answer":"..."}</code>,
      image paths relative to the archive root).</p>
      <input type="file" id="vision-zip-file-input" accept=".zip,application/zip" />
      <div class="hint" id="vision-zip-status"></div>
    </div>

    <div class="card">
      <h3>Import a PDF (one example per page)</h3>
      <p class="hint">Upload a PDF — each page is rasterized to an image and added as a separate
      example sharing the same prompt, so you can then correct answers page-by-page below.</p>
      <div class="inline-form" style="align-items:center;">
        <div class="field" style="flex:1;">
          <label>Prompt (applied to every page)</label>
          <input type="text" id="vision-pdf-prompt" placeholder="Extract vendor, total and date as JSON." />
        </div>
        <div class="field" style="width:110px;">
          <label>DPI</label>
          <input type="number" id="vision-pdf-dpi" value="150" min="50" max="600" />
        </div>
      </div>
      <label>PDF file</label>
      <input type="file" id="vision-pdf-file-input" accept=".pdf,application/pdf" />
      <div class="hint" id="vision-pdf-status"></div>
    </div>
  `;
}

function showEditExampleModal(task, id, input, output) {
  const root = document.getElementById("modal-root");
  root.innerHTML = `
    <div class="modal-backdrop" id="modal-backdrop">
      <div class="modal">
        <h3>Edit example</h3>
        <label>Input</label>
        <textarea id="ee-input" rows="2">${escapeHtml(input)}</textarea>
        <label>Output</label>
        <textarea id="ee-output" rows="2">${escapeHtml(output)}</textarea>
        <div class="modal-actions">
          <button class="btn ghost" id="ee-cancel">Cancel</button>
          <button class="btn primary" id="ee-save">Save changes</button>
        </div>
      </div>
    </div>`;
  document.getElementById("ee-cancel").onclick = closeModal;
  document.getElementById("modal-backdrop").onclick = (e) => {
    if (e.target.id === "modal-backdrop") closeModal();
  };
  document.getElementById("ee-save").onclick = async () => {
    const newInput = document.getElementById("ee-input").value.trim();
    const newOutput = document.getElementById("ee-output").value.trim();
    if (!newInput || !newOutput) {
      toast("Input and output can't be empty.", true);
      return;
    }
    try {
      await api("PUT", `/tasks/${task.id}/examples/${id}`, { input: newInput, output: newOutput });
      closeModal();
      toast("Example updated.");
      renderDatasetPanel(task);
    } catch (e) {
      toast(e.message, true);
    }
  };
}

function renderStats(stats) {
  const balance = stats.label_balance
    ? Object.entries(stats.label_balance)
        .map(([k, v]) => `<span class="source-tag">${escapeHtml(k)}: ${v}</span>`)
        .join("  ")
    : "";
  return `
    <div class="stat-grid">
      <div class="stat"><div class="value">${stats.total}</div><div class="label">Total</div></div>
      <div class="stat"><div class="value">${stats.usable_count}</div><div class="label">Usable</div></div>
      <div class="stat"><div class="value">${stats.duplicates}</div><div class="label">Duplicates</div></div>
      <div class="stat"><div class="value">${stats.flagged}</div><div class="label">Flagged</div></div>
      <div class="stat"><div class="value">${stats.synthetic}</div><div class="label">Synthetic</div></div>
      <div class="stat"><div class="value">${stats.feedback}</div><div class="label">From feedback</div></div>
    </div>
    ${balance ? `<div class="hint">Label balance: ${balance}</div>` : ""}
    <div class="readiness ${stats.ready_to_train ? "ready" : "not-ready"}">
      ${stats.ready_to_train ? "✓ Ready to train" : "✗ Not ready to train"}
      ${stats.readiness_reason ? " — " + escapeHtml(stats.readiness_reason) : ""}
    </div>
  `;
}

// renderPreferenceStats fetches (or reuses already-fetched) preference-pair
// stats and renders the length-bias / readiness diagnostic into the
// dataset panel's #preference-stats slot. Kept separate from the main
// dataset stats fetch since preference pairs are a distinct sub-dataset.
async function renderPreferenceStats(task, stats) {
  const el = document.getElementById("preference-stats");
  if (!el) return;
  if (!stats) {
    try {
      stats = await api("GET", `/tasks/${task.id}/preferences/stats`);
    } catch (e) {
      el.innerHTML = "";
      return;
    }
  }
  if (!stats || stats.total === 0) {
    el.innerHTML = '<p class="muted">No preference pairs yet.</p>';
    return;
  }
  el.innerHTML = `
    <div class="stat-grid">
      <div class="stat"><div class="value">${stats.total}</div><div class="label">Total</div></div>
      <div class="stat"><div class="value">${stats.usable_count}</div><div class="label">Usable</div></div>
      <div class="stat"><div class="value">${stats.duplicates}</div><div class="label">Duplicates</div></div>
      <div class="stat"><div class="value">${stats.flagged}</div><div class="label">Flagged</div></div>
    </div>
    <div class="readiness ${stats.ready_to_train ? "ready" : "not-ready"}">
      ${stats.ready_to_train ? "✓ Ready to train" : "✗ Not ready to train"}
      ${stats.readiness_reason ? " — " + escapeHtml(stats.readiness_reason) : ""}
    </div>
    ${stats.length_bias_warning ? `<p class="hint" style="color:var(--err);">⚠ The chosen answer is longer than the rejected one in most pairs — the model may learn "longer is better" rather than the intended preference.</p>` : ""}
  `;
}

function renderExamplesTable(examples) {
  if (examples.length === 0) return '<p class="muted">No examples yet.</p>';
  const rows = examples
    .slice()
    .reverse()
    .slice(0, 100)
    .map((e) => {
      const cls = [e.flagged ? "flagged" : "", e.duplicate ? "duplicate" : ""].join(" ").trim();
      return `<tr class="${cls}">
        <td>${escapeHtml(truncate(e.input, 60))}</td>
        <td>${escapeHtml(truncate(e.output, 60))}</td>
        <td><span class="source-tag">${e.source}</span></td>
        <td>${e.flagged ? "flagged: " + escapeHtml(e.flag_note || "") : e.duplicate ? "duplicate" : "ok"}</td>
        <td style="white-space:nowrap;">
          <button class="btn small ghost" data-edit-id="${e.id}" data-edit-input="${escapeHtml(e.input)}" data-edit-output="${escapeHtml(e.output)}">Edit</button>
          <button class="btn small danger" data-delete-id="${e.id}">Delete</button>
        </td>
      </tr>`;
    })
    .join("");
  return `<table><thead><tr><th>Input</th><th>Output</th><th>Source</th><th>Status</th><th></th></tr></thead><tbody>${rows}</tbody></table>
  ${examples.length > 100 ? `<p class="hint">Showing latest 100 of ${examples.length}.</p>` : ""}`;
}

// renderVisionGallery renders a vision_lm task's examples as an image
// gallery (thumbnail + prompt + answer) instead of the generic input/output
// table, since a vision_lm example's data lives in its Payload
// ({image, prompt, answer}) rather than the legacy Input/Output fields.
// Editing isn't offered here (the generic PUT /examples/{id} endpoint only
// updates Input/Output, which vision examples don't use) — only delete.
function renderVisionGallery(examples, taskId) {
  if (examples.length === 0) return '<p class="muted">No examples yet — add one above.</p>';
  const cards = examples
    .slice()
    .reverse()
    .slice(0, 60)
    .map((e) => {
      const payload = e.payload || {};
      const cls = [e.flagged ? "flagged" : "", e.duplicate ? "duplicate" : ""].join(" ").trim();
      const imgSrc = payload.image ? `${API}/tasks/${taskId}/blobs/${encodeURIComponent(payload.image)}` : "";
      const statusLabel = e.flagged ? "flagged: " + escapeHtml(e.flag_note || "") : e.duplicate ? "duplicate" : !payload.answer ? "needs answer" : "ok";
      return `
      <div class="job-row ${cls}" style="align-items:flex-start;">
        ${imgSrc ? `<img src="${imgSrc}" alt="page image" style="width:96px;height:96px;object-fit:cover;border-radius:4px;border:1px solid var(--charcoal-3);flex-shrink:0;" />` : ""}
        <div style="flex:1;min-width:0;">
          <div class="hint">Prompt</div>
          <div style="margin-bottom:6px;">${escapeHtml(truncate(payload.prompt || "", 160))}</div>
          <div class="hint">Answer</div>
          <div style="margin-bottom:6px;">${payload.answer ? escapeHtml(truncate(payload.answer, 160)) : '<span class="muted">(none yet)</span>'}</div>
          <span class="source-tag">${escapeHtml(e.source)}</span>
          <span class="source-tag">${statusLabel}</span>
        </div>
        <button class="btn small danger" data-delete-id="${e.id}">Delete</button>
      </div>`;
    })
    .join("");
  return `<div style="display:flex;flex-direction:column;gap:10px;">${cards}</div>
  ${examples.length > 60 ? `<p class="hint" style="margin-top:10px;">Showing latest 60 of ${examples.length}.</p>` : ""}`;
}

// renderASRAddCards renders the asr dataset tab's add/import cards: a
// single audio+transcript form and a ZIP-of-audio bulk import, replacing
// the generic text pair/CSV/JSONL cards, which don't apply to an audio
// dataset. The consent note addresses plan 07's privacy risk (voice data
// is personal/biometric in many jurisdictions).
function renderASRAddCards() {
  return `
    <div class="card">
      <h3>Add an audio example</h3>
      <p class="hint">Upload one audio clip (wav / mp3 / flac / m4a) with its transcript and,
      optionally, a speaker tag (the trainer splits train/eval BY SPEAKER to avoid leakage).</p>
      <div class="field" style="margin-bottom:10px;">
        <label>Audio file</label>
        <input type="file" id="asr-audio-input" accept=".wav,.mp3,.flac,.m4a,.ogg,audio/*" />
      </div>
      <div class="field" style="margin-bottom:10px;">
        <label>Transcript</label>
        <textarea id="asr-text" rows="2" placeholder="Please reset my Acme router to factory settings."></textarea>
      </div>
      <div class="field" style="margin-bottom:10px;">
        <label>Speaker tag (optional)</label>
        <input type="text" id="asr-speaker" placeholder="speaker_01" />
      </div>
      <div class="modal-actions" style="justify-content:flex-start; margin-top:4px;">
        <button class="btn primary" id="add-asr-btn">Add example</button>
      </div>
      <p class="hint">By adding voice recordings you confirm the speakers consented to their
      audio being used for model training. Audio stays local to this deployment.</p>
    </div>

    <div class="card">
      <h3>Import a dataset ZIP</h3>
      <p class="hint">Upload a ZIP archive containing audio files plus a <span class="source-tag">manifest.jsonl</span>
      (one line per clip: <code>{"audio":"audio/call_0193.wav","text":"...","speaker":"...","duration":4.2}</code>,
      audio paths relative to the archive root).</p>
      <input type="file" id="asr-zip-file-input" accept=".zip,application/zip" />
      <div class="hint" id="asr-zip-status"></div>
    </div>
  `;
}

// renderASRGallery renders an asr task's examples as an audio list (inline
// player + transcript) instead of the generic input/output table, since an
// asr example's data lives in its Payload ({audio, text, speaker, duration})
// rather than the legacy Input/Output fields. Editing isn't offered here
// (the generic PUT /examples/{id} endpoint only updates Input/Output) —
// only delete.
function renderASRGallery(examples, taskId) {
  if (examples.length === 0) return '<p class="muted">No examples yet — add one above.</p>';
  const cards = examples
    .slice()
    .reverse()
    .slice(0, 60)
    .map((e) => {
      const payload = e.payload || {};
      const cls = [e.flagged ? "flagged" : "", e.duplicate ? "duplicate" : ""].join(" ").trim();
      const audioSrc = payload.audio ? `${API}/tasks/${taskId}/blobs/${encodeURIComponent(payload.audio)}` : "";
      const statusLabel = e.flagged ? "flagged: " + escapeHtml(e.flag_note || "") : e.duplicate ? "duplicate" : !payload.text ? "needs transcript" : "ok";
      return `
      <div class="job-row ${cls}" style="align-items:flex-start;">
        <div style="flex:1;min-width:0;">
          ${audioSrc ? `<audio controls preload="none" src="${audioSrc}" style="width:100%;max-width:420px;height:32px;margin-bottom:6px;"></audio>` : ""}
          <div class="hint">Transcript</div>
          <div style="margin-bottom:6px;">${payload.text ? escapeHtml(truncate(payload.text, 220)) : '<span class="muted">(none yet)</span>'}</div>
          <span class="source-tag">${escapeHtml(e.source)}</span>
          ${payload.speaker ? `<span class="source-tag">${escapeHtml(payload.speaker)}</span>` : ""}
          ${payload.duration ? `<span class="source-tag">${Number(payload.duration).toFixed(1)}s</span>` : ""}
          <span class="source-tag">${statusLabel}</span>
        </div>
        <button class="btn small danger" data-delete-id="${e.id}">Delete</button>
      </div>`;
    })
    .join("");
  return `<div style="display:flex;flex-direction:column;gap:10px;">${cards}</div>
  ${examples.length > 60 ? `<p class="hint" style="margin-top:10px;">Showing latest 60 of ${examples.length}.</p>` : ""}`;
}

// --- Training panel ---
async function renderTrainingPanel(task) {
  const panel = document.getElementById("panel-training");
  if (!panel) return;
  let jobs, stats, models;
  try {
    [jobs, stats, models] = await Promise.all([
      api("GET", `/tasks/${task.id}/training`),
      api("GET", `/tasks/${task.id}/dataset/stats`).catch(() => null),
      api("GET", "/models").catch(() => []),
    ]);
  } catch (e) {
    panel.innerHTML = `<div class="card">Failed to load training jobs: ${escapeHtml(e.message)}</div>`;
    return;
  }
  jobs = (jobs || []).slice().sort((a, b) => b.version - a.version);
  const hasActive = jobs.some((j) => j.status === "queued" || j.status === "running");
  const ready = stats && stats.ready_to_train;
  const recommendedModel = stats && stats.recommended_model ? stats.recommended_model : null;

  const modelOptions = (models || [])
    .map(
      (m) =>
        `<option value="${escapeHtml(m.name)}">${escapeHtml(m.name)} · ${m.params_billions}B params · ${m.family}</option>`
    )
    .join("");

  panel.innerHTML = `
    <div class="card">
      <h3>Start fine-tuning</h3>
      <p class="hint">Choose the base model to fine-tune. Distillery pre-selects the recommended open-weights model
      (Llama / Mistral / Qwen family) for your task, but you can override it and run a LoRA/QLoRA fine-tune on any model below.</p>
      <div class="inline-form" style="align-items:flex-end; margin-bottom:12px;">
        <div class="field" style="flex:1;">
          <label>Base model</label>
          <select id="base-model-select" ${hasActive || !ready ? "disabled" : ""} style="width:100%;background:var(--charcoal);color:var(--paper);border:1px solid var(--charcoal-3);border-radius:3px;padding:9px 11px;font-family:var(--font-mono);font-size:13px;">
            ${modelOptions || '<option value="">No models available</option>'}
          </select>
          ${recommendedModel ? `<div class="hint">Recommended for your dataset: <span class="source-tag">${escapeHtml(recommendedModel.name)}</span></div>` : ""}
        </div>
      </div>
      <button class="btn primary" id="start-training-btn" ${hasActive || !ready ? "disabled" : ""}>
        ${hasActive ? "Training in progress…" : "Start fine-tuning run"}
      </button>
      ${!ready && !hasActive ? '<p class="hint">Your dataset isn\'t ready yet — check the Dataset tab.</p>' : ""}
    </div>
    ${task.kind === "causal_lm" ? renderPreferenceTrainingCard(jobs, hasActive) : ""}

    <div class="card">
      <h3>Training runs</h3>
      ${jobs.length === 0 ? '<p class="muted">No training runs yet.</p>' : jobs.map((j) => renderJobRow(j, jobs)).join("")}
    </div>
  `;

  // Pre-select the user's explicit choice (if they've made one) or the
  // auto-recommended model so they always have a sensible default but keep
  // control. The selected value is persisted in state so the 1.5s progress
  // polls don't reset it.
  const sel = document.getElementById("base-model-select");
  let selected = state.selectedBaseModel[task.id];
  if (!selected && recommendedModel) {
    selected = recommendedModel.name;
  }
  if (sel && sel.options.length > 0) {
    const matched = Array.from(sel.options).find((o) => o.value === (selected || ""));
    if (matched) {
      sel.value = matched.value;
    } else if (selected) {
      // The stored selection isn't in the catalog anymore — fall back to first option.
      sel.value = sel.options[0].value;
    }
  }

  // Persist the user's choice on change (not during polling re-renders).
  if (sel) {
    sel.onchange = () => {
      state.selectedBaseModel[task.id] = sel.value;
    };
  }

  const startBtn = document.getElementById("start-training-btn");
  if (startBtn) {
    startBtn.onclick = async () => {
      const baseModel = (document.getElementById("base-model-select") || {}).value || "";
      try {
        await api("POST", `/tasks/${task.id}/training`, { base_model: baseModel });
        state.selectedBaseModel[task.id] = baseModel || "";
        toast(`Fine-tuning started on ${baseModel || "recommended model"}.`);
        renderTrainingPanel(task);
      } catch (e) {
        toast(e.message, true);
      }
    };
  }

  const startPrefBtn = document.getElementById("start-preference-btn");
  if (startPrefBtn) {
    startPrefBtn.onclick = async () => {
      const method = (document.getElementById("pref-method-select") || {}).value || "dpo";
      const parentJobId = (document.getElementById("pref-parent-select") || {}).value || "";
      if (method === "dpo" && !parentJobId) {
        toast("DPO needs a completed fine-tuning run to use as the parent (or switch to ORPO).", true);
        return;
      }
      try {
        await api("POST", `/tasks/${task.id}/training`, { method, parent_job_id: parentJobId });
        toast(`${method.toUpperCase()} preference-tuning run started.`);
        renderTrainingPanel(task);
      } catch (e) {
        toast(e.message, true);
      }
    };
  }

  panel.querySelectorAll("[data-deploy-job]").forEach((btn) => {
    btn.onclick = async () => {
      try {
        const res = await api("POST", `/tasks/${task.id}/training/${btn.dataset.deployJob}/deploy`, { autoscale: true });
        state.apiKeys[res.id] = res.api_key;
        toast(`Deployed v${btn.dataset.deployVersion}. Switch to the Deploy & Test tab to see the API key.`);
        renderDeployPanel(task);
        renderDocsPanel(task);
      } catch (e) {
        // The regression gate blocks deploying a preference-tuned version
        // that regressed vs its parent's SFT eval metric; offer to force it.
        if (/regressed/i.test(e.message) && confirm(`${e.message}\n\nDeploy anyway?`)) {
          try {
            const res2 = await api("POST", `/tasks/${task.id}/training/${btn.dataset.deployJob}/deploy`, { autoscale: true, force: true });
            state.apiKeys[res2.id] = res2.api_key;
            toast(`Deployed v${btn.dataset.deployVersion} (forced past the regression gate).`);
            renderDeployPanel(task);
            renderDocsPanel(task);
          } catch (e2) {
            toast(e2.message, true);
          }
          return;
        }
        toast(e.message, true);
      }
    };
  });

  panel.querySelectorAll("[data-delete-job]").forEach((btn) => {
    btn.onclick = async () => {
      if (!confirm(`Delete fine-tuned model v${btn.dataset.deleteVersion}? This permanently removes this training run and any deployment serving it.`)) return;
      try {
        await api("DELETE", `/training/${btn.dataset.deleteJob}`);
        toast(`Fine-tuned model v${btn.dataset.deleteVersion} deleted.`);
        renderTrainingPanel(task);
        renderDeployPanel(task);
        renderDocsPanel(task);
      } catch (e) {
        toast(e.message, true);
      }
    };
  });
}

// renderPreferenceTrainingCard renders the DPO/ORPO start-run card shown
// on causal_lm tasks. DPO requires a completed parent fine-tuning run
// (its adapter is both the starting policy and, with the adapter disabled,
// the reference model); ORPO can run directly from the base model.
function renderPreferenceTrainingCard(jobs, hasActive) {
  const parentCandidates = jobs.filter((j) => j.status === "completed" && j.kind === "causal_lm");
  const hasParent = parentCandidates.length > 0;
  const parentOptions = parentCandidates
    .map((j) => `<option value="${j.id}">v${j.version} — ${escapeHtml(j.base_model.name)}</option>`)
    .join("");
  return `
    <div class="card">
      <h3>Preference tuning (DPO / ORPO)</h3>
      <p class="hint">Turn feedback corrections or preference pairs into a preference-tuned version.
      DPO continues from a completed fine-tuning run below. ORPO can run straight from the base
      model if you don't have a parent run yet.</p>
      <div class="inline-form" style="align-items:flex-end; margin-bottom:12px;">
        <div class="field">
          <label>Method</label>
          <select id="pref-method-select" ${hasActive ? "disabled" : ""} style="width:auto;background:var(--charcoal);color:var(--paper);border:1px solid var(--charcoal-3);border-radius:3px;padding:9px 11px;font-family:var(--font-mono);font-size:13px;">
            <option value="dpo" ${!hasParent ? "disabled" : ""}>DPO</option>
            <option value="orpo" ${!hasParent ? "selected" : ""}>ORPO</option>
          </select>
        </div>
        <div class="field" style="flex:1;">
          <label>Parent run (required for DPO)</label>
          <select id="pref-parent-select" ${hasActive || !hasParent ? "disabled" : ""} style="width:100%;background:var(--charcoal);color:var(--paper);border:1px solid var(--charcoal-3);border-radius:3px;padding:9px 11px;font-family:var(--font-mono);font-size:13px;">
            ${parentOptions || '<option value="">No completed fine-tuning runs yet</option>'}
          </select>
        </div>
      </div>
      <button class="btn primary" id="start-preference-btn" ${hasActive ? "disabled" : ""}>
        ${hasActive ? "Training in progress…" : "Start preference-tuning run"}
      </button>
      ${!hasParent ? '<p class="hint">No completed fine-tuning run yet — you can still run ORPO from the base model, or finish an SFT run first to unlock DPO.</p>' : ""}
    </div>
  `;
}

// renderJobRow renders one training-run row. allJobs (the full sibling list)
// lets a preference_lm row resolve and display its parent's version number.
function renderJobRow(j, allJobs) {
  let metricsHint = "";
  if (j.metrics) {
    if (j.kind === "preference_lm" && j.metrics.reward_accuracy !== undefined) {
      metricsHint = `reward acc ${(j.metrics.reward_accuracy * 100).toFixed(0)}% · margin ${j.metrics.reward_margin.toFixed(3)} · ${j.metrics.train_examples} pairs`;
      if (j.metrics.regression_checked) {
        metricsHint += j.metrics.regression_passed
          ? ` · <span class="badge" style="background:var(--ok);color:#fff;">regression check passed</span>`
          : ` · <span class="badge" style="background:var(--err);color:#fff;">regression check FAILED</span>`;
      }
    } else if (j.kind === "token_classifier" && j.metrics.entity_f1 !== undefined) {
      metricsHint = `Entity F1: <strong>${(j.metrics.entity_f1 * 100).toFixed(1)}%</strong> (P: ${(j.metrics.entity_precision * 100).toFixed(1)}% · R: ${(j.metrics.entity_recall * 100).toFixed(1)}%) · ${j.metrics.train_examples} examples`;
    } else if (j.kind === "asr" && j.metrics.wer !== undefined) {
      metricsHint = `WER: <strong>${(j.metrics.wer * 100).toFixed(1)}%</strong>`;
      if (j.metrics.base_wer !== undefined && j.metrics.base_wer !== null) {
        const d = j.metrics.delta_wer || 0;
        metricsHint += ` (${d >= 0 ? "-" : "+"}${Math.abs(d * 100).toFixed(1)}pt vs. base ${(j.metrics.base_wer * 100).toFixed(1)}%)`;
      }
      metricsHint += ` &middot; CER ${((j.metrics.cer || 0) * 100).toFixed(1)}%`;
      if (j.metrics.insertions !== undefined) {
        metricsHint += ` &middot; insertions ${(j.metrics.insertions * 100).toFixed(1)}%`;
      }
      if (j.metrics.domain_term_recall !== undefined && j.metrics.domain_term_recall !== null) {
        metricsHint += ` &middot; glossary recall ${(j.metrics.domain_term_recall * 100).toFixed(0)}%`;
      }
      metricsHint += ` &middot; ${j.metrics.train_examples} held-out clips`;
    } else if (j.kind === "vision_lm" && j.metrics.macro_field_f1 !== undefined) {
      metricsHint = `Field F1: <strong>${(j.metrics.macro_field_f1 * 100).toFixed(1)}%</strong>`;
      if (j.metrics.baseline_field_f1 !== undefined && j.metrics.baseline_field_f1 !== null) {
        const delta = j.metrics.delta_field_f1 || 0;
        metricsHint += ` (${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)}pt vs. zero-shot baseline)`;
      }
      metricsHint += ` · JSON valid: ${(j.metrics.json_valid_rate * 100).toFixed(0)}% · ANLS: ${j.metrics.anls.toFixed(2)} · doc acc: ${(j.metrics.document_accuracy * 100).toFixed(0)}% · ${j.metrics.train_examples} examples`;
    } else {
      metricsHint = `loss ${j.metrics.final_loss.toFixed(2)} · acc ${(j.metrics.eval_accuracy * 100).toFixed(0)}% · ${j.metrics.train_examples} examples`;
      if (j.metrics.json_validity_rate !== undefined && j.metrics.json_validity_rate !== null) {
        metricsHint += ` · <span class="badge" style="background:var(--ok);color:#fff;">JSON valid: ${(j.metrics.json_validity_rate * 100).toFixed(0)}%</span>`;
      }
    }
  }
  const metrics = metricsHint
    ? `<span class="hint">${metricsHint}</span>`
    : j.error
    ? `<span class="hint" style="color:var(--err)">${escapeHtml(j.error)}</span>`
    : "";
  let lineageHint = "";
  if (j.kind === "preference_lm") {
    const methodLabel = ((j.preference && j.preference.method) || "").toUpperCase();
    if (j.parent_job_id) {
      const parent = (allJobs || []).find((p) => p.id === j.parent_job_id);
      lineageHint = `${methodLabel} from v${parent ? parent.version : "?"}`;
    } else {
      lineageHint = `${methodLabel} · no parent (trained from base model)`;
    }
  }
  const isRunning = j.status === "running" || j.status === "queued";
  const actions = [];
  if (j.status === "completed") {
    actions.push(`<button class="btn small" data-deploy-job="${j.id}" data-deploy-version="${j.version}">Deploy this version</button>`);
  }
  if (!isRunning) {
    actions.push(`<button class="btn small danger" data-delete-job="${j.id}" data-delete-version="${j.version}">Delete</button>`);
  }
  const fieldTable =
    !isRunning && j.kind === "vision_lm" && j.metrics && j.metrics.field_metrics && Object.keys(j.metrics.field_metrics).length > 0
      ? renderFieldMetricsTable(j.metrics.field_metrics)
      : "";
  return `
    <div class="job-row">
      <div style="flex:1">
        <div><strong>v${j.version}</strong> — ${escapeHtml(j.base_model.name)} <span class="status-pill ${j.status}">${j.status}</span></div>
        ${lineageHint ? `<div class="hint">${escapeHtml(lineageHint)}</div>` : ""}
        ${isRunning ? `
          <div class="gauge-wrap">
            <div class="gauge"><div class="gauge-fill" style="width:${j.progress}%"></div></div>
            <div class="gauge-pct">${j.progress}%</div>
          </div>` : `<div>${metrics}</div>${fieldTable}`}
      </div>
      ${actions.join(" ")}
    </div>
  `;
}

// renderFieldMetricsTable renders per-field exact-match/F1 for a vision_lm
// run's document-extraction fields, so the "which fields need more data"
// question the plan calls out doesn't require exporting metrics.json by hand.
function renderFieldMetricsTable(fieldMetrics) {
  const rows = Object.entries(fieldMetrics)
    .map(
      ([field, m]) =>
        `<tr><td>${escapeHtml(field)}</td><td>${(m.exact_match * 100).toFixed(0)}%</td><td>${(m.f1 * 100).toFixed(0)}%</td><td>${m.support}</td></tr>`
    )
    .join("");
  return `
    <table style="margin-top:8px;">
      <thead><tr><th>Field</th><th>Exact match</th><th>F1</th><th>Support</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

// --- Deploy & Test panel ---
async function renderDeployPanel(task) {
  const panel = document.getElementById("panel-deploy");
  if (!panel) return;
  let deployment = null;
  let latestJob = null;
  try {
    const jobs = await api("GET", `/tasks/${task.id}/training`);
    latestJob = (jobs || []).filter((j) => j.status === "completed").sort((a, b) => b.version - a.version)[0];
  } catch (e) {}
  try {
    deployment = await api("GET", `/tasks/${task.id}/deployments/active`);
  } catch (e) {
    deployment = null;
  }
  const knownKey = deployment ? state.apiKeys[deployment.id] : null;

  panel.innerHTML = `
    <div class="card">
      <h3>Deployment</h3>
      ${
        !latestJob
          ? '<p class="muted">Finish a training run before deploying.</p>'
          : deployment
          ? `
        <div class="job-row">
          <div>
            <div>Endpoint <span class="source-tag">${escapeHtml(deployment.endpoint)}</span></div>
            <div class="hint">Deployed from a training run · autoscale ${deployment.autoscale ? "on" : "off"} · ${deployment.request_count} request(s) served</div>
          </div>
          <span class="status-pill ${deployment.status}">${deployment.status}</span>
        </div>
        ${
          knownKey
            ? `<div class="predict-result" style="margin-top:12px;">
                <div class="hint">API key (shown once — copy it now, it can't be retrieved again):</div>
                <div class="out" style="font-size:13px;">${escapeHtml(knownKey)}</div>
                <div class="modal-actions" style="justify-content:flex-start; margin-top:8px;">
                  <button class="btn small" id="copy-key-btn">Copy key</button>
                </div>
              </div>`
            : `<p class="hint">This deployment's API key was issued earlier in this browser session and is no longer shown.
               Redeploy to issue a fresh key if you've lost it.</p>`
        }
        <div class="modal-actions" style="justify-content:flex-start; margin-top:10px;">
          <button class="btn" id="redeploy-btn">Redeploy latest model</button>
        </div>
      `
          : `
        <p class="hint">Deploy your latest trained model as a live API endpoint with one click.
        You'll get an API key — save it, it's shown only once.</p>
        <label style="display:flex;align-items:center;gap:8px;margin-top:10px;">
          <input type="checkbox" id="autoscale-check" style="width:auto;" checked /> Enable autoscaling
        </label>
        <div class="modal-actions" style="justify-content:flex-start; margin-top:12px;">
          <button class="btn primary" id="deploy-btn">Deploy latest model</button>
        </div>
      `
      }
    </div>

    <div class="card">
      <h3>Test the endpoint</h3>
      ${
        !deployment
          ? '<p class="muted">Deploy a model above to test it here.</p>'
          : task.kind === "asr"
          ? `
        ${
          !knownKey
            ? `<label>API key</label><input type="text" id="predict-key" placeholder="sk_..." />`
            : ""
        }
        <div class="field" style="margin-bottom:10px;">
          <label>Audio file (wav / mp3 / flac / m4a)</label>
          <input type="file" id="asr-predict-file" accept=".wav,.mp3,.flac,.m4a,.ogg,audio/*" />
        </div>
        <div class="field" style="margin-bottom:10px;">
          <label>Language (ISO code, optional — leave blank to auto-detect)</label>
          <input type="text" id="asr-predict-language" placeholder="en" />
        </div>
        <button class="btn primary" id="asr-predict-btn">Transcribe</button>
        <div id="predict-result"></div>
      `
          : task.kind === "vision_lm"
          ? `
        ${
          !knownKey
            ? `<label>API key</label><input type="text" id="predict-key" placeholder="sk_…" />`
            : ""
        }
        <div class="field" style="margin-bottom:10px;">
          <label>Image (drag &amp; drop or choose a file)</label>
          <div id="vision-drop-zone" style="border:1px dashed var(--charcoal-3);border-radius:4px;padding:18px;text-align:center;cursor:pointer;">
            <div class="hint" id="vision-drop-hint">Drop an image here, or click to choose one</div>
            <img id="vision-drop-preview" style="display:none;max-width:100%;max-height:200px;border-radius:4px;margin-top:8px;" />
          </div>
          <input type="file" id="vision-predict-file" accept=".png,.jpg,.jpeg,.gif,image/*" style="display:none;" />
        </div>
        <div class="field" style="margin-bottom:10px;">
          <label>Prompt</label>
          <input type="text" id="vision-predict-prompt" placeholder="Extract vendor, total and date as JSON." />
        </div>
        <button class="btn primary" id="vision-predict-btn">Predict</button>
        <div id="predict-result"></div>
      `
          : `
        ${
          !knownKey
            ? `<label>API key</label><input type="text" id="predict-key" placeholder="sk_…" />`
            : ""
        }
        <div class="inline-form">
          <div class="field">
            <label>Input</label>
            <input type="text" id="predict-input" placeholder="Type a test input…" />
          </div>
          <button class="btn primary" id="predict-btn">Predict</button>
        </div>
        <div id="predict-result"></div>
      `
      }
    </div>

    ${task.kind === "vision_lm" || task.kind === "asr" ? "" : `
    <div class="card">
      <h3>Batch inference</h3>
      <p class="hint">Upload a CSV with an "input" column (or one input per line) and get
      predictions for every row back as a downloadable CSV — the bulk workload this kind
      of deployment is built to handle cheaply.</p>
      ${
        !deployment
          ? '<p class="muted">Deploy a model above to run batch predictions.</p>'
          : `
        ${!knownKey ? `<label>API key</label><input type="text" id="batch-key" placeholder="sk_…" />` : ""}
        <label>File</label>
        <input type="file" id="batch-file-input" accept=".csv,.txt,text/csv,text/plain" />
        <div class="hint" id="batch-file-status"></div>
      `
      }
    </div>
    `}

    <div class="card">
      <h3>Portable export</h3>
      <p class="hint">Download a self-hostable package (Dockerfile + manifest + adapter weights) —
      run it anywhere, no lock-in to this platform.</p>
      <button class="btn" id="export-btn" ${!latestJob ? "disabled" : ""}>Download export package</button>
    </div>

    ${task.kind === "vision_lm" || task.kind === "asr" ? "" : `
    <div class="card">
      <h3>GGUF export (HomeBred-LLM / llama.cpp)</h3>
      <p class="hint">Download your trained model in GGUF format — load it directly into HomeBred-LLM
      or llama.cpp on any machine. The merge + convert + quantize runs on-demand when you click;
      a progress bar will show each step in real time.</p>
      <div class="inline-form">
        <label style="align-self:center; white-space:nowrap;">Quantization</label>
        <select id="gguf-quant-select" style="flex:1;">
          <option value="q4_k_m" selected>Q4_K_M (balanced)</option>
          <option value="q5_k_m">Q5_K_M (higher quality)</option>
          <option value="q8_0">Q8_0 (best quality, bigger)</option>
          <option value="f16">F16 (no quantization)</option>
        </select>
        <button class="btn" id="gguf-export-btn" ${!latestJob ? "disabled" : ""}>Download GGUF model</button>
      </div>
      <div id="gguf-progress-container" style="display:none; margin-top:14px;">
        <div class="gauge-wrap">
          <div class="gauge"><div class="gauge-fill" id="gguf-progress-bar" style="width:0%"></div></div>
          <div class="gauge-pct" id="gguf-progress-pct">0%</div>
        </div>
        <div class="hint" id="gguf-progress-step" style="margin-top:6px;">Starting…</div>
        <div class="hint" id="gguf-progress-detail" style="margin-top:2px;"></div>
      </div>
    </div>
    `}
  `;

  const deployBtn = document.getElementById("deploy-btn");
  if (deployBtn) {
    deployBtn.onclick = async () => {
      const autoscale = document.getElementById("autoscale-check").checked;
      try {
        const res = await api("POST", `/tasks/${task.id}/deploy`, { autoscale });
        state.apiKeys[res.id] = res.api_key;
        toast("Model deployed. Save your API key — it won't be shown again.");
        renderDeployPanel(task);
        renderDocsPanel(task);
      } catch (e) {
        toast(e.message, true);
      }
    };
  }
  const redeployBtn = document.getElementById("redeploy-btn");
  if (redeployBtn) {
    redeployBtn.onclick = async () => {
      try {
        const res = await api("POST", `/tasks/${task.id}/deploy`, { autoscale: deployment.autoscale });
        state.apiKeys[res.id] = res.api_key;
        toast("Redeployed with a fresh API key — save it, it won't be shown again.");
        renderDeployPanel(task);
        renderDocsPanel(task);
      } catch (e) {
        toast(e.message, true);
      }
    };
  }
  const copyKeyBtn = document.getElementById("copy-key-btn");
  if (copyKeyBtn) {
    copyKeyBtn.onclick = () => {
      navigator.clipboard.writeText(knownKey).then(() => toast("API key copied."));
    };
  }

  // --- Vision-language test tab: drag-drop image + prompt -> /predict ---
  const dropZone = document.getElementById("vision-drop-zone");
  if (dropZone) {
    const fileInput = document.getElementById("vision-predict-file");
    const preview = document.getElementById("vision-drop-preview");
    const hint = document.getElementById("vision-drop-hint");
    let selectedFile = null;

    const showPreview = (file) => {
      selectedFile = file;
      preview.src = URL.createObjectURL(file);
      preview.style.display = "block";
      hint.textContent = file.name;
    };

    dropZone.onclick = () => fileInput.click();
    fileInput.onchange = () => {
      if (fileInput.files[0]) showPreview(fileInput.files[0]);
    };
    dropZone.ondragover = (e) => {
      e.preventDefault();
      dropZone.style.borderColor = "var(--accent, #999)";
    };
    dropZone.ondragleave = () => {
      dropZone.style.borderColor = "var(--charcoal-3)";
    };
    dropZone.ondrop = (e) => {
      e.preventDefault();
      dropZone.style.borderColor = "var(--charcoal-3)";
      const file = e.dataTransfer.files[0];
      if (file) showPreview(file);
    };

    const visionPredictBtn = document.getElementById("vision-predict-btn");
    visionPredictBtn.onclick = async () => {
      const prompt = document.getElementById("vision-predict-prompt").value.trim();
      if (!selectedFile) {
        toast("Choose or drop an image first.", true);
        return;
      }
      if (!prompt) {
        toast("A prompt is required.", true);
        return;
      }
      const key = knownKey || document.getElementById("predict-key").value.trim();
      if (!key) {
        toast("An API key is required to call the endpoint.", true);
        return;
      }
      const form = new FormData();
      form.append("file", selectedFile);
      form.append("prompt", prompt);
      try {
        const res = await apiRawOrThrow("POST", `/inference/${deployment.id}/predict`, form, {
          Authorization: "Bearer " + key,
        });
        const data = await res.json();
        const result = data.result || {};
        document.getElementById("predict-result").innerHTML = `
          <div class="predict-result" style="margin-top:12px;">
            <div class="hint">Model output:</div>
            <div class="out" style="white-space:pre-wrap;">${escapeHtml(result.text || "")}</div>
            <div class="conf" style="margin-top:6px;">
              ${result.json_valid ? '<span class="badge" style="background:var(--ok);color:#fff;">valid JSON</span>' : '<span class="hint">not valid JSON</span>'}
            </div>
            ${result.json ? `<div class="hint" style="margin-top:8px;">Parsed JSON:</div><pre class="out">${escapeHtml(JSON.stringify(result.json, null, 2))}</pre>` : ""}
          </div>`;
      } catch (e) {
        toast(e.message, true);
      }
    };
  }

  const asrPredictBtn = document.getElementById("asr-predict-btn");
  if (asrPredictBtn) {
    asrPredictBtn.onclick = async () => {
      const fileInput = document.getElementById("asr-predict-file");
      const file = fileInput && fileInput.files[0];
      if (!file) {
        toast("Choose an audio file first.", true);
        return;
      }
      const key = knownKey || document.getElementById("predict-key").value.trim();
      if (!key) {
        toast("An API key is required to call the endpoint.", true);
        return;
      }
      const form = new FormData();
      form.append("file", file);
      const lang = document.getElementById("asr-predict-language").value.trim();
      if (lang) form.append("language", lang);
      try {
        const res = await apiRawOrThrow("POST", `/inference/${deployment.id}/transcribe`, form, {
          Authorization: "Bearer " + key,
        });
        const data = await res.json();
        const result = data.result || {};
        const segs = Array.isArray(result.segments) ? result.segments : [];
        document.getElementById("predict-result").innerHTML = `
          <div class="predict-result" style="margin-top:12px;">
            <div class="hint">Transcript${result.language ? " (" + escapeHtml(result.language) + ")" : ""}:</div>
            <div class="out" style="white-space:pre-wrap;">${escapeHtml(result.text || "")}</div>
            ${segs.length ? `<div class="hint" style="margin-top:8px;">Segments:</div>
            <table style="margin-top:4px;"><tbody>${segs
              .map((s) => `<tr><td class="hint">${Number(s.start).toFixed(1)}s - ${Number(s.end).toFixed(1)}s</td><td>${escapeHtml(s.text || "")}</td></tr>`)
              .join("")}</tbody></table>` : ""}
          </div>`;
      } catch (e) {
        toast(e.message, true);
      }
    };
  }

  const predictBtn = document.getElementById("predict-btn");
  if (predictBtn) {
    predictBtn.onclick = async () => {
      const input = document.getElementById("predict-input").value.trim();
      if (!input) return;
      const key = knownKey || document.getElementById("predict-key").value.trim();
      if (!key) {
        toast("An API key is required to call the endpoint.", true);
        return;
      }
      try {
        const res = await apiRawOrThrow("POST", `/inference/${deployment.id}/predict`, JSON.stringify({ input }), {
          "Content-Type": "application/json",
          Authorization: "Bearer " + key,
        });
        const data = await res.json();
        if (data.kind === "token_classifier" && data.result && Array.isArray(data.result.entities)) {
          const ents = data.result.entities;
          let highlighted = escapeHtml(input);
          // Highlight spans in reverse order so character offsets remain stable
          const sorted = ents.slice().sort((a, b) => b.start - a.start);
          for (const ent of sorted) {
            const before = highlighted.slice(0, ent.start);
            const inside = highlighted.slice(ent.start, ent.end);
            const after = highlighted.slice(ent.end);
            highlighted = `${before}<mark style="background:#fef08a;padding:2px 4px;border-radius:3px;"><strong>${inside}</strong> <span style="font-size:10px;background:#eab308;color:#000;padding:1px 3px;border-radius:2px;">${escapeHtml(ent.label)} ${(ent.score * 100).toFixed(0)}%</span></mark>${after}`;
          }
          const tableRows = ents.map((e) => `<tr><td>${escapeHtml(e.text)}</td><td><span class="source-tag">${escapeHtml(e.label)}</span></td><td>${e.start}..${e.end}</td><td>${(e.score * 100).toFixed(0)}%</td></tr>`).join("");
          document.getElementById("predict-result").innerHTML = `
            <div class="predict-result" style="margin-top:12px;">
              <div class="hint">Annotated input:</div>
              <div class="out" style="line-height:1.8; margin-bottom:12px;">${highlighted}</div>
              <div class="hint">Predicted entities (${ents.length}):</div>
              <table><thead><tr><th>Text</th><th>Label</th><th>Span</th><th>Confidence</th></tr></thead><tbody>${tableRows || '<tr><td colspan="4" class="muted">No entities predicted</td></tr>'}</tbody></table>
            </div>`;
        } else {
          document.getElementById("predict-result").innerHTML = `
            <div class="predict-result">
              <div class="out">${escapeHtml(data.output || (data.result && data.result.label) || JSON.stringify(data.result || data))}</div>
              <div class="conf">confidence ${((data.confidence || (data.result && data.result.score) || 0.95) * 100).toFixed(0)}%</div>
            </div>`;
        }
      } catch (e) {
        toast(e.message, true);
      }
    };
  }
  const batchInput = document.getElementById("batch-file-input");
  if (batchInput) {
    batchInput.onchange = async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const key = knownKey || (document.getElementById("batch-key") || {}).value?.trim();
      if (!key) {
        toast("An API key is required to call the endpoint.", true);
        return;
      }
      const statusEl = document.getElementById("batch-file-status");
      statusEl.textContent = "Running batch predictions…";
      try {
        const text = await file.text();
        const res = await apiRawOrThrow("POST", `/inference/${deployment.id}/batch`, text, {
          Authorization: "Bearer " + key,
        });
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "predictions.csv";
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
        statusEl.textContent = "Done — predictions.csv downloaded.";
      } catch (err) {
        statusEl.textContent = "";
        toast(err.message, true);
      }
    };
  }
  const exportBtn = document.getElementById("export-btn");
  if (exportBtn) {
    exportBtn.onclick = () => {
      window.location.href = `${API}/tasks/${task.id}/export`;
    };
  }
  const ggufExportBtn = document.getElementById("gguf-export-btn");
  if (ggufExportBtn) {
    ggufExportBtn.onclick = async () => {
      const quant = document.getElementById("gguf-quant-select").value;
      const progressContainer = document.getElementById("gguf-progress-container");
      const progressBar = document.getElementById("gguf-progress-bar");
      const progressPct = document.getElementById("gguf-progress-pct");
      const progressStep = document.getElementById("gguf-progress-step");
      const progressDetail = document.getElementById("gguf-progress-detail");

      // Show progress UI and disable the button while converting.
      progressContainer.style.display = "block";
      ggufExportBtn.disabled = true;
      progressBar.style.width = "0%";
      progressPct.textContent = "0%";
      progressStep.textContent = "Starting…";
      progressDetail.textContent = "";

      try {
        // 1. Start async conversion.
        const startRes = await api("POST", `/tasks/${task.id}/export/gguf/async?quantization=${encodeURIComponent(quant)}`);
        const sessionID = startRes.session_id;

        // 2. Poll progress until ready or error.
        const pollProgress = async () => {
          for (;;) {
            const p = await api("GET", `/tasks/${task.id}/export/gguf/progress/${sessionID}`);
            progressBar.style.width = p.percent + "%";
            progressPct.textContent = p.percent + "%";
            progressStep.textContent = p.step || "";
            progressDetail.textContent = p.detail || "";

            if (p.status === "ready") {
              return p;
            }
            if (p.status === "error") {
              throw new Error(p.error || "Conversion failed");
            }

            // Wait 1s before next poll.
            await new Promise((resolve) => setTimeout(resolve, 1000));
          }
        };

        await pollProgress();

        // 3. Download the ready GGUF file. Use a direct link instead of
        // fetch+blob so the browser streams the (potentially multi-GB) file
        // to disk without loading it all into memory first.
        const a = document.createElement("a");
        a.href = API + `/tasks/${task.id}/export/gguf/download/${sessionID}`;
        a.download = "model.gguf"; // fallback; server sets Content-Disposition
        document.body.appendChild(a);
        a.click();
        a.remove();
        toast("GGUF model download started.");
        progressContainer.style.display = "none";
      } catch (e) {
        toast(e.message, true);
        progressStep.textContent = "Error: " + e.message;
        progressStep.style.color = "var(--err)";
      } finally {
        ggufExportBtn.disabled = false;
      }
    };
  }
}

// --- Feedback panel ---
async function renderFeedbackPanel(task) {
  const panel = document.getElementById("panel-feedback");
  if (!panel) return;
  let list;
  try {
    list = (await api("GET", `/tasks/${task.id}/feedback`)) || [];
  } catch (e) {
    panel.innerHTML = `<div class="card">Failed to load feedback: ${escapeHtml(e.message)}</div>`;
    return;
  }
  const unresolved = list.filter((f) => !f.resolved);

  panel.innerHTML = `
    <div class="card">
      <h3>Report a misprediction</h3>
      <p class="hint">Feed production mispredictions back in — they'll be folded into the dataset
      for the next retrain, closing the continuous improvement loop.</p>
      <label>Input that was mispredicted</label>
      <input type="text" id="fb-input" placeholder="e.g. My account got suspended for no reason" />
      <label>What the model actually said (optional)</label>
      <input type="text" id="fb-actual" placeholder="e.g. billing" />
      <label>What it should have said</label>
      <input type="text" id="fb-expected" placeholder="e.g. account" />
      <div class="modal-actions" style="justify-content:flex-start; margin-top:12px;">
        <button class="btn primary" id="fb-submit-btn">Submit correction</button>
      </div>
    </div>

    <div class="card">
      <h3>Unresolved corrections (${unresolved.length})</h3>
      ${
        unresolved.length === 0
          ? '<p class="muted">No unresolved corrections.</p>'
          : `<table><thead><tr><th>Input</th><th>Actual</th><th>Expected</th></tr></thead><tbody>
        ${unresolved
          .map(
            (f) =>
              `<tr><td>${escapeHtml(truncate(f.input, 50))}</td><td>${escapeHtml(truncate(f.actual_output, 40))}</td><td>${escapeHtml(truncate(f.expected_output, 40))}</td></tr>`
          )
          .join("")}
        </tbody></table>
        <div class="modal-actions" style="justify-content:flex-start; margin-top:14px;">
          <button class="btn primary" id="fold-btn">Fold into dataset & prep retrain</button>
          <button class="btn" id="fold-preference-btn">Fold as preference pairs</button>
        </div>
        <p class="hint">"Fold into dataset" adds the corrections as ordinary training examples for the
        next fine-tune. "Fold as preference pairs" instead turns each (wrong output, correction) pair
        into a <span class="source-tag">{prompt, chosen, rejected}</span> triple for DPO/ORPO tuning.</p>`
      }
    </div>
  `;

  document.getElementById("fb-submit-btn").onclick = async () => {
    const input = document.getElementById("fb-input").value.trim();
    const actual = document.getElementById("fb-actual").value.trim();
    const expected = document.getElementById("fb-expected").value.trim();
    if (!input || !expected) {
      toast("Input and expected output are required.", true);
      return;
    }
    try {
      await api("POST", `/tasks/${task.id}/feedback`, { input, actual_output: actual, expected_output: expected });
      toast("Correction recorded.");
      renderFeedbackPanel(task);
    } catch (e) {
      toast(e.message, true);
    }
  };

  const foldBtn = document.getElementById("fold-btn");
  if (foldBtn) {
    foldBtn.onclick = async () => {
      try {
        const res = await api("POST", `/tasks/${task.id}/feedback/fold`);
        toast(`Added ${res.examples_added} corrected example(s) to the dataset. Start a new fine-tuning run when ready.`);
        renderFeedbackPanel(task);
      } catch (e) {
        toast(e.message, true);
      }
    };
  }

  const foldPreferenceBtn = document.getElementById("fold-preference-btn");
  if (foldPreferenceBtn) {
    foldPreferenceBtn.onclick = async () => {
      try {
        const res = await api("POST", `/tasks/${task.id}/feedback/fold?target=preferences`);
        toast(`Added ${res.preferences_added} preference pair(s). Start a DPO/ORPO run from the Training tab when ready.`);
        renderFeedbackPanel(task);
      } catch (e) {
        toast(e.message, true);
      }
    };
  }
}

// --- API Docs panel ---
async function renderDocsPanel(task) {
  const panel = document.getElementById("panel-docs");
  if (!panel) return;
  let deployment = null;
  try {
    deployment = await api("GET", `/tasks/${task.id}/deployments/active`);
  } catch (e) {
    deployment = null;
  }
  const origin = window.location.origin;
  const endpoint = deployment ? origin + deployment.endpoint : `${origin}/api/v1/inference/{deploymentID}/predict`;
  const keyPlaceholder = (deployment && state.apiKeys[deployment.id]) || "sk_…";

  panel.innerHTML = `
    <div class="card">
      <h3>Calling your deployed model</h3>
      <p class="hint">Every prediction request must include the API key issued at deploy time,
      as either an <span class="source-tag">Authorization: Bearer</span> header or an
      <span class="source-tag">X-API-Key</span> header.</p>
      ${!deployment ? '<p class="muted">Deploy a model to get a live endpoint URL here.</p>' : ""}
      <label>Single prediction (curl)</label>
      <textarea rows="5" readonly style="font-size:12px;">curl -X POST ${endpoint} \\
  -H "Authorization: Bearer ${keyPlaceholder}" \\
  -H "Content-Type: application/json" \\
  -d '{"input": "your text here"}'</textarea>

      <label>Batch prediction — CSV in, CSV out (curl)</label>
      <textarea rows="4" readonly style="font-size:12px;">curl -X POST ${endpoint.replace("/predict", "/batch")} \\
  -H "Authorization: Bearer ${keyPlaceholder}" \\
  --data-binary @inputs.csv -o predictions.csv</textarea>

      <label>Response shape</label>
      <textarea rows="4" readonly style="font-size:12px;">{
  "output": "billing",
  "confidence": 0.87
}</textarea>
    </div>

    <div class="card">
      <h3>Other endpoints</h3>
      <table>
        <thead><tr><th>Method</th><th>Path</th><th>Purpose</th></tr></thead>
        <tbody>
          <tr><td>POST</td><td>/api/v1/tasks</td><td>Create a task</td></tr>
          <tr><td>POST</td><td>/api/v1/tasks/{id}/examples</td><td>Add example pairs</td></tr>
          <tr><td>POST</td><td>/api/v1/tasks/{id}/examples/import</td><td>Bulk import CSV</td></tr>
          <tr><td>POST</td><td>/api/v1/tasks/{id}/examples/import-jsonl</td><td>Bulk import Alpaca/chat JSONL</td></tr>
          <tr><td>POST</td><td>/api/v1/tasks/{id}/training</td><td>Start a fine-tuning run</td></tr>
          <tr><td>POST</td><td>/api/v1/tasks/{id}/deploy</td><td>Deploy the latest model (returns a one-time API key)</td></tr>
          <tr><td>POST</td><td>/api/v1/tasks/{id}/training/{jobId}/deploy</td><td>Deploy a specific version (rollback)</td></tr>
          <tr><td>GET</td><td>/api/v1/tasks/{id}/export</td><td>Download portable export package</td></tr>
          <tr><td>GET</td><td>/api/v1/tasks/{id}/export/gguf?quantization=q4_k_m</td><td>Download trained model in GGUF (sync, blocks until done)</td></tr>
          <tr><td>POST</td><td>/api/v1/tasks/{id}/export/gguf/async?quantization=q4_k_m</td><td>Start async GGUF conversion (returns session_id)</td></tr>
          <tr><td>GET</td><td>/api/v1/tasks/{id}/export/gguf/progress/{sessionId}</td><td>Poll GGUF conversion progress</td></tr>
          <tr><td>GET</td><td>/api/v1/tasks/{id}/export/gguf/download/{sessionId}</td><td>Download completed GGUF file</td></tr>
        </tbody>
      </table>
      <p class="hint">Full reference in the project README.</p>
    </div>
  `;
}

// --- Utils ---
function escapeHtml(s) {
  if (s === undefined || s === null) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
function truncate(s, n) {
  if (!s) return "";
  return s.length > n ? s.slice(0, n) + "…" : s;
}

// --- Init ---
document.getElementById("new-task-btn").onclick = showNewTaskModal;
loadTasks();
