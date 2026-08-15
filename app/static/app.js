const state = {
  jobs: [], selectedId: null, logOffset: 0, outputs: [], loras: [], poller: null,
  locale: localStorage.getItem('heretic-language') || 'zh-TW', pendingDelete: null, importLoaded: false,
  loraTaskSignature: null, hereticVersion: null,
  evalRuns: [], evalPresets: [], evalPresetsOllama: [], evalSignature: null, ollamaModels: [],
};
const EVAL_NOTICE_DEFAULT = '評測與 Heretic 任務共用 GPU，同一時間只能執行一項；若評測的 Ollama 位址指向遠端主機，則不佔本機 GPU，可與 Heretic 任務並行。4-bit 量化模型的分數會與 BF16 略有差異，比較時請使用相同設定。';
const EVAL_NOTICE_OLLAMA = 'GGUF 模式透過 Ollama 的 OpenAI 相容 API 評測，只支援生成式任務（如 gsm8k）；hellaswag、mmlu 等選擇題任務需要 logprobs，Ollama API 不提供，請改用 Safetensors 後端。';
const $ = (selector) => document.querySelector(selector);
let lastToastMessage = null;
let lastToastAt = 0;

function t(key, variables = {}) {
  const catalog = window.I18N[state.locale] || window.I18N['zh-TW'];
  let value = catalog[key] || window.I18N['zh-TW'][key] || key;
  Object.entries(variables).forEach(([name, replacement]) => { value = value.replaceAll(`{${name}}`, replacement); });
  return value;
}

function applyTranslations() {
  document.documentElement.lang = { 'zh-TW': 'zh-Hant', 'zh-CN': 'zh-Hans', en: 'en', ja: 'ja' }[state.locale] || 'zh-Hant';
  document.querySelectorAll('[data-i18n]').forEach((element) => {
    if ((element.id === 'console' && state.selectedId) || (element.id === 'ollamaConsole' && state.importLoaded)) return;
    element.textContent = t(element.dataset.i18n);
  });
  document.querySelectorAll('[data-i18n-placeholder]').forEach((element) => { element.placeholder = t(element.dataset.i18nPlaceholder); });
  $('#languageSelect').value = state.locale;
}

async function setLanguage(locale, persist = true) {
  if (!window.I18N[locale]) return;
  state.locale = locale;
  localStorage.setItem('heretic-language', locale);
  applyTranslations();
  renderJobs();
  renderOutputs();
  if (state.selectedId) updateSelected();
  if (persist) {
    try {
      await api('/api/settings', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ language: locale }) });
      toast(t('languageSaved'));
    } catch (error) { toast(error.message); }
  }
}

function toast(message) {
  const normalized = String(message || '發生未預期的錯誤');
  const now = Date.now();
  if (normalized === lastToastMessage && now - lastToastAt < 30000) return;
  lastToastMessage = normalized; lastToastAt = now;
  const element = $('#toast');
  if (!element) { console.error(normalized); return; }
  element.textContent = normalized; element.classList.add('show');
  window.setTimeout(() => element.classList.remove('show'), 3200);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  const detail = Array.isArray(data?.detail) ? data.detail.map((item) => item.msg).join(', ') : data?.detail;
  if (!response.ok) throw new Error(detail || `HTTP ${response.status}`);
  return data;
}

function escapeHtml(value) {
  const div = document.createElement('div'); div.textContent = value ?? ''; return div.innerHTML;
}

function formatTime(value) {
  return value ? new Intl.DateTimeFormat(state.locale, { dateStyle: 'short', timeStyle: 'short' }).format(new Date(value)) : '—';
}

function formatBytes(value) {
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']; let size = Number(value || 0); let unit = 0;
  while (size >= 1024 && unit < units.length - 1) { size /= 1024; unit += 1; }
  return `${size.toFixed(unit ? 1 : 0)} ${units[unit]}`;
}

function statusLabel(status) {
  return t({ queued: 'statusQueued', running: 'statusRunning', completed: 'statusCompleted', failed: 'statusFailed', cancelled: 'statusCancelled' }[status] || status);
}

function showView(name) {
  document.querySelectorAll('.view').forEach((view) => view.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach((item) => item.classList.toggle('active', item.dataset.view === name));
  $(`#${name}View`).classList.add('active');
  if (name === 'jobs') refreshJobs();
  if (name === 'ollama') { refreshOutputs(); refreshOllamaImport(); }
  if (name === 'lora') { refreshOutputs().then(refreshLoras); refreshLoraTask(); }
  if (name === 'evals') { refreshOutputs().then(refreshEvals); refreshEvalTask(); }
  if (name === 'versions') refreshHereticVersion(false);
}

function renderJobs() {
  $('#jobCount').textContent = state.jobs.length;
  const list = $('#jobList');
  if (!state.jobs.length) { list.innerHTML = `<div class="empty-state">${escapeHtml(t('noJobs'))}</div>`; return; }
  list.innerHTML = state.jobs.map((job) => `
    <button class="job-item ${job.id === state.selectedId ? 'active' : ''}" data-id="${escapeHtml(job.id)}">
      <div class="job-item-top"><strong>${escapeHtml(job.request.model)}</strong><span class="status-badge ${job.status}">${escapeHtml(statusLabel(job.status))}</span></div>
      <small>${formatTime(job.created_at)} · ${job.request.n_trials} trials</small>
    </button>`).join('');
  list.querySelectorAll('.job-item').forEach((item) => item.addEventListener('click', () => selectJob(item.dataset.id)));
}

async function refreshJobs() {
  try {
    state.jobs = await api('/api/jobs'); renderJobs();
    if (state.selectedId) updateSelected();
  } catch (error) { toast(`任務清單更新失敗：${error?.message || error}`); }
}

function renderOutputs() {
  const select = $('#ollamaOutput');
  const previous = select.value;
  $('#outputCount').textContent = state.outputs.length;
  if (!state.outputs.length) {
    select.innerHTML = `<option value="">${escapeHtml(t('noCompleteModels'))}</option>`;
    $('#ollamaOutputHelp').textContent = t('completeOnly');
    $('#outputLibrary').innerHTML = `<div class="empty-state">${escapeHtml(t('noModels'))}</div>`;
    return;
  }
  select.innerHTML = `<option value="">${escapeHtml(t('outputModel'))}</option>` + state.outputs.map((output) =>
    `<option value="${escapeHtml(output.name)}">${escapeHtml(output.name)} · ${formatBytes(output.size)}</option>`
  ).join('');
  if (state.outputs.some((output) => output.name === previous)) select.value = previous;
  $('#ollamaOutputHelp').textContent = t('foundModels', { count: state.outputs.length });
  $('#outputLibrary').innerHTML = state.outputs.map((output) => `
    <article class="model-card">
      <div class="model-icon">◇</div>
      <div class="model-copy"><strong title="${escapeHtml(output.name)}">${escapeHtml(output.name)}</strong><small>${formatBytes(output.size)} · ${output.file_count} ${escapeHtml(t('files'))} · ${escapeHtml(output.architectures?.[0] || 'Unknown')}</small></div>
      <button class="model-delete" data-delete-output="${escapeHtml(output.name)}" title="${escapeHtml(t('deleteModel'))}" aria-label="${escapeHtml(t('deleteModel'))}">×</button>
    </article>`).join('');
  document.querySelectorAll('[data-delete-output]').forEach((button) => button.addEventListener('click', () => openDeleteModal(button.dataset.deleteOutput)));
}

async function refreshOutputs() {
  try { state.outputs = await api('/api/outputs'); renderOutputs(); }
  catch (error) { toast(`模型清單更新失敗：${error?.message || error}`); }
}

function renderLoras() {
  $('#loraCount').textContent = state.loras.length;
  $('#loraLibraryCount').textContent = state.loras.length;
  const select = $('#loraSelect');
  const previous = select.value;
  if (!state.loras.length) {
    $('#loraLibrary').innerHTML = '<div class="empty-state">尚無 LoRA</div>';
    select.innerHTML = '<option value="">尚無 LoRA</option>';
    $('#mergeLoraSelect').innerHTML = '<option value="">尚無 LoRA</option>';
    return;
  }
  select.innerHTML = '<option value="">選擇 LoRA</option>' + state.loras.map((item) =>
    `<option value="${escapeHtml(item.name)}">${escapeHtml(item.name)} · ${escapeHtml(item.format)}</option>`
  ).join('');
  if (state.loras.some((item) => item.name === previous)) select.value = previous;
  $('#loraLibrary').innerHTML = state.loras.map((item) => `
    <article class="model-card">
      <div class="model-icon">⧉</div>
      <div class="model-copy"><strong title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</strong><small>${formatBytes(item.size)} · ${escapeHtml(item.format)} · ${escapeHtml(item.base_model || item.repo_id || 'base model 未知')}</small>${item.ollama_adapter_supported === false ? '<small class="merge-hint">此架構需先合併</small>' : ''}</div>
      <button class="model-delete" data-delete-lora="${escapeHtml(item.name)}" title="刪除 LoRA" aria-label="刪除 LoRA">×</button>
    </article>`).join('');
  document.querySelectorAll('[data-delete-lora]').forEach((button) => button.addEventListener('click', () => deleteLora(button.dataset.deleteLora)));
  renderMergeForm();
  updateLoraImportHint();
}

function renderMergeForm() {
  const mergeSelect = $('#mergeLoraSelect');
  const previous = mergeSelect.value;
  const mergeable = state.loras.filter((item) => item.format === 'safetensors');
  mergeSelect.innerHTML = mergeable.length
    ? '<option value="">選擇 LoRA</option>' + mergeable.map((item) =>
        `<option value="${escapeHtml(item.name)}">${escapeHtml(item.name)}</option>`
      ).join('')
    : '<option value="">尚無 Safetensors LoRA</option>';
  if (mergeable.some((item) => item.name === previous)) mergeSelect.value = previous;
  $('#mergeBaseOptions').innerHTML = state.outputs.map((output) =>
    `<option value="${escapeHtml(output.name)}"></option>`
  ).join('');
  applyMergeSuggestion();
}

function applyMergeSuggestion() {
  const lora = state.loras.find((item) => item.name === $('#mergeLoraSelect').value);
  const help = $('#mergeBaseHelp');
  if (!lora) { help.textContent = '會自動比對 adapter 的 base_model 建議基底；也可輸入 /models 路徑或 Hugging Face model ID（會自動下載至快取）。'; return; }
  const baseInput = $('#mergeBaseSelect');
  if (lora.suggested_base && state.outputs.some((output) => output.name === lora.suggested_base)) {
    if (!baseInput.value) baseInput.value = lora.suggested_base;
    help.textContent = `已依 adapter 設定（${lora.base_model || '未知'}）建議基底：${lora.suggested_base}`;
  } else if (lora.base_model && /^[\w.-]+\/[\w.-]+$/.test(lora.base_model)) {
    if (!baseInput.value) baseInput.value = lora.base_model;
    help.textContent = `找不到符合的本機 output，已填入 adapter 的 HF 基底「${lora.base_model}」；合併時會自動下載至快取（需足夠磁碟空間），也可改選本機 output 或 /models 路徑。`;
  } else {
    help.textContent = lora.base_model
      ? `找不到符合「${lora.base_model}」的本機 output，請從清單選擇、輸入 /models 路徑或 HF model ID。`
      : '此 adapter 未記錄 base model，請確認選擇的基底正確。';
  }
  const output = $('#mergeOutputName');
  if (!output.value && baseInput.value) {
    const baseName = baseInput.value.split('/').filter(Boolean).pop() || 'merged';
    output.value = `${baseName}-${lora.name}`.replace(/[^a-zA-Z0-9._-]/g, '-').slice(0, 120);
  }
}

function updateLoraImportHint() {
  const lora = state.loras.find((item) => item.name === $('#loraSelect').value);
  $('#loraBaseHelp').textContent = lora && lora.ollama_adapter_supported === false
    ? '注意：此 adapter 的架構不在 Ollama 支援清單，直接匯入會失敗，建議改用下方「合併為完整模型」。'
    : '請填寫 Ollama 內已有的模型名稱。';
}

async function refreshLoras() {
  try { state.loras = await api('/api/loras'); renderLoras(); }
  catch (error) { toast(`LoRA 清單更新失敗：${error?.message || error}`); }
}

async function deleteLora(name) {
  if (!window.confirm(`確定永久刪除 LoRA「${name}」？`)) return;
  try {
    const result = await api(`/api/loras/${encodeURIComponent(name)}`, { method: 'DELETE' });
    await refreshLoras();
    toast(`已刪除 ${name}，釋放 ${formatBytes(result.deleted_bytes)}。`);
  } catch (error) { toast(error?.message || error); }
}

async function refreshLoraTask() {
  try {
    const task = await api('/api/loras/task');
    if (!task) return;
    $('#loraTaskPanel').hidden = false;
    $('#loraTaskStatus').className = `status-badge ${task.status}`;
    $('#loraTaskStatus').textContent = statusLabel(task.status);
    $('#loraTaskTitle').textContent = task.operation === 'merge'
      ? `${task.base_model} + ${task.lora_name} → ${task.output_name}`
      : `${task.operation === 'download' ? 'Hugging Face' : task.base_model} → ${task.lora_name}${task.model_name ? ` → ${task.model_name}` : ''}`;
    const percent = task.bytes_total ? Math.min(100, Math.round(task.bytes_completed * 100 / task.bytes_total)) : (task.status === 'completed' ? 100 : 0);
    const phases = { queued: '準備中', downloading: '下載中', uploading: '上傳中', creating: '建立模型', merging: '合併中', completed: '完成', failed: '失敗' };
    $('#loraTaskProgress').textContent = task.bytes_total ? `${phases[task.phase] || task.phase} · ${percent}% · ${formatBytes(task.bytes_completed)} / ${formatBytes(task.bytes_total)}` : (phases[task.phase] || task.phase);
    $('#loraTaskProgressBar').style.width = `${percent}%`;
    const consoleElement = $('#loraConsole');
    const nearBottom = consoleElement.scrollHeight - consoleElement.scrollTop - consoleElement.clientHeight < 80;
    consoleElement.textContent = task.log || task.error || phases[task.phase] || task.phase;
    if (nearBottom) consoleElement.scrollTop = consoleElement.scrollHeight;
    const running = ['queued', 'running'].includes(task.status);
    $('#loraDownloadButton').disabled = running;
    $('#loraImportButton').disabled = running;
    $('#loraMergeButton').disabled = running;
    document.querySelectorAll('[data-delete-lora]').forEach((button) => { button.disabled = running && button.dataset.deleteLora === task.lora_name; });
    const signature = `${task.id}:${task.status}`;
    if (state.loraTaskSignature !== signature && task.status === 'completed') {
      if (task.operation === 'merge') await refreshOutputs();
      await refreshLoras();
    }
    state.loraTaskSignature = signature;
  } catch (error) { toast(`LoRA 狀態更新失敗：${error?.message || error}`); }
}

function formatMetric(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return String(value);
  return number >= 0 && number <= 1 ? `${(number * 100).toFixed(2)}%` : number.toFixed(3);
}

function renderEvalForm() {
  const ollama = $('#evalBackend').value === 'ollama';
  $('#evalModelOptions').innerHTML = state.outputs.map((output) =>
    `<option value="${escapeHtml(output.name)}"></option>`
  ).join('');
  const select = $('#evalModelSelect');
  const previous = select.value;
  select.innerHTML = state.ollamaModels.length
    ? '<option value="">選擇 Ollama 模型</option>' + state.ollamaModels.map((name) =>
        `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`
      ).join('')
    : '<option value="">尚無 Ollama 模型（請確認 API 位址）</option>';
  if (state.ollamaModels.includes(previous)) select.value = previous;
  const grid = $('#evalTaskGrid');
  const presets = ollama ? state.evalPresetsOllama : state.evalPresets;
  const checked = grid.dataset.backend === $('#evalBackend').value
    ? new Set(Array.from(grid.querySelectorAll('input:checked')).map((input) => input.value))
    : new Set(ollama ? ['gsm8k'] : ['hellaswag', 'arc_challenge']);
  grid.innerHTML = presets.map((task) => `
    <label class="check-item"><input type="checkbox" value="${escapeHtml(task)}" ${checked.has(task) ? 'checked' : ''}><code>${escapeHtml(task)}</code></label>`
  ).join('');
  if (presets.length) grid.dataset.backend = $('#evalBackend').value;
}

let ollamaModelsRequest = 0;

async function refreshOllamaModels() {
  // Sequence guard: a slow fetch for a previous address must not overwrite
  // the list for the address currently in the field.
  const requestId = ++ollamaModelsRequest;
  const base = $('#evalBaseUrl').value.trim();
  try {
    const data = await api(`/api/ollama/models${base ? `?base_url=${encodeURIComponent(base)}` : ''}`);
    if (requestId !== ollamaModelsRequest) return;
    state.ollamaModels = data.models || [];
    renderEvalForm();
  } catch (error) {
    if (requestId !== ollamaModelsRequest) return;
    state.ollamaModels = [];
    renderEvalForm();
    toast(`Ollama 模型清單取得失敗：${error?.message || error}`);
  }
}

function renderEvalRuns() {
  $('#evalCount').textContent = state.evalRuns.length;
  $('#evalRunCount').textContent = state.evalRuns.length;
  const list = $('#evalRunList');
  if (!state.evalRuns.length) { list.innerHTML = '<div class="empty-state">尚無評測紀錄</div>'; return; }
  list.innerHTML = state.evalRuns.map((run) => {
    const running = ['queued', 'running'].includes(run.status);
    const options = [
      run.num_fewshot === null ? 'few-shot 預設' : `few-shot ${run.num_fewshot}`,
      run.limit ? `每任務 ${run.limit} 題` : '全部題目',
      run.max_gen_toks ? `生成上限 ${run.max_gen_toks} tokens` : null,
      run.num_concurrent ? `並行 ${run.num_concurrent}` : null,
      run.use_cache ? '續跑快取' : null,
      run.backend === 'ollama' ? 'GGUF（Ollama）' : (run.quantization === 'bnb_4bit' ? '4-bit' : 'BF16'),
    ].filter(Boolean).join(' · ');
    const metrics = Object.entries(run.results || {}).map(([task, values]) => `
      <div class="eval-metric-row"><strong>${escapeHtml(task)}</strong>${Object.entries(values).map(([metric, value]) =>
        `<span>${escapeHtml(metric)} ${escapeHtml(formatMetric(value))}</span>`
      ).join('')}</div>`).join('');
    return `
    <article class="eval-run-card">
      <div class="eval-run-head"><span class="status-badge ${escapeHtml(run.status)}">${escapeHtml(statusLabel(run.status))}</span><strong title="${escapeHtml(run.model_source)}">${escapeHtml(run.model_source)}</strong></div>
      ${running ? '<span></span>' : `<button class="model-delete" data-delete-eval="${escapeHtml(run.id)}" title="刪除評測紀錄" aria-label="刪除評測紀錄">×</button>`}
      <div class="eval-run-meta">${formatTime(run.created_at)} · ${escapeHtml(run.tasks.join(', '))} · ${escapeHtml(options)}</div>
      ${metrics ? `<div class="eval-metrics">${metrics}</div>` : ''}
      ${run.error ? `<div class="eval-run-meta">${escapeHtml(run.error)}</div>` : ''}
    </article>`;
  }).join('');
  list.querySelectorAll('[data-delete-eval]').forEach((button) => button.addEventListener('click', () => deleteEvalRun(button.dataset.deleteEval)));
}

async function refreshEvals() {
  try {
    const data = await api('/api/evals');
    state.evalPresets = data.preset_tasks || [];
    state.evalPresetsOllama = data.preset_tasks_ollama || [];
    state.evalRuns = data.runs || [];
    renderEvalForm();
    renderEvalRuns();
  } catch (error) { toast(`評測清單更新失敗：${error?.message || error}`); }
}

async function refreshEvalTask() {
  try {
    const task = await api('/api/evals/task');
    if (!task) return;
    $('#evalTaskPanel').hidden = false;
    $('#evalTaskStatus').className = `status-badge ${task.status}`;
    $('#evalTaskStatus').textContent = statusLabel(task.status);
    $('#evalTaskTitle').textContent = `${task.model_source} · ${task.tasks.join(', ')}`;
    const consoleElement = $('#evalConsole');
    const nearBottom = consoleElement.scrollHeight - consoleElement.scrollTop - consoleElement.clientHeight < 80;
    consoleElement.textContent = task.log || task.error || statusLabel(task.status);
    if (nearBottom) consoleElement.scrollTop = consoleElement.scrollHeight;
    const running = ['queued', 'running'].includes(task.status);
    $('#evalCancelButton').hidden = !running;
    if (!$('#evalSubmitButton').dataset.unavailable) $('#evalSubmitButton').disabled = running;
    const signature = `${task.id}:${task.status}`;
    if (state.evalSignature && state.evalSignature !== signature && !running) await refreshEvals();
    state.evalSignature = signature;
  } catch (error) { toast(`評測狀態更新失敗：${error?.message || error}`); }
}

async function deleteEvalRun(id) {
  if (!window.confirm('確定刪除此評測紀錄？結果 JSON 也會一併刪除。')) return;
  try {
    await api(`/api/evals/${encodeURIComponent(id)}`, { method: 'DELETE' });
    await refreshEvals();
    toast('已刪除評測紀錄。');
  } catch (error) { toast(error?.message || error); }
}

function renderHereticVersion(version) {
  state.hereticVersion = version;
  if (!version?.available) {
    $('#hereticVersionNotice').textContent = version?.error || 'Heretic 版本管理不可用。';
    $('#updateVersionButton').disabled = true;
    $('#rollbackVersionButton').disabled = true;
    return;
  }
  $('#hereticCommit').textContent = version.short_commit;
  $('#hereticCommit').title = version.commit;
  $('#hereticSubject').textContent = version.subject;
  $('#hereticTracking').textContent = `Slot ${version.active_slot} · ${version.remote}/${version.branch}`;
  $('#hereticCommitTime').textContent = formatTime(version.committed_at);
  $('#hereticLatest').textContent = version.latest_commit ? version.latest_commit.slice(0, 7) : '尚未檢查';
  $('#hereticUpdateState').textContent = version.latest_commit ? (version.update_available ? '有新版本可用' : '目前已是最新版') : '按下檢查更新以連線 GitHub';
  const dirty = Boolean(version.dirty);
  const dirtyFiles = $('#hereticDirtyFiles');
  dirtyFiles.hidden = !dirty;
  dirtyFiles.textContent = dirty ? `未提交修改：\n${version.dirty_files.join('\n')}` : '';
  if (dirty) {
    $('#hereticVersionNotice').textContent = '偵測到未提交的本機修改。為避免覆蓋，更新與退回功能已鎖定。';
  } else if (version.rebuild_required) {
    $('#hereticVersionNotice').textContent = '依賴檔案曾變更，請重新建置 Docker image 後再執行模型任務。';
  } else if (version.managed_patches_applied) {
    const patchNames = (version.managed_patches || []).map((item) => item.name).join('、') || 'Gemma 4 compatibility patch';
    $('#hereticVersionNotice').textContent = `Active Slot ${version.active_slot} 已驗證；managed patch：${patchNames}。`;
  } else if (version.rollback_available) {
    $('#hereticVersionNotice').textContent = `可退回更新前版本 ${version.previous_short_commit || version.previous_commit?.slice(0, 7)}。`;
  } else {
    $('#hereticVersionNotice').textContent = 'Working tree 乾淨，可以安全檢查或更新版本。';
  }
  $('#updateVersionButton').disabled = dirty || !version.update_available;
  $('#rollbackVersionButton').disabled = dirty || !version.rollback_available;
}

async function refreshHereticVersion(checkRemote = false) {
  const checkButton = $('#checkVersionButton');
  if (checkRemote) { checkButton.disabled = true; checkButton.textContent = '檢查中...'; }
  try {
    const version = await api(`/api/heretic/version${checkRemote ? '?check_remote=true' : ''}`);
    renderHereticVersion(version);
  } catch (error) { toast(error.message); }
  finally { if (checkRemote) { checkButton.disabled = false; checkButton.textContent = '檢查更新'; } }
}

async function changeHereticVersion(action) {
  const isRollback = action === 'rollback';
  const prompt = isRollback ? '確定退回更新前的 Heretic 版本？' : '確定將 Heretic 更新至官方 master 最新版本？';
  if (!window.confirm(prompt)) return;
  const button = isRollback ? $('#rollbackVersionButton') : $('#updateVersionButton');
  button.disabled = true;
  const original = button.textContent;
  button.textContent = isRollback ? '退回中...' : '更新中...';
  try {
    const result = await api(`/api/heretic/version/${action}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ confirmation: isRollback ? 'ROLLBACK' : 'UPDATE' }),
    });
    renderHereticVersion(result);
    toast(result.rebuild_required ? `${result.message}；需要重新建置 image。` : result.message);
  } catch (error) { toast(error.message); await refreshHereticVersion(false); }
  finally { button.textContent = original; }
}

function openDeleteModal(name) {
  state.pendingDelete = name;
  $('#deleteModelName').textContent = name;
  $('#deleteModal').hidden = false;
  $('#deleteCancel').focus();
}

function closeDeleteModal() {
  state.pendingDelete = null;
  $('#deleteModal').hidden = true;
}

async function deletePendingOutput() {
  if (!state.pendingDelete) return;
  const name = state.pendingDelete; const button = $('#deleteConfirm');
  button.disabled = true; button.textContent = t('deleting');
  try {
    const result = await api(`/api/outputs/${encodeURIComponent(name)}`, { method: 'DELETE' });
    closeDeleteModal();
    if ($('#ollamaOutput').value === name) $('#ollamaOutput').value = '';
    await refreshOutputs();
    toast(t('modelDeleted', { name, size: formatBytes(result.deleted_bytes) }));
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; button.textContent = t('deletePermanently'); }
}

async function refreshOllamaImport() {
  try {
    const item = await api('/api/ollama/import');
    if (!item) return;
    state.importLoaded = true;
    $('#ollamaImportPanel').hidden = false;
    $('#ollamaStatus').className = `status-badge ${item.status}`;
    $('#ollamaStatus').textContent = statusLabel(item.status);
    $('#ollamaImportTitle').textContent = `${item.output_name} → ${item.model_name}`;
    const percent = item.bytes_total ? Math.min(100, Math.round(item.bytes_completed * 100 / item.bytes_total)) : 0;
    const phase = t({ queued: 'statusQueued', downloading: 'phaseDownloading', converting_bf16: 'phaseConverting', quantizing: 'phaseQuantizing', uploading: 'phaseUploading', creating: 'phaseCreating', completed: 'statusCompleted', failed: 'statusFailed' }[item.phase] || 'phasePreparing');
    $('#ollamaProgress').textContent = item.bytes_total ? `${phase} · ${percent}% · ${formatBytes(item.bytes_completed)} / ${formatBytes(item.bytes_total)}` : phase;
    $('#ollamaProgressBar').style.width = `${percent}%`;
    const consoleElement = $('#ollamaConsole');
    const nearBottom = consoleElement.scrollHeight - consoleElement.scrollTop - consoleElement.clientHeight < 80;
    consoleElement.textContent = item.log || (item.error ? `${statusLabel('failed')}: ${item.error}` : phase);
    if (nearBottom) consoleElement.scrollTop = consoleElement.scrollHeight;
    const running = ['queued', 'running'].includes(item.status);
    $('#ollamaSubmitButton').disabled = running;
    $('#ollamaSubmitButton').textContent = running ? t('importRunning') : t('startImport');
    document.querySelectorAll('[data-delete-output]').forEach((button) => { button.disabled = running && button.dataset.deleteOutput === item.output_name; });
  } catch (error) { toast(`Ollama 狀態更新失敗：${error?.message || error}`); }
}

function selectJob(id) {
  state.selectedId = id; state.logOffset = 0; $('#console').textContent = '';
  renderJobs(); updateSelected(); pollLog();
}

function updateSelected() {
  const job = state.jobs.find((entry) => entry.id === state.selectedId);
  if (!job) return;
  $('#selectedStatus').className = `status-badge ${job.status}`;
  $('#selectedStatus').textContent = statusLabel(job.status);
  $('#selectedTitle').textContent = job.request.model;
  const hereticVersion = job.heretic_slot ? `Heretic ${job.heretic_slot}@${(job.heretic_commit || '').slice(0, 7)}` : 'Heretic legacy';
  $('#jobMeta').textContent = `${job.id} · ${hereticVersion} · ${job.output_directory}`;
  $('#cancelButton').hidden = !['queued', 'running'].includes(job.status);
  $('#retryButton').hidden = job.status !== 'failed';
}

async function pollLog() {
  if (!state.selectedId) return;
  try {
    const data = await api(`/api/jobs/${state.selectedId}/log?offset=${state.logOffset}`);
    if (data.content) {
      const consoleElement = $('#console');
      const nearBottom = consoleElement.scrollHeight - consoleElement.scrollTop - consoleElement.clientHeight < 80;
      consoleElement.textContent += data.content;
      if (nearBottom) consoleElement.scrollTop = consoleElement.scrollHeight;
    }
    state.logOffset = data.next_offset; $('#logPosition').textContent = `${state.logOffset.toLocaleString()} bytes`;
  } catch (_) { /* Next poll reconciles state. */ }
}

function formPayload(form) {
  const values = Object.fromEntries(new FormData(form));
  ['n_trials', 'n_startup_trials', 'max_response_length', 'batch_size', 'max_batch_size', 'lora_rank'].forEach((key) => { values[key] = Number(values[key]); });
  values.offload_outputs_to_cpu = form.elements.offload_outputs_to_cpu.checked;
  values.orthogonalize_direction = form.elements.orthogonalize_direction.checked;
  if (!values.output_name) delete values.output_name;
  if (!values.hf_token) delete values.hf_token;
  if (!values.good_config || !values.good_config.trim()) delete values.good_config;
  if (!values.bad_config || !values.bad_config.trim()) delete values.bad_config;
  return values;
}

// Auto-detect the configs of an entered HF dataset so multi-config datasets
// (which Heretic can't load by ID) can be picked from a list and resolved.
const datasetConfigRequest = { good: 0, bad: 0 };
async function detectDatasetConfigs(side) {
  const repo = $(`#${side}Dataset`).value.trim();
  const datalist = $(`#${side}ConfigOptions`);
  const hint = $(`#${side}DatasetHint`);
  const requestId = ++datasetConfigRequest[side];
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo)) { datalist.innerHTML = ''; hint.hidden = true; return; }
  try {
    const data = await api(`/api/hf/dataset/configs?repo_id=${encodeURIComponent(repo)}`);
    if (requestId !== datasetConfigRequest[side]) return;
    const configs = data.configs || [];
    datalist.innerHTML = configs.map((c) => `<option value="${c.name}">`).join('');
    if (configs.length) {
      hint.hidden = false;
      hint.textContent = t('configsDetected', { count: configs.length, list: configs.map((c) => c.name).join('、') });
      if (data.suggested_column && $(`#${side}Column`).value === 'text') $(`#${side}Column`).value = data.suggested_column;
    } else {
      hint.hidden = true;
    }
  } catch (error) {
    if (requestId !== datasetConfigRequest[side]) return;
    datalist.innerHTML = ''; hint.hidden = true;
  }
}
['good', 'bad'].forEach((side) => {
  $(`#${side}Dataset`).addEventListener('change', () => detectDatasetConfigs(side));
});

document.querySelectorAll('.nav-item').forEach((button) => button.addEventListener('click', () => showView(button.dataset.view)));
$('#languageSelect').addEventListener('change', (event) => setLanguage(event.target.value));
$('#refreshButton').addEventListener('click', refreshJobs);
$('#refreshOutputsButton').addEventListener('click', refreshOutputs);
$('#refreshLorasButton').addEventListener('click', refreshLoras);
$('#refreshVersionButton').addEventListener('click', () => refreshHereticVersion(false));
$('#checkVersionButton').addEventListener('click', () => refreshHereticVersion(true));
$('#updateVersionButton').addEventListener('click', () => changeHereticVersion('update'));
$('#rollbackVersionButton').addEventListener('click', () => changeHereticVersion('rollback'));
$('#deleteCancel').addEventListener('click', closeDeleteModal);
$('#deleteConfirm').addEventListener('click', deletePendingOutput);
$('#deleteModal').addEventListener('click', (event) => { if (event.target.id === 'deleteModal') closeDeleteModal(); });
document.addEventListener('keydown', (event) => { if (event.key === 'Escape' && !$('#deleteModal').hidden) closeDeleteModal(); });
$('#ollamaOutput').addEventListener('change', (event) => {
  const output = state.outputs.find((entry) => entry.name === event.target.value);
  if (event.target.value && !$('#ollamaModelName').value) $('#ollamaModelName').value = event.target.value.toLowerCase().replace(/[^a-z0-9._/-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 80);
  $('#ollamaFormatHelp').textContent = output?.recommended_format === 'gguf' ? t('autoGgufHelp') : t('autoSafeHelp');
});
$('#ollamaSourceMode').addEventListener('change', (event) => {
  const hf = event.target.value === 'hf';
  $('#ollamaOutput').closest('label').hidden = hf;
  $('#ollamaOutput').disabled = hf;
  $('#ollamaOutput').required = !hf;
  $('#ollamaRepoField').hidden = !hf;
  $('#ollamaRevisionField').hidden = !hf;
  $('#ollamaRepoId').disabled = !hf;
  $('#ollamaRepoId').required = hf;
  $('#ollamaRevision').disabled = !hf;
});
$('#ollamaRepoId').addEventListener('change', (event) => {
  const repo = event.target.value.trim();
  if (repo && !$('#ollamaModelName').value) $('#ollamaModelName').value = (repo.split('/').pop() || '').toLowerCase().replace(/[^a-z0-9._-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 80);
});
$('#loraSelect').addEventListener('change', (event) => {
  const item = state.loras.find((entry) => entry.name === event.target.value);
  if (item?.base_model) $('#loraBaseModel').value = item.base_model;
  $('#loraBaseHelp').textContent = item?.base_model ? `Adapter metadata 建議：${item.base_model}` : '請填寫 Ollama 內已有、且與訓練相同的基底模型。';
  if (item && !$('#loraModelName').value) $('#loraModelName').value = item.name.toLowerCase().replace(/[^a-z0-9._/-]+/g, '-').slice(0, 80);
});
$('#jobForm').elements.quantization.addEventListener('change', (event) => { $('#quantNotice').hidden = event.target.value !== 'bnb_4bit'; });
$('#jobForm').addEventListener('submit', async (event) => {
  event.preventDefault(); const button = $('#submitButton'); button.disabled = true; button.textContent = t('creatingJob');
  try {
    const job = await api('/api/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(formPayload(event.target)) });
    if (event.target.elements.hf_token.value) { $('#hfTokenHelp').textContent = t('tokenSaved'); event.target.elements.hf_token.placeholder = t('tokenSaved'); }
    event.target.elements.hf_token.value = ''; state.selectedId = job.id; state.logOffset = 0; showView('jobs'); toast(t('jobCreated'));
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; button.textContent = t('startProcessing'); }
});
$('#cancelButton').addEventListener('click', async () => {
  if (!state.selectedId || !window.confirm(t('confirmCancel'))) return;
  try { await api(`/api/jobs/${state.selectedId}/cancel`, { method: 'POST' }); await refreshJobs(); toast(t('cancelSent')); }
  catch (error) { toast(error.message); }
});
$('#retryButton').addEventListener('click', async () => {
  if (!state.selectedId || !window.confirm(t('confirmRetry'))) return;
  try { await api(`/api/jobs/${state.selectedId}/retry`, { method: 'POST' }); await refreshJobs(); toast(t('retrySent')); }
  catch (error) { toast(error.message); }
});
$('#ollamaForm').addEventListener('submit', async (event) => {
  event.preventDefault(); const button = $('#ollamaSubmitButton'); button.disabled = true; button.textContent = t('creatingImport');
  const values = Object.fromEntries(new FormData(event.target));
  if (!values.quantize) values.quantize = null;
  values.keep_intermediate = event.target.elements.keep_intermediate.checked;
  const hfMode = $('#ollamaSourceMode').value === 'hf';
  if (hfMode && !values.revision) values.revision = 'main';
  try {
    await api(hfMode ? '/api/ollama/import/hf' : '/api/ollama/import', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(values) });
    $('#ollamaImportPanel').hidden = false; await refreshOllamaImport(); toast(t('importStarted'));
  } catch (error) { toast(error.message); button.disabled = false; button.textContent = t('startImport'); }
});

$('#loraDownloadForm').addEventListener('submit', async (event) => {
  event.preventDefault(); const button = $('#loraDownloadButton'); button.disabled = true;
  const values = Object.fromEntries(new FormData(event.target));
  if (!values.filename) delete values.filename;
  if (!values.hf_token) delete values.hf_token;
  try {
    await api('/api/loras/download', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(values) });
    event.target.elements.hf_token.value = ''; $('#loraTaskPanel').hidden = false; await refreshLoraTask(); toast('LoRA 下載已開始');
  } catch (error) { toast(error.message); button.disabled = false; }
});

$('#loraImportForm').addEventListener('submit', async (event) => {
  event.preventDefault(); const button = $('#loraImportButton'); button.disabled = true;
  const values = Object.fromEntries(new FormData(event.target));
  try {
    await api('/api/loras/import', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(values) });
    $('#loraTaskPanel').hidden = false; await refreshLoraTask(); toast('LoRA 正在匯入 Ollama');
  } catch (error) { toast(error.message); button.disabled = false; }
});

$('#loraMergeForm').addEventListener('submit', async (event) => {
  event.preventDefault(); const button = $('#loraMergeButton'); button.disabled = true;
  const values = Object.fromEntries(new FormData(event.target));
  try {
    await api('/api/loras/merge', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(values) });
    $('#loraTaskPanel').hidden = false; await refreshLoraTask(); toast('LoRA 合併已開始');
  } catch (error) { toast(error.message); button.disabled = false; }
});

$('#mergeLoraSelect').addEventListener('change', () => { $('#mergeOutputName').value = ''; applyMergeSuggestion(); });
$('#mergeBaseSelect').addEventListener('change', applyMergeSuggestion);
$('#mergeBaseSelect').addEventListener('input', (event) => {
  if (event.target.value.includes(':')) {
    $('#mergeBaseHelp').textContent = 'Ollama 模型（GGUF）無法作為合併基底；請輸入 outputs 名稱、/models 路徑或 HF model ID（例：Qwen/Qwen3.6-27B）。';
  }
});
$('#loraSelect').addEventListener('change', updateLoraImportHint);

$('#refreshEvalsButton').addEventListener('click', () => {
  refreshOutputs().then(refreshEvals);
  if ($('#evalBackend').value === 'ollama') refreshOllamaModels();
});
$('#evalBackend').addEventListener('change', () => {
  const ollama = $('#evalBackend').value === 'ollama';
  $('#evalBaseUrlField').hidden = !ollama;
  $('#evalBaseUrl').required = ollama;
  $('#evalQuantField').hidden = ollama;
  $('#evalConcurrencyField').hidden = !ollama;
  $('#evalRetriesField').hidden = !ollama;
  $('#evalModelSource').hidden = ollama;
  $('#evalModelSource').required = !ollama;
  $('#evalModelSelect').hidden = !ollama;
  $('#evalModelSelect').required = ollama;
  $('#evalModelHelp').textContent = ollama
    ? '選擇 Ollama 內已有的模型；分數反映該量化版本的實際表現。'
    : '支援 outputs 內的完整模型、/models 內的本機模型，或 Hugging Face model ID（會下載至快取）。';
  if (!$('#evalSubmitButton').dataset.unavailable) {
    $('#evalNotice').textContent = ollama ? EVAL_NOTICE_OLLAMA : EVAL_NOTICE_DEFAULT;
  }
  $('#evalModelSource').value = '';
  const maxGen = $('#evalMaxGenToks');
  if (ollama && !maxGen.value) maxGen.value = '2048';
  if (ollama) refreshOllamaModels();
  renderEvalForm();
});
$('#evalBaseUrl').addEventListener('change', () => { if ($('#evalBackend').value === 'ollama') refreshOllamaModels(); });
let evalBaseUrlDebounce = 0;
$('#evalBaseUrl').addEventListener('input', () => {
  window.clearTimeout(evalBaseUrlDebounce);
  evalBaseUrlDebounce = window.setTimeout(() => {
    if ($('#evalBackend').value === 'ollama') refreshOllamaModels();
  }, 600);
});
// Enter inside the form would submit and start an eval; confirm the URL instead.
$('#evalBaseUrl').addEventListener('keydown', (event) => {
  if (event.key !== 'Enter') return;
  event.preventDefault();
  if ($('#evalBackend').value === 'ollama') refreshOllamaModels();
});
$('#evalForm').addEventListener('submit', async (event) => {
  event.preventDefault(); const button = $('#evalSubmitButton'); button.disabled = true;
  const form = event.target;
  const selected = Array.from($('#evalTaskGrid').querySelectorAll('input:checked')).map((input) => input.value);
  const custom = form.elements.custom_tasks.value.trim();
  const tasks = selected.concat(custom ? [custom] : []).join(',');
  const backend = form.elements.backend.value;
  const payload = {
    model_source: backend === 'ollama' ? $('#evalModelSelect').value : form.elements.model_source.value.trim(),
    tasks,
    batch_size: Number(form.elements.batch_size.value || 0),
    quantization: form.elements.quantization.value,
    backend,
    log_samples: form.elements.log_samples.checked,
    use_cache: form.elements.use_cache.checked,
  };
  if (form.elements.max_gen_toks.value !== '') payload.max_gen_toks = Number(form.elements.max_gen_toks.value);
  if (backend === 'ollama') {
    payload.base_url = form.elements.base_url.value.trim();
    payload.quantization = 'none';
    if (form.elements.num_concurrent.value !== '') payload.num_concurrent = Number(form.elements.num_concurrent.value);
    if (form.elements.max_retries.value !== '') payload.max_retries = Number(form.elements.max_retries.value);
  }
  if (form.elements.num_fewshot.value !== '') payload.num_fewshot = Number(form.elements.num_fewshot.value);
  if (form.elements.limit.value !== '') payload.limit = Number(form.elements.limit.value);
  try {
    await api('/api/evals', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    $('#evalTaskPanel').hidden = false; await refreshEvalTask(); await refreshEvals(); toast('評測已開始');
  } catch (error) { toast(error.message); button.disabled = false; }
});
$('#evalCancelButton').addEventListener('click', async () => {
  if (!window.confirm('確定取消目前的評測？')) return;
  try { await api('/api/evals/cancel', { method: 'POST' }); await refreshEvalTask(); await refreshEvals(); toast('已取消評測'); }
  catch (error) { toast(error?.message || error); }
});

async function initialize() {
  applyTranslations();
  try {
    const [settings, health, system] = await Promise.all([api('/api/settings'), api('/api/health'), api('/api/system')]);
    await setLanguage(settings.language, false);
    $('#healthDot').classList.toggle('ok', health.status === 'ok');
    $('#healthText').textContent = health.heretic_available ? t('serviceReady') : t('hereticMissing');
    $('#gpuInfo').textContent = system.gpu;
    $('#ollamaBaseUrl').value = system.ollama_base_url;
    $('#loraBaseUrl').value = system.ollama_base_url;
    $('#evalBaseUrl').value = system.ollama_base_url;
    if (!system.gguf_tools_available) $('#ollamaNotice').textContent = t('toolsMissing');
    if (system.lm_eval_available === false) {
      $('#evalNotice').textContent = '目前映像缺少 lm-eval，請重新建置 WebUI image（docker compose up --build -d）後再執行評測。';
      const button = $('#evalSubmitButton'); button.disabled = true; button.dataset.unavailable = '1';
    }
    if (system.hf_token_saved) { $('#hfTokenHelp').textContent = t('tokenSaved'); $('#jobForm').elements.hf_token.placeholder = t('tokenSaved'); }
  } catch (_) { $('#healthText').textContent = t('serviceError'); }
  await refreshJobs(); await refreshOutputs(); await refreshLoras(); await refreshOllamaImport(); await refreshLoraTask(); await refreshEvals(); await refreshEvalTask(); await refreshHereticVersion(false);
  state.poller = window.setInterval(async () => { await refreshJobs(); await pollLog(); await refreshOllamaImport(); await refreshLoraTask(); await refreshEvalTask(); }, 2000);
}
initialize();
