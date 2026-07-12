const state = {
  jobs: [], selectedId: null, logOffset: 0, outputs: [], loras: [], poller: null,
  locale: localStorage.getItem('heretic-language') || 'zh-TW', pendingDelete: null, importLoaded: false,
  loraTaskSignature: null, hereticVersion: null,
};
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
  if (name === 'lora') { refreshLoras(); refreshLoraTask(); }
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
    return;
  }
  select.innerHTML = '<option value="">選擇 LoRA</option>' + state.loras.map((item) =>
    `<option value="${escapeHtml(item.name)}">${escapeHtml(item.name)} · ${escapeHtml(item.format)}</option>`
  ).join('');
  if (state.loras.some((item) => item.name === previous)) select.value = previous;
  $('#loraLibrary').innerHTML = state.loras.map((item) => `
    <article class="model-card">
      <div class="model-icon">⧉</div>
      <div class="model-copy"><strong title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</strong><small>${formatBytes(item.size)} · ${escapeHtml(item.format)} · ${escapeHtml(item.base_model || item.repo_id || 'base model 未知')}</small></div>
      <button class="model-delete" data-delete-lora="${escapeHtml(item.name)}" title="刪除 LoRA" aria-label="刪除 LoRA">×</button>
    </article>`).join('');
  document.querySelectorAll('[data-delete-lora]').forEach((button) => button.addEventListener('click', () => deleteLora(button.dataset.deleteLora)));
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
    $('#loraTaskTitle').textContent = `${task.operation === 'download' ? 'Hugging Face' : task.base_model} → ${task.lora_name}${task.model_name ? ` → ${task.model_name}` : ''}`;
    const percent = task.bytes_total ? Math.min(100, Math.round(task.bytes_completed * 100 / task.bytes_total)) : (task.status === 'completed' ? 100 : 0);
    const phases = { queued: '準備中', downloading: '下載中', uploading: '上傳中', creating: '建立模型', completed: '完成', failed: '失敗' };
    $('#loraTaskProgress').textContent = task.bytes_total ? `${phases[task.phase] || task.phase} · ${percent}% · ${formatBytes(task.bytes_completed)} / ${formatBytes(task.bytes_total)}` : (phases[task.phase] || task.phase);
    $('#loraTaskProgressBar').style.width = `${percent}%`;
    const consoleElement = $('#loraConsole');
    const nearBottom = consoleElement.scrollHeight - consoleElement.scrollTop - consoleElement.clientHeight < 80;
    consoleElement.textContent = task.log || task.error || phases[task.phase] || task.phase;
    if (nearBottom) consoleElement.scrollTop = consoleElement.scrollHeight;
    const running = ['queued', 'running'].includes(task.status);
    $('#loraDownloadButton').disabled = running;
    $('#loraImportButton').disabled = running;
    document.querySelectorAll('[data-delete-lora]').forEach((button) => { button.disabled = running && button.dataset.deleteLora === task.lora_name; });
    const signature = `${task.id}:${task.status}`;
    if (state.loraTaskSignature !== signature && task.status === 'completed') await refreshLoras();
    state.loraTaskSignature = signature;
  } catch (error) { toast(`LoRA 狀態更新失敗：${error?.message || error}`); }
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
    const phase = t({ queued: 'statusQueued', converting_bf16: 'phaseConverting', quantizing: 'phaseQuantizing', uploading: 'phaseUploading', creating: 'phaseCreating', completed: 'statusCompleted', failed: 'statusFailed' }[item.phase] || 'phasePreparing');
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
  return values;
}

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
  if (event.target.value && !$('#ollamaModelName').value) $('#ollamaModelName').value = event.target.value.toLowerCase().replace(/[^a-z0-9._/-]+/g, '-').replace(/^-+|-+$/g, '');
  $('#ollamaFormatHelp').textContent = output?.recommended_format === 'gguf' ? t('autoGgufHelp') : t('autoSafeHelp');
});
$('#loraSelect').addEventListener('change', (event) => {
  const item = state.loras.find((entry) => entry.name === event.target.value);
  if (item?.base_model) $('#loraBaseModel').value = item.base_model;
  $('#loraBaseHelp').textContent = item?.base_model ? `Adapter metadata 建議：${item.base_model}` : '請填寫 Ollama 內已有、且與訓練相同的基底模型。';
  if (item && !$('#loraModelName').value) $('#loraModelName').value = item.name.toLowerCase().replace(/[^a-z0-9._/-]+/g, '-');
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
  try {
    await api('/api/ollama/import', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(values) });
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
    if (!system.gguf_tools_available) $('#ollamaNotice').textContent = t('toolsMissing');
    if (system.hf_token_saved) { $('#hfTokenHelp').textContent = t('tokenSaved'); $('#jobForm').elements.hf_token.placeholder = t('tokenSaved'); }
  } catch (_) { $('#healthText').textContent = t('serviceError'); }
  await refreshJobs(); await refreshOutputs(); await refreshLoras(); await refreshOllamaImport(); await refreshLoraTask(); await refreshHereticVersion(false);
  state.poller = window.setInterval(async () => { await refreshJobs(); await pollLog(); await refreshOllamaImport(); await refreshLoraTask(); }, 2000);
}
initialize();
