const video  = document.getElementById('cam');
const canvas = document.getElementById('snap');
const vpCoord = document.getElementById('vp-coord');
const tabs   = document.querySelectorAll('.mode');
const thresholdInput   = document.getElementById('threshold');
const thresholdReadout = document.getElementById('threshold-readout');
let lastVerify = null;  // last /verify response, kept for live re-render

const currentThreshold = () => {
  const v = parseFloat(thresholdReadout.value);
  return Number.isFinite(v) ? v : parseFloat(thresholdInput.value);
};

// Slider → readout
thresholdInput.addEventListener('input', () => {
  thresholdReadout.value = parseFloat(thresholdInput.value).toFixed(3);
  if (lastVerify) renderScore(lastVerify);
});

// Readout → slider (clamped to slider range, allows out-of-range numbers stored but slider thumb pegs)
thresholdReadout.addEventListener('input', () => {
  const v = parseFloat(thresholdReadout.value);
  if (Number.isFinite(v)) {
    const clamped = Math.max(parseFloat(thresholdInput.min), Math.min(parseFloat(thresholdInput.max), v));
    thresholdInput.value = clamped;
  }
  if (lastVerify) renderScore(lastVerify);
});

// Snap to 3-decimal precision on blur, clamp to [0, 1].
thresholdReadout.addEventListener('blur', () => {
  let v = parseFloat(thresholdReadout.value);
  if (!Number.isFinite(v)) v = 0.565;
  v = Math.max(0, Math.min(1, v));
  thresholdReadout.value = v.toFixed(3);
  thresholdInput.value = Math.max(parseFloat(thresholdInput.min), Math.min(parseFloat(thresholdInput.max), v));
  if (lastVerify) renderScore(lastVerify);
});
const panels = {
  enroll: document.getElementById('enroll-section'),
  verify: document.getElementById('verify-section'),
  list:   document.getElementById('list-section'),
};

/* ---------- camera ----------------------------------------------------- */
async function startCam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 960 } },
      audio: false,
    });
    video.srcObject = stream;
    video.addEventListener('loadedmetadata', () => {
      if (vpCoord) vpCoord.textContent = `${video.videoWidth} × ${video.videoHeight}`;
    }, { once: true });
  } catch (e) {
    if (vpCoord) vpCoord.textContent = `camera error`;
    console.error('Camera:', e.message);
  }
}
startCam();

/* ---------- tab switcher ---------------------------------------------- */
tabs.forEach(t => t.addEventListener('click', () => {
  tabs.forEach(x => {
    const active = x === t;
    x.classList.toggle('active', active);
    x.setAttribute('aria-selected', active);
  });
  Object.entries(panels).forEach(([k, el]) =>
    el.classList.toggle('hidden', k !== t.dataset.tab));
  if (t.dataset.tab === 'list') loadList();
}));

/* ---------- capture --------------------------------------------------- */
function flashFrame() {
  const frame = document.querySelector('.frame');
  let f = frame.querySelector('.flash');
  if (!f) {
    f = document.createElement('div');
    f.className = 'flash';
    frame.appendChild(f);
  }
  f.classList.remove('fire');
  // re-trigger animation
  void f.offsetWidth;
  f.classList.add('fire');
}

function snapBase64() {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  // un-mirror so the backend receives the natural-orientation frame
  ctx.save();
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0);
  ctx.restore();
  flashFrame();
  return canvas.toDataURL('image/jpeg', 0.92).split(',')[1];
}

/* ---------- enroll ---------------------------------------------------- */
document.getElementById('enroll-btn').addEventListener('click', async () => {
  const nameEl = document.getElementById('name');
  const name = nameEl.value.trim();
  const out = document.getElementById('enroll-result');
  if (!name) {
    out.innerHTML = `<span style="color: var(--signal);">Identity is required.</span>`;
    nameEl.focus();
    return;
  }
  out.textContent = 'Capturing…';
  try {
    const r = await fetch('/enroll', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, image: snapBase64() }),
    });
    const j = await r.json();
    if (r.ok) {
      out.innerHTML = `
        <div style="display:flex; gap:18px; align-items:baseline;">
          <span style="font-family:var(--serif); font-size:24px; font-style:italic; color:var(--ink);">
            ${escapeHtml(j.name)}
          </span>
          <span style="color: var(--good);">enrolled · id ${j.id ?? '–'}</span>
        </div>`;
      nameEl.value = '';
    } else {
      out.innerHTML = `<span style="color: var(--signal);">Error: ${escapeHtml(j.detail || 'unknown')}</span>`;
    }
  } catch (e) {
    out.innerHTML = `<span style="color: var(--signal);">Network error: ${escapeHtml(e.message)}</span>`;
  }
});

/* ---------- verify ---------------------------------------------------- */
document.getElementById('verify-btn').addEventListener('click', async () => {
  const out = document.getElementById('verify-result');
  out.innerHTML = `
    <div style="font-family:var(--serif); font-size:18px; font-style:italic; color:var(--ink-mute);">
      Comparing against registry…
    </div>`;
  try {
    const r = await fetch('/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: snapBase64() }),
    });
    const j = await r.json();
    if (!r.ok) {
      lastVerify = null;
      out.innerHTML = `<span style="color: var(--signal);">Error: ${escapeHtml(j.detail || 'unknown')}</span>`;
      return;
    }
    lastVerify = j;
    renderScore(j);
  } catch (e) {
    out.innerHTML = `<span style="color: var(--signal);">Network error: ${escapeHtml(e.message)}</span>`;
  }
});

function renderScore(j) {
  const out = document.getElementById('verify-result');
  const score = typeof j.score === 'number' ? j.score : 0;
  // Always evaluate against the live slider, not the server's snapshot.
  const threshold = currentThreshold();
  const matched = score >= threshold;
  const klass = matched ? 'match' : 'nomatch';
  const verdict = matched ? 'Match' : 'No match';
  const fillPct = Math.max(0, Math.min(100, (score + 0.2) * (100 / 1.2))); // -0.2..1 → 0..100
  const thrPct  = Math.max(0, Math.min(100, (threshold + 0.2) * (100 / 1.2)));
  const [intPart, fracPart] = score.toFixed(4).split('.');
  const candidate = matched ? (j.best_match ?? '—') : '—';
  const top = Array.isArray(j.top_matches) ? j.top_matches : [];

  const candidateRows = top.map((m, i) => {
    const isTop = i === 0;
    const crosses = m.score >= threshold;
    const tag = isTop ? (crosses ? 'MATCH' : 'CLOSEST') : `#${i + 1}`;
    const tagKlass = isTop && crosses ? 'tag-match' : (isTop ? 'tag-nomatch' : 'tag-rank');
    return `
      <li class="candidate ${isTop && crosses ? 'is-match' : ''}">
        <span class="cand-score">${m.score.toFixed(4)}</span>
        <span class="cand-name">${escapeHtml(m.name ?? '—')}</span>
        <span class="cand-tag ${tagKlass}">${tag}</span>
      </li>`;
  }).join('');

  out.innerHTML = `
    <div class="score-card ${klass}">
      <div>
        <div class="score-label">cosine similarity · top match</div>
        <p class="score-num">${intPart}<span class="frac">.${fracPart}</span></p>
        <div class="bar-track">
          <div class="bar-fill" style="width:${fillPct}%"></div>
          <div class="bar-mark" style="left:${thrPct}%" data-label="τ ${threshold.toFixed(3)}"></div>
        </div>
      </div>
      <div class="score-meta">
        <div class="score-label">${matched ? 'identity' : 'no identity above threshold'}</div>
        <div class="score-name">
          ${escapeHtml(candidate)}
          <span class="verdict">${verdict}</span>
        </div>
        ${top.length > 1 ? `
          <div class="score-label" style="margin-top:18px;">top ${top.length} candidates</div>
          <ul class="candidate-list">${candidateRows}</ul>
        ` : ''}
      </div>
    </div>`;
}

/* ---------- registry -------------------------------------------------- */
async function loadList() {
  const ul = document.getElementById('enrolled-list');
  ul.innerHTML = `<li class="li-empty">Loading registry…</li>`;
  try {
    const r = await fetch('/enrolled');
    const items = await r.json();
    if (!Array.isArray(items) || items.length === 0) {
      ul.innerHTML = `<li class="li-empty">No identities enrolled yet.</li>`;
      return;
    }
    ul.innerHTML = '';
    items.forEach((it, i) => {
      const li = document.createElement('li');
      li.innerHTML = `
        <span class="li-index">${String(i + 1).padStart(2, '0')}</span>
        <span class="li-name">${escapeHtml(it.name)}</span>
        <button data-id="${it.id}" aria-label="Delete ${escapeHtml(it.name)}">Delete</button>`;
      li.querySelector('button').addEventListener('click', async () => {
        await fetch(`/enrolled/${it.id}`, { method: 'DELETE' });
        loadList();
      });
      ul.appendChild(li);
    });
  } catch (e) {
    ul.innerHTML = `<li class="li-empty">Error loading registry: ${escapeHtml(e.message)}</li>`;
  }
}
document.getElementById('refresh-btn').addEventListener('click', loadList);

/* ---------- utilities ------------------------------------------------- */
function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, c =>
    ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[c]));
}
