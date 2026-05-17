const video = document.getElementById('cam');
const canvas = document.getElementById('snap');
const tabs = document.querySelectorAll('.tab');
const panels = {
  enroll: document.getElementById('enroll-section'),
  verify: document.getElementById('verify-section'),
  list:   document.getElementById('list-section'),
};

async function startCam() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
}
startCam().catch(e => alert('Camera error: ' + e.message));

tabs.forEach(t => t.addEventListener('click', () => {
  tabs.forEach(x => x.classList.toggle('active', x === t));
  Object.entries(panels).forEach(([k, el]) => el.classList.toggle('hidden', k !== t.dataset.tab));
  if (t.dataset.tab === 'list') loadList();
}));

function snapBase64() {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  return canvas.toDataURL('image/jpeg', 0.9).split(',')[1];
}

document.getElementById('enroll-btn').addEventListener('click', async () => {
  const name = document.getElementById('name').value.trim();
  const el = document.getElementById('enroll-result');
  if (!name) { el.textContent = 'Enter a name first.'; return; }
  el.textContent = 'Capturing...';
  try {
    const r = await fetch('/enroll', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, image: snapBase64() }),
    });
    const j = await r.json();
    el.textContent = r.ok ? `Enrolled "${j.name}".` : `Error: ${j.detail}`;
  } catch (e) {
    el.textContent = 'Network error: ' + e.message;
  }
});

document.getElementById('verify-btn').addEventListener('click', async () => {
  const el = document.getElementById('verify-result');
  el.textContent = 'Capturing...';
  try {
    const r = await fetch('/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: snapBase64() }),
    });
    const j = await r.json();
    if (!r.ok) { el.textContent = `Error: ${j.detail}`; return; }
    if (j.matched) {
      el.innerHTML = `<strong class="match">Match: ${j.best_match}</strong>` +
        `<br />Score: ${j.score.toFixed(3)} (threshold ${j.threshold})`;
    } else {
      el.innerHTML = `<strong class="nomatch">No match</strong>` +
        `<br />Best candidate: ${j.best_match ?? '-'} (score ${j.score.toFixed(3)})`;
    }
  } catch (e) {
    el.textContent = 'Network error: ' + e.message;
  }
});

async function loadList() {
  const ul = document.getElementById('enrolled-list');
  ul.innerHTML = '<li>Loading...</li>';
  const r = await fetch('/enrolled');
  const items = await r.json();
  ul.innerHTML = '';
  for (const it of items) {
    const li = document.createElement('li');
    li.innerHTML = `<span>${it.name}</span> <button data-id="${it.id}">Delete</button>`;
    li.querySelector('button').addEventListener('click', async () => {
      await fetch(`/enrolled/${it.id}`, { method: 'DELETE' });
      loadList();
    });
    ul.appendChild(li);
  }
  if (!items.length) ul.innerHTML = '<li>No enrolled faces yet.</li>';
}

document.getElementById('refresh-btn').addEventListener('click', loadList);
