const dropArea = document.getElementById('dropArea');
const fileInput = document.getElementById('fileInput');
const processing = document.getElementById('processing');
const resultPanel = document.getElementById('resultPanel');
const uploadZone = document.getElementById('uploadZone');

dropArea.addEventListener('dragover', (e) => { e.preventDefault(); dropArea.classList.add('drag-over'); });
dropArea.addEventListener('dragleave', () => dropArea.classList.remove('drag-over'));
dropArea.addEventListener('drop', (e) => {
  e.preventDefault();
  dropArea.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) uploadImage(file);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) uploadImage(fileInput.files[0]);
});

document.getElementById('newDetection').addEventListener('click', () => {
  resultPanel.hidden = true;
  uploadZone.hidden = false;
  fileInput.value = '';
});

document.getElementById('clearHistory').addEventListener('click', async () => {
  if (!confirm('Clear all detection history?')) return;
  await fetch('/api/history', { method: 'DELETE' });
  loadHistory();
});

async function uploadImage(file) {
  uploadZone.hidden = true;
  resultPanel.hidden = true;
  processing.hidden = false;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/detect', { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Detection failed');
    }
    const data = await res.json();
    processing.hidden = true;
    showResult(data);
    loadHistory();
  } catch (e) {
    processing.hidden = true;
    uploadZone.hidden = false;
    alert('Error: ' + e.message);
  }
}

function showResult(data) {
  resultPanel.hidden = false;

  document.getElementById('plateText').textContent = data.plate_text || 'Not detected';
  document.getElementById('confVal').textContent = data.confidence
    ? (data.confidence * 100).toFixed(1) + '%'
    : '—';

  // Candidates
  const list = document.getElementById('candidatesList');
  list.innerHTML = '';
  if (data.candidates && data.candidates.length) {
    data.candidates.forEach(c => {
      const li = document.createElement('li');
      li.innerHTML = `<span class="cand-text">${esc(c.text)}</span><span class="cand-conf">${(c.confidence * 100).toFixed(1)}%</span>`;
      list.appendChild(li);
    });
    document.getElementById('candidatesWrap').hidden = false;
  } else {
    document.getElementById('candidatesWrap').hidden = true;
  }

  // Annotated image
  if (data.annotated_b64) {
    document.getElementById('annotatedImg').src = 'data:image/png;base64,' + data.annotated_b64;
  }

  // Plate crop
  const cropPanel = document.getElementById('cropPanel');
  if (data.plate_crop_b64) {
    document.getElementById('cropImg').src = 'data:image/png;base64,' + data.plate_crop_b64;
    cropPanel.hidden = false;
  } else {
    cropPanel.hidden = true;
  }
}

async function loadHistory() {
  try {
    const res = await fetch('/api/history');
    const items = await res.json();
    const list = document.getElementById('historyList');

    if (!items.length) {
      list.innerHTML = '<p class="empty">No detections yet.</p>';
      return;
    }

    list.innerHTML = items.map(item => `
      <div class="history-item">
        <span class="history-item__plate">${esc(item.plate_text || '—')}</span>
        <span class="history-item__file">${esc(item.filename)}</span>
        <span class="history-item__conf">${item.confidence ? (item.confidence * 100).toFixed(1) + '%' : '—'}</span>
        <span class="history-item__time">${timeAgo(item.created_at)}</span>
      </div>
    `).join('');
  } catch {}
}

function timeAgo(isoStr) {
  const diff = (Date.now() - new Date(isoStr).getTime()) / 1000;
  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

function esc(str) {
  return String(str ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

loadHistory();
