// ── Tab switching ─────────────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(`tab-${tab.dataset.tab}`).classList.add('active');
  });
});

// ══════════════════════════════════════════════════════════
// INTERPRETER
// ══════════════════════════════════════════════════════════
let cameraOn    = false;
let pollInterval = null;

function toggleCamera() {
  const btn         = document.getElementById('btnCamera');
  const feed        = document.getElementById('videoFeed');
  const placeholder = document.getElementById('cameraPlaceholder');
  const overlay     = document.getElementById('camOverlay');

  if (!cameraOn) {
    fetch('/api/camera', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({action:'start'})
    }).then(() => {
      cameraOn      = true;
      feed.src      = '/video_feed';
      feed.style.display = 'block';
      placeholder.style.display = 'none';
      overlay.style.display     = 'flex';
      btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor"><rect x="2" y="2" width="8" height="8" rx="1"/></svg> Stop Camera`;
      btn.classList.add('stop');
      pollInterval = setInterval(fetchState, 200);
    });
  } else {
    fetch('/api/camera', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({action:'stop'})
    }).then(() => {
      cameraOn = false;
      feed.style.display = 'none';
      feed.src  = '';
      placeholder.style.display = 'flex';
      overlay.style.display     = 'none';
      btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor"><path d="M4 3l7 4-7 4V3z"/></svg> Start Camera`;
      btn.classList.remove('stop');
      clearInterval(pollInterval);
    });
  }
}

function fetchState() {
  fetch('/api/state').then(r => r.json()).then(updateUI).catch(() => {});
}

function updateUI(s) {
  // Detection badge
  const badge = document.getElementById('detectionBadge');
  if (s.detected) {
    badge.textContent    = `${s.detected}  ${s.confidence}%`;
    badge.style.display  = 'block';
  } else {
    badge.style.display  = 'none';
  }

  // Hold bar
  const holdWrap = document.getElementById('holdBarWrap');
  const holdFill = document.getElementById('holdFill');
  if (s.mode === 'LETTER' && s.hold > 0) {
    holdWrap.style.display = 'flex';
    holdFill.style.width   = `${s.hold}%`;
  } else {
    holdWrap.style.display = 'none';
  }

  // Sentence
  document.getElementById('sentenceText').textContent = s.sentence || '—';

  // Spelling
  document.getElementById('spellWord').textContent = (s.word_buffer || '') + '_';

  // Suggestions
  const sugEl = document.getElementById('suggestions');
  sugEl.innerHTML = '';
  (s.suggestions || []).forEach((sug, i) => {
    const btn = document.createElement('button');
    btn.className   = 'sug-btn';
    btn.textContent = `${i+1}. ${sug}`;
    btn.onclick = () => sendActionData('autocomplete', {idx: i});
    sugEl.appendChild(btn);
  });

  // Word mode suggestions — show top predictions as clickable buttons
  const wordSugBar  = document.getElementById('wordSugBar');
  const wordSugBtns = document.getElementById('wordSugBtns');
  if (s.mode === 'WORD' && s.word_suggestions && s.word_suggestions.length) {
    // Only rebuild buttons if suggestions changed (avoids losing click mid-poll)
    const newKey = s.word_suggestions.map(w => w.word).join(',');
    if (wordSugBtns.dataset.key !== newKey) {
      wordSugBtns.dataset.key = newKey;
      wordSugBtns.innerHTML   = '';
      s.word_suggestions.forEach(ws => {
        const btn = document.createElement('button');
        btn.className   = 'word-sug-btn';
        btn.innerHTML   = `${ws.word} <span class="conf">${ws.conf}%</span>`;
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          confirmWordSuggestion(ws.word);
        });
        wordSugBtns.appendChild(btn);
      });
    }
    wordSugBar.style.display = 'flex';
  } else {
    wordSugBtns.dataset.key  = '';
    wordSugBar.style.display = 'none';
  }

  // NLP / final sentence — handle async loading
  const finalDiv = document.getElementById('finalDisplay');
  const finalTxt = document.getElementById('finalText');
  const nlpLabel = document.getElementById('nlpLabel');
  const speakBtn = document.getElementById('btnSpeak');
  const speakLbl = document.getElementById('speakLabel');

  if (s.nlp_loading) {
    nlpLabel.textContent = '⟳ NLP';
    finalTxt.textContent = 'Correcting grammar...';
    finalDiv.style.display = 'flex';
    speakBtn.classList.add('loading');
    speakLbl.textContent = 'Processing...';
  } else if (s.final) {
    nlpLabel.textContent = '✦ NLP';
    finalTxt.textContent = s.final;
    finalDiv.style.display = 'flex';
    speakBtn.classList.remove('loading');
    speakLbl.textContent = 'Speak Sentence';
  } else {
    finalDiv.style.display = 'none';
    speakBtn.classList.remove('loading');
    speakLbl.textContent = 'Speak Sentence';
  }

  // Translated
  const transDiv = document.getElementById('translatedDisplay');
  const transTxt = document.getElementById('translatedText');
  const transLbl = document.getElementById('translatedLang');
  if (s.translated && s.language !== 'English') {
    transTxt.textContent    = s.translated;
    transLbl.textContent    = s.language.substring(0,3).toUpperCase();
    transDiv.style.display  = 'flex';
  } else {
    transDiv.style.display = 'none';
  }

  // Mode buttons
  document.getElementById('btnLetter').classList.toggle('active', s.mode === 'LETTER');
  document.getElementById('btnWord').classList.toggle('active',   s.mode === 'WORD');

  // Recent signs
  const recentEl = document.getElementById('recentList');
  if (s.recent && s.recent.length) {
    recentEl.innerHTML = s.recent.map(w =>
      `<div class="recent-tag"><span>${w}</span><span style="color:var(--text3);font-size:10px">✓</span></div>`
    ).join('');
  } else {
    recentEl.innerHTML = '<span class="empty-hint">Signs will appear here</span>';
  }
}

function confirmWordSuggestion(word) {
  // Visual feedback — hide bar immediately so user knows it registered
  const bar = document.getElementById('wordSugBar');
  if (bar) bar.style.display = 'none';
  document.getElementById('wordSugBtns').dataset.key = '';

  fetch('/api/action', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({cmd:'confirm_word_suggestion', word})
  }).then(r => r.json()).then(d => {
    console.log('Word added:', word, d);
  }).catch(err => {
    console.error('Failed to add word:', err);
    // Show bar again if failed
    if (bar) bar.style.display = 'flex';
  });
}

function sendAction(cmd) {
  // Prevent double-firing generate while NLP is running
  if (cmd === 'generate') {
    const btn = document.getElementById('btnSpeak');
    if (btn && btn.classList.contains('loading')) return;
  }
  fetch('/api/action', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({cmd})
  });
}

function sendActionData(cmd, extra) {
  fetch('/api/action', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({cmd, ...extra})
  });
}

function setMode(m) {
  fetch('/api/action', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({cmd:'mode'})
  }).then(() => {
    document.getElementById('btnLetter').classList.toggle('active', m === 'LETTER');
    document.getElementById('btnWord').classList.toggle('active',   m === 'WORD');
  });
}

function setLanguage(lang) {
  sendActionData('language', {lang});
}

// Keyboard shortcuts
document.addEventListener('keydown', e => {
  const activeTab = document.querySelector('.tab-content.active');
  if (!activeTab || activeTab.id !== 'tab-interpreter') return;
  if (e.code === 'Space')    { e.preventDefault(); sendAction('confirm_word'); }
  if (e.code === 'Enter')    { e.preventDefault(); sendAction('generate'); }
  if (e.code === 'Backspace'){ sendAction('backspace'); }
  if (e.code === 'KeyC' && !e.ctrlKey){ sendAction('clear'); }
  if (e.code === 'KeyM')     { sendAction('mode'); }
  if (e.key === '1') sendActionData('autocomplete', {idx:0});
  if (e.key === '2') sendActionData('autocomplete', {idx:1});
  if (e.key === '3') sendActionData('autocomplete', {idx:2});
  if (e.key === '4') sendActionData('autocomplete', {idx:3});
});

// ══════════════════════════════════════════════════════════
// COLLECT DATA — in-browser collection
// ══════════════════════════════════════════════════════════
let selectedWord     = null;
let collectPoll      = null;
let collectCameraOn  = false;

function loadCollectData() {
  fetch('/api/collect/words').then(r => r.json()).then(groups => {
    const container = document.getElementById('wordGroups');
    container.innerHTML = '';
    groups.forEach(g => {
      const label = document.createElement('div');
      label.className   = 'group-label';
      label.textContent = g.group.toUpperCase();
      container.appendChild(label);
      g.words.forEach(w => {
        const row  = document.createElement('div');
        const done = w.count >= 240;
        row.className   = 'word-row';
        row.dataset.word = w.word;
        row.innerHTML = `
          <span class="word-name">${w.word}</span>
          <span class="word-count">
            ${w.count}
            ${done ? '<span class="check">✓</span>' : '<span style="color:var(--text3)">○</span>'}
          </span>`;
        row.onclick = () => selectWord(w.word, row);
        container.appendChild(row);
      });
    });
  }).catch(() => {});
}

function selectWord(word, rowEl) {
  // Highlight selected row
  document.querySelectorAll('.word-row').forEach(r => r.classList.remove('selected'));
  rowEl.classList.add('selected');

  selectedWord = word;

  // Start collect session via API
  fetch('/api/collect/start', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({word})
  }).then(r => r.json()).then(data => {
    // Show camera feed
    const feed = document.getElementById('collectFeed');
    const ph   = document.getElementById('collectPlaceholder');
    if (!collectCameraOn) {
      feed.src = '/collect_feed';
      feed.style.display = 'block';
      ph.style.display   = 'none';
      collectCameraOn    = true;
    }

    // Show controls
    const ctrl = document.getElementById('collectControls');
    ctrl.style.display = 'flex';
    document.getElementById('collectWordName').textContent = word;

    // Start polling collect status
    if (collectPoll) clearInterval(collectPoll);
    collectPoll = setInterval(fetchCollectStatus, 300);

    // Update UI with existing count
    updateCollectUI({
      status: 'idle',
      clip_count: data.existing || 0,
      target: 30,
      word,
      countdown: 0
    });
  });
}

function fetchCollectStatus() {
  fetch('/api/collect/status').then(r => r.json()).then(updateCollectUI).catch(() => {});
}

function updateCollectUI(s) {
  const fill    = document.getElementById('collectFill');
  const count   = document.getElementById('collectCount');
  const statusT = document.getElementById('collectStatusTxt');
  const btnRec  = document.getElementById('btnRecord');
  const btnSave = document.getElementById('btnSave');

  const pct = Math.min((s.clip_count / s.target) * 100, 100);
  fill.style.width = `${pct}%`;
  count.textContent = `${s.clip_count} / ${s.target}`;

  if (s.status === 'countdown') {
    statusT.textContent  = `Get ready… ${s.countdown}`;
    btnRec.disabled      = true;
    btnSave.style.display = 'none';
  } else if (s.status === 'recording') {
    statusT.textContent  = '● Recording clip...';
    btnRec.disabled      = true;
    btnSave.style.display = 'none';
  } else if (s.status === 'done') {
    statusT.textContent  = `✓ All ${s.target} clips done! Save to continue.`;
    btnRec.disabled      = true;
    btnSave.style.display = 'block';
  } else {
    // idle
    statusT.textContent   = s.clip_count > 0
      ? `${s.clip_count} clips done. Record more or save.`
      : 'Click Record to start';
    btnRec.disabled       = false;
    btnSave.style.display = s.clip_count > 0 ? 'block' : 'none';
  }
}

function recordClip() {
  fetch('/api/collect/record', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({})
  });
}

function saveClips() {
  fetch('/api/collect/save', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({})
  }).then(r => r.json()).then(data => {
    if (data.ok) {
      alert(`✅ Saved ${data.clips} clips for "${data.word}"!`);
      loadCollectData(); // refresh word list
      stopCollect();
    }
  });
}

function stopCollect() {
  fetch('/api/collect/stop', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}' });
  clearInterval(collectPoll);
  collectCameraOn = false;
  const feed = document.getElementById('collectFeed');
  const ph   = document.getElementById('collectPlaceholder');
  feed.style.display = 'none';
  feed.src           = '';
  ph.style.display   = 'flex';
  document.getElementById('collectControls').style.display = 'none';
  document.querySelectorAll('.word-row').forEach(r => r.classList.remove('selected'));
  selectedWord = null;
}

// ══════════════════════════════════════════════════════════
// TRAIN MODEL
// ══════════════════════════════════════════════════════════
function loadTrainData() {
  fetch('/api/model/stats').then(r => r.json()).then(d => {
    document.getElementById('statLetters').textContent = d.letter_classes;
    document.getElementById('statWords').textContent   = d.word_classes;
    document.getElementById('letterChips').innerHTML =
      d.letter_words.map(w => `<span class="chip">${w}</span>`).join('');
    document.getElementById('wordChips').innerHTML =
      d.word_words.map(w => `<span class="chip">${w}</span>`).join('');
  }).catch(() => {});
}

function runScript(type) {
  alert(`Open your terminal and run:\n\npython src/train_own_${type}_model.py\n\nThen refresh this page when done.`);
}

// ── Init ──────────────────────────────────────────────────
loadCollectData();
loadTrainData();