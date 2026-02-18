  let resumeUploaded = false;
  let modalResumeUploaded = false;

  // Upload resume for chatbot
  async function handleUpload(input) {
    const file = input.files[0];
    if (!file) return;

    const zone = document.getElementById('uploadZone');
    const content = document.getElementById('uploadContent');
    const success = document.getElementById('fileSuccess');
    const fileName = document.getElementById('uploadedFileName');

    const formData = new FormData();
    formData.append('resume', file);

    try {
      const res = await fetch('/upload', { method: 'POST', body: formData });
      const data = await res.json();

      if (data.error) {
        alert(data.error);
        return;
      }

      zone.classList.add('has-file');
      content.style.display = 'none';
      success.classList.add('show');
      fileName.textContent = file.name;
      resumeUploaded = true;

      addMessage('ai', `✓ Got it! I've loaded **${file.name}**. What would you like to know?`);
      document.getElementById('emptyState').style.display = 'none';

    } catch(e) {
      console.error(e);
    }
  }

  // Modal resume upload
  async function handleModalUpload(input) {
    const file = input.files[0];
    if (!file) return;

    const zone = document.getElementById('modalUploadZone');
    const label = document.getElementById('modalUploadLabel');
    const icon = document.getElementById('modalUploadIcon');
    const check = document.getElementById('modalFileCheck');

    const formData = new FormData();
    formData.append('resume', file);

    try {
      const res = await fetch('/upload', { method: 'POST', body: formData });
      const data = await res.json();

      if (!data.error) {
        zone.classList.add('has-file');
        label.textContent = file.name;
        icon.textContent = '📄';
        check.style.display = 'flex';
        modalResumeUploaded = true;
      }
    } catch(e) { console.error(e); }
  }

  async function sendMessage() {
    const input = document.getElementById('chatInput');
    const q = input.value.trim();
    if (!q) return;

    document.getElementById('emptyState').style.display = 'none';
    addMessage('user', q);
    input.value = '';

    const typingEl = addTyping();

    try {
      const res = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q })
      });
      const data = await res.json();
      typingEl.remove();
      addMessage('ai', data.answer || 'Sorry, something went wrong.');
    } catch(e) {
      typingEl.remove();
      addMessage('ai', 'Connection error. Please check your server.');
    }
  }

  function askQuestion(q) {
    document.getElementById('chatInput').value = q;
    sendMessage();
  }

  function addMessage(type, text) {
    const chat = document.getElementById('chatSection');
    const div = document.createElement('div');
    div.className = `message ${type}`;

    const avatar = `<div class="avatar ${type === 'ai' ? 'ai' : 'user-av'}">${type === 'ai' ? '✦' : 'U'}</div>`;
    const bubble = `<div class="bubble">${text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')}</div>`;

    div.innerHTML = type === 'ai' ? avatar + bubble : bubble + avatar;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    return div;
  }

  function addTyping() {
    const chat = document.getElementById('chatSection');
    const div = document.createElement('div');
    div.className = 'message ai';
    div.innerHTML = `
      <div class="avatar ai">✦</div>
      <div class="bubble typing">
        <span></span><span></span><span></span>
      </div>`;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    return div;
  }

  function openModal() { document.getElementById('modalOverlay').classList.add('open'); }
  function closeModal() {
    document.getElementById('modalOverlay').classList.remove('open');
    document.getElementById('scoreResult').classList.remove('show');
  }

  async function checkATS() {
    const jd = document.getElementById('jdInput').value.trim();
    if (!jd) { alert('Please paste a job description.'); return; }

    const btn = document.getElementById('checkBtn');
    btn.disabled = true;
    btn.innerHTML = '<div class="spinner"></div>';

    try {
      const res = await fetch('/ats-score', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_description: jd })
      });
      const data = await res.json();

      if (data.error) {
        alert(data.error);
        return;
      }

      displayScore(data);
    } catch(e) {
      alert('Error connecting to server.');
    } finally {
      btn.disabled = false;
      btn.innerHTML = '<span>Check ATS Score</span>';
    }
  }

  function displayScore(data) {
    const score = data['ATS Score'] || 0;
    const matched = data['Matched Skills'] || [];
    const missing = data['Missing Skills'] || [];
    const summary = data['Summary'] || '';

    // Show result
    document.getElementById('scoreResult').classList.add('show');

    // Score number
    document.getElementById('scoreNumber').textContent = score + '%';

    // Sub scores
    document.getElementById('skillScoreVal').textContent = (data['Skill Match Score'] || 0) + '%';
    document.getElementById('semScoreVal').textContent = (data['Semantic Score'] || 0) + '%';
    document.getElementById('expScoreVal').textContent = (data['Experience Score'] || 0) + '%';

    // Ring color
    const ring = document.getElementById('ringFill');
    const scoreNum = document.getElementById('scoreNumber');
    if (score >= 70) { ring.style.stroke = '#22d3a5'; scoreNum.style.color = '#22d3a5'; }
    else if (score >= 50) { ring.style.stroke = '#f59e0b'; scoreNum.style.color = '#f59e0b'; }
    else { ring.style.stroke = '#ff5f7e'; scoreNum.style.color = '#ff5f7e'; }

    // Animate ring
    const circumference = 2 * Math.PI * 54;
    const offset = circumference - (score / 100) * circumference;
    setTimeout(() => { ring.style.strokeDashoffset = offset; }, 100);

    // Chips
    const matchedEl = document.getElementById('matchedChips');
    const missingEl = document.getElementById('missingChips');
    matchedEl.innerHTML = matched.map(s => `<span class="chip green">${s}</span>`).join('');
    missingEl.innerHTML = missing.map(s => `<span class="chip red">${s}</span>`).join('');

    // Summary
    if (summary) {
      const sumBox = document.getElementById('summaryBox');
      sumBox.textContent = summary;
      sumBox.style.display = 'block';
    }

    document.querySelector('.modal').scrollTop = 9999;
  }

// comparison code js
  document.getElementById("compareBtn").addEventListener("click", function(e) {
    e.preventDefault();

    // Hide main content
    document.querySelector("main").style.display = "none";

    // Show comparison section
    document.getElementById("comparisonPage").style.display = "block";
});

// ======================== comparison code ===========================
  let comparisonId = null;
  let candidates = [];

  async function handleMultiUpload(input) {
    const files = Array.from(input.files);
    if (!files.length) return;

    // Create session if needed
    if (!comparisonId) {
      const res = await fetch('/comparison/create', { method: 'POST' });
      const data = await res.json();
      comparisonId = data.comparison_id;
    }

    const formData = new FormData();
    files.forEach(f => formData.append('files[]', f));

    try {
      const res = await fetch(`/comparison/${comparisonId}/upload`, { method: 'POST', body: formData });
      const data = await res.json();

      files.forEach(f => {
        if (!candidates.find(c => c.name === f.name)) {
          candidates.push({ name: f.name });
          addCandidateCard(f.name);
        }
      });

      if (candidates.length > 0) {
        document.getElementById('emptyChat').style.display = 'none';
        document.getElementById('chatPills').style.display = 'flex';
        document.getElementById('suggestionSection').style.display = 'block';
        loadSuggestions();
      }

    } catch(e) { console.error(e); }
  }

  function addCandidateCard(name) {
    const grid = document.getElementById('candidatesGrid');
    const card = document.createElement('div');
    card.className = 'candidate-card';
    card.id = `card-${name}`;
    card.innerHTML = `
      <div class="candidate-icon">📄</div>
      <div class="candidate-info">
        <div class="candidate-name">${name}</div>
        <div class="candidate-status">
          <div class="status-dot"></div> Ready
        </div>
      </div>
      <button class="remove-btn" onclick="removeCandidate('${name}')">×</button>
    `;
    grid.appendChild(card);
  }

  function removeCandidate(name) {
    candidates = candidates.filter(c => c.name !== name);
    const el = document.getElementById(`card-${name}`);
    if (el) el.remove();
  }

  async function loadSuggestions() {
    if (!comparisonId) return;
    try {
      const res = await fetch(`/comparison/${comparisonId}/suggestions`);
      const data = await res.json();
      if (data.suggestions && data.suggestions.length > 0) {
        const pills = document.getElementById('chatPills');
        pills.innerHTML = data.suggestions.map(s =>
          `<span class="chat-pill" onclick="sendComparison('${s}')">${s}</span>`
        ).join('');
      }
    } catch(e) {}
  }

  async function sendComparison(msg) {
    const input = document.getElementById('compInput');
    const message = msg || input.value.trim();
    if (!message) return;
    if (!comparisonId) { alert('Please upload resumes first.'); return; }

    document.getElementById('emptyChat').style.display = 'none';
    addMessage('user', message);
    if (!msg) input.value = '';

    const typing = addTyping();

    try {
      const res = await fetch(`/comparison/${comparisonId}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
      });
      const data = await res.json();
      typing.remove();
      addMessage('ai', data.response || 'Sorry, something went wrong.');
    } catch(e) {
      typing.remove();
      addMessage('ai', 'Connection error. Please check your server.');
    }
  }

  function addMessage(type, text) {
    const chat = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = `message ${type}`;
    const avatar = `<div class="avatar ${type === 'ai' ? 'ai' : 'user-av'}">${type === 'ai' ? '✦' : 'U'}</div>`;
    const bubble = `<div class="bubble">${text}</div>`;
    div.innerHTML = type === 'ai' ? avatar + bubble : bubble + avatar;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    return div;
  }

  function addTyping() {
    const chat = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = 'message ai';
    div.innerHTML = `
      <div class="avatar ai">✦</div>
      <div class="bubble typing">
        <span></span><span></span><span></span>
      </div>`;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    return div;
  }


