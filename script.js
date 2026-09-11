/* ------------------------------------------------------------------ */
/* Mouse-scrub background video                                        */
/* ------------------------------------------------------------------ */

(function setupPageTransitions() {
  const body = document.body;
  if (!body.classList.contains('page-transition')) return;

  requestAnimationFrame(() => body.classList.add('page-ready'));

  document.querySelectorAll('a[href]').forEach((link) => {
    const destination = new URL(link.href, window.location.href);
    const isLocalPage = destination.origin === window.location.origin &&
      destination.pathname !== window.location.pathname &&
      !link.hash;

    if (!isLocalPage) return;

    link.addEventListener('click', (event) => {
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      body.classList.remove('page-ready');
      body.classList.add('page-leaving');
      setTimeout(() => {
        window.location.href = destination.href;
      }, 350);
    });
  });
})();

(function setupScrubVideo() {
  const video = document.getElementById('bg-video');
  if (!video) return;

  const SENSITIVITY = 0.8;
  let prevX = null;
  let targetTime = null;
  let seeking = false;

  function trySeek() {
    if (!video.duration || Number.isNaN(video.duration)) return;
    if (targetTime === null) return;
    if (seeking) return;
    seeking = true;
    video.currentTime = targetTime;
  }

  video.addEventListener('seeked', function onSeeked() {
    seeking = false;
    if (targetTime !== null && Math.abs(video.currentTime - targetTime) > 0.01) {
      trySeek();
    }
  });

  window.addEventListener('mousemove', function onMouseMove(e) {
    if (!video.duration || Number.isNaN(video.duration)) {
      prevX = e.clientX;
      return;
    }

    if (prevX === null) {
      prevX = e.clientX;
      return;
    }

    const delta = e.clientX - prevX;
    prevX = e.clientX;

    if (targetTime === null) targetTime = video.currentTime || 0;

    const offset = (delta / window.innerWidth) * SENSITIVITY * video.duration;
    targetTime = Math.min(Math.max(targetTime + offset, 0), video.duration);
    trySeek();
  });
})();

/* ------------------------------------------------------------------ */
/* Mobile menu toggle                                                  */
/* ------------------------------------------------------------------ */

(function setupMobileMenu() {
  const hamburger = document.getElementById('hamburger');
  const mobileMenu = document.getElementById('mobile-menu');
  if (!hamburger || !mobileMenu) return;

  let open = false;

  function setOpen(value) {
    open = value;
    hamburger.classList.toggle('open', open);
    mobileMenu.classList.toggle('open', open);
    hamburger.setAttribute('aria-expanded', String(open));
  }

  hamburger.addEventListener('click', () => setOpen(!open));

  mobileMenu.querySelectorAll('a').forEach((link) => {
    link.addEventListener('click', () => setOpen(false));
  });
})();

/* ------------------------------------------------------------------ */
/* Typewriter effect                                                   */
/* ------------------------------------------------------------------ */

(function setupTypewriter() {
  const textEl = document.getElementById('typewriter-text');
  const cursorEl = document.getElementById('typewriter-cursor');
  if (!textEl || !cursorEl) return;

  const TEXT =
    "Paste an article. Pick a format. FORMA maps the facts and flags what it can't verify.";
  const SPEED = 38;
  const START_DELAY = 600;

  let i = 0;

  setTimeout(() => {
    const intervalId = setInterval(() => {
      i += 1;
      textEl.textContent = TEXT.slice(0, i);
      if (i >= TEXT.length) {
        clearInterval(intervalId);
        cursorEl.classList.add('hidden');
      }
    }, SPEED);
  }, START_DELAY);
})();

/* ------------------------------------------------------------------ */
/* Action pills fade-in                                                */
/* ------------------------------------------------------------------ */

(function setupPills() {
  const pills = document.getElementById('pills');
  if (!pills) return;

  setTimeout(() => {
    pills.classList.add('visible');
  }, 400);
})();

/* ------------------------------------------------------------------ */
/* Copy email to clipboard                                             */
/* ------------------------------------------------------------------ */

(function setupCopyEmail() {
  const button = document.getElementById('copy-email');
  if (!button) return;

  button.addEventListener('click', () => {
    navigator.clipboard.writeText('hello@formshift.dev').catch(() => {
      /* clipboard not available, ignore */
    });
  });
})();

/* ------------------------------------------------------------------ */
/* Chat box                                                             */
/* ------------------------------------------------------------------ */

(function setupChat() {
  const overlay = document.getElementById('chat-overlay');
  const chatBox = document.getElementById('chat-box');
  const closeBtn = document.getElementById('chat-close');
  const messages = document.getElementById('chat-messages');
  const form = document.getElementById('chat-form');
  const input = document.getElementById('chat-input');
  const triggers = document.querySelectorAll('.chat-trigger');

  if (!overlay || !chatBox || !messages || !form || !input) return;

  // Each pill plays out a short, scripted walk-through of the pipeline
  // (extract -> generate -> verify) instead of a generic support reply.
  // Message "kind" is either 'bot' (conversational) or 'system' (schema /
  // fact-check output, rendered as a mono block).
  const PILL_FLOWS = {
    'Try a press advisory': [
      { kind: 'bot', text: "Pulling facts from the source article first — nothing gets written straight from raw text." },
      {
        kind: 'system',
        text:
          'PRESS ADVISORY — required fields\n' +
          '<span class="ok">✓</span> Headline\n' +
          '<span class="ok">✓</span> Dateline\n' +
          '<span class="ok">✓</span> Body (inverted pyramid)\n' +
          '<span class="ok">✓</span> Contact block',
      },
      { kind: 'bot', text: 'Structure checks out. Draft below — the flagged line is a claim I couldn\'t trace back to the source.' },
      {
        kind: 'system',
        text:
          'FOR IMMEDIATE RELEASE\n' +
          '[Dateline] — [Headline]\n\n' +
          '...body text...\n' +
          '<span class="flag">⚠ unverified:</span> "expected to double by 2027"\n\n' +
          'Media contact: press@source.org',
      },
    ],
    'Try a technical brief': [
      { kind: 'bot', text: "Same extraction step — different schema. Briefs skip the dateline and take a findings table instead." },
      {
        kind: 'system',
        text:
          'TECHNICAL BRIEF — required fields\n' +
          '<span class="ok">✓</span> Executive summary\n' +
          '<span class="ok">✓</span> Key findings (bulleted)\n' +
          '<span class="ok">✓</span> Methodology note\n' +
          '<span class="ok">✓</span> Source references',
      },
      { kind: 'bot', text: "Draft's ready. Every bullet under Key Findings links back to a specific sentence in the source — nothing summarized without a citation." },
    ],
    'See the fact-check': [
      { kind: 'bot', text: "Every generated claim gets compared back against the facts extracted from the source, using an NLI model — entailed, contradicted, or unsupported." },
      {
        kind: 'system',
        text:
          'CLAIM CHECK\n' +
          '<span class="ok">✓ entailed</span>    "revenue rose 12% year-over-year"\n' +
          '<span class="ok">✓ entailed</span>    "the facility opened in March"\n' +
          '<span class="flag">⚠ unsupported</span>  "analysts expect the trend to continue"',
      },
      { kind: 'bot', text: 'Unsupported claims get flagged in the output rather than silently removed — you decide whether to keep, cut, or verify them.' },
    ],
    'How the pipeline works': [
      { kind: 'bot', text: 'Three steps, always in this order:' },
      {
        kind: 'system',
        text:
          '1. Extract   →  structured facts from the source article\n' +
          '2. Generate  →  draft in the target format\'s schema\n' +
          '3. Verify    →  NLI check against the extracted facts',
      },
      { kind: 'bot', text: "Nothing skips step 3 — that's the part a generic \"summarize this\" prompt doesn't do." },
    ],
  };

  function addMessage(html, sender) {
    const bubble = document.createElement('div');
    bubble.className = 'chat-msg ' + sender;
    bubble.innerHTML = html;
    messages.appendChild(bubble);
    messages.scrollTop = messages.scrollHeight;
  }

  function openChat() {
    overlay.classList.add('open');
    chatBox.classList.add('open');
  }

  function closeChat() {
    overlay.classList.remove('open');
    chatBox.classList.remove('open');
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  function playFlow(steps, delay) {
    steps.forEach((step) => {
      setTimeout(() => addMessage(step.text, step.kind), delay);
      delay += step.kind === 'system' ? 700 : 550;
    });
    return delay;
  }

  function startConversation(label) {
    messages.innerHTML = '';
    addMessage(escapeHtml(label), 'user');

    const flow = PILL_FLOWS[label];
    if (flow) {
      playFlow(flow, 500);
    } else {
      setTimeout(
        () => addMessage('Paste a source article and I\'ll walk it through extraction, generation, and verification.', 'bot'),
        500
      );
    }

    openChat();
    setTimeout(() => input.focus(), 350);
  }

  triggers.forEach((button) => {
    button.addEventListener('click', () => startConversation(button.textContent.trim()));
  });

  closeBtn.addEventListener('click', closeChat);
  overlay.addEventListener('click', closeChat);

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const value = input.value.trim();
    if (!value) return;

    addMessage(escapeHtml(value), 'user');
    input.value = '';

    setTimeout(() => {
      addMessage("Got it — drop in the full article text and I'll run it through the pipeline: extract, generate, verify.", 'bot');
    }, 500);
  });
})();
