/* ------------------------------------------------------------------ */
/* Mouse-scrub background video                                      */
/* ------------------------------------------------------------------ */

(function setupPageTransitions() {
  const body = document.body;
  if (!body.classList.contains('page-transition')) return;

  requestAnimationFrame(() => body.classList.add('page-ready'));

  document.querySelectorAll('a[href]').forEach((link) => {
    const destination = new URL(link.href, window.location.href);

    const isLocalPage =
      destination.origin === window.location.origin &&
      destination.pathname !== window.location.pathname &&
      !link.hash;

    if (!isLocalPage) return;

    link.addEventListener('click', (event) => {
      if (
        event.metaKey ||
        event.ctrlKey ||
        event.shiftKey ||
        event.altKey
      ) {
        return;
      }

      event.preventDefault();

      body.classList.remove('page-ready');
      body.classList.add('page-leaving');

      setTimeout(() => {
        window.location.href = destination.href;
      }, 350);
    });
  });
})();


/* ------------------------------------------------------------------ */
/* Background video mouse scrubbing                                  */
/* ------------------------------------------------------------------ */

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

  video.addEventListener('seeked', function () {
    seeking = false;

    if (
      targetTime !== null &&
      Math.abs(video.currentTime - targetTime) > 0.01
    ) {
      trySeek();
    }
  });

  window.addEventListener('mousemove', function (e) {
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

    if (targetTime === null) {
      targetTime = video.currentTime || 0;
    }

    const offset =
      (delta / window.innerWidth) *
      SENSITIVITY *
      video.duration;

    targetTime = Math.min(
      Math.max(targetTime + offset, 0),
      video.duration
    );

    trySeek();
  });
})();


/* ------------------------------------------------------------------ */
/* Mobile menu toggle                                                */
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

  hamburger.addEventListener('click', () => {
    setOpen(!open);
  });

  mobileMenu.querySelectorAll('a').forEach((link) => {
    link.addEventListener('click', () => {
      setOpen(false);
    });
  });
})();


/* ------------------------------------------------------------------ */
/* Typewriter effect                                                  */
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
/* Action pills fade-in                                               */
/* ------------------------------------------------------------------ */

(function setupPills() {
  const pills = document.getElementById('pills');
  if (!pills) return;

  setTimeout(() => {
    pills.classList.add('visible');
  }, 400);
})();


/* ------------------------------------------------------------------ */
/* Copy email to clipboard                                            */
/* ------------------------------------------------------------------ */

(function setupCopyEmail() {
  const button = document.getElementById('copy-email');
  if (!button) return;

  button.addEventListener('click', () => {
    navigator.clipboard
      .writeText('hello@formshift.dev')
      .catch(() => {
        /* Clipboard not available, ignore */
      });
  });
})();


/* ------------------------------------------------------------------ */
/* Chat box                                                            */
/* ------------------------------------------------------------------ */

(function setupChat() {
  const overlay = document.getElementById('chat-overlay');
  const chatBox = document.getElementById('chat-box');
  const closeBtn = document.getElementById('chat-close');
  const messages = document.getElementById('chat-messages');
  const form = document.getElementById('chat-form');
  const input = document.getElementById('chat-input');

  /* File upload elements */
  const fileInput = document.getElementById('chat-file');
  const uploadButton = document.getElementById('chat-upload');

  const triggers = document.querySelectorAll('.chat-trigger');

  if (
    !overlay ||
    !chatBox ||
    !messages ||
    !form ||
    !input
  ) {
    return;
  }


  /* -------------------------------------------------------------- */
  /* Demo flows for informational buttons                           */
  /* -------------------------------------------------------------- */

  const PILL_FLOWS = {
    'Try a press advisory': [
      {
        kind: 'bot',
        text:
          "Pulling facts from the source article first — nothing gets written straight from raw text."
      },

      {
        kind: 'system',
        text:
          'PRESS ADVISORY — required fields\n' +
          '<span class="ok">✓</span> Headline\n' +
          '<span class="ok">✓</span> Dateline\n' +
          '<span class="ok">✓</span> Body (inverted pyramid)\n' +
          '<span class="ok">✓</span> Contact block'
      },

      {
        kind: 'bot',
        text:
          "Structure checks out. Paste an article below and I'll generate the real advisory."
      }
    ],

    'Try a technical brief': [
      {
        kind: 'bot',
        text:
          "Same extraction step — different schema. Briefs focus on findings, technical details and implications."
      },

      {
        kind: 'system',
        text:
          'TECHNICAL BRIEF — required fields\n' +
          '<span class="ok">✓</span> Overview\n' +
          '<span class="ok">✓</span> Key findings\n' +
          '<span class="ok">✓</span> Technical details\n' +
          '<span class="ok">✓</span> Implications'
      },

      {
        kind: 'bot',
        text:
          "Paste an article below and I'll generate the real technical brief."
      }
    ],

    'See the fact-check': [
      {
        kind: 'bot',
        text:
          "Every generated claim gets compared back against source evidence using retrieval and an NLI model."
      },

      {
        kind: 'system',
        text:
          'CLAIM CHECK\n' +
          '<span class="ok">✓ entailed</span>    Supported by source evidence\n' +
          '<span class="flag">⚠ contradicted</span>  Conflicts with source evidence\n' +
          '<span class="flag">⚠ unsupported</span>   Insufficient evidence'
      },

      {
        kind: 'bot',
        text:
          "The live backend exposes these verification results with every generated document."
      }
    ],

    'How the pipeline works': [
      {
        kind: 'bot',
        text:
          'Three core stages, always in this order:'
      },

      {
        kind: 'system',
        text:
          '1. Extract   → structured claims from the source article\n' +
          '2. Generate  → target format using the transformation contract\n' +
          '3. Verify    → evidence retrieval + NLI verification'
      },

      {
        kind: 'bot',
        text:
          "The verification stage is what separates FormShift from a normal summarizer."
      }
    ]
  };


  /* -------------------------------------------------------------- */
  /* Message helpers                                                  */
  /* -------------------------------------------------------------- */

  function addMessage(html, sender) {
    const bubble = document.createElement('div');

    bubble.className = 'chat-msg ' + sender;
    bubble.innerHTML = html;

    messages.appendChild(bubble);
    messages.scrollTop = messages.scrollHeight;

    return bubble;
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
    div.textContent = String(str ?? '');
    return div.innerHTML;
  }


  function playFlow(steps, delay) {
    steps.forEach((step) => {
      setTimeout(() => {
        addMessage(step.text, step.kind);
      }, delay);

      delay += step.kind === 'system'
        ? 700
        : 550;
    });

    return delay;
  }


  /* -------------------------------------------------------------- */
  /* Article file upload                                             */
  /* -------------------------------------------------------------- */

  if (uploadButton && fileInput) {

    uploadButton.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();

      fileInput.click();
    });


    fileInput.addEventListener('change', async () => {

      const file =
        fileInput.files && fileInput.files[0];

      if (!file) {
        return;
      }


      try {

        const text = await file.text();


        if (!text.trim()) {

          addMessage(
            '<strong>Upload failed:</strong> The selected file is empty.',
            'bot'
          );

          fileInput.value = '';
          return;
        }


        /* Put file contents into the article input */
        input.value = text;


        /* Tell user that the file was loaded */
        addMessage(
          `<strong>File loaded:</strong> ${escapeHtml(file.name)}<br>` +
          'Review the article in the input box, then press → to process it.',
          'bot'
        );


        input.focus();

      } catch (error) {

        console.error(
          'File upload error:',
          error
        );

        addMessage(
          '<strong>Upload failed:</strong> Could not read the selected file.',
          'bot'
        );
      }


      /* Reset so the same file can be selected again */
      fileInput.value = '';
    });

  }


  /* -------------------------------------------------------------- */
  /* Selected output format                                          */
  /* -------------------------------------------------------------- */

  let selectedFormat = 'press_advisory';


  /* -------------------------------------------------------------- */
  /* Start conversation                                               */
  /* -------------------------------------------------------------- */

  function startConversation(label) {

    messages.innerHTML = '';

    addMessage(
      escapeHtml(label),
      'user'
    );

    const flow = PILL_FLOWS[label];

    if (flow) {

      playFlow(flow, 500);

    } else {

      setTimeout(() => {

        addMessage(
          "Paste a source article and I'll walk it through extraction, generation, and verification.",
          'bot'
        );

      }, 500);
    }


    openChat();


    setTimeout(() => {
      input.focus();
    }, 350);
  }


  /* -------------------------------------------------------------- */
  /* Existing action pills                                           */
  /* -------------------------------------------------------------- */

  triggers.forEach((button) => {

    button.addEventListener('click', () => {

      const label = button.textContent.trim();


      if (label === 'Try a technical brief') {

        selectedFormat = 'technical_brief';

      } else if (label === 'Try a press advisory') {

        selectedFormat = 'press_advisory';
      }


      startConversation(label);
    });

  });


  /* -------------------------------------------------------------- */
  /* Close chat                                                       */
  /* -------------------------------------------------------------- */

  if (closeBtn) {
    closeBtn.addEventListener('click', closeChat);
  }

  overlay.addEventListener('click', closeChat);


  /* -------------------------------------------------------------- */
  /* Render backend result                                            */
  /* -------------------------------------------------------------- */

  function addPipelineResult(data) {

    const documentData = data.document || {};
    const verification = data.verification || {};


    const keyFindings =
      Array.isArray(documentData.key_findings)
        ? documentData.key_findings
        : [];


    const technicalDetails =
      Array.isArray(documentData.technical_details)
        ? documentData.technical_details
        : [];


    const implications =
      Array.isArray(documentData.implications)
        ? documentData.implications
        : [];


    const formatName =
      data.target_format === 'technical_brief'
        ? 'Technical Brief'
        : 'Press Advisory';


    function listHtml(items) {

      if (!items.length) {

        return '<li>Not provided in source.</li>';
      }


      return items
        .map((item) => {
          return `<li>${escapeHtml(item)}</li>`;
        })
        .join('');
    }


    const html = `
      <div class="assistant-result">

        <h3>
          ${escapeHtml(
            documentData.title ||
            'Generated Document'
          )}
        </h3>


        <p>
          <strong>Format:</strong>
          ${escapeHtml(formatName)}
        </p>


        <h4>Overview</h4>

        <p>
          ${escapeHtml(
            documentData.overview ||
            'Not provided in source.'
          )}
        </p>


        <h4>Key Findings</h4>

        <ul>
          ${listHtml(keyFindings)}
        </ul>


        <h4>Technical Details</h4>

        <ul>
          ${listHtml(technicalDetails)}
        </ul>


        <h4>Implications</h4>

        <ul>
          ${listHtml(implications)}
        </ul>


        <hr>

        <h4>Fact Check</h4>

        <p>
          <strong>Status:</strong>
          <span class="${verification.status === 'PASS' ? 'ok' : 'flag'}">
            ${escapeHtml(verification.status || 'REVIEW')}
          </span>
        </p>

        <p>
          <strong>Source coverage:</strong>
          ${escapeHtml(
            String(
              verification.source_coverage?.weighted_coverage_percent ??
              verification.source_coverage?.coverage_percent ??
              '—'
            )
          )}%
        </p>

        <p>
          <strong>Claims extracted:</strong>
          ${escapeHtml(String(verification.claim_count ?? '—'))}
        </p>

        <p>
          <strong>Contradictions:</strong>
          ${escapeHtml(
            String(
              verification.factual_support?.contradiction ??
              verification.factual_support?.contradictions ??
              0
            )
          )}
        </p>

        <p>
          <strong>Unresolved claims:</strong>
          ${escapeHtml(
            String(
              Array.isArray(verification.unresolved_claims)
                ? verification.unresolved_claims.length
                : 0
            )
          )}
        </p>

        <p>
          <strong>Repairs accepted:</strong>
          ${escapeHtml(
            String(
              Array.isArray(verification.repairs_performed)
                ? verification.repairs_performed.length
                : 0
            )
          )}
        </p>

        <p>
          <strong>Pipeline:</strong>
          ${escapeHtml(
            Array.isArray(verification.pipeline)
              ? verification.pipeline.join(' → ')
              : 'Completed'
          )}
        </p>

        <p class="muted">
          Article ID:
          ${escapeHtml(data.article_id || '—')}
        </p>

      </div>
    `;


    addMessage(
      html,
      'system'
    );
  }


  /* -------------------------------------------------------------- */
  /* REAL BACKEND SUBMISSION                                        */
  /* -------------------------------------------------------------- */

  form.addEventListener('submit', async (e) => {

    e.preventDefault();


    const value = input.value.trim();


    if (!value) {
      return;
    }


    /* Show user's article */
    addMessage(
      escapeHtml(value),
      'user'
    );


    input.value = '';


    /* Show processing message */
    const processingMessage = addMessage(
      'Running the article through extraction, generation, and verification...',
      'bot'
    );


    try {

      const response = await fetch(
        'http://127.0.0.1:8000/api/transform',
        {
          method: 'POST',

          headers: {
            'Content-Type': 'application/json'
          },

          body: JSON.stringify({
            article: value,
            target_format: selectedFormat
          })
        }
      );


      const data = await response.json();


      if (!response.ok) {

        throw new Error(
          data.detail ||
          'The backend could not process the article.'
        );
      }


      /* Remove processing message */
      if (processingMessage) {
        processingMessage.remove();
      }


      /* Render the real generated document */
      addPipelineResult(data);


    } catch (error) {

      console.error(
        'FormShift API error:',
        error
      );


      if (processingMessage) {
        processingMessage.remove();
      }


      addMessage(
        `<strong>Pipeline error</strong><br><br>` +
        `${escapeHtml(error.message)}<br><br>` +
        `Make sure the FastAPI backend and Ollama are running.`,
        'bot'
      );
    }

  });

})();