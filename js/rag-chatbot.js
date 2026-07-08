/* ============================================================
   Taliqa Muhib — Portfolio RAG Assistant
   Client-side Retrieval-Augmented answering:
     1. Loads a curated knowledge base (data/knowledge-base.json)
     2. Builds TF-IDF vectors for each document
     3. Retrieves top matches by cosine similarity
     4. Synthesises a grounded answer from the most relevant
        sentences and cites the sources it used.
   No external APIs, no keys, no hallucinated content —
   answers are drawn only from the knowledge base.
   ============================================================ */
(function () {
  "use strict";

  const KB_URL = "data/knowledge-base.json";
  const TOP_K = 3;
  const MIN_SCORE = 0.06;

  const STOPWORDS = new Set(
    ("a,an,the,and,or,but,if,then,else,for,of,to,in,on,at,by,with,from,as,is," +
      "are,was,were,be,been,being,do,does,did,have,has,had,i,you,he,she,it,we," +
      "they,me,him,her,them,my,your,his,its,our,their,this,that,these,those,what," +
      "which,who,whom,how,when,where,why,can,could,should,would,will,shall,may," +
      "might,must,about,tell,me,know,want,like,please,give,show,does,do,any,some," +
      "there,here,into,over,than,too,very,just,also").split(",")
  );

  /* ---------- Text utilities ---------- */
  function stem(word) {
    return word
      .replace(/(ing|edly|ed|ly|ies|ment|ness|tion|sions|sion|ers|er|es|s)$/g, "")
      .replace(/i$/g, "y");
  }

  function tokenize(text) {
    return (text || "")
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter((w) => w.length > 1 && !STOPWORDS.has(w))
      .map(stem);
  }

  function splitSentences(text) {
    return (text || "")
      .split(/(?<=[.!?])\s+/)
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
  }

  /* ---------- TF-IDF index ---------- */
  const Index = {
    docs: [],
    idf: {},
    vectors: [],
    ready: false,

    build(documents) {
      this.docs = documents;
      const df = {};
      const tfList = [];

      documents.forEach((doc) => {
        const tokens = tokenize(doc.title + " " + doc.title + " " + (doc.tags || "") + " " + doc.text);
        const tf = {};
        tokens.forEach((t) => (tf[t] = (tf[t] || 0) + 1));
        tfList.push({ tf, len: tokens.length });
        Object.keys(tf).forEach((t) => (df[t] = (df[t] || 0) + 1));
      });

      const N = documents.length;
      Object.keys(df).forEach((t) => {
        this.idf[t] = Math.log((N + 1) / (df[t] + 1)) + 1;
      });

      this.vectors = tfList.map(({ tf, len }) => {
        const vec = {};
        let norm = 0;
        Object.keys(tf).forEach((t) => {
          const w = (tf[t] / len) * (this.idf[t] || 0);
          vec[t] = w;
          norm += w * w;
        });
        return { vec, norm: Math.sqrt(norm) || 1 };
      });

      this.ready = true;
    },

    queryVector(tokens) {
      const tf = {};
      tokens.forEach((t) => (tf[t] = (tf[t] || 0) + 1));
      const vec = {};
      let norm = 0;
      const len = tokens.length || 1;
      Object.keys(tf).forEach((t) => {
        const w = (tf[t] / len) * (this.idf[t] || 0);
        vec[t] = w;
        norm += w * w;
      });
      return { vec, norm: Math.sqrt(norm) || 1 };
    },

    search(query) {
      const tokens = tokenize(query);
      if (!tokens.length) return [];
      const q = this.queryVector(tokens);

      const scored = this.vectors.map((d, i) => {
        let dot = 0;
        Object.keys(q.vec).forEach((t) => {
          if (d.vec[t]) dot += q.vec[t] * d.vec[t];
        });
        return { i, score: dot / (q.norm * d.norm) };
      });

      return scored
        .filter((s) => s.score > MIN_SCORE)
        .sort((a, b) => b.score - a.score)
        .slice(0, TOP_K)
        .map((s) => ({ doc: this.docs[s.i], score: s.score, tokens }));
    },
  };

  /* ---------- Answer synthesis (extractive, grounded) ---------- */
  function bestSentences(doc, queryTokens, max) {
    const qset = new Set(queryTokens);
    const sentences = splitSentences(doc.text);
    const scored = sentences.map((s, idx) => {
      const st = tokenize(s);
      let overlap = 0;
      st.forEach((t) => {
        if (qset.has(t)) overlap += 1;
      });
      return { s, score: overlap + (idx === 0 ? 0.5 : 0), idx };
    });
    scored.sort((a, b) => b.score - a.score || a.idx - b.idx);
    const picked = scored.slice(0, max).filter((x) => x.score > 0);
    if (!picked.length) picked.push(scored[0]);
    return picked.sort((a, b) => a.idx - b.idx).map((x) => x.s);
  }

  function synthesize(query, results) {
    if (!results.length) {
      return {
        text:
          "I can only answer questions about Taliqa Muhib's work — her HealthTech and " +
          "data science projects, skills, research, education, and how to get in touch. " +
          "Try asking about the ADHD pathway dashboard, EEG research, her skills, or contact details.",
        sources: [],
      };
    }

    const top = results[0];
    const parts = [];

    // Primary doc: up to 2 sentences.
    bestSentences(top.doc, top.tokens, 2).forEach((s) => parts.push(s));

    // Supporting doc if clearly relevant and distinct.
    if (results[1] && results[1].score > MIN_SCORE * 1.5 && results[1].doc.category === top.doc.category) {
      const extra = bestSentences(results[1].doc, top.tokens, 1)[0];
      if (extra && !parts.includes(extra)) parts.push(extra);
    }

    return {
      text: parts.join(" "),
      sources: results.map((r) => r.doc.title),
    };
  }

  /* ---------- Intent shortcuts ---------- */
  function shortcut(query) {
    const q = query.toLowerCase().trim();
    if (/^(hi|hey|hello|salam|assalam|good (morning|afternoon|evening))\b/.test(q)) {
      return {
        text:
          "Hi! I'm Taliqa's portfolio assistant. Ask me about her HealthTech projects, " +
          "skills, research, education, or how to contact her.",
        sources: [],
      };
    }
    if (/(thank|thanks|cheers)/.test(q)) {
      return { text: "You're welcome! Anything else you'd like to know about Taliqa's work?", sources: [] };
    }
    return null;
  }

  /* ---------- UI ---------- */
  const SUGGESTIONS = [
    "What are her HealthTech projects?",
    "Tell me about the EEG research",
    "What are her skills?",
    "Is she available for remote roles?",
    "How can I contact her?",
  ];

  function el(tag, cls, html) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
    );
  }

  function buildWidget() {
    const root = el("div", "rag-widget");
    root.innerHTML = `
      <button class="rag-launcher" id="rag-launcher" aria-label="Open portfolio assistant" aria-expanded="false">
        <svg class="rag-ic-chat" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" aria-hidden="true">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M8 10h8M8 13h5" stroke-linecap="round"/>
        </svg>
        <span class="rag-launcher-label">Ask about my work</span>
      </button>

      <section class="rag-panel" id="rag-panel" role="dialog" aria-modal="false"
        aria-label="Portfolio assistant" hidden>
        <header class="rag-header">
          <div class="rag-header-info">
            <span class="rag-avatar" aria-hidden="true">TM</span>
            <div>
              <p class="rag-title">Portfolio Assistant</p>
              <p class="rag-sub"><span class="rag-dot"></span>Grounded in Taliqa's work</p>
            </div>
          </div>
          <button class="rag-close" id="rag-close" aria-label="Close assistant">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" aria-hidden="true">
              <path d="M6 6l12 12M18 6L6 18" stroke-linecap="round"/>
            </svg>
          </button>
        </header>

        <div class="rag-messages" id="rag-messages" role="log" aria-live="polite" aria-atomic="false"></div>

        <div class="rag-suggestions" id="rag-suggestions"></div>

        <form class="rag-input" id="rag-form" autocomplete="off">
          <input type="text" id="rag-text" name="rag-text"
            placeholder="Ask about projects, skills, research…"
            aria-label="Type your question" maxlength="240" />
          <button type="submit" aria-label="Send">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
              <path d="M4 12l16-8-6 16-3-7-7-1z" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </button>
        </form>
        <p class="rag-disclaimer">Answers come only from Taliqa's portfolio content.</p>
      </section>`;
    document.body.appendChild(root);
    return root;
  }

  /* ---------- Controller ---------- */
  function init(documents) {
    Index.build(documents);

    buildWidget();
    const launcher = document.getElementById("rag-launcher");
    const panel = document.getElementById("rag-panel");
    const closeBtn = document.getElementById("rag-close");
    const messages = document.getElementById("rag-messages");
    const form = document.getElementById("rag-form");
    const input = document.getElementById("rag-text");
    const suggestBox = document.getElementById("rag-suggestions");
    let greeted = false;

    function scrollDown() {
      messages.scrollTop = messages.scrollHeight;
    }

    function addMessage(role, htmlContent, sources) {
      const row = el("div", `rag-msg rag-msg-${role}`);
      const bubble = el("div", "rag-bubble");
      bubble.innerHTML = htmlContent;
      if (sources && sources.length) {
        const src = el("div", "rag-sources");
        src.innerHTML =
          '<span class="rag-sources-label">Sources</span>' +
          sources.map((s) => `<span class="rag-source">${escapeHtml(s)}</span>`).join("");
        bubble.appendChild(src);
      }
      row.appendChild(bubble);
      messages.appendChild(row);
      scrollDown();
    }

    function typing() {
      const row = el("div", "rag-msg rag-msg-bot rag-typing-row");
      row.innerHTML =
        '<div class="rag-bubble rag-typing"><span></span><span></span><span></span></div>';
      messages.appendChild(row);
      scrollDown();
      return row;
    }

    function renderSuggestions() {
      suggestBox.innerHTML = "";
      SUGGESTIONS.forEach((q) => {
        const chip = el("button", "rag-chip", escapeHtml(q));
        chip.type = "button";
        chip.addEventListener("click", () => handle(q));
        suggestBox.appendChild(chip);
      });
    }

    function answer(query) {
      const sc = shortcut(query);
      if (sc) return sc;
      const results = Index.search(query);
      return synthesize(query, results);
    }

    function handle(query) {
      query = query.trim();
      if (!query) return;
      addMessage("user", escapeHtml(query));
      input.value = "";
      const t = typing();
      const delay = 320 + Math.random() * 260;
      setTimeout(() => {
        t.remove();
        const res = answer(query);
        addMessage("bot", escapeHtml(res.text), res.sources);
      }, delay);
    }

    function openPanel() {
      panel.hidden = false;
      launcher.setAttribute("aria-expanded", "true");
      document.body.classList.add("rag-open");
      if (!greeted) {
        greeted = true;
        renderSuggestions();
        setTimeout(() => {
          addMessage(
            "bot",
            "Hi! I'm Taliqa's portfolio assistant. I can tell you about her HealthTech " +
              "projects, skills, research, and background. What would you like to know?"
          );
        }, 200);
      }
      setTimeout(() => input.focus(), 250);
    }

    function closePanel() {
      panel.hidden = true;
      launcher.setAttribute("aria-expanded", "false");
      document.body.classList.remove("rag-open");
      launcher.focus();
    }

    launcher.addEventListener("click", () =>
      panel.hidden ? openPanel() : closePanel()
    );
    closeBtn.addEventListener("click", closePanel);
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      handle(input.value);
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !panel.hidden) closePanel();
    });
  }

  /* ---------- Bootstrap ---------- */
  function boot() {
    fetch(KB_URL)
      .then((r) => {
        if (!r.ok) throw new Error("KB load failed: " + r.status);
        return r.json();
      })
      .then((data) => init(data.documents || []))
      .catch((err) => {
        // Fail silently in production; log for debugging.
        console.warn("[Portfolio assistant] Could not load knowledge base.", err);
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
