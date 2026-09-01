/* ============================================================
   Taliqa Muhib — HealthTech Portfolio interactions
   ============================================================ */
(function () {
  "use strict";

  const $ = (sel, ctx = document) => ctx.querySelector(sel);
  const $$ = (sel, ctx = document) => Array.from(ctx.querySelectorAll(sel));

  /* ---------- Navbar: scrolled state ---------- */
  const navbar = $("#navbar");
  const onScroll = () => {
    if (window.scrollY > 20) navbar.classList.add("scrolled");
    else navbar.classList.remove("scrolled");

    // Back to top visibility
    backToTop.classList.toggle("show", window.scrollY > 500);
  };

  /* ---------- Mobile navigation ---------- */
  const navToggle = $("#nav-toggle");
  const navLinks = $("#nav-links");

  const closeMenu = () => {
    navLinks.classList.remove("open");
    navToggle.setAttribute("aria-expanded", "false");
  };

  navToggle.addEventListener("click", () => {
    const open = navLinks.classList.toggle("open");
    navToggle.setAttribute("aria-expanded", String(open));
  });

  $$("#nav-links a").forEach((a) => a.addEventListener("click", closeMenu));

  document.addEventListener("click", (e) => {
    if (
      navLinks.classList.contains("open") &&
      !navLinks.contains(e.target) &&
      !navToggle.contains(e.target)
    ) {
      closeMenu();
    }
  });

  /* ---------- Active nav link on scroll (scrollspy) ---------- */
  const sections = $$("main section[id]");
  const navAnchors = $$('#nav-links a[href^="#"]');

  const spy = () => {
    let current = "";
    const offset = 120;
    sections.forEach((sec) => {
      if (window.scrollY >= sec.offsetTop - offset) current = sec.id;
    });
    navAnchors.forEach((a) => {
      a.classList.toggle("active", a.getAttribute("href") === `#${current}`);
    });
  };

  /* ---------- Reveal on scroll ---------- */
  const revealEls = $$(".reveal");
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver(
      (entries, obs) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("visible");
            obs.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12, rootMargin: "0px 0px -40px 0px" }
    );
    revealEls.forEach((el) => io.observe(el));
  } else {
    revealEls.forEach((el) => el.classList.add("visible"));
  }

  /* ---------- Animated counters ---------- */
  const counters = $$(".stat-num");
  const animateCount = (el) => {
    const target = parseInt(el.dataset.count, 10) || 0;
    const suffix = el.dataset.suffix || "";
    const duration = 1200;
    const start = performance.now();
    const step = (now) => {
      const p = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      el.textContent = Math.round(eased * target) + (p === 1 ? suffix : "");
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };

  if ("IntersectionObserver" in window && counters.length) {
    const co = new IntersectionObserver(
      (entries, obs) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            animateCount(entry.target);
            obs.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.6 }
    );
    counters.forEach((c) => co.observe(c));
  } else {
    counters.forEach((c) => (c.textContent = (c.dataset.count || "0") + (c.dataset.suffix || "")));
  }

  /* ---------- Project filters ---------- */
  const filterBtns = $$(".filter-btn");
  const projectCards = $$("#project-grid .project-card");

  filterBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      filterBtns.forEach((b) => {
        b.classList.remove("is-active");
        b.setAttribute("aria-selected", "false");
      });
      btn.classList.add("is-active");
      btn.setAttribute("aria-selected", "true");

      const filter = btn.dataset.filter;
      projectCards.forEach((card) => {
        const tags = card.dataset.tags || "";
        const match = filter === "all" || tags.includes(filter);
        card.classList.toggle("hide", !match);
      });
    });
  });

  /* ---------- Expandable project details ---------- */
  $$(".toggle-details").forEach((btn) => {
    btn.addEventListener("click", () => {
      const card = btn.closest(".project-card");
      const open = card.classList.toggle("open");
      btn.setAttribute("aria-expanded", String(open));
      btn.textContent = open ? "Hide" : "Details";
    });
  });

  /* ---------- Other work toggle ---------- */
  const otherToggle = $("#other-toggle");
  const otherGrid = $("#other-grid");
  if (otherToggle && otherGrid) {
    otherToggle.addEventListener("click", () => {
      const open = otherGrid.hasAttribute("hidden");
      if (open) otherGrid.removeAttribute("hidden");
      else otherGrid.setAttribute("hidden", "");
      otherToggle.setAttribute("aria-expanded", String(open));
    });
  }

  /* ---------- Skills tabs ---------- */
  const skillTabs = $$(".skill-tab");
  const skillPanels = $$(".skill-panel");
  skillTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      skillTabs.forEach((t) => {
        t.classList.remove("is-active");
        t.setAttribute("aria-selected", "false");
      });
      tab.classList.add("is-active");
      tab.setAttribute("aria-selected", "true");

      const key = tab.dataset.skill;
      skillPanels.forEach((panel) => {
        if (panel.dataset.panel === key) {
          panel.removeAttribute("hidden");
          panel.classList.add("is-active");
        } else {
          panel.setAttribute("hidden", "");
          panel.classList.remove("is-active");
        }
      });
    });
  });

  /* ---------- Neural brain interaction ---------- */
  const brainScene = document.querySelector(".brain-scene");
  const brainHotspots = $$(".brain-hotspot");
  const miniPanelHead = document.querySelector("#mini-panel-head");
  const miniPanelTitle = document.querySelector("#mini-panel-title");
  const miniPanelCopy = document.querySelector("#mini-panel-copy");
  const customCursor = document.querySelector(".custom-cursor");

  const regionMeta = {
    eeg: {
      label: "EEG",
      title: "Signal rhythm",
      copy: "Electrical patterns and temporal dynamics"
    },
    fmri: {
      label: "fMRI",
      title: "Functional connectivity",
      copy: "Activation patterns in motion"
    },
    graph: {
      label: "GRAPH",
      title: "Network topology",
      copy: "Connectivity structure and spectral flow"
    },
    imaging: {
      label: "IMAGING",
      title: "Clinical scan layer",
      copy: "Attention maps and anatomical detail"
    },
    neuroai: {
      label: "NEUROAI",
      title: "Biological to artificial",
      copy: "Brain-inspired inference systems"
    },
    connectivity: {
      label: "CONNECTIVITY",
      title: "Connectome map",
      copy: "Neural pathways across regions"
    }
  };

  const activateBrainRegion = (region) => {
    if (!brainScene || !regionMeta[region]) return;
    brainScene.dataset.active = region;
    brainScene.classList.add("is-active");

    if (miniPanelHead) miniPanelHead.textContent = regionMeta[region].label;
    if (miniPanelTitle) miniPanelTitle.textContent = regionMeta[region].title;
    if (miniPanelCopy) miniPanelCopy.textContent = regionMeta[region].copy;
  };

  const clearBrainRegion = () => {
    if (!brainScene) return;
    brainScene.classList.remove("is-active");
    brainScene.dataset.active = "fmri";
    if (miniPanelHead) miniPanelHead.textContent = "fMRI";
    if (miniPanelTitle) miniPanelTitle.textContent = "Functional connectivity";
    if (miniPanelCopy) miniPanelCopy.textContent = "Activation patterns in motion";
  };

  brainHotspots.forEach((hotspot) => {
    const region = hotspot.dataset.region;
    hotspot.addEventListener("mouseenter", () => activateBrainRegion(region));
    hotspot.addEventListener("focus", () => activateBrainRegion(region));
    hotspot.addEventListener("mouseleave", clearBrainRegion);
    hotspot.addEventListener("blur", clearBrainRegion);
    hotspot.addEventListener("click", () => activateBrainRegion(region));
  });

  document.querySelectorAll(".science-module").forEach((module) => {
    const region = module.dataset.module;
    if (!region) return;
    module.addEventListener("mouseenter", () => activateBrainRegion(region));
    module.addEventListener("mouseleave", clearBrainRegion);
    module.addEventListener("focusin", () => activateBrainRegion(region));
    module.addEventListener("focusout", clearBrainRegion);
  });

  if (brainScene) {
    brainScene.addEventListener("pointermove", (event) => {
      const rect = brainScene.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width - 0.5) * 2;
      const y = ((event.clientY - rect.top) / rect.height - 0.5) * 2;
      brainScene.style.setProperty("--brain-shift-x", `${x * 14}px`);
      brainScene.style.setProperty("--brain-shift-y", `${y * 12}px`);
      brainScene.style.setProperty("--brain-tilt", `${x * 6}deg`);
      brainScene.style.setProperty("--brain-glow", `${1 + Math.abs(x) * 0.2}`);
    });

    brainScene.addEventListener("pointerleave", () => {
      brainScene.style.setProperty("--brain-shift-x", "0px");
      brainScene.style.setProperty("--brain-shift-y", "0px");
      brainScene.style.setProperty("--brain-tilt", "0deg");
      brainScene.style.setProperty("--brain-glow", "1");
    });
  }

  if (customCursor) {
    document.addEventListener("pointermove", (event) => {
      customCursor.style.left = `${event.clientX}px`;
      customCursor.style.top = `${event.clientY}px`;
    });
    document.addEventListener("pointerdown", () => customCursor.classList.add("active"));
    document.addEventListener("pointerup", () => customCursor.classList.remove("active"));
  }

  /* ---------- Back to top ---------- */
  const backToTop = $("#back-to-top");
  backToTop.addEventListener("click", () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  /* ---------- Scroll listener (throttled) ---------- */
  let ticking = false;
  window.addEventListener("scroll", () => {
    if (!ticking) {
      window.requestAnimationFrame(() => {
        onScroll();
        spy();
        ticking = false;
      });
      ticking = true;
    }
  });

  // Initial state
  onScroll();
  spy();

  /* ---------- Update footer year automatically if needed ---------- */
  // Keeps the printed year current while preserving the branded label.
})();
