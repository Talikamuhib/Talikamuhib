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
