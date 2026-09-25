"use strict";

const copyButton = document.getElementById("copy-citation");
if (copyButton && navigator.clipboard && window.isSecureContext) {
  copyButton.hidden = false;
  copyButton.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(document.getElementById("bibtex").textContent.trim() + "\n");
      document.getElementById("copy-status").textContent = "Citation copied.";
      copyButton.textContent = "Copied";
      window.setTimeout(() => { copyButton.textContent = "Copy BibTeX"; }, 1800);
    } catch {
      document.getElementById("copy-status").textContent = "Copy was unavailable. Select the citation text or download the BibTeX file.";
      copyButton.textContent = "Select text to copy";
    }
  });
}

const paperNavigation = document.getElementById("paper-navigation");
if (paperNavigation) {
  const menuButton = document.getElementById("section-menu-toggle");
  const currentSection = document.getElementById("current-section");
  const sectionLinks = document.getElementById("paper-section-links");
  const siteHeader = document.querySelector(".site-header");
  const compactNavigation = window.matchMedia("(max-width: 1199px)");
  const sections = Array.from(sectionLinks.querySelectorAll('a[href^="#"]'))
    .map(link => ({link, target: document.getElementById(link.hash.slice(1))}))
    .filter(section => section.target);
  let currentLink = null;
  let pendingFrame = false;
  let headerHeight = -1;

  function setMenuOpen(open, restoreFocus = false) {
    paperNavigation.dataset.open = String(open);
    menuButton.setAttribute("aria-expanded", String(open));
    if (restoreFocus) menuButton.focus();
  }

  function updateCurrentSection() {
    pendingFrame = false;
    const measuredHeader = siteHeader ? siteHeader.getBoundingClientRect().height : 0;
    if (measuredHeader !== headerHeight) {
      headerHeight = measuredHeader;
      document.documentElement.style.setProperty("--site-header-height", `${headerHeight}px`);
    }

    // Only the compact bar offsets the content; an expanded menu is an overlay.
    let navigationHeight = 0;
    if (compactNavigation.matches) {
      const style = window.getComputedStyle(paperNavigation);
      navigationHeight = menuButton.getBoundingClientRect().height
        + (parseFloat(style.paddingTop) || 0) + (parseFloat(style.paddingBottom) || 0)
        + (parseFloat(style.borderTopWidth) || 0) + (parseFloat(style.borderBottomWidth) || 0);
    }
    const readingLine = headerHeight + navigationHeight + 30;
    let selected = sections[0];
    for (const section of sections) {
      const anchorMargin = parseFloat(window.getComputedStyle(section.target).scrollMarginTop) || 0;
      if (section.target.getBoundingClientRect().top <= readingLine + anchorMargin + 1) selected = section;
    }

    // The final section may be too short to reach the reading line.
    const documentHeight = Math.max(document.documentElement.scrollHeight, document.body.scrollHeight);
    if (window.scrollY > 0 && window.scrollY + window.innerHeight >= documentHeight - 2) {
      selected = sections[sections.length - 1];
    }
    if (selected && selected.link !== currentLink) {
      for (const {link} of sections) link.removeAttribute("aria-current");
      selected.link.setAttribute("aria-current", "location");
      currentSection.textContent = selected.link.textContent.trim();
      currentLink = selected.link;
    }
  }

  function scheduleUpdate() {
    if (!pendingFrame) {
      pendingFrame = true;
      window.requestAnimationFrame(updateCurrentSection);
    }
  }

  menuButton.addEventListener("click", () => {
    setMenuOpen(paperNavigation.dataset.open !== "true");
  });
  sectionLinks.addEventListener("click", event => {
    if (event.target.closest('a[href^="#"]')) setMenuOpen(false);
  });
  document.addEventListener("keydown", event => {
    if (event.key === "Escape" && paperNavigation.dataset.open === "true") {
      setMenuOpen(false, true);
    }
  });
  document.addEventListener("pointerdown", event => {
    if (!paperNavigation.contains(event.target)) setMenuOpen(false);
  });
  compactNavigation.addEventListener("change", () => {
    setMenuOpen(false);
    scheduleUpdate();
  });

  paperNavigation.dataset.enhanced = "true";
  setMenuOpen(false);
  menuButton.hidden = false;
  window.addEventListener("scroll", scheduleUpdate, {passive: true});
  window.addEventListener("resize", scheduleUpdate, {passive: true});
  window.addEventListener("hashchange", scheduleUpdate);
  if ("ResizeObserver" in window) {
    const layoutObserver = new ResizeObserver(scheduleUpdate);
    if (siteHeader) layoutObserver.observe(siteHeader);
    layoutObserver.observe(paperNavigation);
    layoutObserver.observe(document.querySelector("main") || document.body);
  }
  scheduleUpdate();
}
