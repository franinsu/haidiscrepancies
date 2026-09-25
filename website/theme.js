"use strict";

// Apply the saved/system theme before styles paint; charts keep their data unchanged.
(() => {
  const key = "haidiscrepancies-theme";
  const system = window.matchMedia("(prefers-color-scheme: dark)");
  const valid = value => value === "light" || value === "dark";
  let preference = null;
  try {
    const saved = localStorage.getItem(key);
    if (valid(saved)) preference = saved;
  } catch { /* The theme still works when browser storage is unavailable. */ }

  const palette = {
    "#8c1515": "--source-human", "#482878": "--source-chatgpt",
    "#31688e": "--source-claude", "#21918c": "--source-gemini",
    "#777c84": "--source-uniform", "#cc79a7": "--family-arithmetic",
    "#e6ab02": "--family-maze", "#6a3d9a": "--family-rooks",
    "#8c564b": "--family-minesweeper", "#17afc2": "--family-sudoku",
  };
  window.StudyTheme = {
    color(value) {
      const variable = palette[String(value).toLowerCase()];
      return variable ? `var(${variable}, ${value})` : value;
    },
  };

  function apply(theme) {
    document.documentElement.dataset.theme = theme;
    document.querySelector('meta[name="theme-color"]')?.setAttribute("content", theme === "dark" ? "#12171e" : "#ffffff");
    document.querySelectorAll("[data-theme-toggle]").forEach(button => {
      const label = theme === "dark" ? "Light mode" : "Dark mode";
      button.querySelector(".theme-label").textContent = label;
      button.setAttribute("aria-label", "Switch to " + label.toLowerCase());
      button.title = "Switch to " + label.toLowerCase();
    });
  }
  const preferredTheme = () => preference || (system.matches ? "dark" : "light");
  apply(preferredTheme());

  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("[data-theme-toggle]").forEach(button => {
      button.hidden = false;
      button.addEventListener("click", () => {
        preference = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
        try { localStorage.setItem(key, preference); } catch { /* Session-only choice. */ }
        apply(preference);
      });
    });
    apply(preferredTheme());
  });
  system.addEventListener("change", () => {
    if (!preference) apply(preferredTheme());
  });
  window.addEventListener("storage", event => {
    if (event.key !== key && event.key !== null) return;
    preference = valid(event.newValue) ? event.newValue : null;
    apply(preferredTheme());
  });
})();
