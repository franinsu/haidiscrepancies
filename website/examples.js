"use strict";

const moduleDescriptions = {
  cue: "Gold outlines draw attention to cells without changing the rules. They mark both valid answers and invalid choices.",
  spatial: "The board is reflected, preserving the puzzle’s solutions. The same solution number follows each answer to its reflected position.",
  formulation: "The same four numbers appear in reverse order. The valid expressions are unchanged.",
  pair: "Try the first puzzle before the target, then compare with the target alone. A related first puzzle is a transformed version of the target; an unrelated one is a different puzzle from the same family.",
  transfer: "First try a primer puzzle with one valid answer, then a target with several. Compare with the same target presented alone. In the study, participants attempted the primer without seeing its solution."
};
const { board, task, solutionText, familyNames } = window.PuzzleBoards;
const main = document.getElementById("example-cards");
const status = document.getElementById("explorer-status");

function element(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}
function button(text, className, action) {
  const node = element("button", className, text);
  node.type = "button";
  node.addEventListener("click", action);
  return node;
}
class ExampleCard {
  constructor(category, data, isModule, order, initialIndex = -1, revealInitially = false) {
    this.category = category;
    this.data = data;
    this.isModule = isModule;
    this.exampleIndex = -1;
    this.variantIndex = isModule ? 1 : 0;
    this.solutionIndex = 0;
    this.revealed = false;
    this.section = element("section", "example-card");
    this.section.id = category.id;
    this.section.setAttribute("aria-labelledby", category.id + "-title");
    const header = element("div", "example-card-heading");
    const heading = element("div");
    heading.append(element("p", "section-label", String(order + 1).padStart(2, "0") + " / " + (isModule ? "Stimulus experiment" : "Puzzle family")));
    const title = element("h2", "", category.label);
    title.id = category.id + "-title";
    heading.append(title);
    const random = button(isModule ? "Another example ↻" : "Another puzzle ↻", "button shuffle-button", () => this.randomize(true));
    random.setAttribute("aria-label", isModule ? "Another " + category.label + " example" : "Another " + category.label + " puzzle");
    header.append(heading, random);
    this.section.append(header);
    if (isModule) this.section.append(element("p", "module-description", moduleDescriptions[category.id]));
    this.meta = element("p", "example-meta");
    this.variants = element("div", "variant-buttons");
    this.variants.setAttribute("role", "group");
    this.variants.setAttribute("aria-label", category.label + " presentation");
    this.stage = element("div", "example-stage");
    this.reveal = button("Reveal solutions & distribution", "button primary reveal-button", () => this.toggleReveal());
    this.reveal.setAttribute("aria-expanded", "false");
    this.reveal.setAttribute("aria-controls", category.id + "-results");
    this.results = element("div", "example-results");
    this.results.id = category.id + "-results";
    this.results.hidden = true;
    this.section.append(this.meta, this.variants, this.stage, this.reveal, this.results);
    main.append(this.section);
    if (initialIndex >= 0) {
      this.loadExample(initialIndex);
      if (revealInitially) this.toggleReveal();
    } else {
      this.randomize(false);
    }
  }
  get variant() { return this.group.variants[this.variantIndex]; }
  get puzzle() { return this.data.puzzles[this.variant.puzzle]; }
  humanCounts(puzzle = this.puzzle) {
    return puzzle.counts[this.data.sources.findIndex(source => source.label === "Human")] || [];
  }
  humanMode(puzzle = this.puzzle) {
    const counts = this.humanCounts(puzzle);
    // Keep the lowest solution number when several answers tie for the mode.
    return Math.max(0, counts.indexOf(Math.max(...counts)));
  }
  answerLabel(puzzle, index) {
    const counts = this.humanCounts(puzzle), maximum = Math.max(...counts);
    if (maximum <= 0 || counts[index] !== maximum) return "Revealed answer";
    return "Most common human answer" + (counts.filter(count => count === maximum).length > 1 ? " (tied)" : "");
  }
  randomize(announce) {
    const count = this.category.examples.length;
    const choices = Array.from({ length: count }, (_, i) => i).filter(i => i !== this.exampleIndex);
    this.loadExample(choices[Math.floor(Math.random() * choices.length)]);
    if (announce) status.textContent = "New " + this.category.label + " puzzle loaded. Solutions are hidden.";
  }
  loadExample(index) {
    this.exampleIndex = index;
    const example = this.category.examples[this.exampleIndex];
    this.group = this.isModule ? example : { id: example, variants: [{ label: "Main study", puzzle: example }] };
    this.useHumanMode = true;
    this.revealed = false;
    this.results.hidden = true;
    this.reveal.textContent = "Reveal solutions & distribution";
    this.reveal.setAttribute("aria-expanded", "false");
    this.variants.replaceChildren();
    if (this.isModule) this.group.variants.forEach((variant, index) => {
      const control = button(variant.label, "button variant-button", () => this.selectVariant(index));
      this.variants.append(control);
    });
    this.selectVariant(this.variantIndex);
  }
  selectVariant(index) {
    this.variantIndex = index;
    // Follow the human mode until the reader explicitly chooses a solution.
    if (this.useHumanMode) this.solutionIndex = this.humanMode();
    [...this.variants.children].forEach((control, i) => control.setAttribute("aria-pressed", String(i === index)));
    this.meta.textContent = (this.isModule ? familyNames[this.puzzle.family] + " · " : "") + "Example " + (this.exampleIndex + 1) + " of " + this.category.examples.length + " · " + this.puzzle.solutions.length + " solution classes";
    this.renderStage();
    this.renderResults();
  }
  renderStage() {
    this.stage.replaceChildren();
    const prompt = task(this.puzzle);
    const instruction = element("div", "example-instructions");
    instruction.append(element("h3", "", prompt.title), element("p", "", prompt.rules));
    this.stage.append(instruction);
    const boards = element("div", "example-boards" + (this.variant.preceding ? " has-context" : ""));
    if (this.variant.preceding) {
      const before = this.data.puzzles[this.variant.preceding];
      const container = element("div", "board-panel context-panel");
      container.append(element("h4", "", this.category.id === "transfer" ? "1. Strategy primer" : "1. First puzzle"));
      container.append(element("p", "board-task", task(before).title));
      const beforeSolution = this.humanMode(before);
      container.append(board(before, this.revealed ? beforeSolution : null));
      if (this.revealed) container.append(element("p", "preceding-answer", this.answerLabel(before, beforeSolution) + ": " + solutionText(before, beforeSolution)));
      boards.append(container);
    }
    const target = element("div", "board-panel");
    const isContext = ["pair", "transfer"].includes(this.category.id);
    if (isContext) {
      target.append(element("h4", "", this.variant.preceding ? "2. Target puzzle" : "Target puzzle"));
      target.append(element("p", "board-task", prompt.title));
    }
    target.append(board(this.puzzle, this.revealed ? this.solutionIndex : null));
    if (this.revealed) {
      const caption = this.puzzle.family === "arithmetic24" ? solutionText(this.puzzle, this.solutionIndex) : "Showing solution " + (this.solutionIndex + 1) + " of " + this.puzzle.solutions.length;
      target.append(element("p", "board-solution-caption" + (this.puzzle.family === "arithmetic24" ? " expression-answer" : ""), caption));
    }
    if (this.puzzle.highlights.length || (this.revealed && this.puzzle.family !== "arithmetic24")) {
      const key = element("div", "board-key");
      if (this.puzzle.highlights.length) key.append(element("span", "cue-key", "Highlighted cell"));
      if (this.revealed && this.puzzle.family !== "arithmetic24") key.append(element("span", "solution-key", "Shown solution"));
      target.append(key);
    }
    boards.append(target);
    this.stage.append(boards);
    this.stage.classList.toggle("context-stage", isContext);
  }
  toggleReveal() {
    this.revealed = !this.revealed;
    this.results.hidden = !this.revealed;
    this.reveal.textContent = this.revealed ? "Hide solutions & distribution" : "Reveal solutions & distribution";
    this.reveal.setAttribute("aria-expanded", String(this.revealed));
    this.renderStage();
    this.renderResults();
    status.textContent = this.category.label + (this.revealed ? ": solutions and answer distribution revealed." : ": solutions and distribution hidden.");
  }
  chooseSolution(index) {
    this.useHumanMode = false;
    this.solutionIndex = (index + this.puzzle.solutions.length) % this.puzzle.solutions.length;
    this.renderStage();
    this.updateSolution();
  }
  updateSolution() {
    this.revealedLabel.textContent = (["pair", "transfer"].includes(this.category.id) ? "Target puzzle · " : "") + this.answerLabel(this.puzzle, this.solutionIndex);
    this.solutionCount.textContent = "Solution " + (this.solutionIndex + 1) + " of " + this.puzzle.solutions.length;
    this.answerText.textContent = solutionText(this.puzzle, this.solutionIndex);
    this.selectionLabel.textContent = this.solutionCount.textContent;
    this.selectionAnswer.textContent = this.answerText.textContent;
    this.solutionPreview.replaceChildren();
    if (this.puzzle.family !== "arithmetic24") this.solutionPreview.append(board(this.puzzle, this.solutionIndex));
    this.solutionPreview.hidden = this.puzzle.family === "arithmetic24";
    this.answerText.classList.toggle("expression-answer", this.puzzle.family === "arithmetic24");
    this.results.querySelectorAll("[data-solution]").forEach(node => {
      const selected = Number(node.dataset.solution) === this.solutionIndex;
      node.classList.toggle("selected-solution", selected);
      if (node.tagName === "BUTTON") node.setAttribute("aria-pressed", String(selected));
    });
    this.results.querySelector(".solution-row.selected-solution").after(this.selectionRow);
  }
  renderResults() {
    this.results.replaceChildren();
    if (!this.revealed) return;
    const viewer = element("div", "solution-viewer");
    this.revealedLabel = element("p", "small-label");
    viewer.append(this.revealedLabel);
    const controls = element("div", "solution-controls");
    const previous = button("←", "button", () => this.chooseSolution(this.solutionIndex - 1));
    previous.setAttribute("aria-label", "Previous " + this.category.label + " solution");
    const next = button("→", "button", () => this.chooseSolution(this.solutionIndex + 1));
    next.setAttribute("aria-label", "Next " + this.category.label + " solution");
    this.solutionCount = element("strong");
    controls.append(previous, this.solutionCount, next);
    this.solutionPreview = element("div", "solution-preview");
    this.solutionPreview.id = this.category.id + "-solution-preview";
    this.answerText = element("p", "revealed-answer");
    this.answerText.id = this.category.id + "-solution-answer";
    this.answerText.setAttribute("aria-live", "polite");
    [previous, next].forEach(control => control.setAttribute("aria-controls", this.solutionPreview.id + " " + this.answerText.id));
    viewer.append(controls, this.solutionPreview, this.answerText);
    if (this.puzzle.family === "arithmetic24") viewer.append(element("p", "solution-class-note", "Some equivalent expressions, such as reordered sums, belong to the same solution class. One example from each class is shown."));
    else viewer.append(element("p", "solution-class-note", "Blue marks show the selected solution." + (this.puzzle.highlights.length ? " Gold outlines show the highlighted cells." : "")));

    const distribution = element("div", "distribution-panel");
    const heading = element("h3", "distribution-title", "Answer distribution");
    const subtitle = this.isModule ? this.variant.label + " · " : "";
    distribution.append(heading, element("p", "distribution-help", subtitle + "Share of valid answers. Select a row to view its solution."));
    const scroll = element("div", "distribution-scroll");
    scroll.tabIndex = 0;
    scroll.setAttribute("role", "region");
    scroll.setAttribute("aria-label", this.category.label + " answer distribution");
    const table = element("table", "distribution-table");
    table.setAttribute("aria-label", this.category.label + ", " + this.variant.label + ": percentage of valid answers by solution");
    const thead = element("thead");
    const head = element("tr");
    const corner = element("th", "solution-heading", "Solution");
    corner.scope = "col";
    head.append(corner);
    const totals = this.puzzle.counts.map(counts => counts.reduce((a, b) => a + b, 0));
    this.data.sources.forEach((source, i) => {
      const label = element("th", "source-heading");
      label.scope = "col";
      label.style.setProperty("--source-color", window.StudyTheme.color(source.color));
      label.append(element("span", "distribution-source", source.label), element("small", "", "n = " + totals[i]));
      head.append(label);
    });
    thead.append(head);
    table.append(thead);
    const tbody = element("tbody");
    this.puzzle.solutions.forEach((solution, index) => {
      const row = element("tr", "solution-row");
      row.dataset.solution = index;
      const label = element("th", "solution-heading");
      label.scope = "row";
      const control = button(String(index + 1), "solution-row-button", () => this.chooseSolution(index));
      control.dataset.solution = index;
      control.setAttribute("aria-label", "Show " + this.category.label + " solution " + (index + 1));
      control.setAttribute("aria-controls", this.solutionPreview.id + " " + this.answerText.id);
      control.addEventListener("keydown", event => {
        let nextIndex;
        if (event.key === "ArrowDown") nextIndex = (index + 1) % this.puzzle.solutions.length;
        if (event.key === "ArrowUp") nextIndex = (index + this.puzzle.solutions.length - 1) % this.puzzle.solutions.length;
        if (event.key === "Home") nextIndex = 0;
        if (event.key === "End") nextIndex = this.puzzle.solutions.length - 1;
        if (nextIndex === undefined) return;
        event.preventDefault();
        this.chooseSolution(nextIndex);
        this.results.querySelectorAll(".solution-row-button")[nextIndex].focus();
      });
      label.append(control);
      row.append(label);
      this.data.sources.forEach((source, sourceIndex) => {
        const count = this.puzzle.counts[sourceIndex][index], total = totals[sourceIndex];
        const percentage = total ? 100 * count / total : 0;
        const cell = element("td", "frequency-cell" + (count === 0 && total ? " zero-frequency" : ""));
        cell.style.setProperty("--source-color", window.StudyTheme.color(source.color));
        const value = element("span", "frequency-value", total ? percentage.toFixed(1) + "%" : "—");
        value.setAttribute("aria-hidden", "true");
        const track = element("span", "frequency-track");
        track.setAttribute("aria-hidden", "true");
        const bar = element("span", "frequency-bar");
        bar.style.width = percentage + "%";
        track.append(bar);
        cell.title = total ? count + " of " + total + " valid answers" : "No valid answers";
        cell.append(value, track, element("span", "sr-only", total ? percentage.toFixed(1) + " percent; " + count + " of " + total + " valid answers" : "No valid answers"));
        row.append(cell);
      });
      row.addEventListener("click", event => { if (!event.target.closest("button")) this.chooseSolution(index); });
      tbody.append(row);
    });
    table.append(tbody);
    scroll.append(table);
    const selectionReadout = element("div", "selection-readout");
    const selectionHeading = element("div", "selection-readout-heading");
    this.selectionLabel = element("strong");
    this.selectionAnswer = element("p");
    const viewSolution = element("a", "", "View solution ↑");
    viewSolution.href = "#" + (this.puzzle.family === "arithmetic24" ? this.answerText.id : this.solutionPreview.id);
    selectionHeading.append(this.selectionLabel, viewSolution);
    selectionReadout.append(selectionHeading, this.selectionAnswer);
    this.selectionRow = element("tr", "mobile-selection-row");
    const selectionCell = element("td");
    selectionCell.colSpan = 5;
    selectionCell.append(selectionReadout);
    this.selectionRow.append(selectionCell);
    distribution.append(scroll);
    let note = "n = correct human or valid model answers. Bars share a 0–100% scale. Rounded percentages may not sum to 100%.";
    if (totals.some(total => total === 0)) note += " — means no valid answers.";
    if (this.isModule) note += " Solution numbers stay matched across versions.";
    distribution.append(element("p", "distribution-note", note));
    this.results.append(viewer, distribution);
    if (["pair", "transfer"].includes(this.category.id)) {
      const contextNote = this.variant.preceding
        ? "The first solution shown is an example: these target-answer frequencies include all valid target responses, even if the first answer was wrong or different. Humans saw one puzzle at a time; models received both in one request."
        : "These answers were collected with the target presented alone, without a preceding puzzle.";
      this.results.append(element("p", "context-distribution-note", contextNote));
    }
    this.updateSolution();
  }
}

async function initializeExamples() {
  const loading = document.getElementById("load-status");
  try {
    const response = await fetch("assets/examples.json");
    if (!response.ok) throw new Error("Study examples unavailable");
    const data = await response.json();
    const mode = document.body.dataset.explorer;
    const params = new URLSearchParams(location.search);
    const requestedPuzzle = mode === "families" ? params.get("puzzle") : null;
    data[mode].forEach((category, index) => {
      const link = element("a", "", category.label);
      link.href = "#" + category.id;
      document.getElementById("example-jumps").append(link);
      const initialIndex = requestedPuzzle ? category.examples.indexOf(requestedPuzzle) : -1;
      new ExampleCard(category, data, mode === "modules", index, initialIndex, params.get("reveal") === "1");
    });
    loading.hidden = true;
    // Follow a family link after the asynchronous cards have been created.
    const anchor = location.hash.slice(1);
    if (anchor && data[mode].some(category => category.id === anchor)) document.getElementById(anchor).scrollIntoView();
  } catch {
    loading.textContent = "The puzzle examples could not be loaded. Please refresh the page to try again.";
  }
}
initializeExamples();
