"use strict";

(() => {
  const overview = document.getElementById("study-overview");
  if (!overview) return;

  const mazeStatus = document.getElementById("maze-distribution-status");

  function element(tag, className = "", text = "") {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
  }

  function svgElement(tag, attributes = {}, text = "") {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
    node.textContent = text;
    return node;
  }

  function mazeDistributions(data) {
    const example = document.getElementById("maze-example");
    const puzzle = data.puzzles[example.dataset.puzzle];
    document.getElementById("maze-route-description").textContent =
      `This maze has ${puzzle.solutions.length} correct routes, each ${puzzle.solutions[0].moves.length} moves long. Here is how often people and models used them.`;
    example.querySelector(".maze-explorer-link").href = `puzzles.html?puzzle=${encodeURIComponent(puzzle.id)}&reveal=1#maze`;
    const pathWidth = 10;
    const totals = puzzle.counts.map(counts => counts.reduce((sum, n) => sum + n, 0));
    const edges = new Map();
    puzzle.solutions.forEach((solution, route) => {
      solution.coordinates.slice(1).forEach((end, i) => {
        const start = solution.coordinates[i];
        const key = [start.join(","), end.join(",")].sort().join("|");
        if (!edges.has(key)) edges.set(key, {key, start, end, counts:totals.map(() => 0)});
        const edge = edges.get(key);
        // Shared segments accumulate every complete route that uses them.
        edge.counts.forEach((_, source) => edge.counts[source] += puzzle.counts[source][route]);
      });
    });

    const root = document.getElementById("maze-distribution-boards");
    const readout = document.getElementById("maze-solution-readout");
    const instruction = readout.textContent;
    const views = [];
    const controls = [];
    const key = document.getElementById("maze-width-key");
    let pinnedSolution = null;
    let hoveredSolution;
    let focusedSolution;
    const percent = (count, source) => (100 * count / totals[source]).toFixed(1) + "%";
    const show = route => {
      views.forEach((view, source) => {
        view.board.classList.toggle("has-selected-solution", route !== null);
        view.trace.style.display = route === null ? "none" : "";
        if (route !== null) {
          const points = puzzle.solutions[route].coordinates.map(view.coordinate).map(point => point.join(",")).join(" ");
          view.trace.querySelectorAll("polyline").forEach(line => line.setAttribute("points", points));
          view.value.textContent = `${percent(puzzle.counts[source][route], source)} chose this solution`;
          view.value.title = `${puzzle.counts[source][route]} of ${totals[source]} correct answers`;
        } else {
          view.value.textContent = "All solutions";
          view.value.removeAttribute("title");
        }
      });
      controls.forEach(({button, index}) => {
        button.classList.toggle("is-selected", route === index);
        button.setAttribute("aria-pressed", String(pinnedSolution === index));
      });
      key.dataset.view = route === null ? "overview" : "solution";
      key.setAttribute("aria-label", route === null ? "Path-width scale" : "Selected complete solution");
      readout.textContent = route === null ? instruction :
        `Solution ${route + 1} of ${puzzle.solutions.length} · ${puzzle.solutions[route].moves.length} moves from S to G. ` +
        data.sources.map((source, i) => `${source.label}: ${puzzle.counts[i][route]}/${totals[i]} (${percent(puzzle.counts[i][route], i)})`).join(" · ");
    };
    const preview = () => show(hoveredSolution !== undefined ? hoveredSolution :
      focusedSolution !== undefined ? focusedSolution : pinnedSolution);
    function bindSolution(target, route) {
      target.addEventListener("pointerenter", event => {
        if (event.pointerType !== "touch") { hoveredSolution = route; preview(); }
      });
      target.addEventListener("pointerleave", event => {
        if (event.pointerType !== "touch") { hoveredSolution = undefined; preview(); }
      });
      target.addEventListener("focus", () => { focusedSolution = route; hoveredSolution = undefined; preview(); });
      target.addEventListener("blur", () => { focusedSolution = undefined; preview(); });
      target.addEventListener("click", () => { pinnedSolution = route; preview(); });
      if (target.namespaceURI === "http://www.w3.org/2000/svg") {
        target.addEventListener("keydown", event => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault(); pinnedSolution = route; preview();
          }
        });
      }
    }

    const solutionControls = document.getElementById("maze-solution-controls");
    solutionControls.append(element("span", "maze-controls-label", "Highlight solution"));
    [null, ...puzzle.solutions.map((_, route) => route)].forEach(route => {
      const button = element("button", "maze-solution-button", route === null ? "All solutions" : String(route + 1));
      button.type = "button";
      button.setAttribute("aria-label", route === null ? "Show all solution frequencies" : `Highlight solution ${route + 1} of ${puzzle.solutions.length}`);
      bindSolution(button, route);
      controls.push({button, index:route});solutionControls.append(button);
    });

    data.sources.forEach((source, sourceIndex) => {
      const card = element("section", "maze-source-panel");
      card.style.setProperty("--maze-source", window.StudyTheme.color(source.color));
      card.append(element("h4", "", source.label),
        element("p", "maze-source-count", `n = ${totals[sourceIndex]} correct answers`));
      const value = element("p", "maze-solution-value", "All solutions");
      card.append(value);
      const board = window.PuzzleBoards.board(puzzle);
      board.classList.add("weighted-maze");
      board.setAttribute("role", "group");
      board.setAttribute("aria-label", `${source.label}: ${puzzle.solutions.length} shortest routes, weighted by frequency among ${totals[sourceIndex]} correct answers`);
      // Use the shared renderer's cell geometry, retaining its exact maze walls.
      const cellRect = board.querySelector(".board-cell");
      const cell = Number(cellRect.getAttribute("width"));
      const margin = Number(cellRect.getAttribute("x"));
      const rows = puzzle.instance.grid.length, cols = puzzle.instance.grid[0].length;
      board.setAttribute("viewBox", `${margin - 4} ${margin - 4} ${cols * cell + 8} ${rows * cell + 8}`);
      board.querySelectorAll(".board-axis").forEach(axis => axis.remove());
      const coordinate = ([r, c]) => [margin + (c - .5) * cell, margin + (r - .5) * cell];
      edges.forEach(edge => {
        const [x1, y1] = coordinate(edge.start), [x2, y2] = coordinate(edge.end);
        const share = edge.counts[sourceIndex] / totals[sourceIndex];
        board.append(svgElement("line", {x1, y1, x2, y2, "stroke-width":pathWidth * share, class:"maze-flow-line"}));
      });
      // Prefer the most frequently chosen route where hit paths overlap; the
      // numbered controls make every solution unambiguous, including zero counts.
      const routeOrder = puzzle.solutions.map((_, route) => route)
        .sort((a, b) => puzzle.counts[sourceIndex][a] - puzzle.counts[sourceIndex][b] || b - a);
      routeOrder.forEach(route => {
        const points = puzzle.solutions[route].coordinates.map(coordinate).map(point => point.join(",")).join(" ");
        const label = `${source.label}, solution ${route + 1}: ${puzzle.counts[sourceIndex][route]} of ${totals[sourceIndex]} correct answers (${percent(puzzle.counts[sourceIndex][route], sourceIndex)})`;
        const hit = svgElement("polyline", {points, class:"maze-route-hit", tabindex:0, role:"button", "aria-label":label, "data-solution":route + 1});
        bindSolution(hit, route);board.append(hit);
      });
      const trace = svgElement("g", {class:"maze-solution-trace", "aria-hidden":"true"});
      trace.append(svgElement("polyline", {class:"maze-solution-halo"}), svgElement("polyline", {class:"maze-solution-line"}));
      trace.style.display = "none";
      board.append(trace);
      views.push({board, trace, value, coordinate});
      // Keep S and G above all paths, including segments with full probability.
      board.querySelectorAll(".board-symbol").forEach(label => {
        const x = Number(label.getAttribute("x")), y = Number(label.getAttribute("y"));
        board.append(svgElement("circle", {cx:x, cy:y - 7, r:12, class:"maze-endpoint"}), label);
      });
      board.append(svgElement("rect", {x:margin, y:margin, width:cols * cell, height:rows * cell, class:"maze-outline"}));
      card.append(board);root.append(card);
    });
    example.addEventListener("keydown", event => {
      if (event.key === "Escape") {
        pinnedSolution = null; hoveredSolution = undefined; focusedSolution = undefined; show(null);
      }
    });
    key.append(element("span", "", "Share of correct answers"));
    [.25, .5, 1].forEach(share => {
      const item = element("span", "maze-width-sample");
      const sample = svgElement("svg", {viewBox:"0 0 54 24", "aria-hidden":"true"});
      sample.append(svgElement("line", {x1:7, y1:12, x2:47, y2:12, "stroke-width":pathWidth * share}));
      item.append(sample, element("span", "", `${100 * share}%`));key.append(item);
    });
    key.append(element("span", "maze-selected-key", "Complete solution highlighted · percentages show how often it was chosen"));
    show(null);
    mazeStatus.hidden = true;
    document.getElementById("maze-distribution-content").hidden = false;
  }

  const status = document.getElementById("overview-status") || element("p", "overview-status", "Loading puzzle previews…");
  status.setAttribute("role", "status");
  if (!status.parentNode) overview.append(status);
  overview.dataset.state = "loading";

  async function initializeOverview() {
    try {
      const response = await fetch("assets/examples.json");
      if (!response.ok) throw new Error("Study examples unavailable");
      const data = await response.json();
      const { board, task } = window.PuzzleBoards;
      const cards = data.families.map(family => {
        const puzzle = data.puzzles[family.examples[0]];
        const card = element("article", "overview-puzzle-card");
        card.dataset.family = family.id;
        const heading = element("h3", "overview-puzzle-title", family.label);
        const preview = element("div", "overview-board");
        preview.append(board(puzzle));
        const instruction = element("p", "overview-puzzle-task", task(puzzle).title);
        const link = element("a", "overview-puzzle-link", "Try " + family.label.toLowerCase() + " →");
        link.href = "puzzles.html#" + family.id;
        card.append(heading, preview, instruction, link);
        return card;
      });
      overview.replaceChildren(...cards);
      overview.dataset.state = "ready";
      status.hidden = true;
      try {
        mazeDistributions(data);
      } catch {
        mazeStatus.textContent = "The maze distributions could not be loaded. Please refresh the page to try again.";
      }
    } catch {
      overview.dataset.state = "error";
      status.hidden = false;
      status.textContent = "The puzzle previews could not be loaded. Please refresh the page to try again.";
      mazeStatus.textContent = "The maze distributions could not be loaded. Please refresh the page to try again.";
    }
  }

  initializeOverview();
})();
