"use strict";

// One renderer for the public overview and the puzzle/stimulus explorers.
(() => {
const familyNames = { arithmetic24: "Arithmetic", maze: "Maze", grid_placement: "Rooks", minesweeper_lite: "Minesweeper", mini_sudoku: "Sudoku" };
function element(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}
function svgElement(tag, attributes = {}, text = "") {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
  if (text) node.textContent = text;
  return node;
}
function task(puzzle) {
  const i = puzzle.instance;
  switch (puzzle.family) {
    case "arithmetic24": return { title: "Make " + i.target + ".", rules: "Use each number exactly once, with only +, −, ×, ÷ and parentheses. Negative or fractional intermediate results are allowed. Do not join numbers into a larger number." };
    case "maze": return { title: "Find a shortest path.", rules: "Go from S to G, moving up, down, left or right through open cells. Dark cells are walls. Find one path with the fewest moves." };
    case "grid_placement": return { title: "Place " + i.m + " rooks.", rules: "Use open cells only. No two rooks may share a row or a column; diagonal alignment is allowed. Find one complete placement." };
    case "minesweeper_lite": return { title: i.target_kind === "safe" ? "Find one definitely safe cell." : "Find one definite mine.", rules: "Choose one ? cell. Each number counts mines in neighboring cells, including diagonals; F is a known mine. Your answer must hold for every mine arrangement consistent with the clues." };
    case "mini_sudoku": return { title: "Place the digit " + i.digit + ".", rules: "Choose one empty cell. The digit must not already appear in its row, column or outlined box. The placement need not be forced or lead to a complete Sudoku solution." };
  }
}
function prettyExpression(value) {
  return value.replaceAll("*", " × ").replaceAll("/", " ÷ ").replaceAll("+", " + ").replaceAll("-", " − ");
}
function solutionText(puzzle, index) {
  const answer = puzzle.solutions[index];
  if (answer.expression) return prettyExpression(answer.expression) + " = " + puzzle.instance.target;
  if (answer.moves) return (answer.coordinates.length - 1) + " moves: " + [...answer.moves].map(move => ({ U: "↑", D: "↓", L: "←", R: "→" }[move])).join(" ");
  if (answer.coordinates) return "Rooks at " + answer.coordinates.map(([r, c]) => "(" + r + ", " + c + ")").join(", ") + ".";
  return "Row " + answer.coordinate[0] + ", column " + answer.coordinate[1] + (puzzle.family === "mini_sudoku" ? ": place " + puzzle.instance.digit + "." : ".");
}

function board(puzzle, solutionIndex = null) {
  const instance = puzzle.instance;
  if (puzzle.family === "arithmetic24") {
    const wrap = element("div", "arithmetic-board");
    wrap.setAttribute("aria-label", "Numbers: " + instance.numbers.join(", "));
    instance.numbers.forEach(number => wrap.append(element("span", "arithmetic-number", String(number))));
    return wrap;
  }
  const grid = instance.grid || instance.board;
  const rows = grid.length;
  const cols = grid[0].length;
  const cell = 40;
  const margin = 24;
  const width = cols * cell + 2 * margin;
  const height = rows * cell + 2 * margin;
  const answer = solutionIndex === null ? null : puzzle.solutions[solutionIndex];
  const selected = new Set((answer?.coordinates || (answer?.coordinate ? [answer.coordinate] : [])).map(([r, c]) => r + "," + c));
  const highlights = new Set(puzzle.highlights.map(([r, c]) => r + "," + c));
  const svg = svgElement("svg", { viewBox: "0 0 " + width + " " + height, class: "puzzle-board", role: "img" });
  svg.setAttribute("aria-label", familyNames[puzzle.family] + ". " + task(puzzle).title + " " + grid.map((row, r) => "Row " + (r + 1) + ": " + [...row].join(" ")).join(". ") + (puzzle.highlights.length ? ". Highlighted cells: " + puzzle.highlights.map(([r, c]) => "row " + r + ", column " + c).join("; ") : "") + (answer ? " Solution: " + solutionText(puzzle, solutionIndex) : ""));
  for (let c = 0; c < cols; c++) svg.append(svgElement("text", { x: margin + (c + .5) * cell, y: 15, class: "board-axis" }, c + 1));
  for (let r = 0; r < rows; r++) {
    svg.append(svgElement("text", { x: 11, y: margin + (r + .5) * cell + 4, class: "board-axis" }, r + 1));
    for (let c = 0; c < cols; c++) {
      const value = grid[r][c];
      const key = (r + 1) + "," + (c + 1);
      const x = margin + c * cell;
      const y = margin + r * cell;
      const blocked = value === "#" || value === "X";
      const marked = selected.has(key);
      const given = (puzzle.family === "minesweeper_lite" && value !== "?") ||
        (puzzle.family === "mini_sudoku" && value !== ".");
      svg.append(svgElement("rect", { x, y, width: cell, height: cell, class: "board-cell" + (given ? " given" : "") + (blocked ? " blocked" : marked ? " solved" : "") }));
      if (highlights.has(key)) svg.append(svgElement("rect", { x: x + 3, y: y + 3, width: cell - 6, height: cell - 6, class: "board-highlight" }));
      let text = [".", "#", "X"].includes(value) ? "" : value;
      if (marked && puzzle.family === "grid_placement") text = "●";
      if (marked && puzzle.family === "mini_sudoku") text = String(instance.digit);
      if (marked && puzzle.family === "minesweeper_lite") text = instance.target_kind === "safe" ? "✓" : "M";
      if (text) svg.append(svgElement("text", { x: x + cell / 2, y: y + cell / 2 + 7, class: "board-symbol" + (marked ? " solution-symbol" : "") }, text));
    }
  }
  if (puzzle.family === "mini_sudoku") {
    for (let r = 0; r <= rows; r += instance.box_rows) svg.append(svgElement("line", { x1: margin, x2: margin + cols * cell, y1: margin + r * cell, y2: margin + r * cell, class: "box-boundary" }));
    for (let c = 0; c <= cols; c += instance.box_cols) svg.append(svgElement("line", { x1: margin + c * cell, x2: margin + c * cell, y1: margin, y2: margin + rows * cell, class: "box-boundary" }));
  }
  if (answer?.moves) {
    const points = answer.coordinates.map(([r, c]) => (margin + (c - .5) * cell) + "," + (margin + (r - .5) * cell)).join(" ");
    svg.append(svgElement("polyline", { points, class: "solution-path" }));
    // Keep the start and goal legible above the path.
    [answer.coordinates[0], answer.coordinates[answer.coordinates.length - 1]].forEach(([r, c]) => {
      svg.append(svgElement("circle", { cx: margin + (c - .5) * cell, cy: margin + (r - .5) * cell, r: 13, fill: "var(--board-solved)" }));
      svg.append(svgElement("text", { x: margin + (c - .5) * cell, y: margin + (r - .5) * cell + 7, class: "board-symbol solution-symbol" }, grid[r - 1][c - 1]));
    });
  }
  return svg;
}

window.PuzzleBoards = Object.freeze({ board, task, solutionText, familyNames });
})();
