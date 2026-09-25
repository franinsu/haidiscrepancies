"use strict";

// Reproduce the paper's lower triangle from frozen, aggregate figure data.
(() => {
  window.drawDifficultyMatrix = (target, data, scope, helpers) => {
    const {svgNode, number, familyColors} = helpers;
    const {puzzles, density} = data.difficulty;
    if (data.sources.length !== 4 || puzzles.length !== 100 || density?.length !== 4) {
      throw new Error("The difficulty matrix needs four sources and 100 puzzles.");
    }

    const width = 1000, height = 956;
    const left = 80, top = 20, size = 188, gap = 28;
    const ticks = [-3, -2, -1, 0, 1, 2, 3];
    const maxDensity = Math.max(...density.flatMap(curve => curve.density)) * 1.10;
    const seen = Array(data.families.length).fill(0);
    const puzzleLabels = puzzles.map(p => `${data.families[p.family]} · puzzle ${++seen[p.family]}`);
    const svg = svgNode("svg", {
      viewBox: `0 0 ${width} ${height}`,
      class: "difficulty-matrix-svg",
      role: "group",
      "aria-label": "Relative difficulty: all six source comparisons, with four density curves",
      "font-family": '-apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif',
      "font-size": 17,
      fill: "var(--plot-ink)",
    });
    svg.append(svgNode("desc", {},
      "The lower triangle compares Human, ChatGPT, Claude, and Gemini on the same 100 puzzles. " +
      "Every scatterplot uses square axes from minus three to three z-scores. " +
      "The diagonal shows the saved density curve and one rug mark per puzzle. " +
      "Tab enters each plot; arrow keys move between puzzles, Home and End jump to its first and last puzzle, " +
      "Enter or Space reopens values, and Escape clears the highlight."));

    const label = (text, x, y, attrs = {}) => svgNode("text", {
      x, y, "text-anchor": "middle", "aria-hidden": "true", ...attrs,
    }, text);
    const line = (x1, y1, x2, y2, attrs = {}) => svgNode("line", {
      x1, y1, x2, y2, stroke: "var(--plot-ink)", "stroke-width": 1.1,
      "aria-hidden": "true", ...attrs,
    });
    const reopenOnKey = (element, show) => {
      element.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          show();
        }
      });
    };

    for (let row = 0; row < 4; row++) {
      for (let col = 0; col <= row; col++) {
        const sourceX = data.sources[col], sourceY = data.sources[row];
        const x0 = left + col * (size + gap), y0 = top + row * (size + gap);
        const bottom = y0 + size;
        const x = z => x0 + (z + 3) * size / 6;
        const y = z => bottom - (z + 3) * size / 6;
        const diagonal = row === col;
        const plot = svgNode("g", {
          class: diagonal ? "difficulty-marginal" : "difficulty-comparison",
          "data-row": row, "data-col": col,
          role: "group",
          "aria-label": diagonal ? `${sourceX.label} relative difficulty density` :
            `${sourceX.label} on the horizontal axis and ${sourceY.label} on the vertical axis, 100 puzzles`,
        });

        if (!diagonal) {
          plot.append(line(x(0), y0, x(0), bottom, {stroke: "var(--plot-zero)"}),
            line(x0, y(0), x0 + size, y(0), {stroke: "var(--plot-zero)"}));
          plot.append(line(x0, y0, x0, bottom));
        }
        plot.append(line(x0, bottom, x0 + size, bottom));

        ticks.forEach(tick => {
          plot.append(line(x(tick), bottom, x(tick), bottom + 4));
          if (row === 3) plot.append(label(tick, x(tick), bottom + 23, {"font-size": 14}));
          if (!diagonal) {
            plot.append(line(x0 - 4, y(tick), x0, y(tick)));
            if (col === 0) plot.append(label(tick, x0 - 9, y(tick) + 5, {
              "text-anchor": "end", "font-size": 14,
            }));
          }
        });
        if (row === 3) plot.append(label(sourceX.label, x0 + size / 2, bottom + 49));
        if (col === 0 && row > 0) plot.append(label(sourceY.label, 0, 0, {
          transform: `translate(${x0 - 48} ${y0 + size / 2}) rotate(-90)`,
        }));

        if (diagonal) {
          const curve = density[col];
          const densityY = value => bottom - value / maxDensity * size;
          const d = curve.grid.map((z, i) => `${i ? "L" : "M"}${x(z)},${densityY(curve.density[i])}`).join(" ");
          const tags = {sources: [col]};
          const area = svgNode("path", {
            d: `${d} L${x(3)},${bottom} L${x(-3)},${bottom} Z`,
            fill: sourceX.color, "fill-opacity": .13, "pointer-events": "none",
          });
          const stroke = svgNode("path", {
            d, fill: "none", stroke: sourceX.color, "stroke-width": 1.8,
            class: "difficulty-density-line", "pointer-events": "none",
          });
          scope.mark(area, tags);
          scope.mark(stroke, tags);
          plot.append(area, stroke);

          puzzles.forEach((puzzle, index) => {
            const rug = line(x(puzzle.z[col]), bottom, x(puzzle.z[col]), bottom - size * .04, {
              stroke: sourceX.color, "stroke-width": .8, "stroke-opacity": .45,
              class: "difficulty-rug", "pointer-events": "none", "data-puzzle": index,
            });
            scope.mark(rug, {puzzle: index, family: puzzle.family, sources: [col]});
            plot.append(rug);
          });

          let curveIndex = Math.floor(curve.grid.length / 2);
          const hit = svgNode("path", {
            d, fill: "none", stroke: "transparent", "stroke-width": 16,
            class: "density-hit difficulty-density-hit", tabindex: 0, role: "button",
          });
          const showCurve = scope.bind(hit, tags, event => {
            if (event?.clientX !== undefined && !(event.type === "click" && event.detail === 0)) {
              const bounds = svg.getBoundingClientRect();
              const fraction = ((event.clientX - bounds.left) * width / bounds.width - x0) / size;
              curveIndex = Math.max(0, Math.min(curve.grid.length - 1,
                Math.round(fraction * (curve.grid.length - 1))));
            }
            return `${sourceX.label} · standardized relative difficulty ${number(curve.grid[curveIndex])}\n` +
              `Probability density ${number(curve.density[curveIndex])}\n100 puzzles · saved density curve`;
          });
          reopenOnKey(hit, showCurve);
          hit.addEventListener("keydown", event => {
            let next;
            if (event.key === "ArrowRight" || event.key === "ArrowUp") next = curveIndex + 10;
            if (event.key === "ArrowLeft" || event.key === "ArrowDown") next = curveIndex - 10;
            if (event.key === "Home") next = 0;
            if (event.key === "End") next = curve.grid.length - 1;
            if (next === undefined) return;
            event.preventDefault();
            curveIndex = Math.max(0, Math.min(curve.grid.length - 1, next));
            showCurve();
          });
          plot.append(hit);
        } else {
          const points = puzzles.map((puzzle, index) => {
            const point = svgNode("circle", {
              cx: x(puzzle.z[col]), cy: y(puzzle.z[row]), r: 3.1,
              fill: familyColors[puzzle.family], "fill-opacity": .9,
              class: "scatter-dot", tabindex: index === 0 ? 0 : -1, role: "button",
              "data-puzzle": index, "data-family": puzzle.family,
            });
            const tags = {puzzle: index, family: puzzle.family, sources: [col, row]};
            scope.mark(point, tags);
            const showPoint = scope.bind(point, tags,
              `${puzzleLabels[index]}\n${sourceX.label}: ${number(puzzle.z[col])}\n` +
              `${sourceY.label}: ${number(puzzle.z[row])}\nStandardized relative difficulty (z-scores)`);
            const select = () => points.forEach((other, i) => other.setAttribute("tabindex", i === index ? 0 : -1));
            point.addEventListener("focus", select);
            point.addEventListener("click", select);
            reopenOnKey(point, showPoint);
            point.addEventListener("keydown", event => {
              let next;
              if (event.key === "ArrowRight") next = index + 1;
              if (event.key === "ArrowLeft") next = index - 1;
              if (event.key === "ArrowDown") next = index + 10;
              if (event.key === "ArrowUp") next = index - 10;
              if (event.key === "Home") next = 0;
              if (event.key === "End") next = points.length - 1;
              if (next === undefined) return;
              event.preventDefault();
              next = (next + points.length) % points.length;
              points[next].focus({preventScroll: true});
              // Keep keyboard movement inside the matrix's local horizontal scroller.
              const bounds = points[next].getBoundingClientRect(), viewport = target.getBoundingClientRect();
              if (bounds.left < viewport.left + 12) target.scrollLeft -= viewport.left + 12 - bounds.left;
              if (bounds.right > viewport.right - 12) target.scrollLeft += bounds.right - viewport.right + 12;
            });
            plot.append(point);
            return point;
          });
        }
        svg.append(plot);
      }
    }

    const legendX = 612, legendY = 65;
    svg.append(label("Puzzle family", legendX + 4, legendY, {"text-anchor": "start", "font-size": 18}));
    data.families.forEach((family, index) => {
      const y = legendY + 32 + index * 43;
      const key = svgNode("g", {class: "difficulty-family-key", tabindex: 0, role: "button"});
      key.append(svgNode("rect", {
        x: legendX - 10, y: y - 21, width: 230, height: 41, rx: 4,
        fill: "transparent", stroke: "transparent", class: "difficulty-family-hit",
      }), svgNode("circle", {cx: legendX + 2, cy: y, r: 4.5, fill: familyColors[index], "fill-opacity": .9}),
      label(family, legendX + 25, y + 6, {"text-anchor": "start", "font-size": 18}));
      const tags = {family: index};
      scope.mark(key, tags);
      const showFamily = scope.bind(key, tags, `${family} · 20 puzzles highlighted across all six comparisons`);
      reopenOnKey(key, showFamily);
      svg.append(key);
    });
    svg.append(label("Standardized relative difficulty (z-score)", width / 2, height - 16, {"font-size": 17}));
    target.append(svg);
  };
})();
