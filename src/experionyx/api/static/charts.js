// Deterministic SVG chart primitives. No randomness, no layout simulation: every coordinate is a
// pure function of the input data (see docs/visualization.md). Every chart renders its metric
// name, unit, population/scope, comparison and uncertainty as visible text, never a bare number.

const NS = "http://www.w3.org/2000/svg";
const W = 640, H = 220, PAD = 36;

function svg(children) {
  const el = document.createElementNS(NS, "svg");
  el.setAttribute("viewBox", `0 0 ${W} ${H}`);
  el.setAttribute("width", "100%");
  el.setAttribute("height", H);
  for (const c of children) el.appendChild(c);
  return el;
}

function line(x1, y1, x2, y2, cls) {
  const el = document.createElementNS(NS, "line");
  el.setAttribute("x1", x1); el.setAttribute("y1", y1);
  el.setAttribute("x2", x2); el.setAttribute("y2", y2);
  el.setAttribute("stroke", "currentColor");
  el.setAttribute("stroke-width", "1");
  el.setAttribute("opacity", cls === "axis" ? "0.4" : "0.9");
  return el;
}

function text(x, y, s, opts = {}) {
  const el = document.createElementNS(NS, "text");
  el.setAttribute("x", x); el.setAttribute("y", y);
  el.setAttribute("font-size", opts.size || "9");
  el.setAttribute("fill", "currentColor");
  el.setAttribute("opacity", opts.muted ? "0.6" : "0.9");
  if (opts.anchor) el.setAttribute("text-anchor", opts.anchor);
  el.textContent = s;
  return el;
}

function scaleLinear(values, lo, hi) {
  const min = Math.min(0, ...values), max = Math.max(1, ...values);
  const span = max - min || 1;
  return (v) => hi - ((v - min) / span) * (hi - lo);
}

function frame(caption) {
  const wrap = document.createElement("div");
  const cap = document.createElement("div");
  cap.className = "chart-caption";
  cap.textContent = caption;
  wrap.appendChild(cap);
  return wrap;
}

// bar: points = [{label, value}]
export function bar(points, { metric, unit }) {
  const wrap = frame(`${metric}${unit ? " (" + unit + ")" : ""}`);
  const els = [line(PAD, H - PAD, W - 10, H - PAD, "axis")];
  const values = points.map((p) => p.value).filter((v) => typeof v === "number");
  const y = scaleLinear(values, 10, H - PAD);
  const bw = Math.max(4, (W - PAD - 20) / Math.max(1, points.length) - 6);
  points.forEach((p, i) => {
    const x = PAD + i * (bw + 6);
    const v = typeof p.value === "number" ? p.value : 0;
    els.push(line(x + bw / 2, H - PAD, x + bw / 2, y(v), "bar"));
    const rect = document.createElementNS(NS, "rect");
    rect.setAttribute("x", x); rect.setAttribute("y", Math.min(y(v), H - PAD));
    rect.setAttribute("width", bw); rect.setAttribute("height", Math.abs(H - PAD - y(v)));
    rect.setAttribute("fill", "currentColor"); rect.setAttribute("opacity", "0.7");
    els.push(rect);
    els.push(text(x + bw / 2, H - PAD + 12, String(p.label).slice(0, 10), { anchor: "middle", size: "8" }));
    els.push(text(x + bw / 2, Math.min(y(v), H - PAD) - 3, String(p.value), { anchor: "middle", size: "8" }));
  });
  wrap.appendChild(svg(els));
  return wrap;
}

// groupedBar: points = [{label, series: {name: value}}]
export function groupedBar(points, seriesNames, { metric, unit }) {
  const wrap = frame(`${metric}${unit ? " (" + unit + ")" : ""} — grouped by ${seriesNames.join(", ")}`);
  const els = [line(PAD, H - PAD, W - 10, H - PAD, "axis")];
  const all = points.flatMap((p) => seriesNames.map((s) => p.series[s])).filter((v) => typeof v === "number");
  const y = scaleLinear(all, 10, H - PAD);
  const groupW = (W - PAD - 20) / Math.max(1, points.length);
  const barW = Math.max(3, groupW / (seriesNames.length + 1) - 2);
  points.forEach((p, gi) => {
    seriesNames.forEach((s, si) => {
      const v = p.series[s];
      const x = PAD + gi * groupW + si * (barW + 2);
      const val = typeof v === "number" ? v : 0;
      const rect = document.createElementNS(NS, "rect");
      rect.setAttribute("x", x); rect.setAttribute("y", Math.min(y(val), H - PAD));
      rect.setAttribute("width", barW); rect.setAttribute("height", Math.abs(H - PAD - y(val)));
      rect.setAttribute("fill", "currentColor");
      rect.setAttribute("opacity", si === 0 ? "0.8" : "0.4");
      els.push(rect);
    });
    els.push(text(PAD + gi * groupW + groupW / 2, H - PAD + 12, String(p.label).slice(0, 10), { anchor: "middle", size: "8" }));
  });
  wrap.appendChild(svg(els));
  return wrap;
}

// lineWithBand: points = [{x, y, lo, hi}] -- lo/hi optional uncertainty band
export function lineWithBand(points, { metric, unit }) {
  const wrap = frame(`${metric}${unit ? " (" + unit + ")" : ""}`);
  if (!points.length) { wrap.appendChild(text(4, 12, "unavailable")); return wrap; }
  const xs = points.map((p) => p.x);
  const ys = points.flatMap((p) => [p.y, p.lo, p.hi].filter((v) => typeof v === "number"));
  const xScale = scaleLinear(xs, PAD, W - 10);
  const yScale = scaleLinear(ys, 10, H - PAD);
  const els = [line(PAD, H - PAD, W - 10, H - PAD, "axis")];
  if (points.some((p) => typeof p.lo === "number" && typeof p.hi === "number")) {
    const poly = document.createElementNS(NS, "polygon");
    const upper = points.map((p) => `${xScale(p.x)},${yScale(p.hi)}`);
    const lower = [...points].reverse().map((p) => `${xScale(p.x)},${yScale(p.lo)}`);
    poly.setAttribute("points", [...upper, ...lower].join(" "));
    poly.setAttribute("fill", "currentColor");
    poly.setAttribute("opacity", "0.15");
    els.push(poly);
  }
  const path = document.createElementNS(NS, "polyline");
  path.setAttribute("points", points.map((p) => `${xScale(p.x)},${yScale(p.y)}`).join(" "));
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", "currentColor");
  path.setAttribute("stroke-width", "1.5");
  els.push(path);
  wrap.appendChild(svg(els));
  return wrap;
}

// scatter: points = [{x, y, label}]
export function scatter(points, { metric, unit }) {
  const wrap = frame(`${metric}${unit ? " (" + unit + ")" : ""}`);
  if (!points.length) { wrap.appendChild(text(4, 12, "unavailable")); return wrap; }
  const xScale = scaleLinear(points.map((p) => p.x), PAD, W - 10);
  const yScale = scaleLinear(points.map((p) => p.y), 10, H - PAD);
  const els = [line(PAD, H - PAD, W - 10, H - PAD, "axis")];
  points.forEach((p) => {
    const c = document.createElementNS(NS, "circle");
    c.setAttribute("cx", xScale(p.x)); c.setAttribute("cy", yScale(p.y));
    c.setAttribute("r", "3"); c.setAttribute("fill", "currentColor"); c.setAttribute("opacity", "0.8");
    els.push(c);
  });
  wrap.appendChild(svg(els));
  return wrap;
}

// Deterministic circular layout for a bounded graph neighborhood: position = f(index), no
// force simulation, no randomness.
export function circularGraph(nodes, edges) {
  const size = 320, r = size / 2 - 30, cx = size / 2, cy = size / 2;
  const pos = {};
  nodes.forEach((n, i) => {
    const angle = (2 * Math.PI * i) / Math.max(1, nodes.length);
    pos[n.id] = [cx + r * Math.cos(angle), cy + r * Math.sin(angle)];
  });
  const els = [];
  edges.forEach((e) => {
    const a = pos[e.from], b = pos[e.to];
    if (!a || !b) return;
    els.push(line(a[0], a[1], b[0], b[1], "edge"));
  });
  nodes.forEach((n) => {
    const [x, y] = pos[n.id];
    const c = document.createElementNS(NS, "circle");
    c.setAttribute("cx", x); c.setAttribute("cy", y); c.setAttribute("r", n.resolved ? "5" : "3");
    c.setAttribute("fill", "currentColor"); c.setAttribute("opacity", n.resolved ? "0.85" : "0.35");
    els.push(c);
    els.push(text(x + 7, y + 3, n.label.slice(0, 14), { size: "7", muted: true }));
  });
  const el = document.createElementNS(NS, "svg");
  el.setAttribute("viewBox", `0 0 ${size} ${size}`);
  el.setAttribute("width", "100%");
  el.setAttribute("height", size);
  els.forEach((e) => el.appendChild(e));
  return el;
}
