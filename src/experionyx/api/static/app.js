import * as charts from "/charts.js";

const app = document.getElementById("app");

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function idLink(id) {
  if (typeof id !== "string" || !/^[a-z]{3}_[0-9a-f]{32}$/.test(id)) return esc(String(id));
  const prefix = id.slice(0, 3);
  const routes = {
    inv: "investigations", exp: "investigations/experiments", run: "investigations/runs",
    rpf: "reliability/profiles", fcl: "failures/clusters", fmd: "failures/modes",
    fan: "faults/analyses", dan: "drift/analyses", sta: "stats/analyses",
    bmk: "benchmark/benchmarks", brs: "benchmark/results", rpa: "reproducibility/attempts",
    gsn: "graph/snapshots", rpt: "reports", dsr: "dossiers",
  };
  const base = routes[prefix];
  return base ? `<a href="#/${base}/${id}"><code>${id}</code></a>` : `<code>${esc(id)}</code>`;
}

function traceChain(...ids) {
  const parts = ids.filter(Boolean).map(idLink);
  return `<div class="trace">${parts.join(" &rarr; ")}</div>`;
}

function fieldList(obj, skip = []) {
  const rows = Object.entries(obj)
    .filter(([k]) => !skip.includes(k))
    .map(([k, v]) => `<dt>${esc(k)}</dt><dd>${renderValue(v)}</dd>`);
  return `<dl class="field-list">${rows.join("")}</dl>`;
}

function renderValue(v) {
  if (v === "unavailable" || v === null || v === undefined) return `<span class="unavailable">unavailable</span>`;
  if (typeof v === "string" && /^[a-z]{3}_[0-9a-f]{32}$/.test(v)) return idLink(v);
  if (Array.isArray(v)) return v.length ? v.map((x) => renderValue(x)).join(", ") : `<span class="unavailable">none</span>`;
  if (typeof v === "object") return `<code>${esc(JSON.stringify(v))}</code>`;
  return esc(String(v));
}

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(`${res.status} ${body.error || ""}: ${body.detail || ""}`);
  }
  return res.json();
}

function card(html) {
  return `<div class="card">${html}</div>`;
}

function table(rows, cols) {
  if (!rows.length) return `<p class="unavailable">no records</p>`;
  const head = cols.map(([, label]) => `<th>${esc(label)}</th>`).join("");
  const body = rows
    .map((r) => `<tr>${cols.map(([key]) => `<td>${renderValue(typeof key === "function" ? key(r) : r[key])}</td>`).join("")}</tr>`)
    .join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

// --- routes ------------------------------------------------------------------------------------

const routes = [
  [/^investigations$/, viewInvestigationList],
  [/^investigations\/(inv_[0-9a-f]{32})$/, viewInvestigation],
  [/^investigations\/experiments\/(exp_[0-9a-f]{32})$/, viewExperiment],
  [/^investigations\/runs\/(run_[0-9a-f]{32})$/, viewRun],
  [/^reliability$/, viewReliabilityList],
  [/^reliability\/profiles\/(rpf_[0-9a-f]{32})$/, viewReliabilityProfile],
  [/^failures$/, viewFailureClusters],
  [/^failures\/clusters\/(fcl_[0-9a-f]{32})$/, viewFailureCluster],
  [/^failures\/modes\/(fmd_[0-9a-f]{32})$/, viewFailureMode],
  [/^faults$/, viewFaultExperiments],
  [/^faults\/analyses\/(fan_[0-9a-f]{32})$/, viewFaultAnalysis],
  [/^drift$/, viewDriftAnalyses],
  [/^drift\/analyses\/(dan_[0-9a-f]{32})$/, viewDriftAnalysis],
  [/^stats$/, viewStatsList],
  [/^stats\/analyses\/(sta_[0-9a-f]{32})$/, viewStatsAnalysis],
  [/^benchmark$/, viewBenchmarkList],
  [/^benchmark\/results\/(brs_[0-9a-f]{32})$/, viewBenchmarkResult],
  [/^reproducibility$/, viewReproducibilityList],
  [/^reproducibility\/attempts\/(rpa_[0-9a-f]{32})$/, viewReproductionAttempt],
  [/^graph$/, viewGraphSnapshots],
  [/^graph\/snapshots\/(gsn_[0-9a-f]{32})$/, viewGraphSnapshot],
  [/^reports$/, viewReportList],
  [/^reports\/(rpt_[0-9a-f]{32})$/, viewReport],
  [/^dossiers$/, viewDossierList],
  [/^dossiers\/(dsr_[0-9a-f]{32})$/, viewDossier],
];

async function router() {
  const hash = (location.hash || "#/investigations").slice(2);
  document.querySelectorAll("nav a").forEach((a) => a.classList.toggle("active", hash.startsWith(a.getAttribute("href").slice(2))));
  for (const [re, fn] of routes) {
    const m = hash.match(re);
    if (m) {
      app.innerHTML = "Loading…";
      try {
        app.innerHTML = await fn(...m.slice(1));
      } catch (e) {
        app.innerHTML = `<p class="error">${esc(e.message)}</p>`;
      }
      return;
    }
  }
  app.innerHTML = `<p>Unknown route.</p>`;
}
window.addEventListener("hashchange", router);
router();

// --- investigations ------------------------------------------------------------------------

async function viewInvestigationList() {
  const page = await api("/api/investigations");
  return card(`<h2>Investigations</h2>` + table(page.items, [["id", "ID"], ["name", "Name"], ["question", "Question"]]));
}

async function viewInvestigation(id) {
  const inv = await api(`/api/investigations/${id}`);
  return card(`<h2>${esc(inv.name)}</h2>${traceChain(inv.id)}${fieldList(inv, ["experiment_ids", "report_ids", "dossier_ids"])}
    <h3>Experiments</h3>${inv.experiment_ids.map(idLink).join("<br>") || '<span class="unavailable">none</span>'}
    <h3>Reports</h3>${inv.report_ids.map(idLink).join("<br>") || '<span class="unavailable">none</span>'}
    <h3>Dossiers</h3>${inv.dossier_ids.map(idLink).join("<br>") || '<span class="unavailable">none</span>'}`);
}

async function viewExperiment(id) {
  const exp = await api(`/api/investigations/experiments/${id}`);
  return card(`<h2>${esc(exp.name)}</h2>${traceChain(exp.investigation_id, exp.id)}${fieldList(exp, ["run_ids"])}
    <h3>Runs</h3>${exp.run_ids.map(idLink).join("<br>") || '<span class="unavailable">none</span>'}`);
}

async function viewRun(id) {
  const run = await api(`/api/investigations/runs/${id}`);
  return card(`<h2>Run</h2>${traceChain(run.experiment_id, run.id)}${fieldList(run, ["outcome", "provenance"])}
    <h3>Outcome</h3>${typeof run.outcome === "object" ? fieldList(run.outcome) : renderValue(run.outcome)}
    <h3>Provenance</h3>${typeof run.provenance === "object" ? fieldList(run.provenance) : renderValue(run.provenance)}`);
}

// --- reliability -------------------------------------------------------------------------------

async function viewReliabilityList() {
  const page = await api("/api/reliability/profiles");
  return card(`<h2>Reliability Profiles</h2>` + table(page.items, [["id", "ID"], ["scope", "Scope"], ["split", "Split"], ["run_id", "Run"]]));
}

async function viewReliabilityProfile(id) {
  const p = await api(`/api/reliability/profiles/${id}`);
  const chart = await api(`/api/viz/reliability/${id}/dimensions`);
  const div = document.createElement("div");
  div.appendChild(charts.bar(chart.points.map((d) => ({ label: d.dimension, value: d.status === "SUPPORTED" ? 1 : 0 })), chart));
  return card(`<h2>Reliability Profile</h2>${traceChain(p.investigation_id, p.run_id, p.id)}
    <p><em>Per-dimension status -- never an aggregate score.</em></p>
    ${div.outerHTML}
    ${fieldList(p, ["references"])}
    <h3>Dimension status</h3>${fieldList(p.dimension_status)}
    <h3>References</h3>` + table(p.references, [["dimension", "Dimension"], ["ref_kind", "Kind"], ["ref_id", "Ref"], ["note", "Note"]]));
}

// --- failures ------------------------------------------------------------------------------

async function viewFailureClusters() {
  const page = await api("/api/failures/clusters");
  return card(`<h2>Failure Clusters</h2>` + table(page.items, [["id", "ID"], ["algorithm", "Algorithm"], ["threshold", "Threshold"]]));
}

async function viewFailureCluster(id) {
  const c = await api(`/api/failures/clusters/${id}`);
  return card(`<h2>Failure Cluster</h2>${traceChain(c.investigation_id, c.id)}${fieldList(c, ["mode_ids"])}
    <h3>Modes</h3>${c.mode_ids.map(idLink).join("<br>") || '<span class="unavailable">none</span>'}`);
}

async function viewFailureMode(id) {
  const m = await api(`/api/failures/modes/${id}`);
  return card(`<h2>${esc(m.title)}</h2>${traceChain(m.investigation_id, m.cluster_id, m.id)}${fieldList(m, ["evidence", "relationships"])}
    <h3>Evidence</h3>${Array.isArray(m.evidence) ? table(m.evidence, [["id", "ID"], ["evidence_kind", "Kind"], ["summary", "Summary"]]) : renderValue(m.evidence)}
    <h3>Relationships</h3>${table(m.relationships, [["subject_id", "Subject"], ["predicate", "Predicate"], ["object_id", "Object"]])}`);
}

// --- faults --------------------------------------------------------------------------------

async function viewFaultExperiments() {
  const page = await api("/api/faults/experiments");
  return card(`<h2>Fault Experiments</h2>` + table(page.items, [["id", "ID"], ["name", "Name"], ["status", "Status"]]));
}

async function viewFaultAnalysis(id) {
  const a = await api(`/api/faults/analyses/${id}`);
  const chart = await api(`/api/viz/faults/${a.fault_experiment_id}/degradation`);
  const div = document.createElement("div");
  if (Array.isArray(chart.points) && chart.points.length) {
    div.appendChild(charts.lineWithBand(chart.points, chart));
  } else {
    div.appendChild(document.createTextNode("degradation curve: unavailable"));
  }
  return card(`<h2>Fault Analysis</h2>${traceChain(a.fault_experiment_id, a.run_id, a.id)}
    <p><em>${esc(chart.comparison)}</em></p>
    ${div.outerHTML}
    ${fieldList(a)}`);
}

// --- drift -----------------------------------------------------------------------------------

async function viewDriftAnalyses() {
  const page = await api("/api/drift/analyses");
  return card(`<h2>Drift Analyses</h2>` + table(page.items, [["id", "ID"], ["baseline_run_id", "Baseline Run"], ["analysis_status", "Status"]]));
}

async function viewDriftAnalysis(id) {
  const a = await api(`/api/drift/analyses/${id}`);
  const chart = await api(`/api/viz/drift/${id}/trajectory`);
  return card(`<h2>Drift Analysis</h2>${traceChain(a.investigation_id, a.baseline_run_id, a.run_id, a.id)}
    <p>${esc(chart.metric)} | population: ${esc(chart.population)} | comparison: ${esc(chart.comparison)}</p>
    <h3>Summary</h3>${typeof chart.points === "object" ? fieldList(chart.points) : renderValue(chart.points)}
    ${fieldList(a, ["spec", "summary"])}`);
}

// --- stats -----------------------------------------------------------------------------------

async function viewStatsList() {
  const page = await api("/api/stats/analyses");
  return card(`<h2>Statistical Analyses</h2>` + table(page.items, [["id", "ID"], ["analysis_kind", "Kind"], ["analysis_status", "Status"]]));
}

async function viewStatsAnalysis(id) {
  const a = await api(`/api/stats/analyses/${id}`);
  const chart = await api(`/api/viz/stats/${id}/effect`);
  return card(`<h2>Statistical Analysis</h2>${traceChain(a.id)}
    <p>${esc(chart.metric)} | population: ${esc(chart.population)}</p>
    <h3>Result</h3>${typeof chart.points === "object" ? fieldList(chart.points) : renderValue(chart.points)}
    <h3>Uncertainty</h3>${typeof chart.uncertainty === "object" ? fieldList(chart.uncertainty) : renderValue(chart.uncertainty)}
    <h3>Config</h3>${fieldList(a.config)}
    <h3>Sources</h3>${fieldList(a.sources)}`);
}

// --- benchmark -------------------------------------------------------------------------------

async function viewBenchmarkList() {
  const page = await api("/api/benchmark/benchmarks");
  return card(`<h2>Benchmarks</h2><p><em>EXPERIONYX never declares a universal "best model."</em></p>` +
    table(page.items, [["id", "ID"], ["name", "Name"], ["model_record_id", "Model"], ["dataset_record_id", "Dataset"]]));
}

async function viewBenchmarkResult(id) {
  const r = await api(`/api/benchmark/results/${id}`);
  return card(`<h2>Benchmark Result</h2>${traceChain(r.benchmark_id, r.run_id, r.id)}${fieldList(r, ["units"])}
    <h3>Units</h3>${table(r.units, [["unit_key", "Unit"], ["unit_kind", "Kind"], ["unit_status", "Status"]])}`);
}

// --- reproducibility -------------------------------------------------------------------------

async function viewReproducibilityList() {
  const page = await api("/api/reproducibility/attempts");
  return card(`<h2>Reproduction Attempts</h2>` + table(page.items, [["id", "ID"], ["target_kind", "Target Kind"], ["outcome", "Outcome"]]));
}

async function viewReproductionAttempt(id) {
  const a = await api(`/api/reproducibility/attempts/${id}`);
  return card(`<h2>Reproduction Attempt</h2>${traceChain(a.investigation_id, a.target_id, a.id)}${fieldList(a)}`);
}

// --- graph -----------------------------------------------------------------------------------

async function viewGraphSnapshots() {
  const page = await api("/api/graph/snapshots");
  return card(`<h2>Graph Snapshots</h2>` + table(page.items, [["id", "ID"], ["node_count", "Nodes"], ["edge_count", "Edges"], ["truncated", "Truncated"]]));
}

async function viewGraphSnapshot(id) {
  const s = await api(`/api/graph/snapshots/${id}`);
  const startId = new URLSearchParams(location.hash.split("?")[1] || "").get("node");
  let neighborhoodHtml = `<p class="unavailable">pick a starting node id via ?node=gnd_...</p>`;
  if (startId) {
    const nb = await api(`/api/graph/snapshots/${id}/nodes/${startId}/related?max_depth=2&max_visited=200`);
    const div = document.createElement("div");
    div.appendChild(charts.circularGraph(nb.nodes, nb.edges));
    neighborhoodHtml = `<p>visited: ${nb.visited_count}, truncated: ${nb.truncated}</p>${div.outerHTML}` +
      table(nb.nodes, [["id", "ID"], ["kind", "Kind"], ["ref_id", "Ref"], ["label", "Label"], ["resolved", "Resolved"]]);
  }
  return card(`<h2>Graph Snapshot</h2>${traceChain(s.investigation_id, s.run_id, s.id)}${fieldList(s, ["summary"])}
    <h3>Summary</h3>${fieldList(s.summary)}
    <h3>Neighborhood (bounded)</h3>${neighborhoodHtml}`);
}

// --- reports / dossiers -----------------------------------------------------------------------

async function viewReportList() {
  const page = await api("/api/reports");
  return card(`<h2>Reports</h2>` + table(page.items, [["id", "ID"], ["title", "Title"], ["report_type", "Type"], ["status", "Status"]]));
}

async function viewReport(id) {
  const r = await api(`/api/reports/${id}`);
  const findings = await api(`/api/reports/${id}/findings`);
  return card(`<h2>${esc(r.title)}</h2>${traceChain(r.investigation_id, r.id)}
    <p><a class="btn" href="/api/reports/${id}/export.md" target="_blank">Export Markdown</a></p>
    ${fieldList(r, ["sections", "finding_ids"])}
    <h3>Claims / Findings</h3>${table(findings.items, [["id", "ID"], ["statement", "Statement"], ["status", "Status"]])}
    <h3>Limitations</h3>${r.limitations.length ? r.limitations.map(esc).join("<br>") : '<span class="unavailable">none recorded</span>'}
    <h3>Evidence gaps</h3>${r.evidence_gaps.length ? r.evidence_gaps.map(esc).join("<br>") : '<span class="unavailable">none recorded</span>'}`);
}

async function viewDossierList() {
  const page = await api("/api/dossiers");
  return card(`<h2>Evidence Dossiers</h2>` + table(page.items, [["id", "ID"], ["research_question", "Question"], ["source_kind", "Source"]]));
}

async function viewDossier(id) {
  const d = await api(`/api/dossiers/${id}`);
  const findings = await api(`/api/dossiers/${id}/findings`);
  return card(`<h2>${esc(d.research_question)}</h2>${traceChain(d.investigation_id, d.id)}
    <p><a class="btn" href="/api/dossiers/${id}/export.md" target="_blank">Export Markdown</a></p>
    ${fieldList(d, ["item_ids", "finding_ids"])}
    <h3>Sufficiency findings</h3>${table(findings.items, [["id", "ID"], ["statement", "Statement"], ["status", "Status"]])}
    <h3>Limitations</h3>${d.limitations.length ? d.limitations.map(esc).join("<br>") : '<span class="unavailable">none recorded</span>'}`);
}
