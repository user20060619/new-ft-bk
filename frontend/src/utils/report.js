// ============================================
// SatQuery AI — Downloadable Report Generator
// ============================================
//
// No PDF library is bundled with this project, so reports are produced
// by opening a print-formatted document in a new window and invoking the
// browser's native print dialog, where "Save as PDF" produces a real PDF
// with no extra dependency.
//
// T13: takes the real AnalysisResponse directly and builds every section
// from it (no hardcoded values, no pre-flattened prop shape assembled by the
// caller) -- Header -> Input -> Question -> Evidence -> Answer -> Computed
// values -> Execution summary -> Warnings/limitations, in that order.

import { resolveEvidenceUrl } from "../services/api";
import { computedLabel, formatParams, formatValue, formatWarning, isGeoTiffFilename } from "./format";

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => {
    switch (char) {
      case "&":
        return "&amp;";
      case "<":
        return "&lt;";
      case ">":
        return "&gt;";
      case '"':
        return "&quot;";
      default:
        return "&#39;";
    }
  });
}

function fmt(value) {
  return value === null || value === undefined || value === "" ? "—" : escapeHtml(value);
}

// --- section builders --------------------------------------------------------

function buildHeaderHtml(response, query) {
  const inputs = response.metadata?.inputs || [];
  const inputRows = inputs
    .map(
      (input) => `
        <tr>
          <td>${fmt(input.filename)}</td>
          <td>${fmt(input.modality)}</td>
          <td>${fmt(input.bands)}</td>
          <td>${fmt(input.dtype)}</td>
          <td>${input.width}&times;${input.height}</td>
          <td>${fmt(input.crs)}</td>
          <td>${input.gsd_m != null ? `${input.gsd_m.toFixed(2)} m` : "—"}</td>
          <td>${fmt(input.date)}</td>
          <td>${input.bounds ? input.bounds.map((b) => b.toFixed(4)).join(", ") : "—"}</td>
        </tr>`
    )
    .join("");

  return `
    <header>
      <span class="brand">SatQuery AI</span>
      <h1>Analysis Report</h1>
      <table class="meta-table">
        <tbody>
          <tr><td>Generated</td><td>${escapeHtml(new Date().toLocaleString())}</td></tr>
          <tr><td>Request ID</td><td>${fmt(response.request_id)}</td></tr>
          <tr><td>Query</td><td>${fmt(query)}</td></tr>
          <tr><td>Input configuration</td><td>${fmt(response.input_config)}</td></tr>
          <tr><td>Intent</td><td>${fmt(response.intent)}</td></tr>
          <tr><td>Status</td><td>${fmt(response.status)}</td></tr>
        </tbody>
      </table>
      ${
        inputRows
          ? `<h3>Input metadata</h3>
      <table class="stats">
        <thead>
          <tr>
            <th>Filename</th><th>Modality</th><th>Bands</th><th>Dtype</th><th>Size</th>
            <th>CRS</th><th>GSD</th><th>Date</th><th>Bounds (W,S,E,N)</th>
          </tr>
        </thead>
        <tbody>${inputRows}</tbody>
      </table>`
          : ""
      }
    </header>`;
}

function buildInputSectionHtml(filePreviews, metadata, evidence) {
  const inputs = metadata?.inputs || [];
  const images = (filePreviews || [])
    .map((src, index) => {
      const filename = inputs[index]?.filename;
      const label = filename || `Input ${index + 1}`;
      // T18: a raw blob preview can't render a GeoTIFF in the report either
      // -- prefer the backend's rendered input_N_preview evidence for one,
      // when it exists; JPG/PNG keep using the local blob, unchanged.
      if (isGeoTiffFilename(filename)) {
        const previewItem = (evidence || []).find((item) => item.id === `input_${index}_preview`);
        if (previewItem) return { src: resolveEvidenceUrl(previewItem.url), label };
      }
      return { src, label };
    })
    .filter((image) => image.src);

  if (images.length === 0) return "";

  const imagesHtml = images
    .map(
      (image) => `
        <figure>
          <img src="${escapeHtml(image.src)}" alt="${escapeHtml(image.label)}" />
          <figcaption>${escapeHtml(image.label)}</figcaption>
        </figure>`
    )
    .join("");

  return `<h3>Input</h3><div class="images">${imagesHtml}</div>`;
}

function buildQuestionHtml(query) {
  return `
    <div class="query-block">
      <span class="label">Question</span>
      <p class="question">${fmt(query)}</p>
    </div>`;
}

function buildEvidenceSectionHtml(response) {
  const reportAssets = new Set(response.report_assets || []);
  const items = (response.evidence || []).filter((item) => reportAssets.has(item.id));
  if (items.length === 0) return "";

  const imagesHtml = items
    .map(
      (item) => `
        <figure>
          <img src="${escapeHtml(resolveEvidenceUrl(item.url))}" alt="${escapeHtml(item.label)}" />
          <figcaption>${escapeHtml(item.label)}</figcaption>
        </figure>`
    )
    .join("");

  return `<h3>Evidence</h3><div class="images">${imagesHtml}</div>`;
}

function buildAnswerHtml(response) {
  // T14: mirrors AnalysisResult.jsx -- answer_points (structured facts from
  // the backend) render as bullets when present; `answer` is the fallback
  // for handlers that don't build a points list.
  const body =
    response.answer_points?.length > 0
      ? `<ul class="answer-points">${response.answer_points
          .map((point) => `<li>${fmt(point)}</li>`)
          .join("")}</ul>`
      : `<p class="answer">${fmt(response.answer)}</p>`;

  return `
    <div class="query-block">
      <span class="label">Answer</span>
      ${body}
    </div>`;
}

function buildComputedSectionHtml(computed) {
  if (!computed || Object.keys(computed).length === 0) return "";

  const entries = Object.entries(computed).filter(
    ([key]) => key !== "regions" && key !== "total_region_count"
  );

  const statsHtml = entries.length
    ? `
      <table class="stats">
        <thead>
          <tr><th>Measurement</th><th>Value</th></tr>
        </thead>
        <tbody>
          ${entries
            .map(
              ([key, value]) => `
            <tr>
              <td>${escapeHtml(computedLabel(key, computed))}</td>
              <td>${escapeHtml(formatValue(value))}</td>
            </tr>`
            )
            .join("")}
        </tbody>
      </table>`
    : "";

  const regions = computed.regions;
  const totalRegionCount = computed.total_region_count;
  const regionsHtml =
    Array.isArray(regions) && regions.length
      ? `
        <h4>Regions</h4>
        <table class="stats">
          <thead>
            <tr><th>#</th><th>Bbox (x, y, w, h)</th><th>Area</th><th>Class change</th></tr>
          </thead>
          <tbody>
            ${regions
              .map(
                (region, index) => `
              <tr>
                <td>${index + 1}</td>
                <td>${region.bbox.join(", ")}</td>
                <td>${
                  region.area_km2 != null
                    ? `${region.area_km2.toFixed(4)} km&sup2;`
                    : `${region.area_px.toLocaleString()} px`
                }</td>
                <td>${fmt(region.dominant_class_change)}</td>
              </tr>`
              )
              .join("")}
          </tbody>
        </table>
        ${
          totalRegionCount > regions.length
            ? `<p class="note">Showing top ${regions.length} of ${totalRegionCount} detected regions.</p>`
            : ""
        }`
      : "";

  if (!statsHtml && !regionsHtml) return "";
  return `<h3>Computed values</h3>${statsHtml}${regionsHtml}`;
}

function buildExecutionSectionHtml(execution) {
  if (!Array.isArray(execution) || execution.length === 0) return "";

  const rows = execution
    .map(
      (step) => `
        <tr>
          <td>${step.step}</td>
          <td>${escapeHtml(step.name)}</td>
          <td>${escapeHtml(step.tool)}</td>
          <td>${escapeHtml(step.method)}</td>
          <td>${escapeHtml(formatParams(step.params))}</td>
          <td>${fmt(step.output_summary)}</td>
          <td>${step.ms}</td>
          <td>${step.fallback ? "Yes" : "No"}</td>
        </tr>`
    )
    .join("");

  return `
    <h3>Execution summary</h3>
    <table class="stats execution-summary">
      <thead>
        <tr>
          <th>#</th><th>Step</th><th>Tool</th><th>Method</th>
          <th>Key params</th><th>Output summary</th><th>ms</th><th>Fallback</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function buildConfidenceHtml(confidence) {
  if (!confidence) return "";
  const routerLine =
    confidence.router == null
      ? "n/a (see basis below)"
      : `${Math.round(confidence.router * 100)}%`;
  const analysisLine = confidence.analysis != null ? `${Math.round(confidence.analysis * 100)}%` : null;

  return `
    <h3>Confidence</h3>
    <table class="meta-table">
      <tbody>
        <tr><td>Router</td><td>${routerLine}</td></tr>
        ${analysisLine ? `<tr><td>Analysis</td><td>${analysisLine}</td></tr>` : ""}
      </tbody>
    </table>
    ${confidence.basis ? `<p class="note">${fmt(confidence.basis)}</p>` : ""}`;
}

function buildWarningsSectionHtml(warnings) {
  if (!Array.isArray(warnings) || warnings.length === 0) return "";
  const items = warnings
    .map((warning) => {
      const { text, code } = formatWarning(warning);
      const codeHtml = code ? ` <span class="warning-code">(${escapeHtml(code)})</span>` : "";
      return `<li>${escapeHtml(text)}${codeHtml}</li>`;
    })
    .join("");
  return `<h3>Warnings / limitations</h3><ul class="warnings">${items}</ul>`;
}

function buildReportHtml({ response, query, filePreviews }) {
  return `<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>SatQuery AI Analysis Report</title>
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 40px;
    color: #1B2530;
    background: #FFFFFF;
    font-family: "Segoe UI", Inter, system-ui, sans-serif;
  }
  header {
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 2px solid #1E5F8C;
  }
  header .brand {
    color: #1E5F8C;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
  }
  header h1 {
    margin: 6px 0 12px;
    font-size: 22px;
  }
  h3 {
    margin: 24px 0 10px;
    font-size: 14px;
  }
  h4 {
    margin: 16px 0 8px;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #66727F;
  }
  .query-block {
    padding: 14px 16px;
    border: 1px solid #DCE8F2;
    border-radius: 10px;
    background: #F4F8FC;
    margin-bottom: 16px;
  }
  .query-block .label {
    display: block;
    margin-bottom: 4px;
    color: #66727F;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
  }
  .query-block .question,
  .query-block .answer {
    margin: 0;
    font-size: 13px;
    line-height: 1.6;
  }
  .query-block .answer-points {
    margin: 0;
    padding-left: 18px;
    font-size: 13px;
    line-height: 1.6;
  }
  .query-block .answer-points li {
    margin-bottom: 4px;
  }
  .query-block .answer-points li:last-child {
    margin-bottom: 0;
  }
  .images {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    margin-top: 8px;
  }
  figure {
    margin: 0;
    width: 220px;
  }
  figure img {
    display: block;
    width: 100%;
    border: 1px solid #CDD3DA;
    border-radius: 8px;
  }
  figcaption {
    margin-top: 6px;
    color: #66727F;
    font-size: 10px;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  table.meta-table, table.stats {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    margin-bottom: 8px;
  }
  table.meta-table td,
  table.stats th,
  table.stats td {
    padding: 8px 10px;
    border-bottom: 1px solid #E5EAF0;
    text-align: left;
  }
  table.meta-table td:first-child {
    color: #66727F;
    width: 180px;
  }
  table.stats th {
    color: #66727F;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  table.execution-summary { font-size: 10.5px; }
  .note {
    font-size: 11px;
    color: #66727F;
    margin: 4px 0 0;
  }
  ul.warnings {
    font-size: 12px;
    line-height: 1.7;
    padding-left: 18px;
  }
  .warning-code {
    color: #94A3B8;
    font-size: 10px;
    font-family: ui-monospace, monospace;
  }
  footer {
    margin-top: 32px;
    padding-top: 12px;
    border-top: 1px solid #E5EAF0;
    color: #94A3B8;
    font-size: 10px;
  }
  @media print {
    body { padding: 0; }
  }
</style>
</head>
<body>
  ${buildHeaderHtml(response, query)}
  ${buildInputSectionHtml(filePreviews, response.metadata, response.evidence)}
  ${buildQuestionHtml(query)}
  ${buildEvidenceSectionHtml(response)}
  ${buildAnswerHtml(response)}
  ${buildComputedSectionHtml(response.computed)}
  ${buildConfidenceHtml(response.confidence)}
  ${buildExecutionSectionHtml(response.execution)}
  ${buildWarningsSectionHtml(response.warnings)}

  <footer>SatQuery AI &middot; AI-Powered Earth Observation &middot; This report reflects computed values from the analysis backend.</footer>
</body>
</html>`;
}

// Opens the report in a new tab and triggers the browser's print dialog
// once every embedded image has finished loading, so "Save as PDF" in
// that dialog produces a complete downloadable PDF.
export function downloadReport({ response, query, filePreviews }) {
  const html = buildReportHtml({ response, query, filePreviews });

  const reportWindow = window.open("", "_blank");

  if (!reportWindow) {
    throw new Error(
      "Your browser blocked the report window. Please allow pop-ups for this site and try again."
    );
  }

  reportWindow.document.open();
  reportWindow.document.write(html);
  reportWindow.document.close();

  const triggerPrint = () => {
    reportWindow.focus();
    reportWindow.print();
  };

  const images = reportWindow.document.images;

  if (!images || images.length === 0) {
    setTimeout(triggerPrint, 300);
    return;
  }

  let settled = 0;

  const onImageSettled = () => {
    settled += 1;
    if (settled >= images.length) {
      setTimeout(triggerPrint, 200);
    }
  };

  Array.from(images).forEach((image) => {
    if (image.complete) {
      onImageSettled();
    } else {
      image.addEventListener("load", onImageSettled);
      image.addEventListener("error", onImageSettled);
    }
  });
}
