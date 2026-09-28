import { useEffect, useState } from "react";
import InputConfigBadge from "./InputConfigBadge";
import ExecutionDetails from "./ExecutionDetails";
import { computedLabel, formatValue as formatComputedValue, formatWarning } from "../utils/format";

// T15: bullet points instead of label/value rows -- each fact is shown only
// when its backing field is non-null, so an un-overridden response (no
// router_suggested_intent/override_reason) reads as just "Router: X%" +
// "Method: ...", not a row of empty dashes.
function ConfidenceBlock({ confidence, intent }) {
  if (!confidence) return null;

  const points = [];

  points.push(
    confidence.router == null
      ? "Router: n/a (overridden)"
      : `Router: ${Math.round(confidence.router * 100)}% for '${intent}'`
  );

  if (confidence.analysis != null) {
    points.push(`Analysis: ${Math.round(confidence.analysis * 100)}%`);
  }

  if (confidence.router_suggested_intent != null) {
    const score =
      confidence.router_suggested_score != null
        ? ` (${Math.round(confidence.router_suggested_score * 100)}%)`
        : "";
    points.push(`Router suggested: '${confidence.router_suggested_intent}'${score}`);
  }

  if (confidence.override_reason) {
    points.push(`Overridden because: ${confidence.override_reason}`);
  }

  if (confidence.method_basis) {
    points.push(`Method: ${confidence.method_basis}`);
  }

  return (
    <ul className="confidence-points">
      {points.map((point, index) => (
        <li key={index}>{point}</li>
      ))}
    </ul>
  );
}

function RegionsTable({ regions, totalRegionCount, label }) {
  if (!Array.isArray(regions) || regions.length === 0) return null;

  return (
    <div className="computed-regions">
      <h4 className="card-title">{label}</h4>
      <table className="regions-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Bbox (x, y, w, h)</th>
            <th>Area</th>
            <th>Class change</th>
          </tr>
        </thead>
        <tbody>
          {regions.map((region, index) => (
            <tr key={index}>
              <td>{index + 1}</td>
              <td>{region.bbox.join(", ")}</td>
              <td>
                {region.area_km2 != null
                  ? `${region.area_km2.toFixed(4)} km²`
                  : `${region.area_px.toLocaleString()} px`}
              </td>
              <td>{region.dominant_class_change || "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {totalRegionCount > regions.length && (
        <p className="regions-note">
          Showing top {regions.length} of {totalRegionCount} detected regions.
        </p>
      )}
    </div>
  );
}

function ComputedTable({ computed }) {
  if (!computed || Object.keys(computed).length === 0) return null;

  const entries = Object.entries(computed).filter(
    ([key]) => key !== "regions" && key !== "total_region_count"
  );
  if (entries.length === 0) return null;

  return (
    <div className="computed-table">
      <h4 className="card-title">Computed values</h4>
      <table>
        <thead>
          <tr>
            <th>Measurement</th>
            <th>Value</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([key, value]) => (
            <tr key={key}>
              <td className="computed-key">{computedLabel(key, computed)}</td>
              <td className={value === null ? "computed-value-na" : "computed-value"}>
                {formatComputedValue(value)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function AnalysisResult({
  response,
  visualizationLayers = [],
  onExploreChanges,
  onDownloadReport,
}) {
  const [activeTab, setActiveTab] = useState("summary");

  // A fresh analysis should always land on Summary, not whichever tab was
  // active for the previous result.
  useEffect(() => {
    setActiveTab("summary");
  }, [response?.request_id]);

  if (!response) return null;

  if (response.status === "failed") {
    return (
      <section className="result-card result-card-failed">
        <div className="result-header">
          <span className="result-label">ANALYSIS FAILED</span>
        </div>
        <p className="result-message error-message">{response.answer}</p>
        {response.warnings?.length > 0 && (
          <p className="failure-code">Code: {response.warnings.join(", ")}</p>
        )}
      </section>
    );
  }

  const hasComputed = Object.entries(response.computed || {}).some(
    ([key]) => key !== "regions" && key !== "total_region_count"
  );
  const hasRegions = (response.computed?.regions || []).length > 0;
  const hasExecution = (response.execution || []).length > 0;
  // T17: "Change regions" only makes sense for the bitemporal `change`
  // intent -- a single-image VQA/locate "location" result has regions too,
  // but nothing "changed" (single image, no before/after).
  const regionsLabel = response.intent === "change" ? "Change regions" : "Detected regions";
  const exploreLabel = response.input_config === "single" ? "Explore Evidence" : "Explore Changes";

  const tabs = [
    { id: "summary", label: "Summary", show: true },
    { id: "computed", label: "Computed values", show: hasComputed },
    { id: "regions", label: regionsLabel, show: hasRegions },
    { id: "execution", label: "Execution details", show: hasExecution },
  ].filter((tab) => tab.show);

  const currentTab = tabs.some((tab) => tab.id === activeTab) ? activeTab : "summary";

  return (
    <section className="result-card">
      <div className="result-header">
        <span className="result-label">ANALYSIS RESULT</span>
        <span className="result-status">
          {response.status === "partial" ? "Partial result" : "Complete"}
        </span>
      </div>

      <div className="result-badges">
        <InputConfigBadge inputConfig={response.input_config} />
        <span className="intent-badge">{response.intent}</span>
      </div>

      <div className="result-tabs" role="tablist">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={currentTab === tab.id}
            className={`result-tab ${currentTab === tab.id ? "active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="result-tab-panel">
        {currentTab === "summary" && (
          <>
            <div className="result-section">
              <h4 className="card-title">Result</h4>
              {/* T14: answer_points (structured facts from the backend) render
                  as clean bullets when present; the frontend never splits
                  `answer` itself into points -- each string is already one
                  complete fact. */}
              {response.answer_points?.length > 0 ? (
                <ul className="result-message result-answer-points">
                  {response.answer_points.map((point, index) => (
                    <li key={index}>{point}</li>
                  ))}
                </ul>
              ) : (
                <p className="result-message">{response.answer}</p>
              )}
            </div>

            {response.confidence && (
              <div className="result-section">
                <h4 className="card-title">Confidence</h4>
                <ConfidenceBlock confidence={response.confidence} intent={response.intent} />
              </div>
            )}

            {response.warnings?.length > 0 && (
              <div className="warnings-list">
                <h4 className="card-title">Warnings</h4>
                <ul>
                  {response.warnings.map((warning, index) => {
                    const { text, code } = formatWarning(warning);
                    return (
                      <li key={index}>
                        {text}
                        {code && <span className="warning-code"> ({code})</span>}
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}

            <div className="result-actions">
              {visualizationLayers.length > 0 && (
                <button type="button" className="explore-button" onClick={onExploreChanges}>
                  <span>{exploreLabel}</span>
                  <span className="explore-arrow">→</span>
                </button>
              )}

              <button
                type="button"
                className="explore-button download-report-button"
                onClick={onDownloadReport}
              >
                <span>Download Report</span>
                <span className="explore-arrow">↓</span>
              </button>
            </div>
          </>
        )}

        {currentTab === "computed" && <ComputedTable computed={response.computed} />}

        {currentTab === "regions" && (
          <RegionsTable
            regions={response.computed?.regions}
            totalRegionCount={response.computed?.total_region_count}
            label={regionsLabel}
          />
        )}

        {currentTab === "execution" && <ExecutionDetails execution={response.execution || []} />}
      </div>
    </section>
  );
}
