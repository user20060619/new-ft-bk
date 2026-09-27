import InputConfigBadge from "./InputConfigBadge";
import EvidenceGallery from "./EvidenceGallery";
import MapView from "../MapView";

function formatKey(key) {
  return key.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatComputedValue(value) {
  if (value === null || value === undefined) return "not computable";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") {
    return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(2);
  }
  if (Array.isArray(value)) return value.join(", ") || "none";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function ConfidenceBlock({ confidence }) {
  if (!confidence) return null;

  return (
    <div className="confidence-block">
      <div className="confidence-row">
        <span>Router confidence</span>
        {/* T10: confidence.router is null when the orchestrator replaced a
            genuinely-unclear router verdict with a structural default --
            show the basis text instead of a misleading 0%. */}
        {confidence.router == null ? (
          <span className="confidence-na">n/a</span>
        ) : (
          <span>{Math.round(confidence.router * 100)}%</span>
        )}
      </div>

      {confidence.analysis != null && (
        <div className="confidence-row">
          <span>Analysis confidence</span>
          <span>{Math.round(confidence.analysis * 100)}%</span>
        </div>
      )}

      {confidence.basis && <p className="confidence-basis">{confidence.basis}</p>}
    </div>
  );
}

function RegionsTable({ regions, totalRegionCount }) {
  if (!Array.isArray(regions) || regions.length === 0) return null;

  return (
    <div className="computed-regions">
      <h4>Change regions</h4>
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
      <h4>Computed values</h4>
      <table>
        <tbody>
          {entries.map(([key, value]) => (
            <tr key={key}>
              <td className="computed-key">{formatKey(key)}</td>
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

export default function AnalysisResult({ response }) {
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

      <p className="result-message">{response.answer}</p>

      <ConfidenceBlock confidence={response.confidence} />

      <ComputedTable computed={response.computed} />
      <RegionsTable
        regions={response.computed?.regions}
        totalRegionCount={response.computed?.total_region_count}
      />

      <EvidenceGallery
        evidence={response.evidence || []}
        regions={response.computed?.regions || []}
        metadata={response.metadata}
      />

      {response.warnings?.length > 0 && (
        <div className="warnings-list">
          <h4>Warnings</h4>
          <ul>
            {response.warnings.map((warning, index) => (
              <li key={index}>{warning}</li>
            ))}
          </ul>
        </div>
      )}

      {(() => {
        const geoInput = (response.metadata?.inputs || []).find((item) => item.bounds);
        if (!geoInput) return null;
        return (
          <div className="evidence-section">
            <h4>Location</h4>
            <MapView bounds={geoInput.bounds} label={geoInput.filename} />
          </div>
        );
      })()}
    </section>
  );
}
