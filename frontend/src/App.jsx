import { useEffect, useState } from "react";
import "./App.css";
import { analyzeImages, resolveEvidenceUrl } from "./services/api";
import { downloadReport } from "./utils/report";
import UploadPanel from "./components/UploadPanel";
import AnalysisResult from "./components/AnalysisResult";
import CompareSlider from "./components/CompareSlider";
import MapView from "./MapView";
import OverlayCanvas from "./OverlayCanvas";
import { buildVisualizationLayers, dateLabel } from "./components/visualizationLayers";

const EMPTY_SLOTS = [null, null];
const EMPTY_MODALITIES = ["", ""];

function historyTypeLabel(entry) {
  const config = entry.result?.input_config;
  if (config === "bitemporal") return "Bitemporal";
  if (config === "optical_sar") return "Optical+SAR";
  if (config === "single") return "Single";
  return entry.result?.intent || "Analysis";
}

function App() {
  // ==========================================
  // UPLOAD STATE (up to 2 files, one modality hint each)
  // ==========================================

  const [files, setFiles] = useState(EMPTY_SLOTS);
  const [filePreviews, setFilePreviews] = useState(EMPTY_SLOTS);
  const [modalities, setModalities] = useState(EMPTY_MODALITIES);

  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [analysisHistory, setAnalysisHistory] = useState(() => {
    try {
      const savedHistory = localStorage.getItem("satquery-analysis-history");
      return savedHistory ? JSON.parse(savedHistory) : [];
    } catch {
      return [];
    }
  });

  const [activeHistoryId, setActiveHistoryId] = useState(null);
  const [historyOpen, setHistoryOpen] = useState(true);

  const [showVisualization, setShowVisualization] = useState(false);
  const [activeLayer, setActiveLayer] = useState(null);

  const [showFollowUpModal, setShowFollowUpModal] = useState(false);
  const [followUpQuery, setFollowUpQuery] = useState("");
  const [followUpLoading, setFollowUpLoading] = useState(false);
  const [followUpError, setFollowUpError] = useState("");
  const [followUpThread, setFollowUpThread] = useState([]);

  const handleClearHistory = () => {
    const confirmed = window.confirm("Clear all analysis history?");
    if (!confirmed) return;

    localStorage.removeItem("satquery-analysis-history");
    setAnalysisHistory([]);
    setActiveHistoryId(null);
  };

  useEffect(() => {
    localStorage.setItem("satquery-analysis-history", JSON.stringify(analysisHistory));
  }, [analysisHistory]);

  useEffect(() => {
    setFollowUpQuery("");
    setFollowUpError("");
    setShowFollowUpModal(false);
  }, [activeHistoryId]);

  // ==========================================
  // IMAGE UPLOAD
  // ==========================================

  const handleFileChange = (index, file) => {
    setFiles((previous) => {
      const next = [...previous];
      next[index] = file;
      return next;
    });
    setFilePreviews((previous) => {
      const next = [...previous];
      next[index] = URL.createObjectURL(file);
      return next;
    });
    setResult(null);
    setError("");
  };

  const handleFileRemove = (index) => {
    setFiles((previous) => {
      const next = [...previous];
      next[index] = null;
      return next;
    });
    setFilePreviews((previous) => {
      const next = [...previous];
      next[index] = null;
      return next;
    });
    setModalities((previous) => {
      const next = [...previous];
      next[index] = "";
      return next;
    });
    setResult(null);
    setError("");
    setActiveHistoryId(null);
  };

  const handleModalityChange = (index, value) => {
    setModalities((previous) => {
      const next = [...previous];
      next[index] = value;
      return next;
    });
  };

  const resetUpload = () => {
    setQuery("");
    setResult(null);
    setError("");
    setFiles(EMPTY_SLOTS);
    setFilePreviews(EMPTY_SLOTS);
    setModalities(EMPTY_MODALITIES);
    setFollowUpThread([]);
    setFollowUpQuery("");
    setFollowUpError("");
    setActiveHistoryId(null);
    setShowVisualization(false);
    setActiveLayer(null);
  };

  // Files and their matching modality hints, in upload order, skipping empty
  // slots -- the backend only applies hints when the hint count exactly
  // matches the file count (backend/backend/orchestrator.py::run_analysis).
  const selectedFilesAndModalities = () => {
    const selectedFiles = [];
    const selectedModalities = [];
    files.forEach((file, index) => {
      if (file) {
        selectedFiles.push(file);
        selectedModalities.push(modalities[index]);
      }
    });
    return { selectedFiles, selectedModalities };
  };

  // ==========================================
  // ANALYSIS
  // ==========================================

  const handleAnalyze = async () => {
    setError("");
    setResult(null);

    const { selectedFiles, selectedModalities } = selectedFilesAndModalities();

    if (selectedFiles.length === 0) {
      setError("Please upload at least one satellite image first.");
      return;
    }

    if (!query.trim()) {
      setError("Please enter a question about the image(s).");
      return;
    }

    try {
      setLoading(true);

      const response = await analyzeImages({
        files: selectedFiles,
        query: query.trim(),
        modalities: selectedModalities,
      });

      setResult(response);

      const historyEntry = {
        id: Date.now(),
        query: query.trim(),
        result: response,
        filePreviews: [...filePreviews],
        modalities: [...modalities],
        followUpThread: [],
        createdAt: new Date().toLocaleString(),
      };

      setAnalysisHistory((previousHistory) => [historyEntry, ...previousHistory]);
      setActiveHistoryId(historyEntry.id);
    } catch (err) {
      if (err.message === "Failed to fetch") {
        setError(
          "Unable to connect to the analysis server. Please make sure the backend is running and try again."
        );
      } else {
        setError(err.message || "Something went wrong while analyzing the image(s).");
      }
    } finally {
      setLoading(false);
    }
  };

  // ==========================================
  // FOLLOW-UP QUESTIONS
  // ==========================================

  const handleFollowUpAsk = async () => {
    const trimmed = followUpQuery.trim();
    if (!trimmed) return;

    const { selectedFiles, selectedModalities } = selectedFilesAndModalities();

    if (selectedFiles.length === 0) {
      setFollowUpError("Please upload at least one image first.");
      return;
    }

    setFollowUpError("");

    try {
      setFollowUpLoading(true);

      const response = await analyzeImages({
        files: selectedFiles,
        query: trimmed,
        modalities: selectedModalities,
      });

      setFollowUpThread((previousThread) => {
        const updatedThread = [...previousThread, { query: trimmed, response }];

        setAnalysisHistory((previousHistory) =>
          previousHistory.map((entry) =>
            entry.id === activeHistoryId ? { ...entry, followUpThread: updatedThread } : entry
          )
        );

        return updatedThread;
      });

      setFollowUpQuery("");
    } catch (err) {
      if (err.message === "Failed to fetch") {
        setFollowUpError(
          "Unable to connect to the analysis server. Please make sure the backend is running and try again."
        );
      } else {
        setFollowUpError(err.message || "Something went wrong while processing the follow-up.");
      }
    } finally {
      setFollowUpLoading(false);
    }
  };

  // ==========================================
  // DOWNLOADABLE REPORT
  // ==========================================

  const handleDownloadReport = () => {
    if (!result) return;

    try {
      const summaryStats = result.computed
        ? Object.entries(result.computed)
            .filter(([key]) => key !== "regions")
            .map(([label, value]) => ({
              label,
              value: typeof value === "object" ? JSON.stringify(value) : String(value),
            }))
        : null;

      const changes = Array.isArray(result.computed?.regions)
        ? result.computed.regions.map((region, index) => ({
            id: index + 1,
            area:
              region.area_km2 != null ? `${region.area_km2} km²` : `${region.area_px} px`,
            x: region.bbox[0],
            y: region.bbox[1],
          }))
        : null;

      downloadReport({
        title: "SatQuery AI Analysis Report",
        subtitle: (result.input_config || "analysis").replace("_", " "),
        createdAt: new Date().toLocaleString(),
        query,
        answer: result.answer,
        summaryStats,
        changes,
        images: filePreviews
          .filter(Boolean)
          .map((src, index) => ({ label: `Uploaded image ${index + 1}`, src })),
      });
    } catch (err) {
      setError(err.message || "Could not generate the report.");
    }
  };

  // ==========================================
  // EXPLORE CHANGES VISUALIZATION
  // ==========================================
  // Every layer is populated only from evidence (or, for a single-image
  // "Input" layer, the actual uploaded file) that genuinely exists in the
  // current response -- a layer whose backing data is missing is left out
  // entirely, never shown as a placeholder.

  const activeVisualizationLayers =
    result && result.status !== "failed"
      ? buildVisualizationLayers({ response: result, filePreviews })
      : [];

  const renderActiveVisualizationLayer = () => {
    const layer = activeVisualizationLayers.find((item) => item.id === activeLayer);

    const unavailable = (
      <div className="visualization-empty">
        <strong>Processing output unavailable</strong>
        <span>
          This visualization will appear once the analysis backend processes
          the current input(s).
        </span>
      </div>
    );

    if (!layer) return unavailable;

    if (layer.kind === "compare") {
      const beforeEvidence = result.evidence.find((item) => item.id === layer.beforeEvidenceId);
      const afterEvidence = result.evidence.find((item) => item.id === layer.afterEvidenceId);
      return (
        <CompareSlider
          beforeSrc={resolveEvidenceUrl(beforeEvidence?.url)}
          afterSrc={resolveEvidenceUrl(afterEvidence?.url)}
          beforeLabel={dateLabel(result.metadata, 0, "Before")}
          afterLabel={dateLabel(result.metadata, 1, "After")}
        />
      );
    }

    if (layer.kind === "regions") {
      const imageEvidence = result.evidence.find((item) => item.id === layer.imageEvidenceId);
      const maskEvidence = result.evidence.find((item) => item.id === layer.maskEvidenceId);
      const overlayRegions = (result.computed?.regions || []).map((region, index) => {
        const [x, y, width, height] = region.bbox;
        return { id: index + 1, x, y, width, height, area: region.area_px };
      });
      return (
        <OverlayCanvas
          imageSrc={resolveEvidenceUrl(imageEvidence?.url)}
          maskSrc={resolveEvidenceUrl(maskEvidence?.url)}
          regions={overlayRegions}
        />
      );
    }

    if (layer.kind === "map") {
      return <MapView bounds={layer.bounds} label={layer.label} />;
    }

    if (layer.kind === "local-image") {
      return <img src={layer.src} alt={layer.name} />;
    }

    // "image"
    const evidence = result.evidence.find((item) => item.id === layer.evidenceId);
    return evidence ? (
      <img src={resolveEvidenceUrl(evidence.url)} alt={layer.name} />
    ) : (
      unavailable
    );
  };

  return (
    <div className={`app ${historyOpen ? "history-open" : "history-closed"}`}>
      {/* ======================================
          HEADER
      ====================================== */}

      <header className="header">
        <div className="header-left">
          <button
            type="button"
            className="history-toggle"
            onClick={() => setHistoryOpen((open) => !open)}
            aria-label={historyOpen ? "Close analysis history" : "Open analysis history"}
            title={historyOpen ? "Close analysis history" : "Open analysis history"}
          >
            {historyOpen ? "‹" : "☰"}
          </button>

          <div>
            <h1>SatQuery AI</h1>
            <p>Intelligent Satellite Change Detection</p>
          </div>
        </div>

        <div className="status">
          <span className="status-dot"></span>
          {loading ? "Analyzing..." : "System Ready"}
        </div>
      </header>

      {/* ======================================
          MAIN
      ====================================== */}

      <main className="main-content">
        {/* ======================================
            ANALYSIS HISTORY
        ====================================== */}

        <aside className={`history-sidebar ${historyOpen ? "open" : "closed"}`}>
          <button type="button" className="new-analysis-button" onClick={resetUpload}>
            <span>＋</span>
            <span>New Analysis</span>
          </button>

          <div className="history-title">
            <span>ANALYSIS HISTORY</span>

            {analysisHistory.length > 0 && (
              <button type="button" className="clear-history-button" onClick={handleClearHistory}>
                Clear All
              </button>
            )}
          </div>

          <div className="history-list">
            {analysisHistory.length === 0 ? (
              <div className="history-empty">
                <span>No previous analyses</span>
                <small>Your analysis questions will appear here.</small>
              </div>
            ) : (
              analysisHistory.map((entry) => (
                <button
                  type="button"
                  key={entry.id}
                  className={`history-item ${activeHistoryId === entry.id ? "active" : ""}`}
                  onClick={() => {
                    setQuery(entry.query);
                    setResult(entry.result);
                    setFilePreviews(entry.filePreviews || EMPTY_SLOTS);
                    setModalities(entry.modalities || EMPTY_MODALITIES);
                    // Actual File objects can't be restored from history
                    // (only their preview URLs/modality choices were saved),
                    // so a follow-up from a restored entry needs a re-upload.
                    setFiles(EMPTY_SLOTS);
                    setFollowUpThread(entry.followUpThread || []);
                    setError("");
                    setActiveHistoryId(entry.id);
                  }}
                >
                  <span className="history-question">
                    <span className="history-type">{historyTypeLabel(entry)}</span>
                    {entry.query}
                  </span>

                  <span className="history-date">{entry.createdAt}</span>
                </button>
              ))
            )}
          </div>
        </aside>

        <section className="intro">
          <p className="tag">SATELLITE INTELLIGENCE</p>

          <h2>
            Understand what changed
            <br />
            <span>from above.</span>
          </h2>

          <p className="description">
            Upload one satellite image for a direct question, or two images of
            the same place (over time, or a co-registered optical + SAR pair)
            to compare them.
          </p>
        </section>

        <UploadPanel
          files={files}
          filePreviews={filePreviews}
          modalities={modalities}
          query={query}
          loading={loading}
          onFileChange={handleFileChange}
          onFileRemove={handleFileRemove}
          onModalityChange={handleModalityChange}
          onQueryChange={setQuery}
          onSubmit={handleAnalyze}
        />

        {loading && <div className="loading-message">Analyzing satellite imagery...</div>}

        {error && <div className="error-message">{error}</div>}

        {result && (
          <button
            type="button"
            className="follow-up-trigger"
            onClick={() => setShowFollowUpModal(true)}
          >
            <span className="follow-up-trigger-label">FOLLOW-UP ANALYSIS</span>
            <span className="follow-up-trigger-title">
              Ask a follow-up question
              {followUpThread.length > 0 && ` (${followUpThread.length})`}
            </span>
            <span className="explore-arrow">→</span>
          </button>
        )}

        <AnalysisResult response={result} />

        {activeVisualizationLayers.length > 0 && (
          <button
            type="button"
            className="explore-button"
            onClick={() => {
              setActiveLayer(activeVisualizationLayers[0].id);
              setShowVisualization(true);
            }}
          >
            <span>Explore Changes</span>
            <span className="explore-arrow">→</span>
          </button>
        )}

        {result && result.status !== "failed" && (
          <button
            type="button"
            className="explore-button download-report-button"
            onClick={handleDownloadReport}
          >
            <span>Download Report</span>
            <span className="explore-arrow">↓</span>
          </button>
        )}
      </main>

      {/* ======================================
          VISUALIZATION WORKSPACE
      ====================================== */}

      {showVisualization && (
        <div className="visualization-overlay">
          <div className="visualization-modal">
            {/* HEADER */}

            <div className="visualization-header">
              <div>
                <span className="section-label">SATELLITE ANALYSIS</span>
                <h2>Change Visualization</h2>
                <p>Explore the processing outputs for this satellite image pair.</p>
              </div>

              <button
                type="button"
                className="close-button"
                onClick={() => setShowVisualization(false)}
              >
                ×
              </button>
            </div>

            {/* BODY */}

            <div className="visualization-body">
              {/* LAYER SIDEBAR */}

              <aside className="visualization-sidebar">
                <div className="sidebar-title">VISUALIZATION LAYERS</div>

                {activeVisualizationLayers.map((layer) => (
                  <button
                    type="button"
                    key={layer.id}
                    className={`layer-button ${activeLayer === layer.id ? "active" : ""}`}
                    onClick={() => setActiveLayer(layer.id)}
                  >
                    <span className="layer-name">{layer.name}</span>
                    <span className="layer-description">{layer.description}</span>
                  </button>
                ))}
              </aside>

              {/* IMAGE AREA */}

              <section className="visualization-view">
                <div className="visualization-toolbar">
                  <div>
                    <span className="view-label">CURRENT VIEW</span>
                    <strong>
                      {activeVisualizationLayers.find((layer) => layer.id === activeLayer)?.name}
                    </strong>
                  </div>

                  {activeLayer === "regions" && result?.computed?.regions && (
                    <div className="view-stat">
                      <strong>
                        {result.computed.total_region_count ?? result.computed.regions.length}
                      </strong>
                      <span>regions</span>
                    </div>
                  )}
                </div>

                <div className="visualization-image">{renderActiveVisualizationLayer()}</div>
              </section>
            </div>

            {/* FOOTER */}

            <div className="visualization-footer">
              <span>SatQuery AI · Earth Observation Analysis</span>

              <button type="button" onClick={() => setShowVisualization(false)}>
                Back to Analysis
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ======================================
          FOLLOW-UP MODAL
      ====================================== */}

      {showFollowUpModal && (
        <div className="followup-overlay">
          <div className="followup-modal">
            <div className="followup-header">
              <div>
                <span className="section-label">FOLLOW-UP ANALYSIS</span>
                <h2>Ask about this analysis</h2>
              </div>

              <button
                type="button"
                className="close-button"
                onClick={() => setShowFollowUpModal(false)}
              >
                ×
              </button>
            </div>

            <div className="followup-thread">
              {followUpThread.length === 0 && !followUpLoading && (
                <div className="followup-empty">
                  <strong>No follow-up questions yet</strong>
                  <span>
                    Ask something else — the model will answer using the
                    currently uploaded image(s).
                  </span>
                </div>
              )}

              {followUpThread.map((entry, index) => (
                <div className="followup-exchange" key={index}>
                  <div className="followup-question">
                    <span>You</span>
                    <p>{entry.query}</p>
                  </div>

                  <div className="followup-answer">
                    <span>SatQuery AI</span>
                    <p>{entry.response.answer}</p>
                  </div>
                </div>
              ))}

              {followUpLoading && <div className="loading-message">Analyzing follow-up...</div>}

              {followUpError && <div className="error-message">{followUpError}</div>}
            </div>

            <div className="followup-composer">
              <input
                type="text"
                placeholder="Ask a follow-up question..."
                value={followUpQuery}
                onChange={(event) => setFollowUpQuery(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    handleFollowUpAsk();
                  }
                }}
                disabled={followUpLoading}
              />

              <button type="button" onClick={handleFollowUpAsk} disabled={followUpLoading}>
                {followUpLoading ? "Asking..." : "Ask →"}
              </button>
            </div>
          </div>
        </div>
      )}

      <footer>
        <span>SatQuery AI</span>
        <span>AI-Powered Earth Observation</span>
      </footer>
    </div>
  );
}

export default App;
