// ============================================
// SatQuery API Service
// ============================================
//
// Talks to the real backend only -- no mock data, no fallback shapes. The
// unified `/analyze` endpoint (backend/backend/main.py) accepts 1-2 files, a
// query, and a per-file modality hint, and always returns the contract shape
// defined in backend/backend/contract.py (AnalysisResponse), whether the
// request succeeded, partially succeeded, or failed validation.

const API_BASE = "http://localhost:8000";

// ============================================
// UNIFIED ANALYSIS (single image, bitemporal pair, or optical+SAR pair)
// ============================================

export async function analyzeImages({ files, query, modalities = [] }) {
  const formData = new FormData();

  files.forEach((file) => {
    formData.append("files", file);
  });
  formData.append("query", query);

  // Always sent, one entry per file (even "" for auto-detect) -- the backend
  // only applies modality hints when the hint count matches the file count,
  // so a partial list would be silently ignored rather than partially applied.
  files.forEach((_, index) => {
    formData.append("modality", modalities[index] ?? "");
  });

  const response = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let message = `Analysis request failed (${response.status})`;
    try {
      const body = await response.json();
      if (typeof body?.answer === "string") {
        message = body.answer;
      } else if (typeof body?.detail === "string") {
        message = body.detail;
      } else if (Array.isArray(body?.detail)) {
        // FastAPI's own request-validation error shape (e.g. a missing
        // required field) -- a list of {msg, loc, ...}, not a plain string.
        message = body.detail.map((item) => item.msg || JSON.stringify(item)).join("; ");
      }
    } catch {
      // response body wasn't JSON -- fall back to the generic message above
    }
    throw new Error(message);
  }

  // Note: the backend reports request-level failures (bad file pairing,
  // unreadable file, etc.) as a 200 response with status: "failed" (per
  // CLAUDE.md's unified response contract), not as an HTTP error -- callers
  // must check `response.status`, not just whether the fetch succeeded.
  return response.json();
}

export function resolveEvidenceUrl(url) {
  if (!url) return null;
  return url.startsWith("/") ? `${API_BASE}${url}` : url;
}
