// Shared value/key formatting -- used by the live result view
// (AnalysisResult.jsx, ExecutionDetails.jsx) and the downloaded report
// (report.js) so a `null` computed value or param reads "not computable" in
// both places, not just one, and neither ever prints "null"/"undefined" or
// "[object Object]".

export function formatKey(key) {
  return String(key)
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

// T18: browsers can't render TIFF in an <img> -- used to decide, from a
// filename alone, whether a raw local blob preview is safe to render or
// needs the backend's rendered input_N_preview evidence instead.
export function isGeoTiffFilename(filename) {
  if (!filename) return false;
  const lower = filename.toLowerCase();
  return lower.endsWith(".tif") || lower.endsWith(".tiff");
}

export function formatValue(value) {
  if (value === null || value === undefined) return "not computable";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") {
    return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(2);
  }
  if (Array.isArray(value)) return value.length ? value.join(", ") : "none";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

// Compact "key: value; key2: value2" rendering of an ExecutionStep's params
// object -- used by both the live execution panel and the report's
// execution table.
export function formatParams(params) {
  if (!params || Object.keys(params).length === 0) return "—";
  return Object.entries(params)
    .map(([key, value]) => `${formatKey(key)}: ${formatValue(value)}`)
    .join("; ");
}

// T16: known `computed` keys -> a human label with units/meaning, covering
// every key the change/vegetation/water/optical_sar/locate handlers produce
// (backend/backend/orchestrator.py). A key not listed here (e.g. a future
// handler's output) falls back to formatKey -- never blank, just less
// polished. Used by both the live Computed values tab (AnalysisResult.jsx)
// and the downloaded report (report.js).
const COMPUTED_LABELS = {
  pct_changed: "Scene changed (% of pixels)",
  changed_pixels: "Changed pixels (count)",
  total_pixels: "Total pixels (count)",
  area_changed_km2: "Changed area (km²)",
  overlap_pct: "Image overlap (% of frame)",
  total_region_count: "Regions detected (count)",

  water_before_pct: "Water cover, before (% of overlap)",
  water_after_pct: "Water cover, after (% of overlap)",
  water_change_pct_points: "Water change (percentage points)",
  water_direction: "Water direction of change",

  vegetation_before_pct: "Vegetation cover, before (% of overlap)",
  vegetation_after_pct: "Vegetation cover, after (% of overlap)",
  vegetation_change_pct_points: "Vegetation change (percentage points)",
  vegetation_direction: "Vegetation direction of change",

  built_up_before_pct: "Built-up cover, before (% of overlap)",
  built_up_after_pct: "Built-up cover, after (% of overlap)",
  built_up_change_pct_points: "Built-up change (percentage points)",
  built_up_direction: "Built-up direction of change",

  is_true_index: "True spectral index (NDVI/NDWI)",
  method: "Method",
  changed_area_pct: "Changed area (% of overlap)",
  index_before: "Index value, before",
  index_after: "Index value, after",
  vegetation_proxy_before: "Vegetation colour-proxy mean, before",
  vegetation_proxy_after: "Vegetation colour-proxy mean, after",
  water_proxy_before: "Water colour-proxy mean, before",
  water_proxy_after: "Water colour-proxy mean, after",

  water_optical_pct: "Water cover, optical (%)",
  water_sar_pct: "Water cover, SAR (%)",
  water_agreement_pct: "Water agreement, optical vs SAR (%)",
  water_union_pct: "Water cover, optical ∪ SAR (%)",
  built_up_optical_pct: "Built-up cover, optical (%)",
  built_up_sar_pct: "Built-up cover, SAR (%)",
  built_up_agreement_pct: "Built-up agreement, optical vs SAR (%)",
  built_up_union_pct: "Built-up cover, optical ∪ SAR (%)",

  built_up_percentage: "Built-up cover (%)",

  // T17: single-image VQA (_vqa_handler) -- all three classes' stats are
  // always included as grounding data, not just the one asked about.
  water_percentage: "Water cover (%)",
  vegetation_percentage: "Vegetation cover (%)",
  water_method: "Water method",
  vegetation_method: "Vegetation method",
  built_up_method: "Built-up method",
  water_is_true_index: "Water: true index (NDWI)",
  vegetation_is_true_index: "Vegetation: true index (NDVI)",
  question_type: "Question type",
  class_asked: "Class asked about",
};

// `pct_change` is ambiguous on its own -- a true NDVI/NDWI relative change
// vs. an uncalibrated colour proxy's -- so unlike every other key its label
// depends on the sibling `is_true_index` value in the same `computed`
// object, never a fixed string (CLAUDE.md rule 3: never call a proxy a real
// index).
function pctChangeLabel(computed) {
  return computed?.is_true_index
    ? "Index mean, relative change (%) (true NDVI/NDWI)"
    : "Proxy mean, relative change (%) — colour proxy, not NDVI";
}

export function computedLabel(key, computed) {
  if (key === "pct_change") return pctChangeLabel(computed);
  return COMPUTED_LABELS[key] || formatKey(key);
}

// T16: machine-readable warning codes (backend/backend/rsio/compat.py,
// contract.py's _KNOWN_ERROR_MESSAGES) shown as plain-English text, with the
// original code kept in small text after it for anyone cross-referencing
// the API response. A warning that isn't a known code (most of them --
// handlers already write full sentences) passes through unchanged.
const WARNING_LABELS = {
  resample_required: "The two images had different sizes and were resampled to a common grid.",
  incompatible_pair: "The provided inputs could not be compared (mismatched CRS or no overlapping extent).",
  missing_band: "A required spectral band is missing from the input.",
  unsupported_format: "The uploaded file format is not supported.",
  model_unavailable: "The required model is not available right now.",
  unreadable_file: "One of the uploaded files could not be read.",
  no_inputs: "At least one input image is required.",
  too_many_inputs: "Too many input images were supplied.",
};

export function formatWarning(warning) {
  const label = WARNING_LABELS[warning];
  return label ? { text: label, code: warning } : { text: warning, code: null };
}
