// Builds the ordered "Explore Changes" layer list from a real AnalysisResponse
// -- every layer is backed by evidence (or, for a single-image "Input" layer,
// the actual uploaded file) that genuinely exists; a layer whose backing data
// is missing is simply left out, never shown as an empty placeholder.
//
// Each layer: { id, name, description, kind }, where `kind` tells the modal
// how to render it:
//   "image"   -> plain <img> from evidence.url (evidenceId)
//   "compare" -> CompareSlider (beforeEvidenceId/afterEvidenceId)
//   "regions" -> OverlayCanvas fed computed.regions (imageEvidenceId/maskEvidenceId)
//   "map"     -> MapView (bounds)

const PER_CLASS_LAYERS = [
  { evidenceId: "water_change_mask", name: "Water Change", description: "Per-class water change mask" },
  { evidenceId: "vegetation_change_mask", name: "Vegetation Change", description: "Per-class vegetation change mask" },
  { evidenceId: "built_up_change_mask", name: "Built-up Change", description: "Per-class built-up change mask" },
];

function hasEvidence(evidence, id) {
  return evidence.some((item) => item.id === id);
}

function imageLayer(id, name, description, evidenceId) {
  return { id, name, description, kind: "image", evidenceId };
}

// Real acquisition date when the file carries one (backend/backend/rsio/
// raster.py's TIFF tag extraction); otherwise a plain fallback label plus
// the real filename -- never a guessed/hardcoded year (T12).
export function dateLabel(metadata, index, fallback) {
  const input = metadata?.inputs?.[index];
  if (!input) return fallback;
  if (input.date) return input.date;
  return input.filename ? `${fallback} · ${input.filename}` : fallback;
}

function findInputBounds(metadata) {
  const input = (metadata?.inputs || []).find((item) => item.bounds);
  return input ? { bounds: input.bounds, label: input.filename } : null;
}

function buildBitemporalLayers(evidence) {
  const layers = [];

  if (hasEvidence(evidence, "before")) layers.push(imageLayer("before", "Before", "Original satellite observation", "before"));
  if (hasEvidence(evidence, "after")) layers.push(imageLayer("after", "After", "Latest satellite observation", "after"));
  if (hasEvidence(evidence, "before") && hasEvidence(evidence, "after")) {
    layers.push({
      id: "compare",
      name: "Before / After",
      description: "Interactive before and after comparison",
      kind: "compare",
      beforeEvidenceId: "before",
      afterEvidenceId: "after",
    });
  }
  if (hasEvidence(evidence, "alignment")) {
    layers.push(imageLayer("alignment", "Alignment", "Registration overlay", "alignment"));
  }
  if (hasEvidence(evidence, "change_overlay")) {
    layers.push(imageLayer("change-overlay", "Change Overlay", "Detected changes on imagery", "change_overlay"));
  }
  if (hasEvidence(evidence, "heatmap")) {
    layers.push(imageLayer("heatmap", "Change Heatmap", "Pixel-level difference intensity", "heatmap"));
  }
  if (hasEvidence(evidence, "change_mask")) {
    layers.push(imageLayer("mask", "Change Mask", "Binary change detection mask", "change_mask"));
  }
  if (hasEvidence(evidence, "change_mask") && hasEvidence(evidence, "after")) {
    layers.push({
      id: "regions",
      name: "Change Regions",
      description: "Detected regions with bounding boxes",
      kind: "regions",
      imageEvidenceId: "after",
      maskEvidenceId: "change_mask",
    });
  }
  if (hasEvidence(evidence, "difference")) {
    layers.push(imageLayer("difference", "Raw Difference", "Absolute pixel difference", "difference"));
  }

  PER_CLASS_LAYERS.forEach(({ evidenceId, name, description }) => {
    if (hasEvidence(evidence, evidenceId)) {
      layers.push(imageLayer(evidenceId, name, description, evidenceId));
    }
  });

  return layers;
}

function buildOpticalSarLayers(evidence) {
  // "instead of change layers" -- one plain layer per evidence item, no
  // before/after/alignment/etc. attempt (those ids never exist here).
  return evidence.map((item) =>
    imageLayer(item.id, item.label, `${item.modality} ${item.kind}`, item.id)
  );
}

function buildSingleImageLayers(evidence, filePreviews) {
  const layers = [];
  (filePreviews || []).forEach((src, index) => {
    if (src) {
      layers.push({
        id: `input-${index}`,
        name: filePreviews.filter(Boolean).length > 1 ? `Input ${index + 1}` : "Input",
        description: "Uploaded image",
        kind: "local-image",
        src,
      });
    }
  });
  evidence.forEach((item) => {
    layers.push(imageLayer(item.id, item.label, `${item.modality} ${item.kind}`, item.id));
  });
  return layers;
}

export function buildVisualizationLayers({ response, filePreviews = [] }) {
  if (!response) return [];

  const evidence = response.evidence || [];
  let layers;

  if (response.input_config === "optical_sar") {
    layers = buildOpticalSarLayers(evidence);
  } else if (response.input_config === "bitemporal") {
    layers = buildBitemporalLayers(evidence);
  } else {
    layers = buildSingleImageLayers(evidence, filePreviews);
  }

  const inputBounds = findInputBounds(response.metadata);
  if (inputBounds) {
    layers.push({
      id: "map",
      name: "Map View",
      description: "Real geographic footprint",
      kind: "map",
      bounds: inputBounds.bounds,
      label: inputBounds.label,
    });
  }

  return layers;
}
