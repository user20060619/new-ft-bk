import { useMemo, useState } from "react";
import { resolveEvidenceUrl } from "../services/api";
import OverlayCanvas from "../OverlayCanvas";
import CompareSlider from "./CompareSlider";
import { dateLabel } from "./visualizationLayers";

const MODALITY_LABELS = { optical: "Optical", sar: "SAR", fused: "Fused", none: "Other" };

function findEvidence(evidence, id) {
  return evidence.find((item) => item.id === id) || null;
}

// computed.regions[].bbox is [x, y, w, h] (rsio contract); OverlayCanvas
// expects {x, y, width, height, area} per region.
function toOverlayRegion(region, index) {
  const [x, y, width, height] = region.bbox;
  return { id: index + 1, x, y, width, height, area: region.area_px };
}

export default function EvidenceGallery({ evidence = [], regions = [], metadata = null }) {
  const [lightboxUrl, setLightboxUrl] = useState(null);

  const before = findEvidence(evidence, "before");
  const after = findEvidence(evidence, "after");
  const changeOverlay = findEvidence(evidence, "change_overlay");
  const changeMask = findEvidence(evidence, "change_mask");

  const overlayRegions = useMemo(() => regions.map(toOverlayRegion), [regions]);

  if (evidence.length === 0) return null;

  const specialIds = new Set(
    [before, after, changeMask].filter(Boolean).map((item) => item.id)
  );
  const remaining = evidence.filter((item) => !specialIds.has(item.id));

  const groups = remaining.reduce((acc, item) => {
    const key = item.modality || "none";
    (acc[key] = acc[key] || []).push(item);
    return acc;
  }, {});

  // The base image for the interactive region view: prefer the plain "after"
  // image (change_mask is drawn on top interactively by OverlayCanvas
  // itself, so a plain base avoids double-drawing boxes that are already
  // baked into change_overlay.jpg).
  const regionBaseImage = after || changeOverlay;

  return (
    <div className="evidence-gallery">
      {before && after && (
        <div className="evidence-section">
          <h4>Before / After</h4>
          <CompareSlider
            beforeSrc={resolveEvidenceUrl(before.url)}
            afterSrc={resolveEvidenceUrl(after.url)}
            beforeLabel={dateLabel(metadata, 0, "Before")}
            afterLabel={dateLabel(metadata, 1, "After")}
          />
        </div>
      )}

      {changeMask && regionBaseImage && (
        <div className="evidence-section">
          <h4>Change regions</h4>
          <OverlayCanvas
            imageSrc={resolveEvidenceUrl(regionBaseImage.url)}
            maskSrc={resolveEvidenceUrl(changeMask.url)}
            regions={overlayRegions}
          />
        </div>
      )}

      {Object.entries(groups).map(([modality, items]) => (
        <div className="evidence-section" key={modality}>
          <h4>{MODALITY_LABELS[modality] || modality}</h4>
          <div className="evidence-thumb-grid">
            {items.map((item) => (
              <button
                key={item.id}
                type="button"
                className="evidence-thumb"
                onClick={() => setLightboxUrl(resolveEvidenceUrl(item.url))}
              >
                <img src={resolveEvidenceUrl(item.url)} alt={item.label} />
                <span className="evidence-thumb-label">{item.label}</span>
                <span className="evidence-thumb-kind">{item.kind}</span>
              </button>
            ))}
          </div>
        </div>
      ))}

      {lightboxUrl && (
        <div className="evidence-lightbox" onClick={() => setLightboxUrl(null)}>
          <img src={lightboxUrl} alt="Evidence detail" />
        </div>
      )}
    </div>
  );
}
