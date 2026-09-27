import { useState } from "react";

// Before/after comparison slider -- extracted from App.jsx's original
// renderComparisonSlider (T11), generalised to take image sources as props
// instead of closing over local upload-preview state. Reuses the existing
// `comparison-*` CSS classes in App.css unchanged.
//
// Fed the backend's own "before"/"after" evidence images (both already
// resampled to the same common grid by _change_handler) rather than the raw
// uploaded files, which may differ in size.
export default function CompareSlider({
  beforeSrc,
  afterSrc,
  beforeLabel = "BEFORE",
  afterLabel = "AFTER",
}) {
  const [comparePosition, setComparePosition] = useState(50);

  if (!beforeSrc || !afterSrc) {
    return (
      <div className="visualization-empty">
        <strong>Comparison unavailable</strong>
        <span>Both a before and after image are needed to compare them.</span>
      </div>
    );
  }

  const showingBefore = comparePosition < 50;

  return (
    <div className="comparison-container">
      <div className="comparison-before-layer">
        <img src={beforeSrc} alt="Before" className="comparison-image" />
      </div>

      <div
        className="comparison-after-layer"
        style={{
          clipPath: `inset(0 ${100 - comparePosition}% 0 0)`,
        }}
      >
        <img
          src={afterSrc}
          alt="After"
          className="comparison-image comparison-after-image"
        />
      </div>

      <div className="comparison-divider" style={{ left: `${comparePosition}%` }}>
        <div className="comparison-handle">↔</div>
      </div>

      <div
        className={`comparison-active-label ${
          showingBefore ? "comparison-active-before" : "comparison-active-after"
        }`}
      >
        {showingBefore ? beforeLabel : afterLabel}
      </div>

      <input
        type="range"
        min="0"
        max="100"
        value={comparePosition}
        onChange={(event) => setComparePosition(Number(event.target.value))}
        className="comparison-range"
        aria-label="Before and after comparison slider"
      />
    </div>
  );
}
