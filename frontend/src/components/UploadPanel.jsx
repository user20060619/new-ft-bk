import ModalitySelector from "./ModalitySelector";
import { isGeoTiffFilename } from "../utils/format";

const REMOVE_ICON = (
  <svg
    width="15"
    height="15"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <path d="M3 6h18" />
    <path d="M8 6V4h8v2" />
    <path d="M19 6l-1 14H6L5 6" />
    <path d="M10 11v5" />
    <path d="M14 11v5" />
  </svg>
);

// T13: suggestion chips are conditional on upload state, each verified live
// against the real backend for the state it's shown in -- a chip that can't
// produce a real result for that state is not offered (e.g. "What changed
// here?" needs two images).
const SINGLE_IMAGE_SUGGESTIONS = [
  "Describe the land-cover and major objects visible in this image.",
  "Where are the buildings?",
];

const OPTICAL_SAR_SUGGESTIONS = [
  "Use the optical and SAR images together to identify built-up and water-covered regions.",
  "Where do optical and SAR agree on water?",
];

const BITEMPORAL_SUGGESTIONS = [
  "What changed here?",
  "Did any buildings appear?",
  "Was vegetation lost?",
];

// T17: VQA mode's own chip set -- the real `_vqa_handler` now answers these.
const VQA_SUGGESTIONS = [
  "Is there any water in this image?",
  "How much of the image is vegetation?",
  "Where are the buildings?",
  "Describe the land-cover and major objects visible in this image.",
];

function suggestionsFor(mode, files, modalities) {
  if (mode === "vqa") return files[0] ? VQA_SUGGESTIONS : [];
  const uploadedCount = files.filter(Boolean).length;
  if (uploadedCount === 1) return SINGLE_IMAGE_SUGGESTIONS;
  if (uploadedCount === 2) {
    const hasSar = files.some((file, index) => file && modalities[index] === "sar");
    return hasSar ? OPTICAL_SAR_SUGGESTIONS : BITEMPORAL_SUGGESTIONS;
  }
  return [];
}

function FileSlot({
  index, label, required, file, preview, modality, disabled, inputFilename,
  onFileChange, onRemove, onModalityChange,
}) {
  // T18: a raw blob preview can't render a GeoTIFF -- prefer the live
  // File's own name (fresh upload), falling back to the response's record
  // of the original filename (a restored history entry has no File object,
  // only its saved metadata) so the placeholder still catches it there too.
  const isGeoTiff = isGeoTiffFilename(file?.name || inputFilename);

  return (
    <div className="upload-card">
      <div className="card-heading">
        <span>{String(index + 1).padStart(2, "0")}</span>
        <h3>
          {label}
          {!required && (
            <span className="optional-tag">
              (<span className="optional-pill">optional</span>)
            </span>
          )}
        </h3>
      </div>

      <div className="image-upload">
        {preview ? (
          <div className="image-preview">
            {isGeoTiff ? (
              <div className="tiff-preview-placeholder">
                <strong>GeoTIFF</strong>
                <span>Preview after analysis</span>
              </div>
            ) : (
              <img src={preview} alt={label} />
            )}
            <button
              type="button"
              className="remove-image-button"
              onClick={onRemove}
              aria-label={`Remove ${label.toLowerCase()}`}
              title="Remove image"
            >
              {REMOVE_ICON}
            </button>
            <div className="image-label">{file?.name || inputFilename}</div>
          </div>
        ) : (
          <label className="upload-placeholder">
            <span className="upload-icon">↑</span>
            <strong>Upload satellite image</strong>
            <small>GeoTIFF/TIFF, PNG, or JPEG</small>
            <input
              type="file"
              accept=".tif,.tiff,.png,.jpg,.jpeg,image/tiff,image/png,image/jpeg"
              onChange={(event) => {
                const selected = event.target.files[0];
                if (selected) onFileChange(selected);
              }}
              disabled={disabled}
            />
          </label>
        )}
      </div>

      {/* Always rendered, never optional -- the only way to mark a
          single-band panchromatic image as optical (auto-detection alone
          can't tell it apart from an unknown single-band source). */}
      <div className="modality-row">
        <span>Modality</span>
        <ModalitySelector value={modality} onChange={onModalityChange} disabled={disabled} />
      </div>
    </div>
  );
}

export default function UploadPanel({
  mode,
  files,
  filePreviews,
  modalities,
  metadata,
  query,
  loading,
  onFileChange,
  onFileRemove,
  onModalityChange,
  onQueryChange,
  onSubmit,
}) {
  const suggestions = suggestionsFor(mode, files, modalities);
  const filesReady = mode === "vqa" ? Boolean(files[0]) : Boolean(files[0] && files[1]);

  return (
    <>
      <section className={`upload-section ${mode === "vqa" ? "vqa-upload-section" : ""}`}>
        <FileSlot
          index={0}
          label="Image 1"
          required
          file={files[0]}
          preview={filePreviews[0]}
          modality={modalities[0]}
          inputFilename={metadata?.inputs?.[0]?.filename}
          disabled={loading}
          onFileChange={(file) => onFileChange(0, file)}
          onRemove={() => onFileRemove(0)}
          onModalityChange={(value) => onModalityChange(0, value)}
        />

        {mode !== "vqa" && (
          <FileSlot
            index={1}
            label="Image 2"
            required
            file={files[1]}
            preview={filePreviews[1]}
            modality={modalities[1]}
            inputFilename={metadata?.inputs?.[1]?.filename}
            disabled={loading}
            onFileChange={(file) => onFileChange(1, file)}
            onRemove={() => onFileRemove(1)}
            onModalityChange={(value) => onModalityChange(1, value)}
          />
        )}
      </section>

      <section className="query-section">
        <label htmlFor="query">Ask about the image(s)</label>

        <div className="query-box">
          <input
            id="query"
            type="text"
            placeholder='Try: "What changed in this area?"'
            value={query}
            onChange={(event) => onQueryChange(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") onSubmit();
            }}
            disabled={loading}
          />

          <button type="button" onClick={onSubmit} disabled={loading || !filesReady}>
            {loading ? "Analyzing..." : "Analyze →"}
          </button>
        </div>

        <p className="query-hint">
          Upload one image for a single-image question, two images of the
          same place over time for change, or a co-registered optical + SAR
          pair.
        </p>

        {suggestions.length > 0 && (
          <div className="query-suggestions">
            {suggestions.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                onClick={() => onQueryChange(suggestion)}
                disabled={loading}
              >
                {suggestion}
              </button>
            ))}
          </div>
        )}
      </section>
    </>
  );
}
