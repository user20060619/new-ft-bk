import ModalitySelector from "./ModalitySelector";

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
// here?" needs two images; a single image with a non-describe/locate query
// hits the still-unimplemented `vqa` handler, a dead end).
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

function suggestionsFor(files, modalities) {
  const uploadedCount = files.filter(Boolean).length;
  if (uploadedCount === 1) return SINGLE_IMAGE_SUGGESTIONS;
  if (uploadedCount === 2) {
    const hasSar = files.some((file, index) => file && modalities[index] === "sar");
    return hasSar ? OPTICAL_SAR_SUGGESTIONS : BITEMPORAL_SUGGESTIONS;
  }
  return [];
}

function FileSlot({ index, label, required, file, preview, modality, disabled, onFileChange, onRemove, onModalityChange }) {
  return (
    <div className="upload-card">
      <div className="card-heading">
        <span>{String(index + 1).padStart(2, "0")}</span>
        <h3>
          {label}
          {!required && <span className="optional-tag"> (optional)</span>}
        </h3>
      </div>

      <div className="image-upload">
        {preview ? (
          <div className="image-preview">
            <img src={preview} alt={label} />
            <button
              type="button"
              className="remove-image-button"
              onClick={onRemove}
              aria-label={`Remove ${label.toLowerCase()}`}
              title="Remove image"
            >
              {REMOVE_ICON}
            </button>
            <div className="image-label">{file?.name}</div>
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
  files,
  filePreviews,
  modalities,
  query,
  loading,
  onFileChange,
  onFileRemove,
  onModalityChange,
  onQueryChange,
  onSubmit,
}) {
  const suggestions = suggestionsFor(files, modalities);

  return (
    <>
      <section className="upload-section">
        <FileSlot
          index={0}
          label="Image 1"
          required
          file={files[0]}
          preview={filePreviews[0]}
          modality={modalities[0]}
          disabled={loading}
          onFileChange={(file) => onFileChange(0, file)}
          onRemove={() => onFileRemove(0)}
          onModalityChange={(value) => onModalityChange(0, value)}
        />

        <FileSlot
          index={1}
          label="Image 2"
          required={false}
          file={files[1]}
          preview={filePreviews[1]}
          modality={modalities[1]}
          disabled={loading}
          onFileChange={(file) => onFileChange(1, file)}
          onRemove={() => onFileRemove(1)}
          onModalityChange={(value) => onModalityChange(1, value)}
        />
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

          <button type="button" onClick={onSubmit} disabled={loading}>
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
