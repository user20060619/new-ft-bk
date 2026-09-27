// Per-file modality hint: auto-detect, optical, or SAR. Always rendered next
// to every uploaded file, never hidden -- it's the only way to mark a
// single-band panchromatic image as optical, since the backend's own
// auto-detection (rsio/raster.py::_guess_modality) can't tell a panchromatic
// optical image apart from an unknown single-band source on its own.
export default function ModalitySelector({ value, onChange, disabled = false }) {
  return (
    <select
      className="modality-selector"
      value={value}
      onChange={(event) => onChange(event.target.value)}
      disabled={disabled}
      aria-label="Modality"
    >
      <option value="">Auto-detect</option>
      <option value="optical">Optical</option>
      <option value="sar">SAR</option>
    </select>
  );
}
