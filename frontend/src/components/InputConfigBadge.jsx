// Small pill labelling how the backend classified the uploaded input(s)
// (backend/backend/rsio/compat.py's input_config: "single" | "bitemporal" |
// "optical_sar"). Renders nothing when input_config is null -- the
// unreadable_file/no_inputs/too_many_inputs failure cases never set it
// (backend/backend/contract.py), so there's nothing honest to label.
const LABELS = {
  single: { text: "Single image", className: "badge-single" },
  bitemporal: { text: "Bitemporal pair", className: "badge-bitemporal" },
  optical_sar: { text: "Optical + SAR pair", className: "badge-optical-sar" },
};

export default function InputConfigBadge({ inputConfig }) {
  const info = LABELS[inputConfig];
  if (!info) return null;

  return (
    <span className={`input-config-badge ${info.className}`}>{info.text}</span>
  );
}
