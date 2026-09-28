import { useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// Draws the real geographic footprint of an uploaded input -- from
// rasterio.warp.transform_bounds against the file's own CRS + geotransform
// (backend/backend/rsio/raster.py), never a filename-based guess. `bounds`
// is [west, south, east, north] in EPSG:4326, or null/undefined when the
// input has no CRS/geotransform (most PNG/JPEG benchmark images) -- in that
// case there is nothing real to draw, so this renders an empty state rather
// than a default location.
export default function MapView({ bounds, label }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const [tilesUnavailable, setTilesUnavailable] = useState(false);

  useEffect(() => {
    if (!containerRef.current || mapRef.current || !bounds) return;

    setTilesUnavailable(false);

    const [west, south, east, north] = bounds;
    const leafletBounds = [
      [south, west],
      [north, east],
    ];

    const map = L.map(containerRef.current, { attributionControl: true });

    const tileLayer = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 18,
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(map);

    // T18: tiles need the network (offline, or the tile host unreachable) --
    // the bounds rectangle below never depended on tiles loading, so this
    // only adds an honest note, it doesn't change anything else.
    tileLayer.on("tileerror", () => setTilesUnavailable(true));

    L.rectangle(leafletBounds, {
      color: "#2563eb",
      fillColor: "#3b82f6",
      fillOpacity: 0.15,
    }).addTo(map);

    map.fitBounds(leafletBounds, { padding: [20, 20] });

    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, [bounds]);

  if (!bounds) {
    return (
      <div className="visualization-empty">
        <strong>Map view unavailable</strong>
        <span>
          This input has no CRS/geotransform, so its real geographic bounds
          are unknown.
        </span>
      </div>
    );
  }

  return (
    <div className="map-view">
      <div ref={containerRef} className="map-view-canvas" />
      {tilesUnavailable && (
        <p className="map-view-offline-note">
          Map tiles unavailable offline — showing footprint only.
        </p>
      )}
      <div className="map-view-caption">
        <strong>{label || "Input footprint"}</strong>
        <span>Real geographic bounds, from the file's own CRS + geotransform.</span>
      </div>
    </div>
  );
}
