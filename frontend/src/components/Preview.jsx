// SVG preview — renders GeoJSON geometry with current styles client-side
// All style changes update this instantly without re-running the CV pipeline

export default function Preview({ geometry, styles }) {
  if (!geometry) return null

  // TODO: convert GeoJSON features to SVG paths using styles
  // For now, show a placeholder
  return (
    <div className="flex items-center justify-center h-full text-gray-400 text-sm">
      <p>Preview will render here once geometry is loaded.</p>
    </div>
  )
}
