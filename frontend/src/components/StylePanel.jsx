// Style panel — line weights, colors, hatch options, download
export default function StylePanel({ styles, geometry, onChange, onDownload }) {
  function updateLayer(layer, key, value) {
    onChange({ ...styles, [layer]: { ...styles[layer], [key]: value } })
  }

  return (
    <div className="p-4 space-y-6">
      <h2 className="text-sm font-semibold tracking-wide uppercase text-gray-500">Style</h2>

      <LayerControl
        label="Roofs"
        style={styles.roofs}
        onChange={(k, v) => updateLayer('roofs', k, v)}
      />
      <LayerControl
        label="Roads"
        style={styles.roads}
        onChange={(k, v) => updateLayer('roads', k, v)}
      />
      <LayerControl
        label="Trees"
        style={styles.trees}
        onChange={(k, v) => updateLayer('trees', k, v)}
      />

      {/* Land types — populated after detection */}
      {styles.land_types.map((lt, i) => (
        <div key={i} className="border-t pt-4">
          <p className="text-xs text-gray-500 mb-1">{lt.label || `Land Type ${i + 1}`}</p>
          <LayerControl
            label={lt.label}
            style={lt}
            onChange={(k, v) => {
              const updated = [...styles.land_types]
              updated[i] = { ...updated[i], [k]: v }
              onChange({ ...styles, land_types: updated })
            }}
          />
        </div>
      ))}

      <button
        onClick={onDownload}
        className="w-full bg-black text-white py-2 rounded-lg text-sm hover:bg-gray-800 mt-4"
      >
        Download DXF
      </button>
    </div>
  )
}

function LayerControl({ label, style, onChange }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium">{label}</label>
        <input
          type="checkbox"
          checked={style.visible !== false}
          onChange={e => onChange('visible', e.target.checked)}
        />
      </div>
      <div className="flex gap-2 items-center">
        <input
          type="color"
          value={style.color || '#000000'}
          onChange={e => onChange('color', e.target.value)}
          className="w-8 h-8 rounded cursor-pointer"
        />
        <input
          type="range"
          min="0.05"
          max="1.0"
          step="0.05"
          value={style.lineWeightMm || 0.25}
          onChange={e => onChange('lineWeightMm', parseFloat(e.target.value))}
          className="flex-1"
        />
        <span className="text-xs text-gray-400 w-10 text-right">
          {(style.lineWeightMm || 0.25).toFixed(2)}mm
        </span>
      </div>
    </div>
  )
}
