import { useEffect, useRef } from 'react'
import mapboxgl from 'mapbox-gl'
import MapboxDraw from '@mapbox/mapbox-gl-draw'

mapboxgl.accessToken = import.meta.env.VITE_MAPBOX_TOKEN

export default function MapPicker({ onBboxChange }) {
  const mapContainerRef = useRef(null)
  const mapRef = useRef(null)

  useEffect(() => {
    const map = new mapboxgl.Map({
      container: mapContainerRef.current,
      style: 'mapbox://styles/mapbox/satellite-v9',
      center: [0, 20],
      zoom: 2,
    })

    const draw = new MapboxDraw({
      displayControlsDefault: false,
      controls: { rectangle: true, trash: true },
    })
    map.addControl(draw)

    map.on('draw.create', updateBbox)
    map.on('draw.update', updateBbox)
    map.on('draw.delete', () => onBboxChange(null))

    function updateBbox() {
      const data = draw.getAll()
      if (!data.features.length) return
      const coords = data.features[0].geometry.coordinates[0]
      const lons = coords.map(c => c[0])
      const lats = coords.map(c => c[1])
      onBboxChange({
        west: Math.min(...lons),
        east: Math.max(...lons),
        south: Math.min(...lats),
        north: Math.max(...lats),
      })
    }

    mapRef.current = map
    return () => map.remove()
  }, [])

  return <div ref={mapContainerRef} className="w-full h-full" />
}
