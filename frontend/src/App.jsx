import { useState } from 'react'
import MapPicker from './components/MapPicker'
import StylePanel from './components/StylePanel'
import Preview from './components/Preview'

// Workflow steps
const STEPS = ['select', 'blocks', 'calibrate', 'generate', 'assign', 'style', 'download']

export default function App() {
  const [step, setStep] = useState('select')
  const [bbox, setBbox] = useState(null)
  const [geometry, setGeometry] = useState(null)   // GeoJSON from backend
  const [styles, setStyles] = useState(defaultStyles())
  const [jobId, setJobId] = useState(null)

  async function handleGenerate() {
    if (!bbox) return
    setStep('generate')
    const res = await fetch('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bbox, style: styles }),
    })
    const job = await res.json()
    setJobId(job.job_id)
    pollJob(job.job_id)
  }

  async function pollJob(id) {
    const res = await fetch(`/api/jobs/${id}`)
    const job = await res.json()
    if (job.status === 'complete') {
      setGeometry(job.geometry)
      setStep('assign')
    } else if (job.status === 'failed') {
      alert('Processing failed: ' + job.error)
      setStep('select')
    } else {
      setTimeout(() => pollJob(id), 2000)
    }
  }

  async function handleDownload() {
    const res = await fetch(`/api/jobs/${jobId}/export`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ style: styles }),
    })
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'site-plan.zip'
    a.click()
  }

  return (
    <div className="flex h-screen">
      {/* Left: Map */}
      <div className="flex-1 relative">
        <MapPicker onBboxChange={setBbox} />
        {step === 'select' && bbox && (
          <button
            onClick={handleGenerate}
            className="absolute bottom-6 left-1/2 -translate-x-1/2 bg-black text-white px-6 py-3 rounded-lg shadow-lg hover:bg-gray-800"
          >
            Generate Site Plan
          </button>
        )}
      </div>

      {/* Center: Preview */}
      {geometry && (
        <div className="w-[500px] border-x border-gray-200 bg-white overflow-auto">
          <Preview geometry={geometry} styles={styles} />
        </div>
      )}

      {/* Right: Style panel */}
      {geometry && (
        <div className="w-72 bg-white border-l border-gray-200 overflow-y-auto">
          <StylePanel
            styles={styles}
            geometry={geometry}
            onChange={setStyles}
            onDownload={handleDownload}
          />
        </div>
      )}
    </div>
  )
}

function defaultStyles() {
  return {
    roofs: { visible: true, color: '#000000', lineWeightMm: 0.25 },
    roads: { visible: true, color: '#333333', lineWeightMm: 0.18 },
    trees: { visible: true, color: '#444444', lineWeightMm: 0.13 },
    land_types: [],
  }
}
