"use client";
import { useState, useRef, useCallback } from "react";
import { Upload, Loader2, AlertCircle, ArrowLeftRight, GitCompare } from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function ImageUploadSlot({
  label, preview, onFile,
}: { label: string; preview: string | null; onFile: (f: File) => void }) {
  const ref = useRef<HTMLInputElement>(null);
  return (
    <div
      className="drop-zone p-6 text-center cursor-pointer"
      onClick={() => ref.current?.click()}
    >
      <input
        ref={ref}
        type="file"
        accept=".jpg,.jpeg,.png,.dcm"
        className="hidden"
        onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
      />
      {preview ? (
        <img src={preview} alt={label} className="max-h-48 mx-auto rounded-xl object-contain" />
      ) : (
        <>
          <Upload className="w-8 h-8 text-slate-500 mx-auto mb-3" />
          <p className="text-slate-400 text-sm font-medium">{label}</p>
          <p className="text-slate-600 text-xs mt-1">JPEG · PNG · DICOM</p>
        </>
      )}
    </div>
  );
}

export default function ComparePage() {
  const [priorFile, setPriorFile] = useState<File | null>(null);
  const [currentFile, setCurrentFile] = useState<File | null>(null);
  const [priorPreview, setPriorPreview] = useState<string | null>(null);
  const [currentPreview, setCurrentPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFile = (which: "prior" | "current") => (f: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const url = e.target?.result as string;
      if (which === "prior") { setPriorFile(f); setPriorPreview(url); }
      else { setCurrentFile(f); setCurrentPreview(url); }
    };
    reader.readAsDataURL(f);
  };

  const compare = async () => {
    if (!priorFile || !currentFile) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const form = new FormData();
      form.append("prior", priorFile);
      form.append("current", currentFile);
      const res = await fetch(`${API_URL}/api/compare`, { method: "POST", body: form });
      if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.detail || `HTTP ${res.status}`); }
      setResult(await res.json());
    } catch (e: any) { setError(e.message); }
    finally { setLoading(false); }
  };

  const ready = priorFile && currentFile;

  return (
    <div className="max-w-5xl mx-auto px-6 py-12">
      <div className="mb-10">
        <h1 className="text-3xl font-bold gradient-text mb-2">Serial X-ray Comparison</h1>
        <p className="text-slate-400 text-sm">
          Upload a prior and current X-ray from the same patient to quantify visual change.
          Results are image-analysis metrics — NOT clinical healing assessments.
        </p>
      </div>

      {/* Upload slots */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div>
          <p className="text-xs text-slate-500 mb-2 font-medium uppercase tracking-wider">Prior X-ray</p>
          <ImageUploadSlot label="Upload prior X-ray" preview={priorPreview} onFile={handleFile("prior")} />
        </div>
        <div>
          <p className="text-xs text-slate-500 mb-2 font-medium uppercase tracking-wider">Current X-ray</p>
          <ImageUploadSlot label="Upload current X-ray" preview={currentPreview} onFile={handleFile("current")} />
        </div>
      </div>

      <div className="flex justify-center mb-10">
        <button onClick={compare} disabled={!ready || loading} className="btn-glow disabled:opacity-40 disabled:cursor-not-allowed">
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <GitCompare className="w-4 h-4" />}
          {loading ? "Comparing..." : "Compare X-rays"}
        </button>
      </div>

      {error && (
        <div className="glass border border-red-500/30 bg-red-500/5 rounded-xl p-4 flex gap-3 mb-6">
          <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
          <p className="text-sm text-red-300">{error}</p>
        </div>
      )}

      {result && (
        <div className="space-y-6">
          {/* Metric cards */}
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {result.embedding_similarity !== null && result.embedding_similarity !== undefined && (
              <MetricCard
                label="Embedding Similarity"
                value={(result.embedding_similarity * 100).toFixed(1) + "%"}
                note="DINOv2 cosine similarity"
              />
            )}
            {Object.entries(result.pixel_level_metrics || {}).map(([k, v]) => (
              <MetricCard key={k} label={k.replace(/_/g, " ")} value={typeof v === "number" ? v.toFixed(4) : String(v)} note="pixel-level metric" />
            ))}
            {Object.entries(result.mask_comparison || {}).map(([k, v]) => (
              <MetricCard key={k} label={k.replace(/_/g, " ")} value={typeof v === "number" ? (v > 1 ? v.toString() : (v * 100).toFixed(1) + "%") : String(v)} note="mask metric" />
            ))}
          </div>

          <div className="glass border border-amber-500/20 bg-amber-500/5 rounded-xl p-4">
            <p className="text-xs text-amber-400 leading-relaxed">
              ⚠️ {result.metric_warning || result.disclaimer}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

function MetricCard({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className="glass rounded-xl p-4">
      <p className="text-xs text-slate-500 capitalize mb-1">{label}</p>
      <p className="text-xl font-bold text-white">{value}</p>
      <p className="text-xs text-slate-600 mt-1">{note}</p>
    </div>
  );
}
