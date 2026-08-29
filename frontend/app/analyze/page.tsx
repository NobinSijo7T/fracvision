"use client";
import { useState, useCallback, useRef } from "react";
import { Upload, X, AlertCircle, CheckCircle, Loader2, Eye } from "lucide-react";
import { ResultsPanel } from "@/components/ResultsPanel";
import { XrayOverlay } from "@/components/XrayOverlay";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function AnalyzePage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((f: File) => {
    setFile(f);
    setResult(null);
    setError(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(f);
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) handleFile(dropped);
  }, [handleFile]);

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API_URL}/api/predict`, { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      setResult(await res.json());
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-6 py-12">
      <div className="mb-10">
        <h1 className="text-3xl font-bold gradient-text mb-2">Analyze X-ray</h1>
        <p className="text-slate-400 text-sm">
          Upload a bone X-ray (JPEG / PNG / DICOM). Max 50 MB. Results for research review only.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Upload panel */}
        <div className="space-y-6">
          <div
            className={`drop-zone p-10 text-center ${dragging ? "active" : ""}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".jpg,.jpeg,.png,.dcm"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
            {preview ? (
              <div className="relative">
                <img
                  src={preview}
                  alt="X-ray preview"
                  className="max-h-64 mx-auto rounded-xl object-contain"
                />
                <button
                  onClick={(e) => { e.stopPropagation(); setFile(null); setPreview(null); setResult(null); }}
                  className="absolute top-2 right-2 w-7 h-7 bg-black/60 rounded-full flex items-center justify-center hover:bg-black/80"
                >
                  <X className="w-3.5 h-3.5 text-white" />
                </button>
              </div>
            ) : (
              <>
                <Upload className="w-10 h-10 text-slate-500 mx-auto mb-4" />
                <p className="text-slate-400 font-medium mb-1">Drop X-ray here</p>
                <p className="text-slate-600 text-sm">or click to browse · JPEG, PNG, DICOM</p>
              </>
            )}
          </div>

          {file && (
            <div className="glass rounded-xl p-4 flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-200 truncate max-w-48">{file.name}</p>
                <p className="text-xs text-slate-500">{(file.size / 1024).toFixed(0)} KB</p>
              </div>
              <button onClick={analyze} disabled={loading} className="btn-glow disabled:opacity-50 disabled:cursor-not-allowed">
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Eye className="w-4 h-4" />}
                {loading ? "Analyzing..." : "Run Analysis"}
              </button>
            </div>
          )}

          {error && (
            <div className="glass border border-red-500/30 bg-red-500/5 rounded-xl p-4 flex gap-3">
              <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
              <p className="text-sm text-red-300">{error}</p>
            </div>
          )}

          {/* Disclaimer */}
          <div className="glass border border-amber-500/20 bg-amber-500/5 rounded-xl p-4">
            <p className="text-xs text-amber-400/80 leading-relaxed">
              ⚠️ This analysis is produced by a research prototype and has NOT been clinically validated.
              Results must be reviewed by a qualified clinician before any clinical application.
            </p>
          </div>
        </div>

        {/* Results panel */}
        <div>
          {result ? (
            <div className="space-y-6">
              <XrayOverlay imageUrl={preview!} detections={result.detections} />
              <ResultsPanel result={result} />
            </div>
          ) : (
            <div className="glass rounded-2xl h-full flex items-center justify-center min-h-64">
              <div className="text-center text-slate-600">
                <Eye className="w-10 h-10 mx-auto mb-3 opacity-30" />
                <p className="text-sm">Analysis results will appear here</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
