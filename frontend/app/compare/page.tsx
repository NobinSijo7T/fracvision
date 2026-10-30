"use client";
import { useState, useRef } from "react";
import { Upload, Loader2, AlertCircle, GitCompare, ArrowLeftRight } from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function ImageUploadSlot({
  label,
  preview,
  onFile,
}: {
  label: string;
  preview: string | null;
  onFile: (f: File) => void;
}) {
  const ref = useRef<HTMLInputElement>(null);
  return (
    <div
      className="border-2 border-dashed border-[#b1dbb8] hover:border-[#0f3e17] bg-[#fffefc] hover:bg-[#e1f4df]/40 rounded-[14px] p-8 text-center cursor-pointer transition-colors"
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
        <img
          src={preview}
          alt={label}
          className="max-h-56 mx-auto rounded-[10px] object-contain bg-black/5"
        />
      ) : (
        <div className="py-6">
          <div className="w-10 h-10 rounded-full bg-[#e1f4df] text-[#0f3e17] flex items-center justify-center mx-auto mb-3">
            <Upload className="w-5 h-5" />
          </div>
          <p className="text-[14px] font-medium text-[#0f3e17]">{label}</p>
          <p className="text-[12px] text-[#222222]/60 mt-1">JPEG · PNG · DICOM</p>
        </div>
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
      if (which === "prior") {
        setPriorFile(f);
        setPriorPreview(url);
      } else {
        setCurrentFile(f);
        setCurrentPreview(url);
      }
    };
    reader.readAsDataURL(f);
  };

  const compare = async () => {
    if (!priorFile || !currentFile) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append("prior", priorFile);
      form.append("current", currentFile);
      const res = await fetch(`${API_URL}/api/compare`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || `HTTP ${res.status}`);
      }
      setResult(await res.json());
    } catch (e: any) {
      setError(e.message || "Failed to compare radiographs.");
    } finally {
      setLoading(false);
    }
  };

  const ready = priorFile && currentFile;

  return (
    <div className="max-w-[1100px] mx-auto px-6 py-10">
      {/* Header */}
      <div className="mb-10">
        <span className="eyebrow-label block mb-2">Longitudinal Sequence</span>
        <h1 className="font-serif text-[42px] md:text-[52px] font-light text-[#0f3e17] leading-tight mb-2">
          Serial Radiograph Comparison
        </h1>
        <p className="text-[14px] text-[#222222]/80 max-w-2xl">
          Upload a prior baseline and current follow-up radiograph for the same anatomical site to quantify radiographic similarity, alignment shifts, and region changes.
        </p>
      </div>

      {/* Upload slots */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div>
          <span className="eyebrow-label block mb-2 text-[#0f3e17]/70">Prior Baseline</span>
          <ImageUploadSlot
            label="Upload prior X-ray"
            preview={priorPreview}
            onFile={handleFile("prior")}
          />
        </div>
        <div>
          <span className="eyebrow-label block mb-2 text-[#0f3e17]/70">Current Follow-up</span>
          <ImageUploadSlot
            label="Upload current X-ray"
            preview={currentPreview}
            onFile={handleFile("current")}
          />
        </div>
      </div>

      {/* CTA Button */}
      <div className="flex justify-center mb-10">
        <button
          onClick={compare}
          disabled={!ready || loading}
          className="bg-[#0f3e17] hover:bg-[#0c2f10] disabled:opacity-40 text-[#fffefc] text-[14px] font-normal px-8 py-3.5 rounded-[14px] inline-flex items-center gap-2.5 transition-colors"
        >
          {loading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <GitCompare className="w-4 h-4" />
          )}
          <span>{loading ? "Comparing Radiographs..." : "Compare Radiographs"}</span>
        </button>
      </div>

      {/* Error Notice */}
      {error && (
        <div className="bg-[#fffefc] border border-red-200 text-red-900 rounded-[14px] p-4 flex items-start gap-3 mb-6">
          <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
          <p className="text-[13px]">{error}</p>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {result.embedding_similarity !== null &&
              result.embedding_similarity !== undefined && (
                <MetricCard
                  label="DINOv2 Feature Similarity"
                  value={(result.embedding_similarity * 100).toFixed(1) + "%"}
                  note="Cosine embedding correlation"
                />
              )}
            {Object.entries(result.pixel_level_metrics || {}).map(([k, v]) => (
              <MetricCard
                key={k}
                label={k.replace(/_/g, " ")}
                value={typeof v === "number" ? v.toFixed(4) : String(v)}
                note="Pixel structural index"
              />
            ))}
            {Object.entries(result.mask_comparison || {}).map(([k, v]) => (
              <MetricCard
                key={k}
                label={k.replace(/_/g, " ")}
                value={
                  typeof v === "number"
                    ? v > 1
                      ? v.toString()
                      : (v * 100).toFixed(1) + "%"
                    : String(v)
                }
                note="Segmentation overlap"
              />
            ))}
          </div>

          {result.metric_warning && (
            <div className="bg-[#e1f4df] border border-[#cfe7d3] rounded-[14px] p-5">
              <p className="text-[12px] text-[#0f3e17] leading-relaxed">
                <strong>Advisory:</strong> {result.metric_warning}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function MetricCard({
  label,
  value,
  note,
}: {
  label: string;
  value: string;
  note: string;
}) {
  return (
    <div className="bg-[#fffefc] border border-[#efeeeb] rounded-[14px] p-5">
      <span className="eyebrow-label block mb-1 text-[#222222]/60">{label}</span>
      <p className="font-serif text-[28px] font-light text-[#0f3e17] leading-tight mb-1">
        {value}
      </p>
      <p className="text-[11px] text-[#222222]/50">{note}</p>
    </div>
  );
}
