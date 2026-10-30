"use client";
import { useState, useCallback, useRef } from "react";
import { Upload, X, AlertCircle, Loader2, Eye } from "lucide-react";
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
      setError(err.message || "Failed to analyze radiograph. Please verify API service.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-[1200px] mx-auto px-6 py-10">
      {/* Header */}
      <div className="mb-10">
        <span className="eyebrow-label block mb-2">Radiographic Intake</span>
        <h1 className="font-serif text-[42px] md:text-[52px] font-light text-[#0f3e17] leading-tight mb-2">
          Analyze Radiograph
        </h1>
        <p className="text-[14px] text-[#222222]/80 max-w-2xl">
          Upload a high-resolution bone radiograph (JPEG, PNG, or DICOM) up to 50 MB. Findings are processed via synchronized detection, segmentation, and feature extraction models.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Upload & Actions (5 cols) */}
        <div className="lg:col-span-5 space-y-5">
          {/* Dropzone */}
          <div
            className={`border-2 border-dashed rounded-[14px] p-8 text-center cursor-pointer transition-colors duration-200 ${
              dragging
                ? "border-[#0f3e17] bg-[#e1f4df]"
                : "border-[#b1dbb8] bg-[#fffefc] hover:bg-[#e1f4df]/50"
            }`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
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
              <div className="relative group">
                <img
                  src={preview}
                  alt="X-ray preview"
                  className="max-h-72 mx-auto rounded-[10px] object-contain bg-black/5"
                />
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    setFile(null);
                    setPreview(null);
                    setResult(null);
                  }}
                  className="absolute top-2 right-2 w-8 h-8 bg-[#0f3e17] text-[#fffefc] rounded-full flex items-center justify-center hover:bg-[#0c2f10] transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <div className="py-8">
                <div className="w-12 h-12 rounded-full bg-[#e1f4df] text-[#0f3e17] flex items-center justify-center mx-auto mb-4">
                  <Upload className="w-5 h-5" />
                </div>
                <p className="text-[15px] font-medium text-[#0f3e17] mb-1">
                  Drop X-ray here or click to browse
                </p>
                <p className="text-[12px] text-[#222222]/60">
                  Supports JPEG, PNG, DICOM · Up to 50 MB
                </p>
              </div>
            )}
          </div>

          {/* Selected File Card & Submit CTA */}
          {file && (
            <div className="bg-[#fffefc] border border-[#efeeeb] rounded-[14px] p-4 flex items-center justify-between gap-4">
              <div className="min-w-0">
                <p className="text-[14px] font-medium text-[#222222] truncate">
                  {file.name}
                </p>
                <p className="text-[12px] text-[#222222]/60">
                  {(file.size / 1024).toFixed(0)} KB
                </p>
              </div>
              <button
                onClick={analyze}
                disabled={loading}
                className="bg-[#0f3e17] hover:bg-[#0c2f10] disabled:opacity-50 text-[#fffefc] text-[14px] font-normal px-5 py-2.5 rounded-[14px] inline-flex items-center gap-2 transition-colors flex-shrink-0"
              >
                {loading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Eye className="w-4 h-4" />
                )}
                <span>{loading ? "Analyzing..." : "Run Analysis"}</span>
              </button>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="bg-[#fffefc] border border-red-200 text-red-900 rounded-[14px] p-4 flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
              <p className="text-[13px] leading-relaxed">{error}</p>
            </div>
          )}


        </div>

        {/* Right Column: Visualization & Results (7 cols) */}
        <div className="lg:col-span-7">
          {result ? (
            <div className="space-y-6">
              <div className="bg-[#b6ced5] rounded-[14px] p-5">
                <XrayOverlay
                  imageUrl={preview!}
                  processedImageHw={result.image_hw}
                  detections={[...result.detections]
                    .sort((a, b) => b.confidence - a.confidence)
                    .slice(0, 8)}
                  segmentation={result.segmentation}
                />
              </div>
              <ResultsPanel result={result} />
            </div>
          ) : (
            <div className="bg-[#b6ced5] rounded-[14px] p-12 min-h-[460px] flex flex-col items-center justify-center text-center">
              <div className="bg-[#fffefc] rounded-[14px] p-8 max-w-sm w-full">
                <div className="w-12 h-12 rounded-full bg-[#e1f4df] text-[#0f3e17] flex items-center justify-center mx-auto mb-4">
                  <Eye className="w-6 h-6" />
                </div>
                <h3 className="font-serif text-[22px] font-light text-[#0f3e17] mb-2">
                  Awaiting Study
                </h3>
                <p className="text-[13px] text-[#222222]/70 leading-relaxed">
                  Upload an X-ray to generate real-time bounding boxes, MedSAM segmentation masks, and calibrated probability matrices.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
