"use client";
import { CheckCircle, AlertTriangle, Info, ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";

interface ResultsPanelProps {
  result: {
    fracture_detected: boolean;
    fracture_confidence: number;
    qc_passed: boolean;
    qc_warnings: string[];
    detections: { label: string; confidence: number; bbox_xyxy: number[] }[];
    classification?: {
      labels: string[];
      probabilities: Record<string, number>;
      confidence: number;
      calibrated: boolean;
    } | null;
    segmentation?: {
      mask_available: boolean;
      area_pixels: number;
      mask_score: number | null;
      model_loaded: boolean;
      warning?: string;
    } | null;
    stage_errors: Record<string, string>;
    disclaimer: string;
  };
}

function ProbBar({ label, prob }: { label: string; prob: number }) {
  const pct = Math.round(prob * 100);
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-slate-400 w-28 truncate">{label}</span>
      <div className="flex-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full bg-gradient-to-r from-violet-500 to-cyan-400 transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs text-slate-400 w-8 text-right">{pct}%</span>
    </div>
  );
}

export function ResultsPanel({ result }: ResultsPanelProps) {
  const [showProbs, setShowProbs] = useState(false);

  const cls = result.classification;
  const seg = result.segmentation;
  const hasErrors = Object.keys(result.stage_errors).length > 0;

  return (
    <div className="space-y-4">
      {/* Detection status */}
      <div className={`glass rounded-xl p-5 border ${result.fracture_detected ? "border-red-500/30 bg-red-500/5" : "border-emerald-500/30 bg-emerald-500/5"}`}>
        <div className="flex items-center gap-3">
          {result.fracture_detected
            ? <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0" />
            : <CheckCircle className="w-5 h-5 text-emerald-400 flex-shrink-0" />
          }
          <div>
            <p className={`font-semibold ${result.fracture_detected ? "text-red-300" : "text-emerald-300"}`}>
              {result.fracture_detected ? "Fracture Signal Detected" : "No Fracture Signal Detected"}
            </p>
            <p className="text-xs text-slate-500 mt-0.5">
              Confidence: {(result.fracture_confidence * 100).toFixed(1)}% · This is NOT a clinical diagnosis
            </p>
          </div>
        </div>
      </div>

      {/* QC */}
      {!result.qc_passed && (
        <div className="glass border border-amber-500/20 rounded-xl p-4">
          <p className="text-xs text-amber-400 font-medium mb-1">Image Quality Warnings</p>
          {result.qc_warnings.map((w, i) => (
            <p key={i} className="text-xs text-slate-400">• {w}</p>
          ))}
        </div>
      )}

      {/* Detections table */}
      {result.detections.length > 0 && (
        <div className="glass rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b border-white/8">
            <p className="text-sm font-medium text-slate-200">Detected Regions ({result.detections.length})</p>
          </div>
          <div className="divide-y divide-white/5">
            {result.detections.map((d, i) => (
              <div key={i} className="px-4 py-3 flex items-center justify-between">
                <span className="text-sm text-slate-300 capitalize">{d.label}</span>
                <div className="flex items-center gap-3">
                  <div className="h-1.5 w-20 bg-white/5 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-violet-400 rounded-full"
                      style={{ width: `${(d.confidence * 100).toFixed(0)}%` }}
                    />
                  </div>
                  <span className="text-xs text-slate-400 w-10 text-right">
                    {(d.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Classification */}
      {cls && (
        <div className="glass rounded-xl overflow-hidden">
          <button
            className="w-full px-4 py-3 flex items-center justify-between border-b border-white/8"
            onClick={() => setShowProbs(!showProbs)}
          >
            <span className="text-sm font-medium text-slate-200">
              Classification {cls.calibrated ? "(calibrated)" : ""}
            </span>
            {showProbs ? <ChevronUp className="w-4 h-4 text-slate-500" /> : <ChevronDown className="w-4 h-4 text-slate-500" />}
          </button>
          {showProbs && (
            <div className="px-4 py-4 space-y-2.5">
              {Object.entries(cls.probabilities)
                .filter(([, v]) => v > 0.01)
                .sort(([, a], [, b]) => b - a)
                .map(([label, prob]) => (
                  <ProbBar key={label} label={label} prob={prob} />
                ))}
            </div>
          )}
          {cls.labels.length > 0 && (
            <div className="px-4 pb-4 flex flex-wrap gap-2">
              {cls.labels.map((l) => (
                <span key={l} className="badge-detected px-2 py-1 rounded-md text-xs font-medium capitalize">{l}</span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Segmentation */}
      {seg && (
        <div className="glass rounded-xl p-4">
          <p className="text-xs font-medium text-slate-300 mb-2">Segmentation</p>
          {seg.warning ? (
            <p className="text-xs text-amber-400/80">{seg.warning}</p>
          ) : (
            <div className="flex gap-6 text-xs text-slate-400">
              <span>Area: <strong className="text-slate-200">{seg.area_pixels.toLocaleString()} px</strong></span>
              {seg.mask_score !== null && (
                <span>Score: <strong className="text-slate-200">{(seg.mask_score * 100).toFixed(1)}%</strong></span>
              )}
            </div>
          )}
        </div>
      )}

      {/* Stage errors */}
      {hasErrors && (
        <div className="glass border border-red-500/20 rounded-xl p-4">
          <p className="text-xs text-red-400 font-medium mb-2">Stage Errors</p>
          {Object.entries(result.stage_errors).map(([stage, err]) => (
            <p key={stage} className="text-xs text-slate-500">
              <span className="text-slate-400 font-medium">{stage}:</span> {err}
            </p>
          ))}
        </div>
      )}

      {/* Disclaimer */}
      <div className="flex items-start gap-2 text-xs text-slate-600">
        <Info className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
        <p className="leading-relaxed">{result.disclaimer}</p>
      </div>
    </div>
  );
}
