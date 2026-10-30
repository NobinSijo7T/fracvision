"use client";
import { CheckCircle, AlertTriangle, ChevronDown, ChevronUp } from "lucide-react";
import { useMemo, useState } from "react";

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
      mask_data_url?: string | null;
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
      <span className="text-[13px] text-[#222222]/80 w-32 truncate">{label}</span>
      <div className="flex-1 h-2 bg-[#e1f4df] rounded-full overflow-hidden">
        <div
          className="h-full rounded-full bg-[#0f3e17] transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-[12px] font-mono text-[#0f3e17] w-10 text-right">{pct}%</span>
    </div>
  );
}

export function ResultsPanel({ result }: ResultsPanelProps) {
  const [showProbs, setShowProbs] = useState(false);

  const cls = result.classification;
  const seg = result.segmentation;
  const hasErrors = Object.keys(result.stage_errors).length > 0;
  const detectionSummary = useMemo(() => {
    const grouped = new Map<string, { count: number; confidence: number }>();
    for (const detection of result.detections) {
      const current = grouped.get(detection.label) ?? { count: 0, confidence: 0 };
      grouped.set(detection.label, {
        count: current.count + 1,
        confidence: Math.max(current.confidence, detection.confidence),
      });
    }
    return Array.from(grouped.entries()).sort(([, a], [, b]) => b.confidence - a.confidence);
  }, [result.detections]);

  return (
    <div className="space-y-4">
      {/* Primary Detection Status Banner */}
      <div
        className={`rounded-[14px] p-6 border transition-colors ${result.fracture_detected
            ? "bg-[#fffefc] border-[#efeeeb]"
            : "bg-[#e1f4df] border-[#cfe7d3]"
          }`}
      >
        <div className="flex items-start gap-4">
          <div
            className={`w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5 ${result.fracture_detected
                ? "bg-[#0f3e17] text-[#fffefc]"
                : "bg-[#0f3e17] text-[#fffefc]"
              }`}
          >
            {result.fracture_detected ? (
              <AlertTriangle className="w-5 h-5 text-[#e1f4df]" />
            ) : (
              <CheckCircle className="w-5 h-5 text-[#e1f4df]" />
            )}
          </div>
          <div>
            <h3 className="font-serif text-[22px] font-light text-[#0f3e17] leading-tight mb-1">
              {result.fracture_detected
                ? "Fracture Signal Detected"
                : "No Fracture Signal Detected"}
            </h3>
            <p className="text-[13px] text-[#222222]/80 leading-relaxed">
              Confidence: <strong className="text-[#0f3e17]">{(result.fracture_confidence * 100).toFixed(1)}%</strong>
            </p>
          </div>
        </div>
      </div>

      {/* QC Warnings */}
      {!result.qc_passed && (
        <div className="bg-[#fffefc] border border-amber-300 rounded-[14px] p-5">
          <p className="text-[11px] uppercase tracking-[0.08em] font-semibold text-amber-800 mb-1.5">
            Image Quality Warnings
          </p>
          {result.qc_warnings.map((w, i) => (
            <p key={i} className="text-[13px] text-[#222222]/80">• {w}</p>
          ))}
        </div>
      )}

      {/* Detected Findings Summary */}
      {result.detections.length > 0 && (
        <div className="bg-[#fffefc] border border-[#efeeeb] rounded-[14px] overflow-hidden">
          <div className="px-5 py-3.5 border-b border-[#efeeeb] flex items-center justify-between">
            <span className="eyebrow-label">Detected Findings</span>
            <span className="text-[12px] text-[#222222]/60">
              {result.detections.length} regions · {detectionSummary.length} classes
            </span>
          </div>
          <div className="p-4 grid grid-cols-1 sm:grid-cols-2 gap-3">
            {detectionSummary.map(([label, finding]) => (
              <div
                key={label}
                className="bg-[#e1f4df]/50 rounded-[10px] p-3 flex items-center justify-between gap-3"
              >
                <div className="min-w-0">
                  <p className="text-[13px] font-medium text-[#0f3e17] capitalize truncate">
                    {label}
                  </p>
                  <p className="text-[11px] text-[#222222]/60">
                    {finding.count} region{finding.count === 1 ? "" : "s"}
                  </p>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <div className="h-2 w-16 bg-[#fffefc] rounded-full overflow-hidden">
                    <div
                      className="h-full bg-[#0f3e17] rounded-full"
                      style={{ width: `${(finding.confidence * 100).toFixed(0)}%` }}
                    />
                  </div>
                  <span className="text-[11px] font-mono text-[#0f3e17]">
                    {(finding.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Classification Probabilities */}
      {cls && (
        <div className="bg-[#fffefc] border border-[#efeeeb] rounded-[14px] overflow-hidden">
          <button
            type="button"
            className="w-full px-5 py-3.5 flex items-center justify-between border-b border-[#efeeeb] text-left hover:bg-[#e1f4df]/20 transition-colors"
            onClick={() => setShowProbs(!showProbs)}
          >
            <span className="eyebrow-label">
              Classification {cls.calibrated ? "(Calibrated)" : ""}
            </span>
            {showProbs ? (
              <ChevronUp className="w-4 h-4 text-[#0f3e17]" />
            ) : (
              <ChevronDown className="w-4 h-4 text-[#0f3e17]" />
            )}
          </button>
          {showProbs && (
            <div className="p-5 space-y-3">
              {Object.entries(cls.probabilities)
                .filter(([, v]) => v > 0.01)
                .sort(([, a], [, b]) => b - a)
                .map(([label, prob]) => (
                  <ProbBar key={label} label={label} prob={prob} />
                ))}
            </div>
          )}
          {cls.labels.length > 0 && (
            <div className="px-5 pb-4 pt-2 flex flex-wrap gap-2">
              {cls.labels.map((l) => (
                <span
                  key={l}
                  className="bg-[#e1f4df] text-[#0f3e17] px-3 py-1 rounded-full text-[12px] font-medium capitalize"
                >
                  {l}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Segmentation Specs */}
      {seg && (
        <div className="bg-[#fffefc] border border-[#efeeeb] rounded-[14px] p-5">
          <div className="flex items-center justify-between gap-4 mb-2">
            <span className="eyebrow-label">MedSAM Segmentation</span>
            <span
              className={`text-[11px] px-2.5 py-0.5 rounded-full font-medium ${seg.mask_available
                  ? "bg-[#e1f4df] text-[#0f3e17]"
                  : "bg-amber-100 text-amber-800"
                }`}
            >
              {seg.mask_available ? "Mask Rendered" : "Unavailable"}
            </span>
          </div>
          {seg.warning ? (
            <p className="text-[12px] text-amber-700">{seg.warning}</p>
          ) : (
            <div className="flex gap-6 text-[13px] text-[#222222]/80 mt-1">
              <span>
                Area: <strong className="text-[#0f3e17]">{seg.area_pixels.toLocaleString()} px²</strong>
              </span>
              {seg.mask_score !== null && (
                <span>
                  Mask Score: <strong className="text-[#0f3e17]">{(seg.mask_score * 100).toFixed(1)}%</strong>
                </span>
              )}
            </div>
          )}
        </div>
      )}

      {/* Stage Errors */}
      {hasErrors && (
        <div className="bg-[#fffefc] border border-red-200 rounded-[14px] p-5">
          <p className="text-[11px] uppercase tracking-[0.08em] font-semibold text-red-800 mb-2">
            Pipeline Stage Warnings
          </p>
          {Object.entries(result.stage_errors).map(([stage, err]) => (
            <p key={stage} className="text-[12px] text-[#222222]/70">
              <span className="font-semibold text-red-700">{stage}:</span> {err}
            </p>
          ))}
        </div>
      )}


    </div>
  );
}
