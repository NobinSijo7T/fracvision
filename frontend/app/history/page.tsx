"use client";
import { Clock, Info, ArrowRight, AlertCircle, CheckCircle, XCircle } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface HistoryRecord {
  id: number;
  image_id: string;
  filename: string | null;
  created_at: string;
  fracture_detected: boolean | null;
  fracture_confidence: number;
  classification_labels: string[];
  segmentation_available: boolean;
  mask_score: number | null;
}

export default function HistoryPage() {
  const [records, setRecords] = useState<HistoryRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_URL}/api/history`)
      .then(async (response) => {
        if (!response.ok) throw new Error(`History request failed (${response.status})`);
        return response.json();
      })
      .then(setRecords)
      .catch((reason) => setError(reason instanceof Error ? reason.message : "Unable to load history"))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="max-w-[1100px] mx-auto px-6 py-10">
      {/* Header */}
      <div className="mb-10">
        <span className="eyebrow-label block mb-2">Longitudinal Records</span>
        <h1 className="font-serif text-[42px] md:text-[52px] font-light text-[#0f3e17] leading-tight mb-2">
          Patient Study History
        </h1>
        <p className="text-[14px] text-[#222222]/80 max-w-2xl">
          Review historical radiograph predictions, serial healing comparisons, and longitudinal audit trails for tracked patient encounters.
        </p>
      </div>

      {/* Advisory Banner */}
      <div className="bg-[#e1f4df] border border-[#cfe7d3] rounded-[14px] p-5 mb-8 flex gap-3.5 items-start">
        <Info className="w-5 h-5 text-[#0f3e17] flex-shrink-0 mt-0.5" />
        <p className="text-[13px] text-[#0f3e17] leading-relaxed">
          Historical records require a connected persistence database. New analyses initiated through the intake engine are automatically indexed here.
        </p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-[14px] p-4 mb-6 flex items-center gap-3 text-red-900 text-[13px]">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          {error}
        </div>
      )}

      {loading ? (
        <div className="bg-[#b6ced5] rounded-[14px] p-16 text-center text-[#0f3e17] text-[14px]">Loading study records…</div>
      ) : records.length === 0 ? (
        <div className="bg-[#b6ced5] rounded-[14px] p-12 md:p-16 flex flex-col items-center justify-center text-center">
          <div className="bg-[#fffefc] rounded-[14px] p-8 md:p-10 max-w-md w-full">
          <div className="w-12 h-12 rounded-full bg-[#e1f4df] text-[#0f3e17] flex items-center justify-center mx-auto mb-4">
            <Clock className="w-6 h-6" />
          </div>
          <h3 className="font-serif text-[24px] font-light text-[#0f3e17] mb-2">
            No Study Records Yet
          </h3>
          <p className="text-[13px] text-[#222222]/70 leading-relaxed mb-6">
            Upload and analyze radiographs to start compiling a patient longitudinal series. Serial studies will be organized here with visual timestamps.
          </p>
          <Link
            href="/analyze"
            className="bg-[#0f3e17] hover:bg-[#0c2f10] text-[#fffefc] text-[14px] font-normal px-6 py-3 rounded-[14px] inline-flex items-center gap-2 transition-colors"
          >
            <span>Analyze First Study</span>
            <ArrowRight className="w-4 h-4" />
          </Link>
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          {records.map((record) => (
            <article key={record.id} className="bg-[#fffefc] border border-[#efeeeb] rounded-[14px] p-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="min-w-0">
                <p className="text-[14px] font-medium text-[#0f3e17] truncate">{record.filename || "Uploaded radiograph"}</p>
                <p className="text-[12px] text-[#222222]/60 mt-1">{new Date(record.created_at).toLocaleString()}</p>
                <div className="flex flex-wrap gap-2 mt-3">
                  {record.classification_labels.slice(0, 4).map((label) => (
                    <span key={label} className="bg-[#e1f4df] text-[#0f3e17] px-2.5 py-1 rounded-full text-[11px] capitalize">{label}</span>
                  ))}
                </div>
              </div>
              <div className="flex items-center gap-5 text-[12px] text-[#222222]/70 shrink-0">
                <span className="inline-flex items-center gap-1.5">
                  {record.fracture_detected ? <XCircle className="w-4 h-4 text-red-700" /> : <CheckCircle className="w-4 h-4 text-[#0f3e17]" />}
                  {record.fracture_detected ? "Fracture signal" : "No fracture signal"}
                </span>
                <span>{Math.round(record.fracture_confidence * 100)}% confidence</span>
                {record.segmentation_available && <span className="text-[#0f3e17]">MedSAM mask</span>}
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}
