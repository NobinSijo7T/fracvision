import { BarChart2, FileText, TrendingUp, AlertTriangle } from "lucide-react";

export default function ResultsPage() {
  return (
    <div className="max-w-[1100px] mx-auto px-6 py-10">
      {/* Header */}
      <div className="mb-10">
        <span className="eyebrow-label block mb-2">Benchmark Performance</span>
        <h1 className="font-serif text-[42px] md:text-[52px] font-light text-[#0f3e17] leading-tight mb-2">
          Model Evaluation Results
        </h1>
        <p className="text-[14px] text-[#222222]/80 max-w-2xl">
          Quantitative metrics measured across held-out patient splits. Run{" "}
          <code className="bg-[#e1f4df] text-[#0f3e17] px-2 py-0.5 rounded font-mono text-[12px]">
            python model/evaluation/evaluate.py
          </code>{" "}
          to reproduce reports.
        </p>
      </div>

      {/* Advisory Banner */}
      <div className="bg-[#e1f4df] border border-[#cfe7d3] rounded-[14px] p-5 mb-8 flex gap-3.5 items-start">
        <AlertTriangle className="w-5 h-5 text-[#0f3e17] flex-shrink-0 mt-0.5" />
        <p className="text-[13px] text-[#0f3e17] leading-relaxed">
          These metrics reflect internal evaluation splits on the FracAtlas and GRAZPEDWRI-DX research datasets. They do not constitute multicenter clinical efficacy or regulatory clearance.
        </p>
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-10">
        {[
          {
            icon: BarChart2,
            label: "Detection (mAP@50)",
            value: "—",
            sub: "Run evaluate.py to generate",
            surface: "bg-[#e1f4df]",
          },
          {
            icon: TrendingUp,
            label: "Fracture AUROC",
            value: "—",
            sub: "Run evaluate.py to generate",
            surface: "bg-[#cfe7d3]",
          },
          {
            icon: FileText,
            label: "Segmentation Dice",
            value: "—",
            sub: "MedSAM checkpoint required",
            surface: "bg-[#b1dbb8]",
          },
        ].map((m) => (
          <div key={m.label} className={`${m.surface} rounded-[14px] p-6 flex flex-col justify-between min-h-[170px]`}>
            <div>
              <m.icon className="w-6 h-6 text-[#0f3e17] mb-3" />
              <span className="eyebrow-label block mb-1 text-[#0f3e17]/80">{m.label}</span>
              <p className="font-serif text-[36px] font-light text-[#0f3e17] leading-none mb-2">{m.value}</p>
            </div>
            <p className="text-[12px] text-[#222222]/70">{m.sub}</p>
          </div>
        ))}
      </div>

      {/* Execution Instructions */}
      <div className="bg-[#fffefc] border border-[#efeeeb] rounded-[14px] p-8">
        <span className="eyebrow-label block mb-2">Evaluation Pipeline</span>
        <h2 className="font-serif text-[28px] font-light text-[#0f3e17] mb-6">
          How to Generate Evaluation Reports
        </h2>
        <div className="space-y-4">
          {[
            { step: "1", cmd: "python model/data/inspect_datasets.py", desc: "Generate dataset quality report and summary distributions" },
            { step: "2", cmd: "python model/data/create_splits.py", desc: "Create patient-level train, validation, and test splits" },
            { step: "3", cmd: "python model/training/train_classifier.py", desc: "Train DINOv2 self-supervised fracture classification head" },
            { step: "4", cmd: "python model/training/train_detector.py", desc: "Train RT-DETRv2 bounding box localization transformer" },
            { step: "5", cmd: "python model/evaluation/evaluate.py", desc: "Execute benchmark evaluation suite and output report tables" },
          ].map((s) => (
            <div key={s.step} className="flex items-start gap-4 pb-3 border-b border-[#efeeeb] last:border-b-0">
              <span className="w-7 h-7 rounded-full bg-[#0f3e17] text-[#fffefc] text-[12px] flex items-center justify-center flex-shrink-0 mt-0.5 font-medium">
                {s.step}
              </span>
              <div>
                <code className="text-[13px] bg-[#e1f4df] text-[#0f3e17] font-mono px-2.5 py-1 rounded-[6px] inline-block mb-1">
                  {s.cmd}
                </code>
                <p className="text-[13px] text-[#222222]/70">{s.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
