import { BarChart2, FileText, TrendingUp, AlertTriangle } from "lucide-react";

export default function ResultsPage() {
  return (
    <div className="max-w-5xl mx-auto px-6 py-12">
      <div className="mb-10">
        <h1 className="text-3xl font-bold gradient-text mb-2">Evaluation Results</h1>
        <p className="text-slate-400 text-sm">
          Model performance metrics on the held-out test split. Run{" "}
          <code className="bg-white/5 px-1.5 py-0.5 rounded text-violet-300 text-xs">
            python model/evaluation/evaluate.py
          </code>{" "}
          to generate results.
        </p>
      </div>

      <div className="glass border border-amber-500/20 bg-amber-500/5 rounded-2xl p-5 mb-8 flex gap-3">
        <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
        <p className="text-sm text-amber-400/90 leading-relaxed">
          These metrics are computed on the research test split and have NOT been externally validated.
          They do not constitute clinical performance evidence. Recovery-time prediction metrics are
          not reported — training is disabled due to absence of clinical outcome labels.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
        {[
          { icon: BarChart2, label: "Detection (mAP@50)", value: "—", sub: "Run evaluate.py to generate", color: "text-violet-400" },
          { icon: TrendingUp, label: "Fracture AUROC", value: "—", sub: "Run evaluate.py to generate", color: "text-cyan-400" },
          { icon: FileText, label: "Segmentation Dice", value: "—", sub: "MedSAM checkpoint required", color: "text-emerald-400" },
        ].map((m) => (
          <div key={m.label} className="glass rounded-2xl p-6">
            <m.icon className={`w-6 h-6 ${m.color} mb-4`} />
            <p className="text-2xl font-bold text-white mb-1">{m.value}</p>
            <p className="text-sm text-slate-300 font-medium mb-1">{m.label}</p>
            <p className="text-xs text-slate-600">{m.sub}</p>
          </div>
        ))}
      </div>

      <div className="glass rounded-2xl p-6">
        <h2 className="text-lg font-semibold text-white mb-4">How to Generate Results</h2>
        <div className="space-y-3">
          {[
            { step: "1", cmd: "python model/data/inspect_datasets.py", desc: "Generate dataset quality report" },
            { step: "2", cmd: "python model/data/create_splits.py", desc: "Create patient-level train/val/test splits" },
            { step: "3", cmd: "python model/training/train_classifier.py", desc: "Train DINOv2 fracture classifier" },
            { step: "4", cmd: "python model/training/train_detector.py", desc: "Train RT-DETRv2 detector" },
            { step: "5", cmd: "python model/evaluation/evaluate.py", desc: "Run evaluation and generate reports" },
          ].map((s) => (
            <div key={s.step} className="flex items-start gap-4">
              <span className="w-6 h-6 rounded-full bg-violet-600/20 border border-violet-500/30 text-violet-400 text-xs flex items-center justify-center flex-shrink-0 mt-0.5 font-bold">
                {s.step}
              </span>
              <div>
                <code className="text-xs bg-white/5 text-violet-300 px-2 py-1 rounded block mb-0.5">{s.cmd}</code>
                <p className="text-xs text-slate-500">{s.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
