import { Brain, Shield, Database, Code2, ExternalLink } from "lucide-react";

const models = [
  { name: "DINOv2", year: "2023", org: "Meta AI", paper: "https://arxiv.org/abs/2304.07193", desc: "Self-supervised vision transformer. Used for rich feature extraction from X-ray images without requiring annotated pre-training data." },
  { name: "Mamba", year: "2023", org: "Carnegie Mellon / Princeton", paper: "https://arxiv.org/abs/2312.00752", desc: "Selective state space model for efficient long-range sequence modeling. Applied to patient visit sequences for longitudinal healing pattern analysis." },
  { name: "RT-DETRv2", year: "2024", org: "Baidu", paper: "https://arxiv.org/abs/2407.17140", desc: "Real-time detection transformer without NMS post-processing. Used for fracture localization with bounding boxes and confidence scores." },
  { name: "MedSAM", year: "2024", org: "Harvard / UNC", paper: "https://www.nature.com/articles/s41467-024-44824-z", desc: "Medical Segment Anything Model fine-tuned on 1M+ medical image-mask pairs. Generates pixel-level fracture segmentation masks from bounding-box prompts." },
];

const datasets = [
  { name: "FracAtlas", records: "4,085", annotations: "COCO JSON · YOLO · Pascal VOC", labels: "fractured, body part, view", url: "https://figshare.com/articles/dataset/The_dataset/22363012" },
  { name: "GRAZPEDWRI-DX", records: "20,328", annotations: "Pascal VOC", labels: "fracture, pronatorsign, periostealreaction, ...", url: "https://figshare.com/articles/dataset/GRAZPEDWRI-DX/14825193" },
];

export default function AboutPage() {
  return (
    <div className="max-w-4xl mx-auto px-6 py-12">
      <div className="mb-12">
        <h1 className="text-3xl font-bold gradient-text mb-2">About FracVision</h1>
        <p className="text-slate-400">
          A modular research prototype for multimodal X-ray fracture analysis using 2023–2024 foundation models.
        </p>
      </div>

      {/* Disclaimer */}
      <div className="glass border border-red-500/30 bg-red-500/5 rounded-2xl p-6 mb-10 flex gap-4">
        <Shield className="w-6 h-6 text-red-400 flex-shrink-0 mt-0.5" />
        <div>
          <p className="font-semibold text-red-300 mb-1">Research Prototype — Not for Clinical Use</p>
          <p className="text-sm text-slate-400 leading-relaxed">
            FracVision has NOT been clinically validated. All outputs must be reviewed by a qualified clinician.
            Recovery-time prediction is disabled — neither dataset provides verified time-to-union labels.
            This system must not be used for autonomous clinical decision-making.
          </p>
        </div>
      </div>

      {/* Models */}
      <section className="mb-12">
        <h2 className="text-xl font-semibold text-white mb-6 flex items-center gap-2">
          <Brain className="w-5 h-5 text-violet-400" /> Foundation Models
        </h2>
        <div className="space-y-4">
          {models.map((m) => (
            <div key={m.name} className="glass rounded-xl p-5">
              <div className="flex items-start justify-between gap-4 mb-2">
                <div>
                  <span className="font-semibold text-white">{m.name}</span>
                  <span className="text-xs text-slate-500 ml-2">{m.year} · {m.org}</span>
                </div>
                <a href={m.paper} target="_blank" rel="noopener noreferrer" className="text-violet-400 hover:text-violet-300 transition-colors">
                  <ExternalLink className="w-4 h-4" />
                </a>
              </div>
              <p className="text-sm text-slate-400 leading-relaxed">{m.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Datasets */}
      <section className="mb-12">
        <h2 className="text-xl font-semibold text-white mb-6 flex items-center gap-2">
          <Database className="w-5 h-5 text-cyan-400" /> Datasets
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {datasets.map((d) => (
            <div key={d.name} className="glass rounded-xl p-5">
              <div className="flex items-center justify-between mb-3">
                <span className="font-semibold text-white">{d.name}</span>
                <a href={d.url} target="_blank" rel="noopener noreferrer" className="text-xs text-violet-400 hover:underline flex items-center gap-1">
                  Dataset <ExternalLink className="w-3 h-3" />
                </a>
              </div>
              <div className="space-y-1.5 text-xs text-slate-400">
                <p><span className="text-slate-500">Records:</span> {d.records}</p>
                <p><span className="text-slate-500">Annotations:</span> {d.annotations}</p>
                <p><span className="text-slate-500">Labels:</span> {d.labels}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Tech Stack */}
      <section>
        <h2 className="text-xl font-semibold text-white mb-6 flex items-center gap-2">
          <Code2 className="w-5 h-5 text-emerald-400" /> Technology Stack
        </h2>
        <div className="glass rounded-xl overflow-hidden">
          {[
            ["ML Framework", "PyTorch 2.3"],
            ["Feature Extraction", "DINOv2 (torch.hub)"],
            ["Detection", "RT-DETRv2 (Hugging Face transformers)"],
            ["Segmentation", "MedSAM (segment-anything)"],
            ["Sequence Modeling", "Mamba SSM (CUDA) / Pure-PyTorch fallback"],
            ["Backend", "FastAPI + SQLAlchemy + PostgreSQL"],
            ["Frontend", "Next.js 14 + TypeScript + Tailwind CSS"],
            ["Experiment Tracking", "MLflow"],
            ["Medical Imaging", "pydicom + SimpleITK + MONAI"],
          ].map(([label, tech]) => (
            <div key={label} className="flex items-center gap-4 px-5 py-3 border-b border-white/5 last:border-0">
              <span className="text-xs text-slate-500 w-36">{label}</span>
              <span className="text-sm text-slate-300 font-medium">{tech}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
