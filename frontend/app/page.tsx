import Link from "next/link";
import { Activity, Upload, GitCompare, ArrowRight, Zap, Shield, Brain } from "lucide-react";

const features = [
  {
    icon: Upload,
    title: "Fracture Detection",
    desc: "RT-DETRv2 (2024) detects and localizes fractures with bounding boxes and confidence scores.",
    color: "from-violet-500 to-purple-600",
  },
  {
    icon: Zap,
    title: "Region Segmentation",
    desc: "MedSAM (2024) generates pixel-level fracture masks from detection prompts.",
    color: "from-cyan-500 to-blue-600",
  },
  {
    icon: Brain,
    title: "Visual Intelligence",
    desc: "DINOv2 (2023) extracts rich visual features for classification and comparison.",
    color: "from-emerald-500 to-teal-600",
  },
  {
    icon: GitCompare,
    title: "Longitudinal Analysis",
    desc: "Mamba (2023) SSM models healing patterns across serial X-ray sequences.",
    color: "from-rose-500 to-pink-600",
  },
];

const models = [
  { name: "DINOv2", year: "2023", role: "Feature Extraction" },
  { name: "Mamba", year: "2023", role: "Longitudinal Modeling" },
  { name: "RT-DETRv2", year: "2024", role: "Fracture Detection" },
  { name: "MedSAM", year: "2024", role: "Segmentation" },
];

export default function HomePage() {
  return (
    <div className="relative min-h-screen overflow-hidden">
      {/* Background gradient orbs */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-violet-600/15 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-cyan-600/10 rounded-full blur-3xl" />
      </div>

      <div className="relative max-w-6xl mx-auto px-6 pt-20 pb-32">
        {/* Hero */}
        <div className="text-center mb-20">
          <div className="inline-flex items-center gap-2 glass px-4 py-2 rounded-full text-sm text-violet-300 mb-8 border border-violet-500/20">
            <span className="w-2 h-2 rounded-full bg-violet-400 animate-pulse" />
            Research Prototype · Not for Clinical Use
          </div>

          <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
            <span className="gradient-text">FracVision</span>
          </h1>
          <p className="text-xl md:text-2xl text-slate-400 mb-4 font-light">
            Multimodal X-ray Fracture Analysis
          </p>
          <p className="text-base text-slate-500 max-w-2xl mx-auto mb-10">
            A research prototype using state-of-the-art 2023–2024 foundation models to detect,
            segment, and track bone fractures in X-ray images. All outputs require clinician review.
          </p>

          <div className="flex flex-wrap gap-4 justify-center">
            <Link href="/analyze" className="btn-glow pulse-ring">
              <Upload className="w-4 h-4" />
              Analyze X-ray
              <ArrowRight className="w-4 h-4" />
            </Link>
            <Link href="/compare" className="btn-secondary">
              <GitCompare className="w-4 h-4" />
              Compare Series
            </Link>
          </div>
        </div>

        {/* Model cards */}
        <div className="flex flex-wrap justify-center gap-3 mb-20">
          {models.map((m) => (
            <div key={m.name} className="glass px-5 py-3 rounded-xl flex items-center gap-3">
              <div className="w-2 h-2 rounded-full bg-violet-400" />
              <div>
                <p className="font-semibold text-sm text-white">{m.name}</p>
                <p className="text-xs text-slate-500">{m.year} · {m.role}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Feature grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-20">
          {features.map((f) => (
            <div key={f.title} className="glass rounded-2xl p-6 hover:border-white/20 transition-all duration-300 group">
              <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${f.color} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300`}>
                <f.icon className="w-5 h-5 text-white" />
              </div>
              <h3 className="font-semibold text-white mb-2">{f.title}</h3>
              <p className="text-sm text-slate-400 leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>

        {/* Clinical Disclaimer */}
        <div className="glass border border-red-500/20 rounded-2xl p-6 bg-red-500/5">
          <div className="flex items-start gap-3">
            <Shield className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold text-red-300 mb-1">Clinical Disclaimer</p>
              <p className="text-sm text-slate-400 leading-relaxed">
                This system is a <strong className="text-slate-300">research prototype</strong> and has{" "}
                <strong className="text-slate-300">NOT</strong> been clinically validated. All outputs
                must be reviewed by a qualified clinician before any clinical application. Recovery-time
                prediction is disabled — no verified clinical outcome labels are available in the training datasets.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
