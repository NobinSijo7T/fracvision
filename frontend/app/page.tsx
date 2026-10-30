import Link from "next/link";
import { Check, ChevronDown, Plus, ArrowRight, Activity, Layers, GitCompare, Eye, ShieldAlert } from "lucide-react";

export default function HomePage() {
  return (
    <div className="bg-[#fffefc] min-h-screen text-[#222222] pb-24">
      {/* Container - max 1200px centered */}
      <div className="max-w-[1200px] mx-auto px-6 pt-6">

        {/* HERO SECTION - Two-column split layout matching reference image */}
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-5 mb-16 items-stretch">
          
          {/* Left Column (Approx 5 cols / 42%) */}
          <div className="lg:col-span-5 flex flex-col gap-4 justify-between">
            
            {/* Top Keylime Panel */}
            <div className="bg-[#e1f4df] rounded-[14px] p-8 md:p-12 flex-1 flex flex-col justify-center">
              <h1 className="font-serif text-[48px] md:text-[62px] leading-[1.08] font-light text-[#0f3e17] tracking-[-0.03em] mb-6">
                Fewer clicks.<br />
                Earlier healing.
              </h1>
              <p className="text-[15px] leading-[1.55] text-[#222222] font-normal max-w-[420px]">
                Detect bone fractures in seconds, localize cortical disruptions with MedSAM segmentation, and track serial healing with an AI-native system built for radiographic precision.
              </p>
            </div>

            {/* Bottom Forest Button Container */}
            <Link
              href="/analyze"
              className="group bg-[#0f3e17] hover:bg-[#0c2f10] text-[#fffefc] rounded-[14px] p-7 md:p-8 flex items-center justify-between transition-colors"
            >
              <span className="text-[18px] md:text-[20px] font-normal tracking-tight">
                Analyze a Radiograph
              </span>
              <div className="w-10 h-10 rounded-full flex items-center justify-center transition-transform group-hover:translate-x-1">
                <svg
                  width="28"
                  height="28"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.75"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="text-[#fffefc]"
                >
                  <line x1="4" y1="12" x2="20" y2="12"></line>
                  <polyline points="14 6 20 12 14 18"></polyline>
                </svg>
              </div>
            </Link>

          </div>

          {/* Right Column (Approx 7 cols / 58%) - Slate Hush Panel */}
          <div className="lg:col-span-7 bg-[#b6ced5] rounded-[14px] p-8 md:p-11 flex flex-col justify-center relative overflow-hidden">
            
            {/* Inner 3 white floating cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-stretch relative">
              
              {/* Connector line between DETECT and SEGMENT on md+ screens */}
              <div className="hidden md:block absolute top-[45%] left-[29%] right-[36%] h-[2px] border-t-2 border-dashed border-[#0f3e17]/30 z-0 pointer-events-none" />

              {/* CARD 1: DETECT */}
              <div className="flex flex-col items-center">
                <div className="bg-[#fffefc] rounded-[14px] p-6 w-full flex-1 flex flex-col items-center text-center justify-center min-h-[260px] relative z-10">
                  {/* Bone / Radiograph Icon */}
                  <div className="w-12 h-12 rounded-full overflow-hidden mb-3 border border-[#efeeeb] bg-[#e1f4df] text-[#0f3e17] flex items-center justify-center">
                    <Activity className="w-6 h-6 stroke-[1.75]" />
                  </div>
                  <h3 className="text-[15px] font-medium text-[#222222] mb-1">
                    Study #89421
                  </h3>
                  <p className="text-[11px] text-[#222222]/60 mb-3">
                    Distal Radius · PA View
                  </p>
                  <div className="inline-flex items-center gap-1.5 bg-[#e1f4df] text-[#0f3e17] px-3 py-1 rounded-full text-[12px] font-normal">
                    <Check className="w-3.5 h-3.5 stroke-[2.5]" />
                    <span>Fracture detected</span>
                  </div>
                </div>
                {/* Pill tag underneath */}
                <div className="mt-3.5">
                  <span className="bg-[#fffefc] text-[#0f3e17] text-[12px] font-medium px-4 py-1.5 rounded-full inline-block">
                    DETECT
                  </span>
                </div>
              </div>

              {/* CARD 2: SEGMENT */}
              <div className="flex flex-col items-center">
                <div className="bg-[#fffefc] rounded-[14px] p-5 w-full flex-1 flex flex-col justify-between min-h-[260px] text-left relative z-10">
                  <div>
                    <h3 className="font-serif text-[18px] font-normal text-[#222222] mb-1">
                      Fracture Mask
                    </h3>
                    <div className="text-[11px] text-[#222222]/70 space-y-0.5 mb-2.5">
                      <p>Type: Colles Displaced</p>
                      <p>Coverage: 1,280 px²</p>
                    </div>

                    {/* Selector */}
                    <div className="border border-[#efeeeb] rounded-[7px] px-2.5 py-1.5 text-[11px] text-[#222222] flex items-center justify-between mb-2.5 bg-[#fffefc]">
                      <span>MedSAM Overlay</span>
                      <ChevronDown className="w-3.5 h-3.5 text-[#222222]/60" />
                    </div>

                    <p className="text-[11px] text-[#222222]/80 italic mb-2">
                      Cortical margin mapped
                    </p>
                  </div>

                  <button className="w-full bg-[#0f3e17] hover:bg-[#0c2f10] text-[#fffefc] text-[11px] font-normal py-2 px-3 rounded-[7px] flex items-center justify-center gap-1 transition-colors">
                    <Plus className="w-3 h-3" />
                    <span>Inspect Mask</span>
                  </button>
                </div>
                {/* Pill tag underneath */}
                <div className="mt-3.5">
                  <span className="bg-[#fffefc] text-[#0f3e17] text-[12px] font-medium px-4 py-1.5 rounded-full inline-block">
                    SEGMENT
                  </span>
                </div>
              </div>

              {/* CARD 3: TRACK */}
              <div className="flex flex-col items-center">
                <div className="bg-[#fffefc] rounded-[14px] p-5 w-full flex-1 flex flex-col justify-between min-h-[260px] relative z-10">
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-[17px] font-semibold text-[#222222]">
                        96.8%
                      </span>
                      <span className="bg-[#e1f4df] text-[#0f3e17] text-[10px] font-medium px-1.5 py-0.5 rounded-full">
                        Calibrated
                      </span>
                    </div>

                    <div className="text-[10px] text-[#222222]/60 border-t border-[#efeeeb] pt-1.5 mb-3">
                      <div className="flex justify-between">
                        <span>Healing alignment</span>
                      </div>
                      <div className="flex justify-between text-[9px] text-[#222222]/50 mt-0.5">
                        <span>Visit 1 → Visit 2</span>
                        <span>0.912 Cosine</span>
                      </div>
                    </div>
                  </div>

                  {/* SVG Healing Trajectory Graph with dotted vertical line */}
                  <div className="h-20 w-full relative">
                    <svg
                      viewBox="0 0 160 80"
                      className="w-full h-full overflow-visible"
                    >
                      {/* Dotted target marker */}
                      <line
                        x1="80"
                        y1="5"
                        x2="80"
                        y2="75"
                        stroke="#222222"
                        strokeWidth="1"
                        strokeDasharray="2,2"
                        opacity="0.4"
                      />
                      {/* Area wash */}
                      <path
                        d="M 10 55 Q 50 50, 80 35 T 150 25 L 150 75 L 10 75 Z"
                        fill="#e1f4df"
                        opacity="0.7"
                      />
                      {/* Main Trend Line */}
                      <path
                        d="M 10 55 Q 50 50, 80 35 T 150 25"
                        fill="none"
                        stroke="#b1dbb8"
                        strokeWidth="2.5"
                      />
                      {/* Secondary baseline */}
                      <path
                        d="M 10 65 Q 60 62, 100 58 T 150 50"
                        fill="none"
                        stroke="#b6ced5"
                        strokeWidth="1.5"
                      />
                    </svg>
                  </div>
                </div>
                {/* Pill tag underneath */}
                <div className="mt-3.5">
                  <span className="bg-[#fffefc] text-[#0f3e17] text-[12px] font-medium px-4 py-1.5 rounded-full inline-block">
                    TRACK
                  </span>
                </div>
              </div>

            </div>
          </div>

        </section>

        {/* PARTNER / INSTITUTIONAL PROOF STRIP */}
        <section className="py-12 border-b border-[#efeeeb] mb-20 text-center">
          <p className="text-[13px] text-[#222222]/60 font-normal mb-8">
            Powered by foundation architectures & benchmark datasets
          </p>
          <div className="flex flex-wrap items-center justify-center gap-8 md:gap-14 text-[#222222] opacity-85">
            <span className="font-serif tracking-tight text-[16px] font-semibold">
              Meta AI · DINOv2
            </span>
            <span className="font-serif text-[17px] tracking-tight font-normal">
              Baidu · RT-DETRv2
            </span>
            <span className="font-sans text-[15px] font-bold tracking-tight flex items-center gap-1">
              <span className="text-[#0f3e17]">M</span> MedSAM · Harvard / UNC
            </span>
            <span className="font-mono text-[13px] border border-[#222222] px-2.5 py-0.5 tracking-wider font-bold">
              MAMBA SSM
            </span>
            <span className="font-serif text-[18px] font-semibold">
              FracAtlas
            </span>
            <span className="text-[13px] font-semibold tracking-wider flex items-center gap-1 uppercase">
              GRAZPEDWRI-DX
            </span>
          </div>
        </section>

        {/* RADIOGRAPHIC MODULES SECTION - Starting with Keylime panel */}
        <section className="bg-[#e1f4df] rounded-[14px] p-8 md:p-14 mb-20">
          <div className="max-w-2xl mb-12">
            <span className="eyebrow-label block mb-2">Multimodal Radiology Pipeline</span>
            <h2 className="font-serif text-[40px] md:text-[54px] font-light leading-[1.1] text-[#0f3e17] mb-4">
              Detection, Masking & Tracking
            </h2>
            <p className="text-[15px] leading-relaxed text-[#222222]/85">
              Designed around four synchronized foundation models to eliminate clinical triage lag, highlight subtle cortical disruptions, and quantify serial changes.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            {/* Module 1: RT-DETRv2 Detection */}
            <div className="bg-[#fffefc] rounded-[14px] p-7 flex flex-col justify-between min-h-[260px]">
              <div>
                <div className="w-10 h-10 rounded-full bg-[#e1f4df] text-[#0f3e17] flex items-center justify-center mb-4">
                  <Activity className="w-5 h-5" />
                </div>
                <h3 className="font-serif text-[22px] font-light text-[#0f3e17] mb-2">
                  Fracture Localization
                </h3>
                <p className="text-[13px] text-[#222222]/80 leading-relaxed">
                  Real-time transformer detection predicting acute bone fractures, pronator fat pad signs, and periosteal reactions with calibrated probability metrics.
                </p>
              </div>
              <div className="pt-4 border-t border-[#efeeeb] flex items-center justify-between text-[12px] text-[#0f3e17] font-medium">
                <span>RT-DETRv2 Engine</span>
                <span>Sub-Second</span>
              </div>
            </div>

            {/* Module 2: MedSAM Segmentation */}
            <div className="bg-[#fffefc] rounded-[14px] p-7 flex flex-col justify-between min-h-[260px]">
              <div>
                <div className="w-10 h-10 rounded-full bg-[#cfe7d3] text-[#0f3e17] flex items-center justify-center mb-4">
                  <Layers className="w-5 h-5" />
                </div>
                <h3 className="font-serif text-[22px] font-light text-[#0f3e17] mb-2">
                  Pixel-Level Masking
                </h3>
                <p className="text-[13px] text-[#222222]/80 leading-relaxed">
                  Medical Segment Anything Model generating tight pixel-level segmentation masks around complex fracture lines and cortical fragment displacements.
                </p>
              </div>
              <div className="pt-4 border-t border-[#efeeeb] flex items-center justify-between text-[12px] text-[#0f3e17] font-medium">
                <span>MedSAM Checkpoint</span>
                <span>Fine-Tuned</span>
              </div>
            </div>

            {/* Module 3: Mamba Longitudinal Tracking */}
            <div className="bg-[#fffefc] rounded-[14px] p-7 flex flex-col justify-between min-h-[260px]">
              <div>
                <div className="w-10 h-10 rounded-full bg-[#b1dbb8] text-[#0f3e17] flex items-center justify-center mb-4">
                  <GitCompare className="w-5 h-5" />
                </div>
                <h3 className="font-serif text-[22px] font-light text-[#0f3e17] mb-2">
                  Serial Visit Modeling
                </h3>
                <p className="text-[13px] text-[#222222]/80 leading-relaxed">
                  Selective state-space modeling comparing prior and follow-up radiographs to monitor alignment restoration, callus formation, and healing trajectories.
                </p>
              </div>
              <div className="pt-4 border-t border-[#efeeeb] flex items-center justify-between text-[12px] text-[#0f3e17] font-medium">
                <span>Mamba SSM + DINOv2</span>
                <span>Longitudinal</span>
              </div>
            </div>
          </div>
        </section>

        {/* WORKFLOW HIGHLIGHT - Sage Mist Panel */}
        <section className="bg-[#b1dbb8] rounded-[14px] p-8 md:p-14 mb-20">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
            <div className="lg:col-span-6">
              <span className="eyebrow-label block mb-2 text-[#0f3e17]">
                Clinical Workflow
              </span>
              <h2 className="font-serif text-[38px] md:text-[50px] font-light leading-[1.12] text-[#0f3e17] mb-4">
                Instant radiologist assistance without friction
              </h2>
              <p className="text-[15px] text-[#222222] leading-relaxed mb-8 max-w-md">
                Upload raw radiographs in standard JPEG, PNG, or DICOM formats. FracVision immediately runs image quality control, bounding-box localization, and interactive segmentation.
              </p>
              <div className="flex flex-wrap gap-3">
                <Link
                  href="/analyze"
                  className="bg-[#0f3e17] hover:bg-[#0c2f10] text-[#fffefc] text-[14px] px-6 py-3.5 rounded-[14px] inline-flex items-center gap-2 transition-colors"
                >
                  Analyze Radiograph
                  <ArrowRight className="w-4 h-4" />
                </Link>
                <Link
                  href="/compare"
                  className="bg-[#fffefc] hover:bg-[#e1f4df] text-[#0f3e17] text-[14px] px-6 py-3.5 rounded-[14px] inline-flex items-center gap-2 transition-colors"
                >
                  Serial Comparison
                </Link>
              </div>
            </div>

            <div className="lg:col-span-6">
              <div className="bg-[#fffefc] rounded-[14px] p-7 space-y-4">
                <div className="flex items-center justify-between border-b border-[#efeeeb] pb-3">
                  <span className="text-[13px] font-medium text-[#0f3e17]">
                    Active Study #89421
                  </span>
                  <span className="bg-[#e1f4df] text-[#0f3e17] text-[11px] font-medium px-2.5 py-0.5 rounded-full">
                    Quality Passed
                  </span>
                </div>
                <div className="space-y-3 text-[13px]">
                  <div className="flex items-center justify-between py-1 border-b border-[#efeeeb]/60">
                    <span className="text-[#222222]/70">Anatomical Region</span>
                    <span className="font-medium text-[#0f3e17]">Distal Radius & Ulna</span>
                  </div>
                  <div className="flex items-center justify-between py-1 border-b border-[#efeeeb]/60">
                    <span className="text-[#222222]/70">Primary Finding</span>
                    <span className="font-medium text-[#0f3e17]">Transverse Radius Fracture (96.8%)</span>
                  </div>
                  <div className="flex items-center justify-between py-1">
                    <span className="text-[#222222]/70">MedSAM Mask</span>
                    <span className="font-medium text-[#0f3e17]">1,280 px² Rendered</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* FAQ SECTION - Two-column serif question left, sans answer right */}
        <section className="mb-20">
          <div className="mb-12">
            <span className="eyebrow-label block mb-2">QUICK ANSWERS</span>
            <h2 className="font-serif text-[40px] font-light text-[#0f3e17]">
              Frequently Asked Questions
            </h2>
          </div>

          <div className="divide-y divide-[#efeeeb] border-t border-b border-[#efeeeb]">
            {[
              {
                q: "What image formats and anatomical views does FracVision support?",
                a: "FracVision accepts standard medical X-ray formats including DICOM (.dcm), PNG, and JPEG. Preprocessing automatically standardizes pixel intensity with CLAHE contrast enhancement and square aspect normalization.",
              },
              {
                q: "How does the MedSAM segmentation pipeline operate?",
                a: "When a fracture candidate is localized by RT-DETRv2, its bounding coordinates act as an automatic spatial prompt for the MedSAM model, which generates a pixel-level boundary mask directly over the radiograph.",
              },
              {
                q: "How does serial comparison evaluate bone healing over time?",
                a: "The comparison tool processes prior and current radiographs side-by-side using DINOv2 deep visual embeddings and Mamba sequence modeling to measure structural similarity, bone alignment, and regional changes.",
              },
              {
                q: "Is FracVision approved for primary clinical diagnosis?",
                a: "No. FracVision is an investigational research prototype designed to explore foundation model capabilities in radiographic triage. All outputs must be validated by a licensed radiologist or orthopedic specialist.",
              },
            ].map((faq, idx) => (
              <div key={idx} className="py-8 grid grid-cols-1 md:grid-cols-12 gap-6 items-start">
                <div className="md:col-span-5">
                  <h3 className="font-serif text-[26px] md:text-[30px] font-light text-[#0f3e17] leading-[1.2]">
                    {faq.q}
                  </h3>
                </div>
                <div className="md:col-span-7">
                  <p className="text-[14px] text-[#222222] leading-relaxed font-normal">
                    {faq.a}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* BOTTOM CTA CARD */}
        <div className="bg-[#e1f4df] rounded-[14px] p-8 md:p-12 flex flex-col md:flex-row items-center justify-between gap-6">
          <div>
            <h3 className="font-serif text-[32px] md:text-[38px] font-light text-[#0f3e17] mb-2">
              Ready to inspect a radiograph?
            </h3>
            <p className="text-[14px] text-[#222222]/80">
              Upload an X-ray to test real-time fracture detection, MedSAM mask overlay, and feature extraction.
            </p>
          </div>
          <Link
            href="/analyze"
            className="bg-[#0f3e17] hover:bg-[#0c2f10] text-[#fffefc] text-[14px] px-8 py-4 rounded-[14px] inline-flex items-center gap-2 transition-colors flex-shrink-0"
          >
            <span>Launch Live Analysis</span>
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>

      </div>
    </div>
  );
}
