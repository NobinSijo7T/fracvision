import { Clock, Info } from "lucide-react";

export default function HistoryPage() {
  return (
    <div className="max-w-4xl mx-auto px-6 py-12">
      <div className="mb-10">
        <h1 className="text-3xl font-bold gradient-text mb-2">Patient History</h1>
        <p className="text-slate-400 text-sm">
          View and compare serial X-ray sequences for longitudinal tracking.
          Connect the backend database to load patient study history.
        </p>
      </div>

      <div className="glass border border-blue-500/20 bg-blue-500/5 rounded-2xl p-5 mb-8 flex gap-3">
        <Info className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
        <p className="text-sm text-blue-400/90 leading-relaxed">
          Patient history requires a connected PostgreSQL database and submitted predictions.
          Start the backend (<code className="bg-white/5 px-1 rounded text-xs">uvicorn backend.main:app</code>),
          analyze X-rays via the Analyze page, and they will appear here.
        </p>
      </div>

      <div className="glass rounded-2xl flex flex-col items-center justify-center py-24 text-center">
        <Clock className="w-12 h-12 text-slate-700 mb-4" />
        <p className="text-slate-500 font-medium mb-1">No history entries yet</p>
        <p className="text-slate-600 text-sm max-w-xs">
          Upload and analyze X-rays to build patient history. Serial sequences will appear here for longitudinal comparison.
        </p>
      </div>
    </div>
  );
}
