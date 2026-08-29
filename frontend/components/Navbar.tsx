"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, Upload, GitCompare, BarChart2, Clock, Info } from "lucide-react";

const links = [
  { href: "/", label: "Home", icon: Activity },
  { href: "/analyze", label: "Analyze", icon: Upload },
  { href: "/compare", label: "Compare", icon: GitCompare },
  { href: "/results", label: "Results", icon: BarChart2 },
  { href: "/history", label: "History", icon: Clock },
  { href: "/about", label: "About", icon: Info },
];

export function Navbar() {
  const pathname = usePathname();
  return (
    <nav className="fixed top-0 inset-x-0 z-50 glass border-b border-white/8 px-6 h-16 flex items-center justify-between">
      <Link href="/" className="flex items-center gap-2.5">
        <div className="w-8 h-8 rounded-lg bg-violet-600 flex items-center justify-center shadow-[0_0_16px_rgba(139,92,246,0.5)]">
          <Activity className="w-4 h-4 text-white" />
        </div>
        <span className="font-bold text-lg gradient-text">FracVision</span>
        <span className="text-[10px] text-amber-400 bg-amber-400/10 border border-amber-400/20 px-1.5 py-0.5 rounded-md font-medium">
          RESEARCH
        </span>
      </Link>

      <div className="hidden md:flex items-center gap-1">
        {links.map(({ href, label, icon: Icon }) => (
          <Link
            key={href}
            href={href}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-150 ${
              pathname === href
                ? "bg-violet-600/20 text-violet-300 border border-violet-500/20"
                : "text-slate-400 hover:text-slate-200 hover:bg-white/5"
            }`}
          >
            <Icon className="w-3.5 h-3.5" />
            {label}
          </Link>
        ))}
      </div>

      <div className="text-xs text-red-400/70 hidden lg:block border border-red-500/20 bg-red-500/5 px-3 py-1.5 rounded-lg">
        ⚠️ Not for clinical use
      </div>
    </nav>
  );
}
