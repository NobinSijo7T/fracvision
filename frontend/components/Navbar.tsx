"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Home" },
  { href: "/analyze", label: "Analyze" },
  { href: "/compare", label: "Compare" },
  { href: "/history", label: "History" },
];

export function Navbar() {
  const pathname = usePathname();

  return (
    <header className="fixed top-0 inset-x-0 z-50 bg-[#fffefc]/90 backdrop-blur-md border-b border-[#efeeeb] transition-all">
      <div className="max-w-[1240px] mx-auto px-6 h-20 flex items-center justify-between">
        {/* Brand Logo - Botanical green icon + FracVision wordmark in display serif */}
        <Link href="/" className="flex items-center gap-3 group">
          <div className="w-9 h-9 rounded-lg bg-[#0f3e17] flex items-center justify-center p-2 text-[#fffefc]">
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="w-5 h-5 text-[#e1f4df]"
            >
              {/* Botanical curved leaf emblem */}
              <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z" />
              <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12" />
            </svg>
          </div>
          <span className="font-serif text-[28px] font-light leading-none text-[#0f3e17] tracking-tight">
            FracVision
          </span>
        </Link>

        {/* Navigation Links: Home, Analyze, Compare, History */}
        <nav className="hidden md:flex items-center gap-8">
          {links.map(({ href, label }) => {
            const isActive = pathname === href;
            return (
              <Link
                key={href}
                href={href}
                className={`text-[14px] font-normal transition-colors duration-150 py-1.5 px-2 rounded-[7px] ${
                  isActive
                    ? "text-[#0f3e17] font-medium"
                    : "text-[#222222] hover:text-[#0f3e17]"
                }`}
              >
                {label}
              </Link>
            );
          })}
        </nav>

        {/* Right CTA - Forest Ink Pill Button */}
        <div className="flex items-center gap-4">
          <Link
            href="/analyze"
            className="inline-flex items-center justify-center bg-[#0f3e17] hover:bg-[#0c2f10] text-[#fffefc] text-[14px] font-normal rounded-full px-5 py-2.5 transition-colors"
          >
            Analyze X-ray
          </Link>
        </div>
      </div>
    </header>
  );
}
