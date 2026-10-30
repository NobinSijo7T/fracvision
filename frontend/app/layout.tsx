import type { Metadata } from "next";
import { Inter, Cormorant_Garamond } from "next/font/google";
import "./globals.css";
import { Navbar } from "@/components/Navbar";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const cormorant = Cormorant_Garamond({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600"],
  variable: "--font-cormorant",
  display: "swap",
});

export const metadata: Metadata = {
  title: "FracVision — Multimodal X-ray Fracture Analysis",
  description:
    "AI-assisted fracture detection and recovery analysis using DINOv2, RT-DETRv2, MedSAM, and Mamba. Research prototype.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} ${cormorant.variable}`}>
      <body className="font-sans bg-[#fffefc] text-[#222222] min-h-screen antialiased selection:bg-[#e1f4df] selection:text-[#0f3e17]">
        <Navbar />
        <main className="pt-20">{children}</main>
      </body>
    </html>
  );
}
