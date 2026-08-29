"use client";
import { useRef, useEffect } from "react";

interface Detection {
  label: string;
  bbox_xyxy: number[];
  confidence: number;
}

interface XrayOverlayProps {
  imageUrl: string;
  detections: Detection[];
}

const LABEL_COLORS: Record<string, string> = {
  fracture: "#f87171",           // red-400
  pronatorsign: "#fb923c",       // orange-400
  periostealreaction: "#facc15", // yellow-400
  text: "#94a3b8",               // slate-400
};

function getColor(label: string): string {
  return LABEL_COLORS[label] ?? "#a78bfa"; // violet-400 fallback
}

export function XrayOverlay({ imageUrl, detections }: XrayOverlayProps) {
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas || !img.complete) return;

    const draw = () => {
      const { naturalWidth: nw, naturalHeight: nh, clientWidth: cw, clientHeight: ch } = img;
      canvas.width = cw;
      canvas.height = ch;
      const scaleX = cw / nw;
      const scaleY = ch / nh;

      const ctx = canvas.getContext("2d")!;
      ctx.clearRect(0, 0, cw, ch);

      for (const det of detections) {
        const [x1, y1, x2, y2] = det.bbox_xyxy;
        const color = getColor(det.label);
        const rx = x1 * scaleX;
        const ry = y1 * scaleY;
        const rw = (x2 - x1) * scaleX;
        const rh = (y2 - y1) * scaleY;

        // Shadow glow
        ctx.shadowColor = color;
        ctx.shadowBlur = 10;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(rx, ry, rw, rh);
        ctx.shadowBlur = 0;

        // Corner accents
        const cs = 12;
        ctx.lineWidth = 3;
        [[rx, ry, rx + cs, ry, rx, ry + cs],
         [rx + rw - cs, ry, rx + rw, ry, rx + rw, ry + cs],
         [rx, ry + rh - cs, rx, ry + rh, rx + cs, ry + rh],
         [rx + rw - cs, ry + rh, rx + rw, ry + rh, rx + rw, ry + rh - cs],
        ].forEach(([x0, y0, xm, ym, xe, ye]) => {
          ctx.beginPath();
          ctx.moveTo(x0, y0);
          ctx.lineTo(xm, ym);
          ctx.lineTo(xe, ye);
          ctx.stroke();
        });

        // Label tag
        const tag = `${det.label} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = "bold 11px Inter, sans-serif";
        const textW = ctx.measureText(tag).width;
        ctx.fillStyle = color + "cc";
        ctx.fillRect(rx - 1, ry - 22, textW + 10, 20);
        ctx.fillStyle = "#fff";
        ctx.fillText(tag, rx + 4, ry - 6);
      }
    };

    img.addEventListener("load", draw);
    draw();
    return () => img.removeEventListener("load", draw);
  }, [imageUrl, detections]);

  return (
    <div className="glass rounded-2xl overflow-hidden relative">
      <img
        ref={imgRef}
        src={imageUrl}
        alt="X-ray with detections"
        className="w-full max-h-80 object-contain"
        id="xray-overlay-image"
      />
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
        id="xray-overlay-canvas"
      />
    </div>
  );
}
