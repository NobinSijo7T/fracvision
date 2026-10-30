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
  /** Dimensions of the square image passed through the model pipeline. */
  processedImageHw?: number[];
  segmentation?: {
    mask_available: boolean;
    mask_data_url?: string | null;
  } | null;
}

const LABEL_COLORS: Record<string, string> = {
  fracture: "#b91c1c",           // deep clinical crimson
  pronatorsign: "#c2410c",       // rust orange
  periostealreaction: "#b45309", // deep amber
  text: "#0f3e17",               // forest ink
};

function getColor(label: string): string {
  return LABEL_COLORS[label] ?? "#0f3e17";
}

export function XrayOverlay({ imageUrl, detections, processedImageHw, segmentation }: XrayOverlayProps) {
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const detectionKey = detections
    .map((d) => `${d.label}:${d.confidence}:${d.bbox_xyxy.join(",")}`)
    .join("|");
  const maskAvailable = Boolean(segmentation?.mask_available);
  const maskDataUrl = segmentation?.mask_data_url ?? null;
  const processedKey = processedImageHw?.join("x") ?? "";

  useEffect(() => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas || !img.complete) return;

    const draw = () => {
      const { naturalWidth: nw, naturalHeight: nh, clientWidth: cw, clientHeight: ch } = img;
      if (!nw || !nh || !cw || !ch) return;
      canvas.width = cw;
      canvas.height = ch;

      // The image element uses object-contain. Its box can be much wider than
      // the visible portrait X-ray, so map into the actual rendered image area.
      const displayScale = Math.min(cw / nw, ch / nh);
      const displayWidth = nw * displayScale;
      const displayHeight = nh * displayScale;
      const displayOffsetX = (cw - displayWidth) / 2;
      const displayOffsetY = (ch - displayHeight) / 2;

      // Detection and MedSAM coordinates come from the square, padded model
      // input. Remove that preprocessing padding before drawing on the source.
      const modelWidth = processedImageHw?.[1] || nw;
      const modelHeight = processedImageHw?.[0] || nh;
      const modelScale = Math.min(modelWidth / nw, modelHeight / nh);
      const contentWidth = nw * modelScale;
      const contentHeight = nh * modelScale;
      const padLeft = (modelWidth - contentWidth) / 2;
      const padTop = (modelHeight - contentHeight) / 2;

      const ctx = canvas.getContext("2d")!;
      ctx.clearRect(0, 0, cw, ch);

      // MedSAM transparent PNG mask sized to the processed X-ray
      if (segmentation?.mask_available && segmentation.mask_data_url) {
        const maskImage = new Image();
        maskImage.onload = () => {
          ctx.globalAlpha = 0.85;
          ctx.drawImage(
            maskImage,
            padLeft,
            padTop,
            contentWidth,
            contentHeight,
            displayOffsetX,
            displayOffsetY,
            displayWidth,
            displayHeight,
          );
          ctx.globalAlpha = 1;
          drawDetections(ctx, nw, nh, padLeft, padTop, modelScale, displayScale, displayOffsetX, displayOffsetY);
        };
        maskImage.src = segmentation.mask_data_url;
        return;
      }

      drawDetections(ctx, nw, nh, padLeft, padTop, modelScale, displayScale, displayOffsetX, displayOffsetY);
    };

    const drawDetections = (
      ctx: CanvasRenderingContext2D,
      sourceWidth: number,
      sourceHeight: number,
      padLeft: number,
      padTop: number,
      modelScale: number,
      displayScale: number,
      displayOffsetX: number,
      displayOffsetY: number,
    ) => {
      for (const det of detections) {
        const [x1, y1, x2, y2] = det.bbox_xyxy;
        if (![x1, y1, x2, y2].every(Number.isFinite)) continue;

        // Convert model-square pixels -> original-image pixels -> canvas pixels.
        const sourceX1 = (x1 - padLeft) / modelScale;
        const sourceY1 = (y1 - padTop) / modelScale;
        const sourceX2 = (x2 - padLeft) / modelScale;
        const sourceY2 = (y2 - padTop) / modelScale;
        if (sourceX2 <= 0 || sourceY2 <= 0 || sourceX1 >= sourceWidth || sourceY1 >= sourceHeight) continue;
        const clippedX1 = Math.max(0, sourceX1);
        const clippedY1 = Math.max(0, sourceY1);
        const clippedX2 = Math.min(sourceWidth, sourceX2);
        const clippedY2 = Math.min(sourceHeight, sourceY2);
        const color = getColor(det.label);
        const rx = displayOffsetX + clippedX1 * displayScale;
        const ry = displayOffsetY + clippedY1 * displayScale;
        const rw = (clippedX2 - clippedX1) * displayScale;
        const rh = (clippedY2 - clippedY1) * displayScale;

        // Clean hairline box without blurry neon glow
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(rx, ry, rw, rh);

        // Corner tick marks
        const cs = 10;
        ctx.lineWidth = 3;
        [
          [rx, ry, rx + cs, ry, rx, ry + cs],
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

        // Crisp botanical paper tag
        const tag = `${det.label} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = "500 11px Inter, sans-serif";
        const textW = ctx.measureText(tag).width;
        ctx.fillStyle = "#fffefc";
        ctx.fillRect(rx, ry - 22, textW + 12, 20);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.strokeRect(rx, ry - 22, textW + 12, 20);
        ctx.fillStyle = color;
        ctx.fillText(tag, rx + 6, ry - 7);
      }
    };

    img.addEventListener("load", draw);
    draw();
    return () => img.removeEventListener("load", draw);
  }, [imageUrl, detectionKey, maskAvailable, maskDataUrl, processedKey]);

  return (
    <div className="bg-[#fffefc] border border-[#efeeeb] rounded-[14px] overflow-hidden relative p-3">
      <img
        ref={imgRef}
        src={imageUrl}
        alt="X-ray with detections"
        className="w-full max-h-88 object-contain rounded-[10px] mx-auto bg-black/5"
        id="xray-overlay-image"
      />
      <canvas
        ref={canvasRef}
        className="absolute inset-3 w-[calc(100%-24px)] h-[calc(100%-24px)] pointer-events-none"
        id="xray-overlay-canvas"
      />
    </div>
  );
}
