"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { formatTime } from "@/lib/api";

export interface WaveformSection {
  id: string;
  start_s: number;
  end_s: number;
  bars: number;
  bar_start: number;
  bar_end: number;
}

interface WaveformPlayerProps {
  audioUrl: string;
  sections?: WaveformSection[];
  activeSectionId?: string | null;
  onSectionClick?: (section: WaveformSection) => void;
  highlightRanges?: Array<{ start: number; end: number; color?: string }>;
  height?: number;
  barWidth?: number;
}

import type WaveSurfer from "wavesurfer.js";
import type RegionsPlugin from "wavesurfer.js/dist/plugins/regions.esm.js";

type RegionHandle = { id: string; play: () => void };

const SECTION_COLORS = [
  "rgba(92, 124, 250, 0.15)",
  "rgba(16, 185, 129, 0.15)",
  "rgba(245, 158, 11, 0.15)",
  "rgba(239, 68, 68, 0.15)",
  "rgba(168, 85, 247, 0.15)",
];

export default function WaveformPlayer({
  audioUrl,
  sections = [],
  activeSectionId,
  onSectionClick,
  highlightRanges = [],
  height = 80,
  barWidth = 2,
}: WaveformPlayerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WaveSurfer | null>(null);
  const regionsRef = useRef<RegionsPlugin | null>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [ready, setReady] = useState(false);

  const sectionsRef = useRef(sections);
  sectionsRef.current = sections;
  const onSectionClickRef = useRef(onSectionClick);
  onSectionClickRef.current = onSectionClick;

  useEffect(() => {
    if (!containerRef.current) return;

    let cancelled = false;

    const init = async () => {
      const WaveSurfer = (await import("wavesurfer.js")).default;
      const RegionsPlugin = (await import("wavesurfer.js/dist/plugins/regions.esm.js")).default;

      if (cancelled) return;

      const regions = RegionsPlugin.create();
      regionsRef.current = regions;

      const ws = WaveSurfer.create({
        container: containerRef.current!,
        height,
        barWidth,
        barGap: 1,
        barRadius: 2,
        cursorColor: "#5c7cfa",
        cursorWidth: 2,
        waveColor: "#3f3f46",
        progressColor: "#5c7cfa",
        normalize: true,
        plugins: [regions],
      });

      ws.on("ready", () => {
        if (cancelled) return;
        setDuration(ws.getDuration());
        setReady(true);

        sectionsRef.current.forEach((sec) => {
          const color = SECTION_COLORS[sec.bars % SECTION_COLORS.length] || SECTION_COLORS[0];
          regions.addRegion({
            id: sec.id,
            start: sec.start_s,
            end: sec.end_s,
            color: activeSectionId === sec.id ? "rgba(92, 124, 250, 0.35)" : color,
            drag: false,
            resize: false,
            content: `${sec.bars}b`,
          });
        });

        highlightRanges.forEach((r, i) => {
          regions.addRegion({
            id: `hl-${i}`,
            start: r.start,
            end: r.end,
            color: r.color || "rgba(16, 185, 129, 0.3)",
            drag: false,
            resize: false,
          });
        });
      });

      ws.on("timeupdate", (t: number) => setCurrentTime(t));
      ws.on("play", () => setPlaying(true));
      ws.on("pause", () => setPlaying(false));
      ws.on("finish", () => setPlaying(false));

      regions.on("region-clicked", (region: RegionHandle, e: MouseEvent) => {
        e.stopPropagation();
        if (onSectionClickRef.current) {
          const sec = sectionsRef.current.find((s) => s.id === region.id);
          if (sec) onSectionClickRef.current(sec);
        }
        region.play();
      });

      ws.load(audioUrl);
      wsRef.current = ws;
    };

    init();

    return () => {
      cancelled = true;
      if (wsRef.current) {
        wsRef.current.destroy();
        wsRef.current = null;
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [audioUrl]);

  const togglePlay = useCallback(() => {
    if (wsRef.current) wsRef.current.playPause();
  }, []);

  return (
    <div className="space-y-2">
      <div
        ref={containerRef}
        className="rounded-lg overflow-hidden bg-zinc-900/50 border border-zinc-800"
      />
      <div className="flex items-center gap-3">
        <button
          onClick={togglePlay}
          disabled={!ready}
          className="w-9 h-9 flex items-center justify-center rounded-full bg-resonance-600 hover:bg-resonance-700 disabled:opacity-40 transition-colors"
        >
          {playing ? (
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
              <rect x="6" y="4" width="4" height="16" />
              <rect x="14" y="4" width="4" height="16" />
            </svg>
          ) : (
            <svg className="w-4 h-4 ml-0.5" fill="currentColor" viewBox="0 0 24 24">
              <polygon points="5,3 19,12 5,21" />
            </svg>
          )}
        </button>
        <span className="text-xs text-zinc-400 font-mono tabular-nums">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
      </div>
    </div>
  );
}