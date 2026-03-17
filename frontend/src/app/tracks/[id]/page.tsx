"use client";

import { useEffect, useState, use } from "react";
import { useRouter } from "next/navigation";
import { getTrack, getStreamUrl, deleteTrack, formatTime, type TrackDetail, type TrackSection } from "@/lib/api";
import WaveformPlayer from "@/components/WaveformPlayer";

function FeatureBar({ label, value, min, max, unit }: { label: string; value: number; min: number; max: number; unit?: string }) {
  const pct = Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-zinc-500">{label}</span>
        <span className="text-zinc-300 font-mono">{value.toFixed(2)}{unit || ""}</span>
      </div>
      <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
        <div className="h-full bg-resonance-500 rounded-full transition-all" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function TrackDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const router = useRouter();
  const [track, setTrack] = useState<TrackDetail | null>(null);
  const [streamUrl, setStreamUrl] = useState<string | null>(null);
  const [activeSection, setActiveSection] = useState<TrackSection | null>(null);
  const [barsFilter, setBarsFilter] = useState<number | null>(8);
  const [labelFilter, setLabelFilter] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const [t, url] = await Promise.all([getTrack(id), getStreamUrl(id)]);
        setTrack(t);
        setStreamUrl(url);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : "Failed to load track");
      } finally {
        setLoading(false);
      }
    })();
  }, [id]);

  if (loading) return <p className="text-zinc-500 py-12 text-center">Loading track...</p>;
  if (error) return <p className="text-red-400 py-12 text-center">{error}</p>;
  if (!track) return null;

  const sections = track.sections.filter((s) => {
    if (barsFilter && s.bars !== barsFilter) return false;
    if (labelFilter && s.section_label !== labelFilter) return false;
    return true;
  });
  const barsValues = [...new Set(track.sections.map((s) => s.bars))].sort((a, b) => a - b);
  const labelValues = [...new Set(track.sections.map((s) => s.section_label).filter(Boolean))] as string[];

  const handleDelete = async () => {
    await deleteTrack(track.id);
    router.push("/");
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <button onClick={() => router.push("/")} className="text-sm text-zinc-500 hover:text-zinc-300 mb-2 block transition-colors">
            &larr; Back to Library
          </button>
          <h1 className="text-2xl font-bold tracking-tight">{track.original_filename}</h1>
          <div className="flex items-center gap-4 text-sm text-zinc-400 mt-1">
            {track.bpm && <span>{Math.round(track.bpm)} BPM</span>}
            {track.key && <span>{track.key} {track.scale}</span>}
            {track.beats_per_bar && <span>{track.beats_per_bar}/4</span>}
            {track.duration_s && <span>{Math.floor(track.duration_s / 60)}:{String(Math.floor(track.duration_s % 60)).padStart(2, "0")}</span>}
            <span className="text-zinc-600">{track.sections.length} sections</span>
          </div>
        </div>
        <button onClick={handleDelete} className="text-sm text-zinc-500 hover:text-red-400 transition-colors">
          Delete Track
        </button>
      </div>

      {/* Waveform */}
      {streamUrl && (
        <WaveformPlayer
          audioUrl={streamUrl}
          sections={sections.filter((s, i) => {
            // Only show non-overlapping regions on the waveform
            if (i === 0) return true;
            const prev = sections[i - 1];
            return s.start_s >= prev.end_s - 0.1;
          })}
          activeSectionId={activeSection?.id || null}
          onSectionClick={(sec) => setActiveSection(sec as unknown as TrackSection)}
        />
      )}

      {/* Bars filter */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-zinc-500">Filter by bars:</span>
        <button
          onClick={() => setBarsFilter(null)}
          className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
            !barsFilter ? "bg-resonance-600 text-white" : "bg-zinc-800 text-zinc-400 hover:text-white"
          }`}
        >
          All
        </button>
        {barsValues.map((b) => (
          <button
            key={b}
            onClick={() => setBarsFilter(b)}
            className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
              barsFilter === b ? "bg-resonance-600 text-white" : "bg-zinc-800 text-zinc-400 hover:text-white"
            }`}
          >
            {b} bars
          </button>
        ))}
      </div>

      {/* Section label filter */}
      {labelValues.length > 0 && (
        <div className="flex items-center gap-2">
          <span className="text-xs text-zinc-500">Section type:</span>
          <button
            onClick={() => setLabelFilter(null)}
            className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
              !labelFilter ? "bg-resonance-600 text-white" : "bg-zinc-800 text-zinc-400 hover:text-white"
            }`}
          >
            All
          </button>
          {labelValues.map((l) => (
            <button
              key={l}
              onClick={() => setLabelFilter(l)}
              className={`px-3 py-1 rounded text-xs font-medium transition-colors capitalize ${
                labelFilter === l ? "bg-resonance-600 text-white" : "bg-zinc-800 text-zinc-400 hover:text-white"
              }`}
            >
              {l}
            </button>
          ))}
        </div>
      )}

      {/* Sections grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {sections.map((sec) => (
          <div
            key={sec.id}
            onClick={() => setActiveSection(sec)}
            className={`bg-zinc-900/50 border rounded-xl p-4 cursor-pointer transition-all ${
              activeSection?.id === sec.id
                ? "border-resonance-500 bg-resonance-500/5"
                : "border-zinc-800 hover:border-zinc-600"
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">
                  {formatTime(sec.start_s)} - {formatTime(sec.end_s)}
                </span>
                {sec.section_label && (
                  <span className="text-[10px] uppercase tracking-wider font-semibold px-1.5 py-0.5 rounded bg-resonance-600/20 text-resonance-400">
                    {sec.section_label}
                  </span>
                )}
              </div>
              <span className="text-xs text-zinc-500 bg-zinc-800 px-2 py-0.5 rounded">
                {sec.bars} bars
              </span>
            </div>
            <div className="text-xs text-zinc-500 mb-3">
              Bars {sec.bar_start}-{sec.bar_end}
              {sec.key && <span className="ml-2">{sec.key} {sec.scale}</span>}
              {sec.stereo_features && (
                <span className="ml-2">Stereo: {Math.round((1 - sec.stereo_features.mid_side_ratio) * 100)}% width</span>
              )}
            </div>
            <div className="space-y-2">
              <FeatureBar label="HF Perc" value={sec.hf_perc_ratio} min={0} max={0.15} />
              <FeatureBar label="RMS" value={sec.rms_dbfs} min={-40} max={0} unit=" dB" />
              <FeatureBar label="Crest" value={sec.crest_db} min={0} max={20} unit=" dB" />
              <FeatureBar label="Flatness" value={sec.flatness} min={0} max={0.5} />
            </div>
            {sec.band_energies && (
              <div className="mt-3 pt-3 border-t border-zinc-800">
                <p className="text-[10px] uppercase tracking-wider text-zinc-600 mb-1.5">Spectral Profile</p>
                <div className="flex gap-0.5 items-end h-10">
                  {["sub", "low", "low_mid", "mid", "high_mid", "presence", "brilliance", "air"].map((band) => {
                    const val = sec.band_energies?.[band] ?? -96;
                    const pct = Math.max(0, Math.min(100, ((val + 60) / 60) * 100));
                    return (
                      <div key={band} className="flex-1 flex flex-col items-center gap-0.5">
                        <div className="w-full bg-zinc-800 rounded-sm overflow-hidden" style={{ height: "32px" }}>
                          <div className="w-full bg-resonance-500/60 rounded-sm" style={{ height: `${pct}%`, marginTop: `${100 - pct}%` }} />
                        </div>
                        <span className="text-[8px] text-zinc-600">{band.replace("_", "")}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {sections.length === 0 && (
        <p className="text-center text-zinc-500 py-8">No sections found for this filter.</p>
      )}
    </div>
  );
}