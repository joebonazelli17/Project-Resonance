"use client";

import { useCallback, useEffect, useMemo, useState, use } from "react";
import { useRouter } from "next/navigation";
import {
  compareSections,
  deleteTrack,
  formatTime,
  getReferenceCoach,
  getStreamUrl,
  getTrack,
  type ComparisonResult,
  type ReferenceCoachResponse,
  type TrackDetail,
  type TrackSection,
} from "@/lib/api";
import WaveformPlayer from "@/components/WaveformPlayer";

const BAND_NAMES = ["sub", "low", "low_mid", "mid", "high_mid", "presence", "brilliance", "air"];
const A_WEIGHT = [-34.6, -16.1, -3.5, 0, 1.2, 1.0, -1.1, -6.6];

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

function prettyLabel(label: string | null | undefined): string {
  if (!label) return "Unknown";
  return label.replaceAll("_", " ");
}

function prettyMatchBasis(matchBasis: string): string {
  if (matchBasis === "same_label") return "same label";
  if (matchBasis === "fallback_any_label") return "fallback match";
  return "any label";
}

function similarityPct(similarity: number): string {
  return `${(similarity * 100).toFixed(1)}%`;
}

function severityClass(severity: string): string {
  if (severity === "warning") return "border-amber-500/30 bg-amber-500/5";
  if (severity === "suggestion") return "border-resonance-500/30 bg-resonance-500/5";
  return "border-zinc-700 bg-zinc-800/40";
}

export default function TrackDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const router = useRouter();
  const [track, setTrack] = useState<TrackDetail | null>(null);
  const [streamUrl, setStreamUrl] = useState<string | null>(null);
  const [activeSection, setActiveSection] = useState<TrackSection | null>(null);
  const [seekCounter, setSeekCounter] = useState(0);
  const [barsFilter, setBarsFilter] = useState<number | null>(8);
  const [labelFilter, setLabelFilter] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [coach, setCoach] = useState<ReferenceCoachResponse | null>(null);
  const [coachLoading, setCoachLoading] = useState(false);
  const [coachError, setCoachError] = useState<string | null>(null);
  const [comparison, setComparison] = useState<ComparisonResult | null>(null);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [comparisonError, setComparisonError] = useState<string | null>(null);

  const coachBars = barsFilter || 8;

  const loadTrackPage = useCallback(async () => {
    const [nextTrack, nextStreamUrl] = await Promise.all([getTrack(id), getStreamUrl(id)]);
    return { nextTrack, nextStreamUrl };
  }, [id]);

  const loadCoach = useCallback(async (bars: number) => {
    setCoachLoading(true);
    setCoachError(null);
    try {
      setCoach(await getReferenceCoach(id, bars));
    } catch (e: unknown) {
      setCoachError(e instanceof Error ? e.message : "Failed to load reference coach");
    } finally {
      setCoachLoading(false);
    }
  }, [id]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const { nextTrack, nextStreamUrl } = await loadTrackPage();
        if (!cancelled) {
          setTrack(nextTrack);
          setStreamUrl(nextStreamUrl);
        }
      } catch (e: unknown) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Failed to load track");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    })();
    return () => { cancelled = true; };
  }, [loadTrackPage]);

  useEffect(() => {
    if (track?.status !== "ready") {
      setCoach(null);
      return;
    }
    void loadCoach(coachBars);
  }, [track?.status, coachBars, loadCoach]);

  useEffect(() => {
    if (!track) return;
    const needsTrackRefresh = track.status !== "ready";
    const needsCoachRefresh = track.status === "ready" && coach !== null && !coach.matching_ready;
    if (!needsTrackRefresh && !needsCoachRefresh) return;

    const timer = window.setTimeout(async () => {
      try {
        const { nextTrack, nextStreamUrl } = await loadTrackPage();
        setTrack(nextTrack);
        setStreamUrl(nextStreamUrl);
        if (nextTrack.status === "ready") {
          await loadCoach(coachBars);
        }
      } catch {
        return;
      }
    }, 5000);

    return () => window.clearTimeout(timer);
  }, [track, coach, coachBars, loadTrackPage, loadCoach]);

  const sections = useMemo(() => {
    if (!track) return [];
    return track.sections
      .filter((section) => {
        if (barsFilter && section.bars !== barsFilter) return false;
        if (labelFilter && section.section_label !== labelFilter) return false;
        return true;
      })
      .sort((a, b) => a.start_s - b.start_s || a.bars - b.bars);
  }, [track, barsFilter, labelFilter]);

  const barsValues = useMemo(() => {
    if (!track) return [];
    return [...new Set(track.sections.map((section) => section.bars))].sort((a, b) => a - b);
  }, [track]);

  const labelValues = useMemo(() => {
    if (!track) return [];
    return [...new Set(track.sections.map((section) => section.section_label).filter(Boolean))] as string[];
  }, [track]);

  const waveformSections = useMemo(() => {
    if (!track) return [];
    const waveformBars = barsFilter || Math.max(...barsValues, 8);
    return track.sections
      .filter((section) => section.bars === waveformBars && (!labelFilter || section.section_label === labelFilter))
      .sort((a, b) => a.start_s - b.start_s)
      .reduce<TrackSection[]>((kept, section) => {
        if (kept.length === 0 || section.start_s >= kept[kept.length - 1].end_s - 0.1) {
          kept.push(section);
        }
        return kept;
      }, []);
  }, [track, barsFilter, barsValues, labelFilter]);

  useEffect(() => {
    if (sections.length === 0) {
      setActiveSection(null);
      return;
    }
    if (!activeSection || !sections.some((section) => section.id === activeSection.id)) {
      setActiveSection(sections[0]);
    }
  }, [sections, activeSection]);

  const activeCoachSection = useMemo(() => {
    if (!coach || !activeSection) return null;
    return coach.sections.find((section) => section.query_section_id === activeSection.id) || null;
  }, [coach, activeSection]);
  const activeAnchorMatchId = activeCoachSection?.anchor_match?.section_id || null;

  useEffect(() => {
    if (!activeSection || !activeAnchorMatchId) {
      setComparison(null);
      setComparisonError(null);
      return;
    }

    let cancelled = false;
    setComparisonLoading(true);
    setComparisonError(null);
    void compareSections(activeSection.id, activeAnchorMatchId)
      .then((result) => {
        if (!cancelled) {
          setComparison(result);
        }
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setComparisonError(e instanceof Error ? e.message : "Failed to compare sections");
          setComparison(null);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setComparisonLoading(false);
        }
      });

    return () => { cancelled = true; };
  }, [activeSection, activeAnchorMatchId]);

  const handleDelete = async () => {
    if (!track || !confirm("Are you sure you want to delete this track?")) return;
    await deleteTrack(track.id);
    router.push("/");
  };

  if (loading) return <p className="text-zinc-500 py-12 text-center">Loading track...</p>;
  if (error) return <p className="text-red-400 py-12 text-center">{error}</p>;
  if (!track) return null;

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <button onClick={() => router.push("/")} className="text-sm text-zinc-500 hover:text-zinc-300 mb-2 block transition-colors">
            &larr; Back to Library
          </button>
          <h1 className="text-2xl font-bold tracking-tight">{track.original_filename}</h1>
          <div className="flex flex-wrap items-center gap-4 text-sm text-zinc-400 mt-1">
            {track.bpm && <span>{Math.round(track.bpm)} BPM</span>}
            {track.key && <span>{track.key} {track.scale}</span>}
            {track.beats_per_bar && <span>{track.beats_per_bar}/4</span>}
            {track.duration_s && <span>{Math.floor(track.duration_s / 60)}:{String(Math.floor(track.duration_s % 60)).padStart(2, "0")}</span>}
            {track.mastering_state && (
              <span className="px-2 py-0.5 rounded-full text-xs bg-zinc-800 text-zinc-300">
                {track.mastering_state}
              </span>
            )}
            <span className="text-zinc-600">{track.sections.length} sections</span>
          </div>
        </div>
        <button onClick={handleDelete} className="text-sm text-zinc-500 hover:text-red-400 transition-colors">
          Delete Track
        </button>
      </div>

      {track.status !== "ready" && (
        <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 px-4 py-3 text-sm text-amber-200">
          Analysis is still running. This page will refresh automatically while the track finishes processing.
        </div>
      )}

      {streamUrl && (
        <WaveformPlayer
          audioUrl={streamUrl}
          sections={waveformSections}
          activeSectionId={activeSection?.id || null}
          activeStartS={activeSection?.start_s ?? null}
          seekCounter={seekCounter}
          onSectionClick={(section) => {
            setActiveSection(section as unknown as TrackSection);
            setSeekCounter((count) => count + 1);
          }}
        />
      )}

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
        {barsValues.map((bars) => (
          <button
            key={bars}
            onClick={() => setBarsFilter(bars)}
            className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
              barsFilter === bars ? "bg-resonance-600 text-white" : "bg-zinc-800 text-zinc-400 hover:text-white"
            }`}
          >
            {bars} bars
          </button>
        ))}
      </div>

      {labelValues.length > 0 && (
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs text-zinc-500">Section type:</span>
          <button
            onClick={() => setLabelFilter(null)}
            className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
              !labelFilter ? "bg-resonance-600 text-white" : "bg-zinc-800 text-zinc-400 hover:text-white"
            }`}
          >
            All
          </button>
          {labelValues.map((label) => (
            <button
              key={label}
              onClick={() => setLabelFilter(label)}
              className={`px-3 py-1 rounded text-xs font-medium transition-colors capitalize ${
                labelFilter === label ? "bg-resonance-600 text-white" : "bg-zinc-800 text-zinc-400 hover:text-white"
              }`}
            >
              {prettyLabel(label)}
            </button>
          ))}
        </div>
      )}

      <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-5 space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold">Reference Coach</h2>
            <p className="text-sm text-zinc-400">
              Same-label {coachBars}-bar matching, anchor-track selection, and mastering-aware guidance.
            </p>
          </div>
          <span className="text-xs text-zinc-400 bg-zinc-800 px-2 py-1 rounded-full">
            {coachBars}-bar view
          </span>
        </div>

        {coachLoading && (
          <p className="text-sm text-zinc-400">Building reference coach...</p>
        )}

        {!coachLoading && coachError && (
          <p className="text-sm text-red-400">{coachError}</p>
        )}

        {!coachLoading && !coachError && track.status !== "ready" && (
          <p className="text-sm text-zinc-400">
            Reference matching will appear once analysis is ready.
          </p>
        )}

        {!coachLoading && !coachError && track.status === "ready" && coach && !coach.matching_ready && (
          <div className="rounded-lg border border-zinc-700 bg-zinc-800/40 px-4 py-3 text-sm text-zinc-300">
            Section analysis is ready. CLAP embeddings are still finishing, so reference matching will appear automatically in a moment.
            <span className="block text-xs text-zinc-500 mt-1">
              Pending sections: {coach.pending_embedding_sections} / {coach.total_sections}
            </span>
          </div>
        )}

        {!coachLoading && !coachError && track.status === "ready" && coach?.anchor_track && (
          <div className="grid gap-3 md:grid-cols-3">
            <div className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-4">
              <p className="text-xs uppercase tracking-wider text-zinc-500 mb-1">Anchor track</p>
              <p className="font-medium text-zinc-100">{coach.anchor_track.filename}</p>
              <p className="text-xs text-zinc-500 mt-1">
                {coach.anchor_track.mastering_state || "unknown"} mastering state
              </p>
            </div>
            <div className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-4">
              <p className="text-xs uppercase tracking-wider text-zinc-500 mb-1">Average similarity</p>
              <p className="font-mono text-xl text-zinc-100">{similarityPct(coach.anchor_track.avg_similarity)}</p>
            </div>
            <div className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-4">
              <p className="text-xs uppercase tracking-wider text-zinc-500 mb-1">Coverage</p>
              <p className="font-mono text-xl text-zinc-100">{Math.round(coach.anchor_track.coverage_ratio * 100)}%</p>
              <p className="text-xs text-zinc-500 mt-1">
                {coach.anchor_track.matched_sections} / {coach.total_sections} sections matched
              </p>
            </div>
          </div>
        )}

        {!coachLoading && !coachError && track.status === "ready" && coach && !coach.anchor_track && (
          <p className="text-sm text-zinc-400">
            No coherent anchor track found yet for this bar view.
          </p>
        )}

        {!coachLoading && !coachError && track.status === "ready" && coach && activeSection && (
          <div className="space-y-4 pt-2">
            <div className="flex flex-wrap items-center gap-2 text-sm">
              <span className="text-zinc-400">Selected section:</span>
              <span className="font-medium text-zinc-100">
                {formatTime(activeSection.start_s)} - {formatTime(activeSection.end_s)}
              </span>
              <span className="text-zinc-500">Bars {activeSection.bar_start}-{activeSection.bar_end}</span>
              {activeSection.section_label && (
                <span className="text-[10px] uppercase tracking-wider font-semibold px-1.5 py-0.5 rounded bg-resonance-600/20 text-resonance-400">
                  {prettyLabel(activeSection.section_label)}
                </span>
              )}
            </div>

            {!activeCoachSection && (
              <p className="text-sm text-zinc-400">
                Reference Coach is currently following {coachBars}-bar sections. Select a matching section or switch the bar filter.
              </p>
            )}

            {activeCoachSection && !activeCoachSection.anchor_match && (
              <p className="text-sm text-zinc-400">
                No anchor-section match was found for this section yet.
              </p>
            )}

            {activeCoachSection?.anchor_match && (
              <div className="grid gap-4 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,0.85fr)]">
                <div className="space-y-4">
                  <div className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-4 space-y-3">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="text-xs uppercase tracking-wider text-zinc-500 mb-1">Best anchor match</p>
                        <p className="font-medium text-zinc-100">{activeCoachSection.anchor_match.filename}</p>
                        <p className="text-xs text-zinc-500 mt-1">
                          {formatTime(activeCoachSection.anchor_match.start_s)} - {formatTime(activeCoachSection.anchor_match.end_s)}
                          {" · "}Bars {activeCoachSection.anchor_match.bar_start}-{activeCoachSection.anchor_match.bar_end}
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="font-mono text-lg text-zinc-100">{similarityPct(activeCoachSection.anchor_match.similarity)}</p>
                        <p className="text-xs text-zinc-500">{prettyMatchBasis(activeCoachSection.anchor_match.match_basis)}</p>
                      </div>
                    </div>

                    <div className="flex flex-wrap items-center gap-2 text-xs text-zinc-400">
                      {activeCoachSection.anchor_match.section_label && (
                        <span className="px-2 py-1 rounded bg-zinc-800 text-zinc-300">
                          {prettyLabel(activeCoachSection.anchor_match.section_label)}
                        </span>
                      )}
                      {activeCoachSection.query_section_label_confidence !== null && (
                        <span className="px-2 py-1 rounded bg-zinc-800 text-zinc-300">
                          label confidence {Math.round(activeCoachSection.query_section_label_confidence * 100)}%
                        </span>
                      )}
                    </div>
                  </div>

                  {activeCoachSection.alternate_matches.length > 0 && (
                    <div className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-4">
                      <p className="text-xs uppercase tracking-wider text-zinc-500 mb-3">Alternate references</p>
                      <div className="space-y-2">
                        {activeCoachSection.alternate_matches.map((match) => (
                          <div key={match.section_id} className="flex items-center justify-between gap-3 text-sm">
                            <div className="min-w-0">
                              <p className="truncate text-zinc-200">{match.filename}</p>
                              <p className="text-xs text-zinc-500">
                                {formatTime(match.start_s)} - {formatTime(match.end_s)}
                                {match.section_label && ` · ${prettyLabel(match.section_label)}`}
                              </p>
                            </div>
                            <span className="font-mono text-zinc-300">{similarityPct(match.similarity)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                <div className="space-y-4">
                  <div className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-4">
                    <p className="text-xs uppercase tracking-wider text-zinc-500 mb-3">Component deltas</p>
                    {comparison?.stem_balance_delta ? (
                      <div className="grid grid-cols-2 gap-2">
                        {Object.entries(comparison.stem_balance_delta).map(([stem, delta]) => (
                          <div key={stem} className="rounded border border-zinc-800 bg-zinc-900/70 px-3 py-2">
                            <p className="text-xs text-zinc-500 capitalize">{stem}</p>
                            <p className={`font-mono text-sm ${delta > 0 ? "text-amber-300" : delta < 0 ? "text-sky-300" : "text-zinc-300"}`}>
                              {delta > 0 ? "+" : ""}{delta.toFixed(1)} dB
                            </p>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-sm text-zinc-400">Component comparison will appear after the section pair is evaluated.</p>
                    )}
                  </div>

                  <div className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-4">
                    <p className="text-xs uppercase tracking-wider text-zinc-500 mb-3">Guidance</p>
                    {comparisonLoading && <p className="text-sm text-zinc-400">Comparing this section pair...</p>}
                    {!comparisonLoading && comparisonError && <p className="text-sm text-red-400">{comparisonError}</p>}
                    {!comparisonLoading && !comparisonError && comparison && comparison.recommendations.length === 0 && (
                      <p className="text-sm text-zinc-400">
                        This section is already broadly aligned with the anchor reference.
                      </p>
                    )}
                    {!comparisonLoading && !comparisonError && comparison && comparison.recommendations.length > 0 && (
                      <div className="space-y-2">
                        {comparison.recommendations.map((recommendation, index) => (
                          <div key={`${recommendation.type}-${index}`} className={`rounded-lg border px-3 py-2 text-sm ${severityClass(recommendation.severity)}`}>
                            <p className="text-zinc-100">{recommendation.message}</p>
                            <div className="flex flex-wrap items-center gap-2 mt-2 text-[10px] uppercase tracking-wider text-zinc-500">
                              <span>{recommendation.type}</span>
                              {recommendation.band && <span>{recommendation.band}</span>}
                              {recommendation.stem && <span>{recommendation.stem}</span>}
                              <span>{recommendation.severity}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {sections.map((section) => (
          <div
            key={section.id}
            onClick={() => {
              setActiveSection(section);
              setSeekCounter((count) => count + 1);
            }}
            className={`bg-zinc-900/50 border rounded-xl p-4 cursor-pointer transition-all ${
              activeSection?.id === section.id
                ? "border-resonance-500 bg-resonance-500/5"
                : "border-zinc-800 hover:border-zinc-600"
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">
                  {formatTime(section.start_s)} - {formatTime(section.end_s)}
                </span>
                {section.section_label && (
                  <span className="text-[10px] uppercase tracking-wider font-semibold px-1.5 py-0.5 rounded bg-resonance-600/20 text-resonance-400">
                    {prettyLabel(section.section_label)}
                  </span>
                )}
              </div>
              <span className="text-xs text-zinc-500 bg-zinc-800 px-2 py-0.5 rounded">
                {section.bars} bars
              </span>
            </div>
            <div className="text-xs text-zinc-500 mb-3">
              Bars {section.bar_start}-{section.bar_end}
              {section.key && <span className="ml-2">{section.key} {section.scale}</span>}
              {section.stereo_features && (
                <span className="ml-2">Stereo: {Math.round((1 - section.stereo_features.mid_side_ratio) * 100)}% width</span>
              )}
            </div>
            <div className="space-y-2">
              <FeatureBar label="HF Perc" value={section.hf_perc_ratio} min={0} max={0.15} />
              <FeatureBar label="RMS" value={section.rms_dbfs} min={-40} max={0} unit=" dB" />
              <FeatureBar label="Crest" value={section.crest_db} min={0} max={20} unit=" dB" />
              <FeatureBar label="Flatness" value={section.flatness} min={0} max={0.5} />
            </div>
            {section.band_energies && (
              <div className="mt-3 pt-3 border-t border-zinc-800">
                <p className="text-[10px] uppercase tracking-wider text-zinc-600 mb-1.5">Spectral Profile</p>
                <div className="flex gap-0.5 items-end h-10">
                  {(() => {
                    const raw = BAND_NAMES.map((band) => section.band_energies?.[band] ?? -96);
                    const weighted = raw.map((value, index) => value + A_WEIGHT[index]);
                    const maxVal = Math.max(...weighted);
                    return BAND_NAMES.map((band, index) => {
                      const relative = weighted[index] - maxVal;
                      const pct = Math.max(0, Math.min(100, ((relative + 30) / 30) * 100));
                      return (
                        <div key={band} className="flex-1 flex flex-col items-center gap-0.5">
                          <div className="w-full bg-zinc-800 rounded-sm overflow-hidden" style={{ height: "32px" }}>
                            <div className="w-full bg-resonance-500/60 rounded-sm" style={{ height: `${pct}%`, marginTop: `${100 - pct}%` }} />
                          </div>
                          <span className="text-[8px] text-zinc-600">{band.replace("_", "")}</span>
                        </div>
                      );
                    });
                  })()}
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