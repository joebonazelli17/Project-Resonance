"use client";

import { useState } from "react";
import {
  searchSimilar,
  searchByText,
  searchByStems,
  formatTime,
  type SearchResponse,
  type SearchWindowResult,
  type TextSearchResponse,
  type TextSearchMatch,
  type StemWeights,
  type StemSearchResponse,
} from "@/lib/api";

type SearchMode = "text" | "stems" | "audio";

const DEFAULT_WEIGHTS: StemWeights = { mix: 0.2, drums: 0.3, bass: 0.2, vocals: 0.2, other: 0.1 };

function StemSlider({ label, value, onChange }: { label: string; value: number; onChange: (v: number) => void }) {
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-zinc-400 w-14">{label}</span>
      <input
        type="range"
        min={0}
        max={100}
        value={Math.round(value * 100)}
        onChange={(e) => onChange(Number(e.target.value) / 100)}
        className="flex-1 h-1.5 bg-zinc-700 rounded-full appearance-none cursor-pointer accent-resonance-500"
      />
      <span className="text-xs text-zinc-500 font-mono w-10 text-right">{Math.round(value * 100)}%</span>
    </div>
  );
}

function MatchList({ matches }: { matches: TextSearchMatch[] }) {
  if (matches.length === 0) {
    return <p className="text-center text-zinc-500 py-8">No matches found.</p>;
  }
  return (
    <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl overflow-hidden">
      <div className="divide-y divide-zinc-800/50">
        {matches.map((m, i) => (
          <div key={i} className="px-5 py-3 flex items-center justify-between hover:bg-zinc-800/30 transition-colors">
            <div className="min-w-0">
              <p className="font-medium text-sm truncate">{m.filename}</p>
              <div className="flex items-center gap-3 text-xs text-zinc-500 mt-0.5">
                <span>{formatTime(m.start_s)} - {formatTime(m.end_s)}</span>
                <span>{m.bars} bars</span>
                <span>Bars {m.bar_start}-{m.bar_end}</span>
                {m.bpm && <span>{Math.round(m.bpm)} BPM</span>}
                {m.key && <span>{m.key} {m.scale}</span>}
              </div>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-20 h-2 bg-zinc-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-resonance-500 rounded-full"
                  style={{ width: `${Math.max(0, Math.min(100, m.similarity * 100))}%` }}
                />
              </div>
              <span className="text-xs text-zinc-400 w-12 text-right">
                {(m.similarity * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function SearchPage() {
  const [mode, setMode] = useState<SearchMode>("text");
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Text search
  const [textQuery, setTextQuery] = useState("");
  const [textBars, setTextBars] = useState<number | undefined>(undefined);
  const [textResults, setTextResults] = useState<TextSearchResponse | null>(null);

  // Stem search
  const [stemQuery, setStemQuery] = useState("");
  const [stemBars, setStemBars] = useState<number | undefined>(undefined);
  const [stemWeights, setStemWeights] = useState<StemWeights>({ ...DEFAULT_WEIGHTS });
  const [stemResults, setStemResults] = useState<StemSearchResponse | null>(null);

  // Audio search
  const [file, setFile] = useState<File | null>(null);
  const [bars, setBars] = useState(4);
  const [audioResults, setAudioResults] = useState<SearchResponse | null>(null);

  const handleTextSearch = async () => {
    if (!textQuery.trim()) return;
    setSearching(true); setError(null);
    try { setTextResults(await searchByText(textQuery.trim(), textBars)); }
    catch (e: unknown) { setError(e instanceof Error ? e.message : "Search failed"); }
    finally { setSearching(false); }
  };

  const handleStemSearch = async () => {
    if (!stemQuery.trim()) return;
    setSearching(true); setError(null);
    try { setStemResults(await searchByStems(stemQuery.trim(), stemWeights, stemBars)); }
    catch (e: unknown) { setError(e instanceof Error ? e.message : "Search failed"); }
    finally { setSearching(false); }
  };

  const handleAudioSearch = async () => {
    if (!file) return;
    setSearching(true); setError(null);
    try { setAudioResults(await searchSimilar(file, bars)); }
    catch (e: unknown) { setError(e instanceof Error ? e.message : "Search failed"); }
    finally { setSearching(false); }
  };

  const updateWeight = (key: keyof StemWeights, val: number) => {
    setStemWeights((prev) => ({ ...prev, [key]: val }));
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Reference Search</h1>
        <p className="text-zinc-400 mt-1">Find similar sections by text, stems, or audio</p>
      </div>

      {/* Mode tabs */}
      <div className="flex gap-1 bg-zinc-900 rounded-lg p-1 w-fit">
        {(["text", "stems", "audio"] as SearchMode[]).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors capitalize ${
              mode === m ? "bg-resonance-600 text-white" : "text-zinc-400 hover:text-white"
            }`}
          >
            {m === "stems" ? "Stem Search" : m === "text" ? "Text Search" : "Audio Search"}
          </button>
        ))}
      </div>

      {/* Text search panel */}
      {mode === "text" && (
        <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-6 space-y-4">
          <label className="text-sm text-zinc-400">Describe what you are looking for</label>
          <input
            type="text"
            value={textQuery}
            onChange={(e) => setTextQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleTextSearch()}
            placeholder={"e.g. \"dark minimal techno kick\", \"bright pluck synth with reverb\""}
            className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-3 text-sm placeholder-zinc-600 focus:border-resonance-500 focus:outline-none transition-colors"
          />
          <div className="flex items-center gap-3">
            <label className="text-sm text-zinc-400">Bars:</label>
            <select value={textBars || ""} onChange={(e) => setTextBars(e.target.value ? Number(e.target.value) : undefined)} className="bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm">
              <option value="">All</option>
              {[2, 4, 8, 16].map((b) => <option key={b} value={b}>{b} bars</option>)}
            </select>
            <button onClick={handleTextSearch} disabled={!textQuery.trim() || searching} className="px-5 py-2 bg-resonance-600 hover:bg-resonance-700 disabled:opacity-50 rounded-lg text-sm font-medium transition-colors">
              {searching ? "Searching..." : "Search"}
            </button>
          </div>
          {error && <p className="text-red-400 text-sm">{error}</p>}
        </div>
      )}

      {/* Stem search panel */}
      {mode === "stems" && (
        <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-6 space-y-5">
          <div>
            <label className="text-sm text-zinc-400">Describe the sound</label>
            <input
              type="text"
              value={stemQuery}
              onChange={(e) => setStemQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleStemSearch()}
              placeholder={"e.g. \"punchy house groove with tight hats\""}
              className="w-full mt-2 bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-3 text-sm placeholder-zinc-600 focus:border-resonance-500 focus:outline-none transition-colors"
            />
          </div>
          <div>
            <label className="text-sm text-zinc-400 mb-3 block">Stem weights -- control what matters most</label>
            <div className="space-y-2">
              <StemSlider label="Mix" value={stemWeights.mix} onChange={(v) => updateWeight("mix", v)} />
              <StemSlider label="Drums" value={stemWeights.drums} onChange={(v) => updateWeight("drums", v)} />
              <StemSlider label="Bass" value={stemWeights.bass} onChange={(v) => updateWeight("bass", v)} />
              <StemSlider label="Vocals" value={stemWeights.vocals} onChange={(v) => updateWeight("vocals", v)} />
              <StemSlider label="Other" value={stemWeights.other} onChange={(v) => updateWeight("other", v)} />
            </div>
          </div>
          <div className="flex items-center gap-3">
            <label className="text-sm text-zinc-400">Bars:</label>
            <select value={stemBars || ""} onChange={(e) => setStemBars(e.target.value ? Number(e.target.value) : undefined)} className="bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm">
              <option value="">All</option>
              {[2, 4, 8, 16].map((b) => <option key={b} value={b}>{b} bars</option>)}
            </select>
            <button onClick={handleStemSearch} disabled={!stemQuery.trim() || searching} className="px-5 py-2 bg-resonance-600 hover:bg-resonance-700 disabled:opacity-50 rounded-lg text-sm font-medium transition-colors">
              {searching ? "Searching..." : "Search"}
            </button>
          </div>
          {error && <p className="text-red-400 text-sm">{error}</p>}
        </div>
      )}

      {/* Audio search panel */}
      {mode === "audio" && (
        <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-6 space-y-4">
          <div className="flex flex-col sm:flex-row gap-4">
            <label className="flex-1 border-2 border-dashed border-zinc-700 hover:border-resonance-500 rounded-lg p-4 text-center cursor-pointer transition-colors">
              <p className="text-sm text-zinc-300">{file ? file.name : "Choose a query track"}</p>
              <input type="file" className="hidden" accept=".wav,.mp3,.flac,.m4a,.aiff,.aif" onChange={(e) => setFile(e.target.files?.[0] || null)} />
            </label>
            <div className="flex items-center gap-3">
              <label className="text-sm text-zinc-400">Bars:</label>
              <select value={bars} onChange={(e) => setBars(Number(e.target.value))} className="bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm">
                {[2, 4, 8, 16].map((b) => <option key={b} value={b}>{b} bars</option>)}
              </select>
              <button onClick={handleAudioSearch} disabled={!file || searching} className="px-5 py-2 bg-resonance-600 hover:bg-resonance-700 disabled:opacity-50 rounded-lg text-sm font-medium transition-colors">
                {searching ? "Searching..." : "Search"}
              </button>
            </div>
          </div>
          {error && <p className="text-red-400 text-sm">{error}</p>}
        </div>
      )}

      {/* Results */}
      {mode === "text" && textResults && (
        <div className="space-y-4">
          <p className="text-sm text-zinc-400">{textResults.matches.length} matches for &ldquo;{textResults.query}&rdquo;</p>
          <MatchList matches={textResults.matches} />
        </div>
      )}

      {mode === "stems" && stemResults && (
        <div className="space-y-4">
          <div className="flex items-center gap-4 text-sm text-zinc-400">
            <span>{stemResults.matches.length} matches for &ldquo;{stemResults.query}&rdquo;</span>
            <span className="text-xs text-zinc-600">
              weights: {Object.entries(stemResults.weights).map(([k, v]) => `${k}=${Math.round(Number(v) * 100)}%`).join(", ")}
            </span>
          </div>
          <MatchList matches={stemResults.matches} />
        </div>
      )}

      {mode === "audio" && audioResults && (
        <div className="space-y-4">
          <div className="flex items-center gap-4 text-sm text-zinc-400">
            <span>{audioResults.results.length} windows analyzed</span>
            {audioResults.query_bpm && <span>{Math.round(audioResults.query_bpm)} BPM</span>}
            {audioResults.query_key && <span>{audioResults.query_key}</span>}
          </div>
          {audioResults.results.map((win: SearchWindowResult, wi: number) => (
            <div key={wi} className="bg-zinc-900/50 border border-zinc-800 rounded-xl overflow-hidden">
              <div className="px-5 py-3 bg-zinc-800/50 border-b border-zinc-800 flex items-center gap-4">
                <span className="text-sm font-medium">Window {wi + 1}: {formatTime(win.query_start_s)} - {formatTime(win.query_end_s)}</span>
                <span className="text-xs text-zinc-500">Bars {win.query_bar_start}-{win.query_bar_end}</span>
              </div>
              <div className="divide-y divide-zinc-800/50">
                {win.matches.map((m, mi) => (
                  <div key={mi} className="px-5 py-3 flex items-center justify-between hover:bg-zinc-800/30 transition-colors">
                    <div className="min-w-0">
                      <p className="font-medium text-sm truncate">{m.match_filename}</p>
                      <div className="flex items-center gap-3 text-xs text-zinc-500 mt-0.5">
                        <span>{formatTime(m.match_start_s)} - {formatTime(m.match_end_s)}</span>
                        <span>Bars {m.match_bar_start}-{m.match_bar_end}</span>
                        {m.match_bpm && <span>{Math.round(m.match_bpm)} BPM</span>}
                        {m.match_key && <span>{m.match_key}</span>}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-20 h-2 bg-zinc-800 rounded-full overflow-hidden">
                        <div className="h-full bg-resonance-500 rounded-full" style={{ width: `${Math.max(0, m.similarity * 100)}%` }} />
                      </div>
                      <span className="text-xs text-zinc-400 w-12 text-right">{(m.similarity * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}