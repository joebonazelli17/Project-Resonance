"use client";

import { useState } from "react";
import { searchSimilar, formatTime, type SearchResponse, type SearchWindowResult } from "@/lib/api";

export default function SearchPage() {
  const [file, setFile] = useState<File | null>(null);
  const [bars, setBars] = useState(4);
  const [searching, setSearching] = useState(false);
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    if (!file) return;
    setSearching(true);
    setError(null);
    try {
      const res = await searchSimilar(file, bars);
      setResults(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Search failed");
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Reference Search</h1>
        <p className="text-zinc-400 mt-1">Find similar sections in your library</p>
      </div>

      {/* Query panel */}
      <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-6 space-y-4">
        <div className="flex flex-col sm:flex-row gap-4">
          <label className="flex-1 border-2 border-dashed border-zinc-700 hover:border-resonance-500 rounded-lg p-4 text-center cursor-pointer transition-colors">
            <p className="text-sm text-zinc-300">
              {file ? file.name : "Choose a query track"}
            </p>
            <input
              type="file"
              className="hidden"
              accept=".wav,.mp3,.flac,.m4a,.aiff,.aif"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
          </label>
          <div className="flex items-center gap-3">
            <label className="text-sm text-zinc-400">Bars:</label>
            <select
              value={bars}
              onChange={(e) => setBars(Number(e.target.value))}
              className="bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm"
            >
              {[2, 4, 8, 16].map((b) => (
                <option key={b} value={b}>{b} bars</option>
              ))}
            </select>
            <button
              onClick={handleSearch}
              disabled={!file || searching}
              className="px-5 py-2 bg-resonance-600 hover:bg-resonance-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-sm font-medium transition-colors"
            >
              {searching ? "Searching..." : "Search"}
            </button>
          </div>
        </div>
        {error && <p className="text-red-400 text-sm">{error}</p>}
      </div>

      {/* Results */}
      {results && (
        <div className="space-y-4">
          <div className="flex items-center gap-4 text-sm text-zinc-400">
            <span>{results.results.length} windows analyzed</span>
            {results.query_bpm && <span>{Math.round(results.query_bpm)} BPM</span>}
            {results.query_key && <span>{results.query_key}</span>}
          </div>

          {results.results.map((win: SearchWindowResult, wi: number) => (
            <div key={wi} className="bg-zinc-900/50 border border-zinc-800 rounded-xl overflow-hidden">
              <div className="px-5 py-3 bg-zinc-800/50 border-b border-zinc-800 flex items-center gap-4">
                <span className="text-sm font-medium">
                  Window {wi + 1}: {formatTime(win.query_start_s)} - {formatTime(win.query_end_s)}
                </span>
                <span className="text-xs text-zinc-500">
                  Bars {win.query_bar_start}-{win.query_bar_end}
                </span>
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
                        <div
                          className="h-full bg-resonance-500 rounded-full"
                          style={{ width: `${Math.max(0, m.similarity * 100)}%` }}
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
          ))}
        </div>
      )}
    </div>
  );
}