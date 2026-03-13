"use client";

import { useEffect, useState, useCallback } from "react";
import { listTracks, type Track } from "@/lib/api";

export default function TracksPage() {
  const [tracks, setTracks] = useState<Track[]>([]);

  const refresh = useCallback(async () => {
    try {
      const all = await listTracks();
      setTracks(all.filter((t) => t.status === "ready"));
    } catch (e) {
      console.error(e);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold tracking-tight">Tracks</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {tracks.map((track) => (
          <div
            key={track.id}
            className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-5 hover:border-resonance-500/50 transition-colors"
          >
            <p className="font-medium truncate">{track.original_filename}</p>
            <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-zinc-400">
              <div>
                <span className="text-zinc-600">BPM</span>
                <p className="text-zinc-200 font-medium">{track.bpm ? Math.round(track.bpm) : "--"}</p>
              </div>
              <div>
                <span className="text-zinc-600">Key</span>
                <p className="text-zinc-200 font-medium">{track.key || "--"} {track.scale || ""}</p>
              </div>
              <div>
                <span className="text-zinc-600">Duration</span>
                <p className="text-zinc-200 font-medium">
                  {track.duration_s ? `${Math.floor(track.duration_s / 60)}:${String(Math.floor(track.duration_s % 60)).padStart(2, "0")}` : "--"}
                </p>
              </div>
              <div>
                <span className="text-zinc-600">Time Sig</span>
                <p className="text-zinc-200 font-medium">{track.beats_per_bar || 4}/4</p>
              </div>
            </div>
          </div>
        ))}
      </div>
      {tracks.length === 0 && (
        <p className="text-center text-zinc-500 py-12">
          No analyzed tracks yet. Upload tracks from the Library page.
        </p>
      )}
    </div>
  );
}