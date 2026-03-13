"use client";

import { useEffect, useState, useCallback } from "react";
import { listTracks, uploadTrack, deleteTrack, type Track } from "@/lib/api";

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    ready: "bg-emerald-500/20 text-emerald-400",
    analyzing: "bg-amber-500/20 text-amber-400 animate-pulse",
    pending: "bg-zinc-500/20 text-zinc-400",
    failed: "bg-red-500/20 text-red-400",
  };
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${colors[status] || colors.pending}`}>
      {status}
    </span>
  );
}

export default function LibraryPage() {
  const [tracks, setTracks] = useState<Track[]>([]);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setTracks(await listTracks());
    } catch (e) {
      console.error("Failed to load tracks", e);
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 5000);
    return () => clearInterval(interval);
  }, [refresh]);

  const handleUpload = async (files: FileList | null) => {
    if (!files) return;
    setUploading(true);
    try {
      for (const file of Array.from(files)) {
        await uploadTrack(file);
      }
      await refresh();
    } catch (e) {
      console.error("Upload failed", e);
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteTrack(id);
      await refresh();
    } catch (e) {
      console.error("Delete failed", e);
    }
  };

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Library</h1>
          <p className="text-zinc-400 mt-1">{tracks.length} tracks indexed</p>
        </div>
      </div>

      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => { e.preventDefault(); setDragOver(false); handleUpload(e.dataTransfer.files); }}
        className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
          dragOver ? "border-resonance-500 bg-resonance-500/5" : "border-zinc-700 hover:border-zinc-500"
        }`}
      >
        <div className="space-y-2">
          <p className="text-zinc-300 text-lg">
            {uploading ? "Uploading..." : "Drop audio files here or click to upload"}
          </p>
          <p className="text-zinc-500 text-sm">WAV, MP3, FLAC, AIFF, M4A</p>
          <label className="inline-block mt-3 px-5 py-2 bg-resonance-600 hover:bg-resonance-700 rounded-lg text-sm font-medium cursor-pointer transition-colors">
            Choose Files
            <input
              type="file"
              className="hidden"
              multiple
              accept=".wav,.mp3,.flac,.m4a,.aiff,.aif"
              onChange={(e) => handleUpload(e.target.files)}
              disabled={uploading}
            />
          </label>
        </div>
      </div>

      <div className="space-y-2">
        {tracks.map((track) => (
          <div
            key={track.id}
            className="flex items-center justify-between bg-zinc-900/50 border border-zinc-800 rounded-lg px-5 py-4 hover:bg-zinc-800/50 transition-colors group"
          >
            <div className="flex items-center gap-4 min-w-0">
              <div className="w-10 h-10 rounded-lg bg-resonance-600/20 flex items-center justify-center flex-shrink-0">
                <svg className="w-5 h-5 text-resonance-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                </svg>
              </div>
              <div className="min-w-0">
                <p className="font-medium truncate">{track.original_filename}</p>
                <div className="flex items-center gap-3 text-xs text-zinc-500 mt-0.5">
                  {track.bpm && <span>{Math.round(track.bpm)} BPM</span>}
                  {track.key && <span>{track.key} {track.scale}</span>}
                  {track.duration_s && <span>{Math.floor(track.duration_s / 60)}:{String(Math.floor(track.duration_s % 60)).padStart(2, "0")}</span>}
                </div>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <StatusBadge status={track.status} />
              <button
                onClick={() => handleDelete(track.id)}
                className="opacity-0 group-hover:opacity-100 text-zinc-500 hover:text-red-400 transition-all text-sm"
              >
                Delete
              </button>
            </div>
          </div>
        ))}
        {tracks.length === 0 && (
          <p className="text-center text-zinc-500 py-12">
            No tracks yet. Upload some audio files to get started.
          </p>
        )}
      </div>
    </div>
  );
}