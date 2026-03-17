const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000/api";

export interface Track {
  id: string;
  filename: string;
  original_filename: string;
  duration_s: number | null;
  bpm: number | null;
  key: string | null;
  scale: string | null;
  beats_per_bar: number | null;
  status: string;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface TrackSection {
  id: string;
  track_id: string;
  start_s: number;
  end_s: number;
  bars: number;
  bar_start: number;
  bar_end: number;
  bpm: number | null;
  key: string | null;
  scale: string | null;
  hf_perc_ratio: number;
  rms_dbfs: number;
  peak_dbfs: number;
  crest_db: number;
  flatness: number;
  section_label: string | null;
  section_label_confidence: number | null;
  band_energies: Record<string, number> | null;
  band_crest: Record<string, number> | null;
  band_transient_density: Record<string, number> | null;
  stereo_features: { correlation: number; mid_side_ratio: number; width_by_band: Record<string, number> } | null;
}

export interface EnergyCurve {
  times: number[];
  lufs: number[];
  centroid: number[];
  onset_density: number[];
  low_ratio: number[];
}

export interface TrackWithCurve extends Track {
  energy_curve: EnergyCurve | null;
}

export interface TrackDetail extends Track {
  sections: TrackSection[];
}

export interface SearchMatch {
  match_track_id: string;
  match_filename: string;
  match_section_id: string;
  match_start_s: number;
  match_end_s: number;
  match_bars: number;
  match_bar_start: number;
  match_bar_end: number;
  match_bpm: number | null;
  match_key: string | null;
  similarity: number;
}

export interface SearchWindowResult {
  query_start_s: number;
  query_end_s: number;
  query_bars: number;
  query_bar_start: number;
  query_bar_end: number;
  matches: SearchMatch[];
}

export interface SearchResponse {
  query_track_id: string;
  query_bpm: number | null;
  query_key: string | null;
  bars: number;
  results: SearchWindowResult[];
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

export async function listTracks(): Promise<Track[]> {
  return request("/tracks");
}

export async function getTrack(id: string): Promise<TrackDetail> {
  return request(`/tracks/${id}`);
}

export async function uploadTrack(file: File): Promise<Track> {
  const form = new FormData();
  form.append("file", file);
  return request("/tracks/upload", { method: "POST", body: form });
}

export async function deleteTrack(id: string): Promise<void> {
  await request(`/tracks/${id}`, { method: "DELETE" });
}

export async function getTrackEnergy(id: string): Promise<TrackWithCurve> {
  return request(`/tracks/${id}/energy`);
}

export interface ComparisonRecommendation {
  band?: string;
  freq_range?: string;
  type: string;
  message: string;
  severity: string;
}

export interface ComparisonResult {
  section_a: string;
  section_b: string;
  mastering_state_a: string;
  mastering_state_b: string;
  mastering_mismatch: boolean;
  spectral_shape_delta: Record<string, number>;
  band_crest_a: Record<string, number>;
  band_crest_b: Record<string, number>;
  transient_density_a: Record<string, number>;
  transient_density_b: Record<string, number>;
  stereo_a: Record<string, unknown>;
  stereo_b: Record<string, unknown>;
  dynamics_delta: { rms_dbfs: number; peak_dbfs: number; crest_db: number };
  recommendations: ComparisonRecommendation[];
}

export async function compareSections(a: string, b: string): Promise<ComparisonResult> {
  return request(`/search/compare/${a}/${b}`);
}

export async function getStreamUrl(id: string): Promise<string> {
  const data = await request<{ url: string }>(`/tracks/${id}/stream`);
  // URL may be a presigned S3 URL or a relative /api/tracks/{id}/audio path
  if (data.url.startsWith("/")) {
    return `${API_BASE.replace("/api", "")}${data.url}`;
  }
  return data.url;
}

export async function searchSimilar(
  file: File,
  bars: number = 4,
  hopBars: number = 2,
  k: number = 5
): Promise<SearchResponse> {
  const form = new FormData();
  form.append("file", file);
  return request(`/search?bars=${bars}&hop_bars=${hopBars}&k=${k}`, {
    method: "POST",
    body: form,
  });
}

export interface TextSearchMatch {
  track_id: string;
  filename: string;
  section_id: string;
  start_s: number;
  end_s: number;
  bars: number;
  bar_start: number;
  bar_end: number;
  bpm: number | null;
  key: string | null;
  scale: string | null;
  similarity: number;
}

export interface TextSearchResponse {
  query: string;
  matches: TextSearchMatch[];
}

export async function searchByText(
  query: string,
  bars?: number,
  k: number = 10
): Promise<TextSearchResponse> {
  const params = new URLSearchParams({ q: query, k: String(k) });
  if (bars) params.set("bars", String(bars));
  return request(`/search/text?${params}`);
}

export interface StemWeights {
  mix: number;
  drums: number;
  bass: number;
  vocals: number;
  other: number;
}

export interface StemSearchResponse {
  query: string;
  weights: Record<string, number>;
  matches: TextSearchMatch[];
}

export async function searchByStems(
  query: string,
  weights: StemWeights,
  bars?: number,
  k: number = 10
): Promise<StemSearchResponse> {
  return request("/search/stems", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, weights, bars: bars || null, k }),
  });
}

export function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(1);
  return `${m}:${s.padStart(4, "0")}`;
}