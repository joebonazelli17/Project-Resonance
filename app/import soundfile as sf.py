# quick probe: which file is failing for PySoundFile?
import soundfile as sf, glob, os

bad = []
for p in glob.glob("data/corpus/*"):
    try:
        i = sf.info(p)
        # try opening and seeking a bit (mimic librosa offset/duration)
        with sf.SoundFile(p) as f:
            f.seek(min(44100 * 30, len(f)))  # seek ~30s if available
    except Exception as e:
        bad.append((os.path.basename(p), repr(e)))

print("bad files:", bad)
