# app/clean_tags.py
import os, re
from mutagen.aiff import AIFF
from mutagen.id3 import ID3, TIT2, TPE1, TPE2, TCOM

FOLDER = "/Users/lbonazelli/Documents/This is IT/Fresh Restart - All Tracks/Platinum Notes - For Reference"

PREFIX = re.compile(r'^\d{3}\.\s*')

def clean_text_list(text_list):
    changed = False
    new = []
    for t in text_list:
        nt = PREFIX.sub("", t)
        changed |= (nt != t)
        new.append(nt)
    return new, changed

def fix_frame(tags: ID3, frame_id: str, ctor):
    if frame_id in tags:
        frame = tags.get(frame_id)
        if hasattr(frame, "text"):
            new_text, changed = clean_text_list(frame.text)
            if changed:
                # set a proper Frame object
                tags.setall(frame_id, [ctor(encoding=3, text=new_text)])
            return changed
    return False

def main():
    for name in os.listdir(FOLDER):
        if not name.lower().endswith(".aiff"):
            continue
        path = os.path.join(FOLDER, name)
        try:
            audio = AIFF(path)
            if not isinstance(audio.tags, ID3):
                # Skip files without ID3 tags
                continue
            tags = audio.tags
            changed = False
            changed |= fix_frame(tags, "TPE1", TPE1)  # Artist
            changed |= fix_frame(tags, "TPE2", TPE2)  # Album Artist
            changed |= fix_frame(tags, "TCOM", TCOM)  # Composer
            changed |= fix_frame(tags, "TIT2", TIT2)  # Title
            if changed:
                audio.save(v2_version=3)  # write ID3v2.3 for max compatibility
                print(f"Cleaned tags in: {name}")
        except Exception as e:
            print(f"Error with {name}: {e}")

if __name__ == "__main__":
    main()
