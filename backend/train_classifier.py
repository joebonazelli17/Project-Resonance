"""Train section classifier inside Docker using API data."""
import json, pickle, urllib.request
import numpy as np
from sklearn.ensemble import RandomForestClassifier

TRACKS = {
    "tears": "f9e709dd-f16a-40c2-894d-91e5cac31d21",
    "sirens": "a73d215f-a1d9-409b-9aef-2a11ea88a34c",
    "passion": "2c0d51ac-f63d-4c41-8d07-799a2b29ebe2",
    "glasshouse": "8a6b185e-7257-4ae8-9190-d53ff81eeeca",
}
GT = {
    "tears": [(1,"intro"),(9,"verse"),(17,"buildup"),(25,"drop"),(37,"buildup"),(41,"breakdown"),(49,"verse"),(57,"verse"),(65,"buildup"),(73,"buildup"),(81,"buildup"),(89,"drop"),(121,"outro")],
    "sirens": [(1,"intro"),(17,"breakdown"),(25,"buildup"),(33,"drop"),(53,"buildup"),(57,"breakdown"),(73,"breakdown"),(81,"buildup"),(91,"drop"),(115,"buildup"),(123,"drop"),(155,"buildup"),(163,"drop"),(179,"outro")],
    "passion": [(1,"intro"),(17,"intro"),(31,"buildup"),(33,"drop"),(65,"breakdown"),(73,"breakdown"),(81,"breakdown"),(89,"buildup"),(105,"drop"),(169,"outro")],
    "glasshouse": [(1,"verse"),(17,"verse"),(25,"buildup"),(33,"drop"),(41,"drop"),(49,"drop"),(65,"breakdown"),(73,"verse"),(81,"buildup"),(89,"drop"),(113,"outro")],
}

def gtl(secs, bar):
    l = None
    for b, lb in secs:
        if bar >= b: l = lb
        else: break
    return l

def pct(vals):
    if not vals: return {"p25":0,"p50":0,"p75":0}
    s=sorted(vals); n=len(s)
    return {"p25":s[n//4],"p50":s[n//2],"p75":s[3*n//4]}

X_all, y_all = [], []
for name, tid in TRACKS.items():
    resp = urllib.request.urlopen(f"http://127.0.0.1:8000/api/tracks/{tid}")
    data = json.loads(resp.read())
    gt = GT[name]; dur = data["duration_s"]
    secs = sorted([s for s in data["sections"] if s["bars"]==8], key=lambda s:s["start_s"])
    if not secs: continue
    rp = pct([s["rms_dbfs"] for s in secs])
    rr = max(rp["p75"]-rp["p25"], 1e-6)
    sp = {}
    for st in ["drums","bass","vocals","other"]:
        v = [s.get("stem_energies",{}).get(st,-96) for s in secs if s.get("stem_energies",{}).get(st,-96)>-90]
        sp[st] = pct(v) if v else {"p25":-40,"p50":-25,"p75":-10}
    def sn(val, p):
        r=p["p75"]-p["p25"]
        return -1.0 if r<8 else (val-p["p25"])/r
    rl = [s["rms_dbfs"] for s in secs]
    for i, sec in enumerate(secs):
        mb = (sec["bar_start"]+sec["bar_end"])/2
        gl = gtl(gt, mb+1)
        if not gl: continue
        se = sec.get("stem_energies") or {}
        pos = sec["start_s"]/max(dur,1e-6)
        en = (sec["rms_dbfs"]-rp["p25"])/rr
        dn=sn(se.get("drums",-96),sp["drums"]); bn=sn(se.get("bass",-96),sp["bass"])
        vn=sn(se.get("vocals",-96),sp["vocals"]); on=sn(se.get("other",-96),sp["other"])
        rd = (rl[i]-rl[i-1]) if i>=1 else 0.0
        rt = (rl[i]-rl[i-3])/3.0 if i>=3 else rd if i>=1 else 0.0
        n2 = rl[i+2] if i+2<len(rl) else rl[i]
        fd = (n2-rl[i])/2.0
        X_all.append([en,pos,dn,bn,vn,on,rd,rt,fd,sec["crest_db"],sec["hf_perc_ratio"],sec["flatness"],
                       1.0 if dn>0.7 else 0.0, 1.0 if bn>0.7 else 0.0,
                       1.0 if vn>0.4 else 0.0, 1.0 if dn<0.15 else 0.0, pos*en])
        y_all.append(gl)

X=np.array(X_all); y=np.array(y_all)
print(f"Training on {len(X)} samples")
model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced")
model.fit(X, y)
with open("/app/models/section_classifier.pkl","wb") as f: pickle.dump(model,f)
preds = model.predict(X)
print(f"Saved. Training acc: {sum(preds==y)}/{len(y)} = {sum(preds==y)/len(y)*100:.1f}%")
