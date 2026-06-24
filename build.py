#!/usr/bin/env python3
"""Build the RAG corpus: embed a curated space-weather knowledge base with Gemini,
then write src/corpus.js (chunks + vectors) for the Worker to retrieve against."""
import json, os, urllib.request, math

KEY = None
for line in open(os.path.expanduser("~/.gemini-dsremo.env")):
    if line.startswith("GEMINI_API_KEY="):
        KEY = line.strip().split("=", 1)[1]
EMB_MODEL = "gemini-embedding-001"
DIM = 768

# Curated, factual space-weather knowledge base. Each entry is a self-contained fact.
CORPUS = [
    ("What space weather is", "NASA/NOAA",
     "Space weather refers to conditions on the Sun and in the solar wind, magnetosphere, ionosphere, and thermosphere that can affect technology and human activity. Its main drivers are the solar wind, solar flares, and coronal mass ejections."),
    ("The Kp index", "NOAA SWPC",
     "The planetary K-index (Kp) is a global measure of geomagnetic disturbance on a scale of 0 to 9, derived from magnetometer stations worldwide. Kp values below 5 are quiet to unsettled; Kp of 5 or higher indicates a geomagnetic storm."),
    ("NOAA G-scale (geomagnetic storms)", "NOAA SWPC",
     "NOAA rates geomagnetic storms G1 (minor, Kp 5) through G5 (extreme, Kp 9). G1 can cause weak power-grid fluctuations and aurora at high latitudes; G5 can cause widespread voltage-control problems, satellite disruption, and aurora seen near the equator."),
    ("NOAA R-scale (radio blackouts)", "NOAA SWPC",
     "The R-scale rates radio blackouts caused by solar flares, from R1 (minor) to R5 (extreme). Flares emit X-rays that ionize the dayside upper atmosphere, degrading or blacking out high-frequency (HF) radio used by aviation and maritime communication."),
    ("NOAA S-scale (radiation storms)", "NOAA SWPC",
     "The S-scale rates solar radiation storms (S1 minor to S5 extreme) caused by energetic protons from the Sun. They pose radiation risk to astronauts and high-altitude aircraft crews and can damage satellite electronics."),
    ("Solar wind", "NASA",
     "The solar wind is a continuous stream of charged particles (mostly protons and electrons) flowing from the Sun's corona at typically 300 to 800 km/s. Faster, denser solar wind transfers more energy into Earth's magnetosphere and drives stronger geomagnetic activity."),
    ("Coronal mass ejections (CMEs)", "NASA",
     "A coronal mass ejection is a large eruption of plasma and magnetic field from the Sun's corona. An Earth-directed CME can arrive in 1 to 3 days and, if its magnetic field points southward, couple strongly with Earth's field to trigger a geomagnetic storm."),
    ("Solar flares", "NASA",
     "A solar flare is a sudden, intense burst of radiation from the Sun's surface, classified by X-ray brightness as A, B, C, M, or X (X being the strongest). The radiation travels at light speed and reaches Earth in about 8 minutes, causing radio blackouts on the sunlit side."),
    ("Interplanetary magnetic field and Bz", "NOAA SWPC",
     "The solar wind carries the Sun's magnetic field, called the interplanetary magnetic field (IMF). Its north-south component, Bz, is critical: a strongly southward (negative) Bz lets solar-wind energy pour into the magnetosphere, the key trigger for major geomagnetic storms."),
    ("Auroras", "NASA",
     "Auroras (the northern and southern lights) occur when charged particles funneled by Earth's magnetic field collide with atmospheric gases, making them glow. Stronger geomagnetic storms push the aurora to lower latitudes, occasionally visible far from the poles."),
    ("The Carrington Event of 1859", "NASA",
     "The Carrington Event was the most intense geomagnetic storm on record, caused by a powerful CME. Auroras were seen near the equator and telegraph systems failed, some giving operators shocks. A similar storm today could cause trillions of dollars in damage to power and satellite infrastructure."),
    ("The Gannon storm of May 2024", "NOAA SWPC",
     "In May 2024, a series of CMEs produced the strongest geomagnetic storm since 2003, reaching G5 (extreme). Auroras were visible across much of the world at unusually low latitudes, and some GPS-based precision agriculture and high-frequency radio were disrupted."),
    ("The Halloween storms of 2003", "NOAA SWPC",
     "In late October 2003, a cluster of X-class flares and CMEs produced multiple G5 storms. They forced the rerouting of aircraft, caused a power outage in Sweden, and damaged instruments on several satellites."),
    ("Impacts on satellites", "NOAA SWPC",
     "Space weather degrades satellites through surface and deep-dielectric charging, single-event upsets from energetic particles, and increased atmospheric drag during storms that heats and expands the upper atmosphere, lowering orbits and complicating tracking."),
    ("Impacts on the power grid", "NOAA SWPC",
     "Geomagnetic storms induce geomagnetically induced currents (GICs) in long conductors like power lines, which can saturate and damage high-voltage transformers and cause voltage instability. The 1989 Quebec storm blacked out the province for about nine hours."),
    ("Impacts on GPS and navigation", "NOAA SWPC",
     "Space weather disturbs the ionosphere, delaying and scattering GPS signals and reducing positioning accuracy. Precision applications like surveying, aviation approaches, and agriculture are most affected during storms."),
    ("Impacts on aviation", "NOAA SWPC",
     "During solar radiation storms and radio blackouts, polar flights may be rerouted to avoid HF-radio loss and elevated radiation exposure to crew and passengers. Airlines monitor NOAA alerts to plan high-latitude routes."),
    ("The solar cycle", "NASA",
     "The Sun follows an approximately 11-year activity cycle. Near solar maximum, sunspots, flares, and CMEs are far more frequent, so space-weather risk rises. Solar Cycle 25 reached its maximum around 2024-2025."),
    ("Sunspots", "NASA",
     "Sunspots are cooler, darker regions of intense magnetic activity on the Sun's surface. They are the source regions for most flares and CMEs, so sunspot number is a basic indicator of how active the Sun is."),
    ("The Dst index", "NOAA",
     "The disturbance storm-time (Dst) index measures the strength of the ring current around Earth during storms; large negative Dst values indicate intense geomagnetic storms. It complements Kp as a storm-intensity measure."),
    ("Earth's magnetosphere", "NASA",
     "The magnetosphere is the region around Earth dominated by its magnetic field, which deflects most of the solar wind. Storms compress and reconfigure it, energizing particles and driving aurora and induced currents."),
    ("The ionosphere", "NOAA",
     "The ionosphere is the electrically charged layer of the upper atmosphere that reflects HF radio and carries GPS signals. Solar radiation and storms change its density, which is why space weather affects communication and navigation."),
    ("Monitoring spacecraft (DSCOVR, ACE, SOHO)", "NOAA/NASA",
     "Spacecraft at the L1 Lagrange point, about 1.5 million km sunward, measure the solar wind before it reaches Earth. DSCOVR and ACE provide roughly 15 to 60 minutes of warning of incoming conditions; SOHO and SDO image the Sun itself."),
    ("Forecasting lead time", "NOAA SWPC",
     "Flares are detected as they happen (about 8 minutes after emission). CMEs give 1 to 3 days of warning from when they leave the Sun, but their geomagnetic impact is only confirmed minutes ahead by L1 monitors measuring the arriving Bz."),
    ("Geomagnetically induced currents (GICs)", "NOAA",
     "GICs are slow, quasi-DC currents driven into grounded conductors (power lines, pipelines, railways) by the rapidly changing magnetic field of a storm. They are the main mechanism by which space weather threatens power grids."),
    ("Solar energetic particles (SEPs)", "NASA",
     "Solar energetic particles are protons and ions accelerated to near-light speed by flares and CME shocks. They arrive within minutes to hours, create radiation storms, and are a major hazard for astronauts and polar aviation."),
    ("The aurora oval and viewing", "NOAA SWPC",
     "Aurora forms in an oval around each magnetic pole that expands toward the equator as Kp rises. As a rough guide, higher Kp means the aurora may be visible from lower-latitude locations, which is why aurora apps key off the Kp forecast."),
    ("Why space weather matters economically", "NASA/NOAA",
     "Modern infrastructure (power grids, satellites, GPS timing, aviation, and communications) is increasingly exposed to space weather. Studies estimate an extreme, Carrington-class storm could cause economic damage on the scale of a major natural disaster."),
]

def embed_batch(texts, task_type):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMB_MODEL}:batchEmbedContents?key={KEY}"
    reqs = [{"model": f"models/{EMB_MODEL}", "content": {"parts": [{"text": t}]},
             "taskType": task_type, "outputDimensionality": DIM} for t in texts]
    body = json.dumps({"requests": reqs}).encode()
    r = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=120) as resp:
        return [e["values"] for e in json.load(resp)["embeddings"]]

def norm(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [round(x / n, 6) for x in v]

def main():
    texts = [f"{title}. {text}" for title, _src, text in CORPUS]
    vecs = embed_batch(texts, "RETRIEVAL_DOCUMENT")
    out = []
    for (title, src, text), v in zip(CORPUS, vecs):
        out.append({"title": title, "source": src, "text": text, "vec": norm(v)})
    meta = {"model": EMB_MODEL, "dim": DIM, "count": len(out)}
    with open("/home/ashutosh/Music/code/space-weather-rag/src/corpus.js", "w") as f:
        f.write("// Auto-generated by build.py — curated space-weather KB, embedded with Gemini.\n")
        f.write("export const META = " + json.dumps(meta) + ";\n")
        f.write("export const CORPUS = " + json.dumps(out) + ";\n")
    print(f"embedded {len(out)} chunks at dim {DIM}; wrote src/corpus.js")

if __name__ == "__main__":
    main()
