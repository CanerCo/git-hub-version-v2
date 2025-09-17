#!/usr/bin/env python3
"""
EXMARaLDA Search UI (Streamlit) - Optimized VERSION (with sae IDF fallback)
Performance improvements:
1. Pre-computed geocoding cache with German cities
2. Batch database queries
3. Optimized text processing with caching
4. Early filtering and lazy loading
5. Reduced reduntant computations
6. Safe BM25 IDF fallback when precomputed IDF is missing (fixes "No files to show.")
"""

from __future__ import annotations 

import pandas as pd
import html # colorful matches
import numpy as np # for scientific and mathematical calculations
import unicodedata # py lib for Unicode characters
import argparse # parse command-line arguments
import gzip # read/write gzip-compressed files
import json
from pathlib import Path # path handling in an OS-agnostic way, better handling paths esp for city names extractions from .exb files
import heapq # get top-k items efficiently (nlargest)
from typing import Dict, Iterable, List, Tuple, Set, Optional
import re # regular expressions
import sqlite3 # SQLite # DELETE 
from functools import lru_cache # caching decorator for functions, it is used for memoizing
import streamlit as st
import pydeck as pdk


# Pre-computed German cities coordinates to avoid API calls, check parse_cities.py on how we did that
# Used in the function (get_city_coordinates)
GERMAN_CITIES_COORDS = {'Achsheim': (48.4805903, 10.8135775), 'Achstetten': (48.2605412, 9.8971061), 'Adelberg': (48.7628274, 9.5975232), 'Ailingen': (47.6883971, 9.4903528), 'Alfdorf': (48.8439828, 9.7189427), 'Altheim': (48.5808015, 10.0271338), 'Arlesried': (48.0778147, 10.3482261), 'Asselfingen': (48.5290962, 10.1945447), 'Attenhausen': (50.2918502, 7.8769892), 'Auggen': (47.7864318, 7.5970495), 'Aulendorf': (47.9528162, 9.6409331), 'Bachenau': (49.2793334, 9.193018), 'Baiersbronn': (48.5069519, 8.3720124), 'Balgheim': (48.0665256, 8.7646788), 'Ballendorf': (48.5548852, 10.0789077), 'Ballersdorf': (48.7041813, 11.158265), 'Bartholomä': (48.7542906, 9.989245), 'Bebenhausen': (48.5610315, 9.0591401), 'Beimbach': (49.2444984, 9.9814661), 'Beltersrot': (49.1767563, 9.6788454), 'Berlichingen': (49.3269972, 9.4869475), 'Beuren': (49.7272641, 6.9163463), 'Beyerberg': (49.1123554, 10.5049232), 'Biberach': (48.0984413, 9.7899938), 'Bittelbronn': (48.3839642, 8.7717917), 'Blankenburg': (51.7902676, 10.9551991), 'Bondorf': (48.5215073, 8.83361), 'Brettach': (49.1575853, 9.4582484), 'Brombach': (50.9763223, 7.2573513), 'Bräunisheim': (48.6056842, 9.9489348), 'Buchau': (48.0641747, 9.6106661), 'Buttenhausen': (48.3604996, 9.4780772), 'Böhen': (47.8816355, 10.3006322), 'Böhmenkirch': (48.6843886, 9.9320657), 'Bühl': (48.6945066, 8.134423), 'Daxlanden': (49.0014111, 8.3386057), 'Deilingen': (48.1757794, 8.7843485), 'Dettingen': (48.4368485, 8.9343816), 'Diepoldshofen': (47.845911, 9.9457502), 'Dietenheim': (48.2111094, 10.0700937), 'Donaustetten': (48.3306392, 9.9350202), 'Dornstadt': (48.4656356, 9.9431286), 'Döttingen': (49.2213567, 9.7716929), 'Dühren': (49.2455922, 8.8359282), 'Ellenberg': (49.6627831, 7.1421771), 'Emerkingen': (48.2117809, 9.6571276), 'Erdmannhausen': (48.9404684, 9.2961577), 'Erkenbrechtsweiler': (48.557005, 9.4317314), 'Eschach': (48.889384, 9.8711892), 'Eschenau': (49.5741708, 11.1983828), 'Ettenhausen': (49.3538126, 9.8784508), 'Flein': (49.1005935, 9.2138586), 'Flochberg': (48.8495387, 10.366157), 'Forst': (49.1545, 8.5819), 'Frankenbach': (49.1594238, 9.1717549), 'Frauenriedhausen': (48.5978981, 10.4021442), 'Frommenhausen': (48.4298554, 8.8756521), 'Gellmersbach': (49.1814701, 9.2949652), 'Gemmrigheim': (49.0259996, 9.1560581), 'Goldburghausen': (48.8705166, 10.4278574), 'Grab': (49.0380281, 9.5788774), 'Greuthof': (49.0814183, 9.42838), 'Großkuchen': (48.7523628, 10.2301911), 'Grunbach': (48.8302659, 8.6751721), 'Gutmadingen': (47.9106423, 8.615817), 'Göllsdorf': (48.1617099, 8.6576007), 'Gönningen': (48.4336967, 9.1501492), 'Haslach': (47.9906144, 7.8169345), 'Hausach': (48.2836305, 8.174972), 'Hayingen': (48.2751679, 9.4781135), 'Heimsheim': (48.8058986, 8.8619207), 'Hengstfeld': (49.2217469, 10.1013014), 'Herkheim': (48.8233985, 10.4877123), 'Hettlingen': (48.532765, 10.6668097), 'Hinterheubach': (49.4859274, 8.77682), 'Hinterlangenbach': (48.5921425, 8.2524655), 'Hirrlingen': (48.4102084, 8.888044), 'Hirsau': (48.734346, 8.7352364), 'Hohenhaslach': (48.9992281, 9.019298), 'Hohenstadt': (48.544391, 9.6635527), 'Häfnerhaslach': (49.0242421, 8.9193232), 'Höchstberg': (50.2411692, 7.0029945), 'Illmensee': (47.864499, 9.3766192), 'Iptingen': (48.8854377, 8.898631), 'Isingen': (48.2820943, 8.7453053), 'Jungingen': (48.3285552, 9.0417577), 'Kaisersbach': (48.9298109, 9.6389475), 'Kirchberg': (49.9431629, 7.4083878), 'Kleinkuchen': (48.7356816, 10.2432274), 'Klosterreichenbach': (48.5254278, 8.3973121), 'Kniebis': (48.4720808, 8.3106312), 'Kreuzthal': (47.7158765, 10.1204467), 'Kronau': (49.2190434, 8.6328799), 'Kusterdingen': (48.5027717, 9.1137828), 'Königsbronn': (48.7389826, 10.112308), 'Königshofen': (49.5471848, 9.7315312), 'Kösingen': (48.7586139, 10.4102467), 'Langenau': (48.4995693, 10.1210951), 'Langenordnach': (47.9554337, 8.1975907), 'Lauben': (48.057695, 10.2903168), 'Lauterbach': (51.0691221, 10.3551323), 'Lehengericht': (48.2685746, 8.3371012), 'Leimiß': (48.5823426, 8.3105641), 'Leupolz': (47.7366223, 10.3530889), 'Liggersdorf': (47.8857879, 9.1095474), 'Lindenbuch': (48.3482229, 8.4525984), 'Marlen': (48.5218664, 7.8237061), 'Marxheim': (48.7405104, 10.9444427), 'Mengen': (48.0498964, 9.3321246), 'Merklingen': (48.5101058, 9.7557738), 'Meßbach': (50.4585326, 12.11149), 'Michelfeld': (49.0968691, 9.6771416), 'Mindelaltheim': (48.4636897, 10.4091533), 'Missen-Wilhams': (47.5976807, 10.1101868), 'Mittellangenbach': (48.587009, 8.2805047), 'Mitteltal': (48.5194614, 8.3232093), 'Mooshausen': (47.971838, 10.0849323), 'Munzingen': (47.9696412, 7.698233), 'Mähringen': (48.436836, 9.9404851), 'Mönchsdeggingen': (48.7754142, 10.5807045), 'Münster': (51.9625101, 7.6251879), 'Neipperg': (49.10549, 9.0485553), 'Neubulach': (48.6617473, 8.6958614), 'Neuenstein': (49.2049886, 9.5813213), 'Neuhausen': (50.675, 13.4667), 'Neuler': (48.9278463, 10.0688239), 'Neumünster': (54.0703296, 9.9884451), 'Obenhausen': (48.2360078, 10.1778862), 'Oberbergen': (48.0953714, 7.6550304), 'Oberderdingen': (49.0626654, 8.8021324), 'Obergriesheim': (49.2639589, 9.2036205), 'Oberreitnau': (47.5932325, 9.6831748), 'Oberschondorf': (48.0515255, 11.0820881), 'Obersimonswald': (48.0789717, 8.094862), 'Obersteinach': (49.1998766, 9.852938), 'Ochsenburg': (49.0749176, 8.8946844), 'Opfenbach': (47.6282079, 9.840786), 'Oppelsbohm': (48.8608214, 9.4687336), 'Pappelau': (48.3680338, 9.8074368), 'Pfrungen': (47.8811036, 9.3987213), 'Plieningen': (48.7113947, 9.1988651), 'Ratzenhofen': (48.6920172, 11.8058536), 'Reutti': (48.361732, 10.07218), 'Rheinsheim': (49.2356233, 8.418401), 'Rietheim': (48.0391822, 8.7806127), 'Rodt': (51.0231459, 7.4892043), 'Romishorn': (48.3635521, 8.4260699), 'Rottweil': (48.1678244, 8.626979), 'Rudelstetten': (48.8553978, 10.644875), 'Schanbach': (48.7586999, 9.377079), 'Schiltach': (48.2901774, 8.3436346), 'Schlier': (47.768365, 9.6749149), 'Schramberg': (48.225478, 8.3852168), 'Schupfholz': (48.0752873, 7.8220615), 'Schwenningen': (48.0657049, 8.5361259), 'Schwieberdingen': (48.8749991, 9.0778484), 'Schwörsheim': (48.9128917, 10.6224079), 'Serres': (48.8935905, 8.8774214), 'Sommersbach': (47.7186776, 9.9947808), 'St. Roman': (48.3310732, 8.2918829), 'Stuttgart': (48.7784485, 9.1800132), 'Suppingen': (48.4559055, 9.7143444), 'Thalheim': (48.0057, 9.0381), 'Tieringen': (48.1989984, 8.8770634), 'Tomerdingen': (48.4813003, 9.9129965), 'Trollenberg': (48.3625034, 8.4523183), 'Tübingen': (48.5203263, 9.053596), 'Unterbergen': (48.232029, 10.9431986), 'Untergruppenbach': (49.089717, 9.275092), 'Utzmemmingen': (48.8289513, 10.4361142), 'Volzenhäuser': (48.5722151, 8.3127597), 'Vorderlangenbach': (48.5863755, 8.2956322), 'Vorderwestermurr': (48.9545179, 9.5841092), 'Vöhrenbach': (48.0458156, 8.3039184), 'Waldhausen': (48.7856, 9.63771), 'Waldmannshofen': (49.5326594, 10.0670991), 'Weiler': (50.1416289, 7.0766082), 'Weisweil': (48.2010438, 7.6756268), 'Weltersberg': (49.4156126, 6.7938749), 'Wengen': (47.6776211, 10.1482133), 'Wildenstein': (50.7427076, 13.1345221), 'Willstätt': (48.5415314, 7.8934991), 'Wittelshofen': (49.0619697, 10.4829365), 'Wolfach': (48.2985845, 8.222608), 'Wurmlingen': (48.0026634, 8.7765991), 'Zweiflingen': (49.2564, 9.51806), 'Zwickgabel': (48.5869462, 8.3177941), 'Öhringen': (49.2005034, 9.5024397), 'Ötigheim': (48.8894679, 8.2383062)}

GEOCODER_AVAILABLE = True # WHERE TO USE?

# Used in the parse_city function
# Match a single character present in the list below [A-Za-zÄÖÜäöüß\-]
CITY_RE = re.compile(r"([A-Za-zÄÖÜäöüß\-]+)") # WHERE TO USE? #CAN BE ENHANCED FURTHER 

# Colorizing matches
def _colorize_html(text: str, pattern: re.Pattern, cls: str = "hl") -> str:
    """HTML-escape everything, then wrap matches in span class = hl """
    out, last = [], 0
    for m in pattern.finditer(text or ""):
        out.append(html.escape(text[last:m.start()]))
        out.append(f"<span class='{cls}'>{html.escape(m.group(1))}</span>")
        last = m.end()
    out.append(html.escape(text[last:]))
    return "".join(out)
    

# Used prepare_city_points function, for getting ascii version of city names
def to_english_chars(text: str | None) -> str | None:
    """
    Return ASCII-only version of text ((ä→a, ö→o, ü→u, ß→ss).
    """
    if text is None:
        return None
    if text.isascii(): # fast-forward
        return text
    text = text.replace("ß", "ss")
    text = text.replace("ẞ", "SS")
    return unicodedata.normalize("NFKD", text).encode("ascii","ignore").decode("ascii")

def parse_city(file_name: str) -> str | None: # WHERE to USE?
    """
    Extract city name from .exb filename
    """
    if file_name.startswith("St_Roman"):
        return "St. Roman"
    elif file_name.startswith("Missen"):
        return "Missen-Wilhams"
    m = CITY_RE.match(file_name) 
    return m.group(1) if m else None # leading token of letters/umlauts/hyphens at the start of the filename

#@st.cache_data(show_spinner=False, ttl=86400) cache disabled1
def get_city_coordinates(city: str) -> Optional[Tuple[float,float]]: # Usage: in preparing city points 
    """
    get coordinates for a city using pre-computed cache
    """
    # Normalize city name
    normalized = unicodedata.normalize("NFC", city)
    
    # Direct lookup in our cache
    coords = GERMAN_CITIES_COORDS.get(normalized)
    if coords:
        return coords
    
    # Fallback: try without special characters
    ascii_city = to_english_chars(normalized)
    if ascii_city and ascii_city != normalized:
        coords = GERMAN_CITIES_COORDS.get(ascii_city)
        if coords:
            return coords
        
    # If not found, return approx coords for Germany center
    return (51.1657, 10.4515) # Geo center of Germany

def prepare_city_points(results_df: pd.DataFrame): # Usage
    """
    Group results by city and compute the number of unique files per city.
    Returns a DataFrame with lat/lon, match count, and display labels.
    OPTIMIZED: Pre-computed coordinates instead of Nominatim API  callas
    """
    tmp = results_df.dropna(subset=["file"]).copy()
    tmp["city"] = tmp["file"].apply(parse_city)
    tmp = tmp.dropna(subset=["city"])
    
    # Count unique files per city
    counts = tmp.groupby("city", as_index=False).agg(matches=("file", "nunique"))
    
    
    # Get Coordinates from cache (no API calls)
    coords_data = []
    for city in counts["city"]:
        coords = get_city_coordinates(city)    
        if coords:
            coords_data.append({"city": city, "lat": coords[0], "lon": coords[1]})
            
    geocoded_df = pd.DataFrame(coords_data)
    df = counts.merge(geocoded_df, on="city", how="inner")

    # Bubble radius (log scale)
    df["radius"] = 1000 * (1 + df["matches"].apply(lambda x: np.log1p(x)))

    df["city_label"] = df.apply(lambda r: f"{r['city']} ({int(r['matches'])})", axis=1) 

    df["city_label_ascii"] = df["city_label"].apply(to_english_chars)

    return df

# MAP CONFIGURATIONS
def render_map(df_points: pd.DataFrame, top_n_labels: int = 200):
    """
    Renders a map with bubbles and labels.
    """
    if df_points.empty:
        st.info("No city data to display on map.")
        return
    
    center_lat = float(df_points["lat"].mean())
    center_lon = float(df_points["lon"].mean())
    view_state = pdk.ViewState(latitude=center_lat,longitude=center_lon, zoom=5.5,pitch=0)
    
    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=df_points,
        get_position="[lon, lat]",
        get_radius="radius",
        # RGBA - Red , Green, Blue, Alpha (opacity)
        get_fill_color="[200,30,0,160]"
    )
    
    # Show city labels only for the top-N cities by file count (Do we need that?)
    df_labelled = df_points.nlargest(top_n_labels, "matches")
    city_labels = pdk.Layer(
        "TextLayer",
        data=df_labelled,
        pickable=False,
        get_position = "[lon, lat]",
        get_text="city_label_ascii",
        get_size=10,
        get_text_anchor="'start'",
        get_aligment_baseline="'top'",
        get_color=[0,0,255,500],
        get_pixel_offset="[6, -6]",        
    )
    
    tooltip = {"html": "<b>{city}</b><br/>Files: {matches}", "style": {"color": "white"}}
    
    deck = pdk.Deck(
        initial_view_state=view_state,
        map_style="light", # we can use customize maps too
        layers=[scatter, city_labels],
        tooltip=tooltip,
    )
    st.pydeck_chart(deck)
    
# --------- Tokenizer -----------
# \w matches any word character (equivalent to [a-zA-Z0-9_])
# + matches the previous token between one and unlimited times
TOK_RE = re.compile(r"\w+", re.UNICODE)

@lru_cache(maxsize=1024)
# lowercase each word
def tokenize(s: str) -> Tuple[str, ...]: # Return tuple for hashability
    return tuple(TOK_RE.findall((s or "").lower()))

# generate all contiguous n-grams (up to nmax) from a token sequence
def iter_ngrams(tokens: Tuple[str, ...], nmax: int) -> Iterable[str]: # Where to use? # ngram sidebar: n-gram(2) = ["neu", "neu hause", "hause"]
    L = len(tokens)
    for n in range(1, nmax + 1):
        if L < n:
            break
        for i in range(0, L - n + 1):
            yield " ".join(tokens[i: i + n])
            
# -------- OPTIMIZED BACKENDS (PROGRAM'S CORE) -------
class JsonIndex:
    def __init__(self, root: Path):
        self.root = root
        with gzip.open(root / "postings.json.gz", "rt", encoding="utf-8") as f:
            self.postings: Dict[str, Dict[int, float]] = json.load(f)
        with gzip.open(root / "doclen.json.gz", "rt", encoding="utf-8") as f:
            self.doclen: Dict[int, int] = {int(k): int(v) for k, v in json.load(f).items()}
        meta = json.loads((root / "idf.json").read_text(encoding="utf-8"))
        self.N = int(meta["N"])
        self.avgdl = float(meta["avgdl"])
        self.idf: Dict[str, float] = {k: float(v) for k, v in meta["idf"].items()}
        self.docstore: Dict[int, dict] = {}
        with gzip.open(root / "docstore.jsonl.gz", "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    self.docstore[int(rec["doc_id"])] = rec
        self._file_text_cache: Dict[str, str] = {}
        # Pre-compute file-to-docs mapping for faster access #NEW FEATURE FOR CACHING
        self._file_docs_cache: Dict[str, List[dict]] = {}
        
    def get_postings(self, term: str) -> Dict[int, float]:
        return self.postings.get(term) or {}              
    def get_doc(self, doc_id:int) -> dict:
        return self.docstore.get(doc_id, {})
    def doc_length(self, doc_id: int) -> int:
        return self.doclen.get(doc_id, 0)
    @lru_cache(maxsize=512)
    def file_text(self, file_name: str) -> str:
        v = self._file_text_cache.get(file_name)
        if v is not None:
            return v
        joined = "\n".join(rec.get("text", "") for rec in self.docstore.values() if rec.get("file") == file_name)
        self._file_text_cache[file_name] = joined
        return joined
    def docs_for_file(self, file_name: str) -> List[dict]:
        if file_name in self._file_docs_cache:
            return self._file_docs_cache[file_name]

        def keyf(r):
            stt = r.get("start_time")
            return (float(stt) if stt is not None else float("inf"), r.get("doc_id", 0))

        docs = [rec for rec in sorted(self.docstore.values(), key=keyf) if rec.get("file") == file_name]
        self._file_docs_cache[file_name] = docs
        return docs

# Optimized Scoring & Snippets
@lru_cache(maxsize=2048)
def bm25_score(doc_id: int,q_terms_tuple: Tuple[str,...], k1: float =1.2, b: float = 0.75 ) -> float:
    """Placeholder (unused): Cached BM25 scoring function"""
    # This woul need the index passed in - simplified for demonstration
    # in practice, you'd pass the necessary data or use a class method
    return 0.0

def _build_plural_regex_piece(token: str) -> str:
    base = re.escape(token)
    if token.endswith("in"):
        suf = r"(?:nen|en|e|n|s|er)?"
    else:
        suf = r"(?:e|en|er|ern|es|n|s|ens)?"
    return f"{base}{suf}"

@lru_cache(maxsize=256)
def make_highlighter(tokens_tuple: Tuple[str, ...], plural_mode: bool) -> re.Pattern:
    base_terms = [t for t in tokens_tuple if t]
    if not base_terms:
        return re.compile(r"(?!x)x")
    if not plural_mode:
        pat = r"(\b(?:%s)\b)" % "|".join(sorted({re.escape(t) for t in base_terms}, key=len, reverse=True))
        return re.compile(pat, re.IGNORECASE | re.UNICODE)
    else:
        pieces = [_build_plural_regex_piece(t) for t in set(base_terms)]
        pat = r"(\b(?:%s)\b)" % "|".join(sorted(set(pieces), key=len, reverse=True))
        return re.compile(pat, re.IGNORECASE | re.UNICODE)
    
def find_spans(text: str, pattern: re.Pattern) -> List[Tuple[int, int]]:
    return [m.span() for m in pattern.finditer(text or "")]

SENT_END_CHARS = ".!?…\n"

def _prev_boundary(text:str, idx:int) -> int:
    '''find the index of the neares sentence-ending ch that appears before position idx in text.'''
    best = -1
    for ch in SENT_END_CHARS:
        pos = text.rfind(ch, 0, idx)
        if pos > best:
            best = pos
    return best

def _next_boundary(text: str, idx: int) -> int:
    best = len(text) - 1
    found = False
    for ch in SENT_END_CHARS:
        pos = text.find(ch, idx)
        if pos != -1:
            best = min(best, pos)
            found = True
    return best if found else (len(text) - 1)

def _window_for_match(text: str, start: int, end: int, extra_sentences: int=0) -> Tuple[int, int]:
    a = _prev_boundary(text, start)
    b = _next_boundary(text, end)
    a = 0 if a == -1 else a + 1
    b = min(len(text), b + 1)
    for _ in range(extra_sentences):
        pa = _prev_boundary(text, a)
        if pa != -1:
            a = pa + 1
        nb = _next_boundary(text, b)
        if nb < len(text) - 1:
            b = min(len(text), nb + 1)
    while a > 0 and not text[a].isspace():
        a -= 1
    while b < len(text) and not text[b - 1].isspace():
        b += 1
        if b >= len(text):
            b = len(text)
            break
    return (max(0,a), min(len(text), b))

def _merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    ranges.sort()
    merged = [ranges[0]]
    for a, b in ranges[1:]:
        la, lb = merged[-1]
        if a <= lb + 1:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged

@lru_cache(maxsize=128)
def make_all_snippets_sentence_complete(text: str, pattern_str: str, extra_sentences: int =0) -> Tuple[str, ...]:
    """Cached snippet generation - pattern_str is the regex pattern as string for caching"""
    if not text:
        return tuple()
    pattern = re.compile(pattern_str, re.IGNORECASE | re.UNICODE)
    spans = find_spans(text, pattern)
    if not spans: 
        return tuple()
    
    windows = [_window_for_match(text, s,e ,extra_sentences=extra_sentences) for (s, e) in spans]
    merged = _merge_ranges(windows)
    out = []
    for a, b in merged:
        frag = text[a:b]
        frag = pattern.sub(r"**\1**", frag)
        prefix = "… " if a > 0 else ""
        suffix = " …" if b < len(text) else ""
        out.append(prefix + frag + suffix)
    return tuple(out)

@lru_cache(maxsize=512)
def count_occurrences(text: str, pattern_str: str) -> int:
    """cached occurence counting"""
    pattern = re.compile(pattern_str, re.IGNORECASE | re.UNICODE)
    return len(find_spans(text, pattern))

def _sec_to_hhmmss(val) -> str:
    """Convert seconds (float/str) to HH:MM:SS"""
    secs = int(round(float(val)))
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def _fmt_time(val) -> str:
    """Format time values as HH:MM:SS (or empty if no value)"""
    if val is None or val == "":
        return ""
    try:
        return _sec_to_hhmmss(val)
    except Exception:
        return str(val)
    
def _fmt_timeline_range(start_id: str, end_id: str, start_time, end_time) -> str:
    sid = start_id or "?"
    eid = end_id or ""
    id_part = f"{sid}–{eid}" if eid and eid != sid else sid

    stime = _fmt_time(start_time)
    etime = _fmt_time(end_time)
    if stime or etime:
        if etime and etime != stime:
            time_part = f"{stime}–{etime}" if stime else etime
        else:
            time_part = stime or etime
        return f" (id: {id_part}, time: {time_part})"
    else:
        return f" (id: {id_part})"
    
# -------- Query Expansion for plural mode ----------
GERMAN_PLURAL_SUFFIXES = ("e", "en", "er", "ern", "es", "n", "s", "ens")

@lru_cache(maxsize=256)
def expand_variants(token: str) -> frozenset[str]:
    forms: Set[str] = {token}
    for suf in GERMAN_PLURAL_SUFFIXES:
        forms.add(token + suf)
    if token.endswith("in"):
        forms.add(token + "nen")
    return frozenset(forms)

# --------------- Args & setup ---------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--index",default="index_unique", help="Index dir (JSON) or .db (SQLite)")
    ap.add_argument("--ngrams", type=int, default=1)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--extra-sentences", type=int, default=0, help="Include this many extra sentences on each side of a match in overview snippets.")
    return ap.parse_args()

st.set_page_config(page_title="EXMARaLDA Search", layout="wide")

@st.cache_resource(show_spinner=False)
def load_index(index_path: str):
    p = Path(index_path)
    idx = JsonIndex(p)
    backend = "json"
    return {
        "backend": backend,
        "postings_fn": idx.get_postings,
        "idf_fn": lambda t: idx.idf.get(t, 0.0),
        "doclen_fn": idx.doc_length,
        "get_doc": idx.get_doc,
        "get_file_text": idx.file_text,
        "get_docs_for_file": idx.docs_for_file,
        "N": idx.N, "avgdl": idx.avgdl,
    }
    
def _build_context_bundles(events: List[dict], hilite: re.Pattern, extra: int) -> List[str]:
    """Return formatted bundles. Each bundle is a block of lines."""
    bundles = []
    windows_seen = set()
    hit_idxs = [i for i, ev in enumerate(events) if hilite.search(ev.get("text", "") or "")]

    for i in hit_idxs:
        start = max(0, i - extra)
        end = min(len(events) - 1, i + extra)
        win_key = (start, end)
        if win_key in windows_seen:
            continue
        windows_seen.add(win_key)

        lines = []
        for j in range(start, end + 1):
            ev = events[j]
            txt = (ev.get("text", "") or "").strip()
            if not txt:
                continue
            #txt = hilite.sub(r"**\1**", txt)
            safe_txt = _colorize_html(txt, hilite)
            sid = ev.get("start_id") or ev.get("start") or "?"
            eid = ev.get("end_id") or ev.get("end") or ""
            stime = ev.get("start_time")
            etime = ev.get("end_time")
            #suffix = _fmt_timeline_range(sid, eid, stime, etime)
            suffix_html = html.escape(_fmt_timeline_range(sid, eid, stime, etime))
            lines.append(f"- {safe_txt}{suffix_html}")
        if lines:
            bundles.append("\n".join(lines))
    return bundles

def run_ui():
    args = parse_args()
    
    # Progress indicator
    with st.spinner("Loading index..."):
        idx = load_index(args.index)
    
    # Title of the page
    st.title("Exmaralda Archive Search")
    
    # style css for highlighting (yellow background red foreground)
    st.markdown("""
        <style>
        .hl {
        background: yellow;   /*  yellow */
        color: red;        /* deep red text */
        padding: 0 .15em;
        border-radius: .2em;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Add cleanup button for debugging
    if st.sidebar.button("🔄 Clear Cache & Restart", help="Clear all caches and restart the session (good for debugging)"):
        # Clear all caches
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
    
    with st.sidebar:
        st.subheader("Options - Configurations")
        ngrams = st.slider("Max n-gram size", 1, 8, args.ngrams, help="Number of words to search together.")
        topk = st.slider("Top K files", 5, 690, args.topk, help="How many top results to show (max 690)")
        extra = st.slider("Extra context (event lines)", 0, 10, 0, help="Context provider")
        plural_mode = st.checkbox("Include plural/inflected forms", value=False, help="Match plural/inflected variants of words")
        exact_phrase = st.checkbox("Match whole phrase only", value=False, help="Treat multi-word queries as an exact phrase (ignore individual words)")
        
    q = st.text_input("Query", placeholder="e.g., arbeit bauer feldarbeit").strip()
    if not q:
        st.stop()
        
    # Progress indicator for search
    with st.spinner("Searching..."):
        q_tokens = tokenize(q)
        # If "Exact phrase" is enabled and the query has multiple tokens,
        # treat the entire query as a single term for both retrieval and highlighting.
        if 'exact_phrase' in locals() and exact_phrase and len(q_tokens) > 1:
            phrase_term = " ".join(q_tokens)
            hilite = make_highlighter((phrase_term,), plural_mode=False)  # ignore plural_mode for exact phrases
            q_terms = [phrase_term]
        else:
            hilite = make_highlighter(q_tokens, plural_mode=plural_mode)
            # For retrieval/scoring, combine original n-grams + (optional) expanded 1-gram variants
            q_terms = list(iter_ngrams(q_tokens, min(ngrams, max(1, len(q_tokens)))))
            if plural_mode:
                expanded: Set[str] = set()
                for t in q_tokens:
                    expanded.update(expand_variants(t))
                q_terms = list(set(q_terms) | expanded)

        postings_fn = idx["postings_fn"]
        idf_fn = idx["idf_fn"]
        doclen_fn = idx["doclen_fn"]
        get_doc = idx["get_doc"]
        get_file_text = idx["get_file_text"]
        get_docs_for_file = idx["get_docs_for_file"]
        N = idx["N"]
        avgdl = idx["avgdl"]
                
        if not q_terms:
            st.warning("Empty query.")
            st.stop()        
        
        # ------- Canditate  collection + local postings cache ------
        postings_cache: Dict[str, Dict[int, float]] = {}
        candidates: set[int] = set()
        for t in q_terms:
            p = postings_fn(t) or {}
            if p:
                postings_cache[t] = p
                candidates.update(p.keys())
        if not candidates:
            st.warning("No matches.")
            st.stop()
            
        # ----------- Safe IDF fallback (fixes No files to show) --------
        def safe_idf(term: str) -> float:
            """Use precomputed IDF if available; otherwise compute BM25 IDF from DF."""
            idf_val = idf_fn(term)
            if idf_val and idf_val > 0.0:
                return idf_val
            p = postings_cache.get(term, {})
            df = len(p)
            if df <= 0 or N <= 0:
                # Fallback constant so matches still score > 0
                return 1.0
            # BM25 idf = ln( (N - df + 0.5) / (df + 0.5) + 1 )
            return float(np.log(1.0 + (N - df + 0.5) / (df + 0.5)))
        
        # -------- Scoring with early termination --------
        best_per_file: Dict[str, Tuple[float, int]] = {}

        # Process all candidates to ensure we don't miss any files
        for did in candidates:
            # Calculate BM25 score
            dl = doclen_fn(did) or 1
            norm = 1.2 * (1.0 - 0.75 + 0.75 * (dl / (avgdl or 1.0)))
            score = 0.0
            # Only iterate over terms that have postings
            for t in postings_cache.keys():
                p = postings_cache[t]
                tf = p.get(did)
                if not tf:
                    continue
                idf = safe_idf(t)
                score += idf * ((1.2 + 1.0) * tf) / (tf + norm)
                
            if score > 0:  # Only process documents with positive scores
                doc = get_doc(int(did))
                fn = doc.get("file")
                if not fn:
                    continue
                prev = best_per_file.get(fn)
                if (prev is None) or (score > prev[0]):
                    best_per_file[fn] = (score, did)
        if not best_per_file:
            st.warning("No files to show.")
            st.stop()        
            
        # Get top files
        top_files = heapq.nlargest(topk, best_per_file.items(), key=lambda kv: kv[1][0])
    # Process results
    with st.spinner("Processing results..."):
        rows = []
        all_ud_keys: set[str] = set()
        pattern_str = hilite.pattern  # Cache the pattern string for reuse

        for file_name, (score, did) in top_files:
            doc = get_doc(int(did))
            full_text = get_file_text(file_name)

            # Use cached functions
            frequency = count_occurrences(full_text, pattern_str)
            snippets = make_all_snippets_sentence_complete(full_text, pattern_str, extra_sentences=0)
            snippet_field = "\n\n".join([f"{i+1}. {s}" for i, s in enumerate(snippets)]) if snippets else ""

            kw = doc.get("keywords") or ""
            keywords_display = hilite.sub(r"**\1**", kw) if (kw and hilite.search(kw)) else kw

            ref_url = doc.get("reference") or ""

            row = {
                "doc_id": did,
                "score": round(score, 4),
                "file": file_name,
                "speaker": doc.get("speaker"),
                "keywords": keywords_display,
                "snippet": snippet_field,
                "frequency": frequency,
                "reference-file-name": ref_url,
            }
            ud = doc.get("udmeta_kv") or {}
            for k, v in ud.items():
                col_name = f"{k}".capitalize()
                row[col_name] = v
                all_ud_keys.add(col_name)

            rows.append(row)
    st.caption(f"Showing {len(rows)} file(s).")
    st.dataframe(rows, use_container_width=True)
    
    # Map visualization
    with st.spinner("Generating map..."):
        results_df = pd.DataFrame(rows)
        df_points = prepare_city_points(results_df)
        
        st.subheader("Map of matched cities")
        render_map(df_points)
        
        with st.expander("City counts", expanded=False):
            st.dataframe(df_points[["city", "matches"]].sort_values("matches", ascending=False), hide_index=True)
    st.markdown("""
---
### Detailed snippets
Event-level context bundles. Each line has its own timeline.
""")
    
    # Lazy load detailed snippets only when expanded
    for row in rows:
        file_name = row["file"]
        events = get_docs_for_file(file_name)
        bundles = _build_context_bundles(events, hilite, extra)
        if not bundles:
            continue # do not render any expander
        
        with st.expander(f"{file_name} — {len(bundles)} bundle(s), {row['frequency']} total hit(s)", expanded=False):
            if row.get("reference-file-name"):
                st.markdown(f"**Reference** {row['reference-file-name']}")
                for i, block in enumerate(bundles, 1):
                    st.markdown(f"**Bundle {i}**:")
                    st.markdown(block, unsafe_allow_html=True)
if __name__ == "__main__":
    run_ui()