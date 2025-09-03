from __future__ import annotations
import re

try:
    import minml_rs as _rs
    RS_OK = True
except Exception:
    RS_OK = False

_DUP_WORD_RE = re.compile(r"\b(\w+)\s+\1\b", re.I)

def _collapse_dupes(s: str) -> str:
    return _DUP_WORD_RE.sub(r"\1", s).strip()

def stopword_remove(text: str) -> str:
    if RS_OK:
        return _rs.stopword_remove(text)
    # simple fallback: keep interrogatives
    keep = {"what","why","when","how","where","who","which"}
    stop = {"the","a","an","of","in","on","for","to","and","or","with","that","this","those","these",
            "is","are","be","been","it","as","at","by","from","into","about","than","then","there",
            "we","you","they","i","he","she","them","his","her","our","your","their","was","were",
            "do","did","does","can","could","should","would","may","might","will","shall","have","has","had",
            "please","kindly","just","some","any","no","not","if","but"}
    out = []
    for tok in text.split():
        low = re.sub(r"^\W+|\W+$", "", tok.lower())
        if low in keep or low not in stop:
            out.append(tok)
    return _collapse_dupes(" ".join(out))

def keywords_compress(text: str, max_keywords: int = 3) -> str:
    if RS_OK:
        return _rs.keywords_compress(text, int(max_keywords))
    # fallback heuristic similar to Rust path
    words = text.split()
    actions = {"explain","summarize","compare"}
    stop_words = {"the","a","an","of","in","on","for","to","and","or","with","that","this","those","these",
                  "is","are","be","been","it","as","at","by","from","into","about","than","then","there",
                  "we","you","they","i","he","she","them","his","her","our","your","their","was","were",
                  "do","did","does","can","could","should","would","may","might","will","shall","have","has","had",
                  "please","kindly","just","some","any","no","not","if","but"}
    content = []
    for i, w in enumerate(words):
        lw = re.sub(r"^\W+|\W+$", "", w.lower())
        if i==0 and lw in actions:
            continue
        if lw and lw not in stop_words:
            content.append(lw)
    # naive bigrams
    phrases = []
    i = 0
    while i < len(content):
        if i+1 < len(content):
            phrases.append(f"{content[i]} {content[i+1]}")
            i += 2
        else:
            phrases.append(content[i]); i += 1
    # cap + fold tails
    phrases = list(dict.fromkeys(phrases))[:max(1, int(max_keywords))]
    # simple fold: "x y", "z y" -> "x/z y"
    tails = {}
    for p in phrases:
        toks = p.split()
        if len(toks)>=2:
            head, tail = " ".join(toks[:-1]), toks[-1]
            tails.setdefault(tail, []).append(head)
        else:
            tails.setdefault(p, []).append("")
    folded = []
    for tail, heads in tails.items():
        heads = [h for h in heads if h]
        folded.append(f"{'/'.join(dict.fromkeys(heads))} {tail}".strip("/ ").strip())
    out = " ".join(folded)
    return _collapse_dupes(re.sub(r"\s+", " ", out).strip())

def keywords_decompress(text: str) -> str:
    if RS_OK:
        return _rs.keywords_decompress(text)
    toks = text.split()
    if not toks: return ""
    actions = {"explain":"Explain","summarize":"Summarize","compare":"Compare"}
    act = toks[0].lower() if toks[0].lower() in actions else "explain"
    subject = " ".join(toks[1:] if toks[0].lower() in actions else toks).strip() or "this topic"
    return _collapse_dupes(f"{actions[act]} {subject}.")

def shorthand(text: str, level: int = 3) -> str:
    if RS_OK:
        return _rs.shorthand(text, level)
    # fallback
    if level < 3:
        return text
    s = text
    replacements = {
        "please": "pls",
        "explain": "expln",
        "concept": "concpt",
        "terms": "trms",
        "beginner": "bginner",
        "machine learning": "ML",
        "artificial intelligence": "AI",
        "and": "&",
        "implementation": "implmntn",
        "function": "func",
        "variable": "var",
        "parameter": "param",
        "parameters": "params",
        "return": "ret",
        "object": "obj",
        "class": "cls",
        "module": "mod",
        "library": "lib",
        "framework": "frmwrk",
        "database": "db",
        "configuration": "config",
        "environment": "env",
        "development": "dev",
        "production": "prod",
        "neural network": "NN",
        "deep learning": "DL",
        "natural language": "NL",
        "processing": "proc",
        "application": "app",
        "programming": "prog",
        "algorithm": "algo",
        "algorithms": "algos"
    }
    for a, b in replacements.items():
        s = re.sub(rf"\b{re.escape(a)}\b", b, s, flags=re.I)

    # Simple vowel removal for long words
    if level >= 3:
        # Remove alternating vowels from long words
        s = re.sub(r"\b([b-df-hj-np-tv-z])([aeiou])([b-df-hj-np-tv-z])([aeiou])([b-df-hj-np-tv-z]+)\b",
                   r"\1\3\5", s, flags=re.I)

    return _collapse_dupes(s)
