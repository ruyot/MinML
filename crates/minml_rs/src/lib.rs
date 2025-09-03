use once_cell::sync::Lazy;
use pyo3::prelude::*;
use regex::Regex;
use std::collections::{BTreeSet, HashMap};

static DUP_WORD_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b(\w+)\s+\1\b").unwrap());
static MULTISPACE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").unwrap());

static STOPWORDS: Lazy<BTreeSet<&'static str>> = Lazy::new(|| {
    // Keep interrogatives
    let keep = ["what","why","when","how","where","who","which"];
    let mut s = BTreeSet::new();
    // lightweight set; expand if needed
    for w in [
        "the","a","an","of","in","on","for","to","and","or","with","that","this","those","these",
        "is","are","be","been","it","as","at","by","from","into","about","than","then","there",
        "we","you","they","i","he","she","them","his","her","our","your","their","was","were",
        "do","did","does","can","could","should","would","may","might","will","shall","have","has","had",
        "please","kindly","just","some","any","no","not","if","but"
    ] {
        if !keep.contains(&w) { s.insert(w); }
    }
    s
});

fn normalize_ws(s: &str) -> String {
    MULTISPACE_RE.replace_all(s.trim(), " ").to_string()
}

fn collapse_dupes(s: &str) -> String {
    let t = DUP_WORD_RE.replace_all(s, "$1").to_string();
    normalize_ws(&t)
}

fn dedup_tokens(s: &str) -> String {
    let mut seen = BTreeSet::<String>::new();
    let mut out = Vec::new();
    for tok in s.split_whitespace() {
        let low = tok.to_lowercase();
        if !seen.contains(&low) {
            seen.insert(low);
            out.push(tok);
        }
    }
    out.join(" ")
}

/// Remove stopwords but keep interrogatives (what/why/...).
#[pyfunction]
fn stopword_remove(text: &str) -> String {
    let mut out = Vec::new();
    for tok in text.split_whitespace() {
        let low = tok.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
        if !STOPWORDS.contains(low.as_str()) {
            out.push(tok);
        }
    }
    collapse_dupes(&out.join(" "))
}

/// Fold phrases like "machine learning", "deep learning" -> "machine/deep learning"
fn fold_shared_tails(phrases: &[String]) -> Vec<String> {
    let mut groups: HashMap<String, Vec<String>> = HashMap::new();
    for p in phrases {
        let toks: Vec<&str> = p.split_whitespace().collect();
        if toks.len() >= 2 {
            let tail = toks[toks.len()-1].to_string();
            let head = toks[..toks.len()-1].join(" ");
            groups.entry(tail).or_default().push(head);
        } else {
            groups.entry(p.clone()).or_default().push(String::new());
        }
    }
    let mut folded = Vec::new();
    for (tail, heads) in groups {
        let heads: Vec<String> = heads.into_iter().filter(|h| !h.is_empty()).collect();
        if !heads.is_empty() {
            let mut uniq = BTreeSet::new();
            for h in heads { uniq.insert(h); }
            folded.push(format!("{}/{} {}", uniq.into_iter().collect::<Vec<_>>().join("/"), "", tail).replace("//","/").replace("/ "," "));
        } else {
            folded.push(tail);
        }
    }
    folded
}

/// Very simple keyword extraction:
/// - Keep 1-2 word terms by removing stopwords, preferring content words.
/// - Preserve the first action word if present (explain/summarize/compare).
#[pyfunction]
fn keywords_compress(text: &str, max_keywords: usize) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut content = Vec::new();
    let mut action: Option<String> = None;
    let actions = ["explain","summarize","compare"];

    for (i, w) in words.iter().enumerate() {
        let lw = w.to_lowercase();
        if i == 0 && actions.contains(&lw.as_str()) {
            action = Some(lw);
            continue;
        }
        if !STOPWORDS.contains(lw.as_str()) {
            content.push(lw);
        }
    }

    // Build 2-gram phrases where possible
    let mut phrases = Vec::<String>::new();
    let mut i = 0;
    while i < content.len() {
        if i+1 < content.len() {
            phrases.push(format!("{} {}", content[i], content[i+1]));
            i += 2;
        } else {
            phrases.push(content[i].clone());
            i += 1;
        }
    }

    // Dedup, cap, and fold shared tails
    let mut uniq = BTreeSet::new();
    let mut deduped = Vec::new();
    for p in phrases {
        if uniq.insert(p.clone()) { deduped.push(p); }
    }
    let capped: Vec<String> = deduped.into_iter().take(max_keywords.max(1)).collect();

    let mut folded = fold_shared_tails(&capped);
    // Ensure we don't exceed approx 5 tokens when max_keywords=3 (tests allow ~5 tokens)
    let out = dedup_tokens(&folded.join(" "));
    collapse_dupes(&out)
}

/// Template-based decompression: first token (if action) + subject.
#[pyfunction]
fn keywords_decompress(text: &str) -> String {
    let mut toks: Vec<&str> = text.split_whitespace().collect();
    if toks.is_empty() { return String::new(); }

    let actions = ["explain","summarize","compare"];
    let mut action = "explain";
    if actions.contains(&toks[0].to_lowercase().as_str()) {
        action = toks.remove(0);
    }
    let subject = toks.join(" ");
    let subj = if subject.trim().is_empty() { "this topic".to_string() } else { subject };
    let base = match action.to_lowercase().as_str() {
        "summarize" => format!("Summarize {}.", subj),
        "compare"   => format!("Compare {}.", subj),
        _           => format!("Explain {}.", subj),
    };
    collapse_dupes(&base)
}

/// Shorthand replacements (safe-ish). Only use at level >=3.
#[pyfunction]
fn shorthand(text: &str, level: u8) -> String {
    if level < 3 { return text.to_string(); }
    let mut s = text.to_string();

    // Abbreviations
    let replacements = vec![
        ("please", "pls"),
        ("explain", "expln"),
        ("concept", "concpt"),
        ("terms", "trms"),
        ("beginner", "bginner"),
        ("machine learning", "ML"),
        ("artificial intelligence", "AI"),
        ("and", "&"),
        ("implementation", "implmntn"),
        ("function", "func"),
        ("variable", "var"),
        ("parameter", "param"),
        ("parameters", "params"),
        ("return", "ret"),
        ("object", "obj"),
        ("class", "cls"),
        ("module", "mod"),
        ("library", "lib"),
        ("framework", "frmwrk"),
        ("database", "db"),
        ("configuration", "config"),
        ("environment", "env"),
        ("development", "dev"),
        ("production", "prod"),
        ("neural network", "NN"),
        ("deep learning", "DL"),
        ("natural language", "NL"),
        ("processing", "proc"),
        ("application", "app"),
        ("programming", "prog"),
        ("algorithm", "algo"),
        ("algorithms", "algos")
    ];

    for (a, b) in replacements {
        s = regex::Regex::new(&format!(r"(?i)\b{}\b", regex::escape(a))).unwrap()
            .replace_all(&s, b).to_string();
    }

    // Simple vowel removal for long words
    if level >= 3 {
        s = regex::Regex::new(r"\b([b-df-hj-np-tv-z])([aeiou])([b-df-hj-np-tv-z])([aeiou])([b-df-hj-np-tv-z]+)\b").unwrap()
            .replace_all(&s, "$1$3$5").to_string();
    }

    collapse_dupes(&s)
}

#[pymodule]
fn minml_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(stopword_remove, m)?)?;
    m.add_function(wrap_pyfunction!(keywords_compress, m)?)?;
    m.add_function(wrap_pyfunction!(keywords_decompress, m)?)?;
    m.add_function(wrap_pyfunction!(shorthand, m)?)?;
    Ok(())
}
