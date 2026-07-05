// Ask Space Weather — a small retrieval-augmented-generation (RAG) assistant.
// Questions are embedded with Gemini, matched against a curated, pre-embedded space-weather
// knowledge base by cosine similarity (in-Worker), and answered by Gemini grounded ONLY in the
// retrieved passages, with citations. No vector DB needed — the corpus ships with the Worker.
import { CORPUS, META } from "./corpus.js";

const EMB = "gemini-embedding-001";
// Try these in order; fall through on overload (503/429/5xx) so a busy model doesn't break the app.
const GEN_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash-001"];
const TOP_K = 4;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function embedQuery(q, key) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${EMB}:embedContent?key=${key}`;
  const r = await fetch(url, { method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ model: `models/${EMB}`, content: { parts: [{ text: q }] }, taskType: "RETRIEVAL_QUERY", outputDimensionality: META.dim }) });
  if (!r.ok) throw new Error("embed failed: " + r.status);
  const v = (await r.json()).embedding.values;
  const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
  return v.map((x) => x / n);
}

function topMatches(qvec, k) {
  return CORPUS.map((c, i) => ({ i, score: c.vec.reduce((s, x, j) => s + x * qvec[j], 0) }))
    .sort((a, b) => b.score - a.score).slice(0, k).map((m) => ({ ...CORPUS[m.i], score: m.score }));
}

async function generate(question, matches, key) {
  const context = matches.map((m, i) => `[${i + 1}] ${m.title} (source: ${m.source})\n${m.text}`).join("\n\n");
  const prompt = `You are a careful space-weather assistant. Answer the question using ONLY the numbered context below. Cite the passages you use inline like [1] or [2]. If the answer is not in the context — for example a real-time forecast like "the next solar storm" — say plainly that you can't answer that from this knowledge base (and, if relevant, point to what the live dashboard or NOAA provides) rather than guessing. Keep it to a short, clear paragraph.\n\nContext:\n${context}\n\nQuestion: ${question}\n\nAnswer:`;
  const body = JSON.stringify({ contents: [{ parts: [{ text: prompt }] }], generationConfig: { temperature: 0.2, maxOutputTokens: 900 } });
  let lastErr = "";
  for (const model of GEN_MODELS) {
    for (let attempt = 0; attempt < 2; attempt++) {
      let r;
      try {
        r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${key}`,
          { method: "POST", headers: { "content-type": "application/json" }, body });
      } catch (e) { lastErr = `${model}: ${e.message}`; await sleep(300); continue; }
      if (r.ok) {
        const data = await r.json();
        const parts = ((data.candidates || [])[0]?.content?.parts) || [];
        const text = parts.map((p) => p.text || "").join("").trim();
        if (text) return { answer: text, model };
        lastErr = `${model}: empty`; break;
      }
      lastErr = `${model}: ${r.status}`;
      if (r.status === 429 || r.status >= 500) { await sleep(400 * (attempt + 1)); continue; } // transient → retry
      break; // hard error (e.g. 400) → next model
    }
  }
  throw new Error(lastErr || "all models unavailable");
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const FAVICON_SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64"><rect width="64" height="64" rx="14" fill="#F59E0B"/><circle cx="32" cy="32" r="12" fill="#fff"/></svg>`;
    if (url.pathname === "/favicon.svg" || url.pathname === "/favicon.ico") return new Response(FAVICON_SVG, { headers: { "content-type": "image/svg+xml", "cache-control": "public, max-age=86400" } });
    if (url.pathname === "/robots.txt") return new Response("User-agent: *\nAllow: /\nSitemap: https://ask.dsremo.com/sitemap.xml\n", { headers: { "content-type": "text/plain; charset=utf-8" } });
    if (url.pathname === "/sitemap.xml") return new Response('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n<url><loc>https://ask.dsremo.com/</loc><changefreq>weekly</changefreq><priority>1.0</priority></url>\n</urlset>\n', { headers: { "content-type": "application/xml; charset=utf-8" } });
    try {
      if (url.pathname === "/api/ask" && request.method === "POST") {
        if (!env.GEMINI_API_KEY) return json({ error: "model not configured" }, 503);
        const { question } = await request.json();
        if (!question || !question.trim()) return json({ error: "empty question" }, 400);
        const qvec = await embedQuery(question.trim(), env.GEMINI_API_KEY);
        const matches = topMatches(qvec, TOP_K);
        const sources = matches.map((m) => ({ title: m.title, source: m.source, score: Number(m.score.toFixed(3)) }));
        try {
          const { answer, model } = await generate(question.trim(), matches, env.GEMINI_API_KEY);
          return json({ answer, model, sources });
        } catch (e) {
          // Retrieval still succeeded — return the sources plus a friendly note so the UI never shows a raw 503.
          return json({ busy: true, answer: "", sources,
            note: "The answer model is briefly overloaded — your top sources are below. Please tap Ask again in a few seconds." }, 200);
        }
      }
      if (url.pathname === "/" || url.pathname === "") return html(page());
      return new Response("Not found", { status: 404 });
    } catch (err) { return json({ error: String(err && err.message) }, 500); }
  },
};

function page() {
  const examples = [
    "What is the Kp index and when does a geomagnetic storm start?",
    "How much warning do we get before a CME hits Earth?",
    "How can space weather damage the power grid?",
    "What was the Carrington Event?",
  ];
  return `<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ask Space Weather — a cited RAG assistant</title>
<meta name="description" content="Ask Space Weather — a cited RAG assistant answering solar-wind and space-weather questions with sources.">
<link rel="canonical" href="https://ask.dsremo.com/">
<meta name="robots" content="index, follow">
<meta name="theme-color" content="#F59E0B">
<meta property="og:type" content="website">
<meta property="og:site_name" content="Ask Space Weather">
<meta property="og:title" content="Ask Space Weather — a cited RAG assistant">
<meta property="og:description" content="A cited RAG assistant answering solar-wind and space-weather questions with sources.">
<meta property="og:url" content="https://ask.dsremo.com/">
<meta property="og:image" content="https://ask.dsremo.com/favicon.svg">
<meta name="twitter:card" content="summary">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="icon" href="/favicon.ico" sizes="any">
<link rel="apple-touch-icon" href="/favicon.svg">
<style>
  :root{--bg:#0a0e17;--panel:#121826;--line:#1f2937;--ink:#e7ecf3;--muted:#8b96a8;--accent:#a78bfa;--accent2:#6cb6ff;}
  *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;line-height:1.6}
  .wrap{max-width:760px;margin:0 auto;padding:36px 20px 64px}
  h1{font-size:clamp(26px,4vw,38px);margin:0 0 4px;letter-spacing:-.02em}
  .sub{color:var(--muted);margin:0 0 24px;font-size:15px}
  .ask{display:flex;gap:10px;flex-wrap:wrap}
  input{flex:1;min-width:240px;background:var(--panel);border:1px solid var(--line);color:var(--ink);border-radius:10px;padding:13px 15px;font-size:15px}
  input:focus{outline:2px solid var(--accent);outline-offset:1px}
  button{background:var(--accent);color:#10081f;border:0;border-radius:10px;padding:13px 22px;font-weight:700;font-size:15px;cursor:pointer}
  button:disabled{opacity:.5;cursor:default}
  .ex{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}
  .chip{font-size:13px;color:var(--muted);background:var(--panel);border:1px solid var(--line);border-radius:999px;padding:7px 13px;cursor:pointer}
  .chip:hover{color:var(--ink);border-color:var(--accent)}
  .answer{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:22px;margin-top:24px;display:none}
  .answer.show{display:block}
  .answer .a{font-size:16px;white-space:pre-wrap}
  .sources{margin-top:18px;border-top:1px solid var(--line);padding-top:14px}
  .sources .k{font-family:ui-monospace,Menlo,monospace;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
  .src{font-size:13.5px;color:var(--muted);margin-top:8px}
  .src b{color:var(--ink)}
  .spin{color:var(--muted);font-size:14px}
  footer{margin-top:34px;color:var(--muted);font-size:12.5px;font-family:ui-monospace,Menlo,monospace}
  a{color:var(--accent2);text-decoration:none}a:hover{text-decoration:underline}
</style></head><body>
<div class="wrap">
  <h1>Ask Space Weather</h1>
  <p class="sub">A retrieval-augmented assistant. It answers from a curated knowledge base of ${META.count} space-weather passages — and cites them — instead of guessing.</p>
  <div class="ask">
    <input id="q" placeholder="Ask about solar storms, auroras, the Kp index…" autocomplete="off">
    <button id="go">Ask</button>
  </div>
  <div class="ex" id="ex">${examples.map((e) => `<span class="chip">${e}</span>`).join("")}</div>
  <div class="answer" id="answer"><div class="a" id="a"></div><div class="sources" id="sources"></div></div>
  <footer>RAG: Gemini embeddings + in-Worker cosine search + ${"gemini-2.5-flash"}, grounded in a curated KB · built by Ashutosh Tiwari · <a href="https://spaceweather.dsremo.com">live dashboard</a></footer>
</div>
<script>
  const q=document.getElementById("q"), go=document.getElementById("go"), ans=document.getElementById("answer"), aEl=document.getElementById("a"), srcEl=document.getElementById("sources");
  async function ask(){
    const question=q.value.trim(); if(!question) return;
    go.disabled=true; ans.classList.add("show"); aEl.innerHTML='<span class="spin">Searching the knowledge base…</span>'; srcEl.innerHTML="";
    try{
      const r=await fetch("/api/ask",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({question})});
      const d=await r.json();
      const renderSources=(arr)=> arr&&arr.length ? '<div class="k">Sources</div>'+arr.map((s,i)=>'<div class="src">['+(i+1)+'] <b>'+s.title+'</b> — '+s.source+' <span style="opacity:.6">· match '+s.score+'</span></div>').join("") : "";
      if(d.error){ aEl.textContent="Couldn't reach the model — please try again."; }
      else if(d.busy){ aEl.textContent=d.note; srcEl.innerHTML=renderSources(d.sources); }
      else{
        aEl.textContent=d.answer + (d.model? "" : "");
        srcEl.innerHTML=renderSources(d.sources)+(d.model?'<div class="src" style="margin-top:10px;opacity:.6">answered by '+d.model+'</div>':'');
      }
    }catch(e){ aEl.textContent="Error: "+e.message; }
    go.disabled=false;
  }
  go.addEventListener("click",ask);
  q.addEventListener("keydown",e=>{if(e.key==="Enter")ask();});
  document.getElementById("ex").addEventListener("click",e=>{if(e.target.classList.contains("chip")){q.value=e.target.textContent;ask();}});
</script>
</body></html>`;
}
function json(o, status=200){ return new Response(JSON.stringify(o,null,2),{status,headers:{"content-type":"application/json","cache-control":"no-store"}}); }
function html(b){ return new Response(b,{headers:{"content-type":"text/html; charset=utf-8","cache-control":"public, max-age=300"}}); }
