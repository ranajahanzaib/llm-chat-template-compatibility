import { useEffect, useState } from "react";
type Finding = {
  code: string;
  severity: string;
  evidence: string;
  interpretation: string;
  suggested_action: string;
};
type Prompt = {
  source_runtime: string;
  rendered_text: string;
  token_ids: number[];
  token_pieces: string[];
  special_token_positions: number[];
  provenance: string;
};
type TokenOp = {
  tag: string;
  a_start: number;
  a_end: number;
  b_start: number;
  b_end: number;
  a: number[];
  b: number[];
};
type Experiment = {
  experiment_id: string;
  synthetic: boolean;
  mode: string;
  compatibility: string;
  conclusion: string;
  prompts: Record<string, Prompt>;
  token_comparison: {
    token_ids_semantically_comparable?: boolean;
    token_diff?: {
      operations: TokenOp[];
      first_divergence?: { a: number; b: number } | null;
    };
    crossover_matrix?: Record<
      string,
      { status: string; reason?: string; cells?: Record<string, string> }
    >;
  };
  diagnostic_findings: Finding[];
  generations: Record<
    string,
    {
      output_text: string;
      finish_reason: string | null;
      effective_generation_settings: Record<string, unknown>;
    }
  >;
};
type Run = { id: string };
export default function App() {
  const [e, setE] = useState<Experiment | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [selected, setSelected] = useState("demo");
  const [error, setError] = useState("");
  const loadDemo = () =>
    fetch("/api/demo")
      .then((r) => {
        if (!r.ok) throw new Error("Could not load demo");
        return r.json();
      })
      .then(setE)
      .catch((x) => setError(String(x)));
  const loadRun = (id: string) =>
    fetch(`/api/experiments/${encodeURIComponent(id)}`)
      .then((r) => {
        if (!r.ok) throw new Error(`Could not load experiment ${id}`);
        return r.json();
      })
      .then(setE)
      .catch((x) => setError(String(x)));
  useEffect(() => {
    loadDemo();
    fetch("/api/experiments")
      .then((r) => r.json())
      .then((data) => setRuns(data))
      .catch(() => setError("Could not list local experiment artifacts."));
  }, []);
  return (
    <main>
      <header>
        <div>
          <span className="eyebrow">LOCAL INFERENCE DEBUGGER</span>
          <h1>
            LLM Chat Template Compatibility
          </h1>
          <p>
            Find the prompt and token differences behind inconsistent local LLM
            behavior.
          </p>
        </div>
        <label className="selector">
          Experiment
          <select
            value={selected}
            onChange={(ev) => {
              const id = ev.target.value;
              setSelected(id);
              setError("");
              id === "demo" ? loadDemo() : loadRun(id);
            }}
          >
            <option value="demo">Synthetic demo</option>
            {runs.map((run) => (
              <option value={run.id} key={run.id}>
                {run.id}
              </option>
            ))}
          </select>
        </label>
      </header>
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
      {e && (
        <>
          <div className={e.synthetic ? "banner" : "banner real"}>
            {e.synthetic
              ? "Synthetic demonstration — not measured model inference."
              : `Stored ${e.mode} experiment · ${e.experiment_id}`}
          </div>
          <section className="summary">
            <div>
              <small>MODE</small>
              <strong>{e.mode} comparison</strong>
            </div>
            <div>
              <small>COMPATIBILITY</small>
              <strong>{e.compatibility}</strong>
            </div>
            <div>
              <small>FINDINGS</small>
              <strong>{e.diagnostic_findings.length} observations</strong>
            </div>
          </section>
          <section>
            <h2>Prompt paths</h2>
            <div className="grid">
              {Object.entries(e.prompts).map(([id, p]) => (
                <article key={id}>
                  <div className="cardhead">
                    <b>{p.source_runtime}</b>
                    <span>{p.token_ids.length} observed tokens</span>
                  </div>
                  <pre>{p.rendered_text}</pre>
                  <small>{p.provenance}</small>
                  <div className="tokens">
                    {p.token_ids.map((t, i) => (
                      <code
                        key={i}
                        className={
                          p.special_token_positions?.includes(i)
                            ? "special"
                            : ""
                        }
                        title={`token index ${i}; ${p.special_token_positions?.includes(i) ? "special token" : "ordinary token"}`}
                      >
                        <b>{t}</b>
                        <span>{p.token_pieces?.[i] ?? ""}</span>
                      </code>
                    ))}
                  </div>
                </article>
              ))}
            </div>
          </section>
          {e.token_comparison.token_diff && (
            <section>
              <h2>Token alignment</h2>
              <p className="muted">
                {e.token_comparison.token_ids_semantically_comparable
                  ? "Token IDs are comparable on checked vectors."
                  : "Raw token IDs are shown; cross-runtime semantics are unverified."}
                {e.token_comparison.token_diff.first_divergence &&
                  ` First divergence: prompt A index ${e.token_comparison.token_diff.first_divergence.a}, prompt B index ${e.token_comparison.token_diff.first_divergence.b}.`}
              </p>
              <div className="alignment">
                {e.token_comparison.token_diff.operations.map((op, i) => (
                  <article className={`op ${op.tag}`} key={i}>
                    <b>{op.tag}</b>
                    <small>
                      A [{op.a_start}:{op.a_end}] · B [{op.b_start}:{op.b_end}]
                    </small>
                    <code>A: {op.a.join(" ") || "∅"}</code>
                    <code>B: {op.b.join(" ") || "∅"}</code>
                  </article>
                ))}
              </div>
            </section>
          )}
          {e.token_comparison.crossover_matrix && (
            <section>
              <h2>Crossover matrix</h2>
              <div className="grid">
                {Object.entries(e.token_comparison.crossover_matrix).map(
                  ([prompt, cell]) => (
                    <article key={prompt}>
                      <b>
                        {prompt} prompt · {cell.status}
                      </b>
                      {cell.reason && <p>{cell.reason}</p>}
                      {cell.cells &&
                        Object.entries(cell.cells).map(([runtime, text]) => (
                          <p key={runtime}>
                            {runtime}: {text}
                          </p>
                        ))}
                    </article>
                  ),
                )}
              </div>
            </section>
          )}
          <section>
            <h2>Generation output</h2>
            <div className="grid">
              {Object.entries(e.generations).map(([id, g]) => (
                <article key={id}>
                  <b>{id}</b>
                  <pre>{g.output_text}</pre>
                  <small>Finish: {g.finish_reason ?? "unavailable"}</small>
                </article>
              ))}
            </div>
          </section>
          <section>
            <h2>Diagnostic findings</h2>
            {e.diagnostic_findings.map((f) => (
              <article className="finding" key={`${f.code}-${f.evidence}`}>
                <span>{f.code}</span>
                <div>
                  <b>{f.evidence}</b>
                  <p>{f.interpretation}</p>
                  <small>Next: {f.suggested_action}</small>
                </div>
              </article>
            ))}
          </section>
          <footer>
            {e.conclusion}
            <p>
              Observed differences support investigation; they do not establish
              that a template caused an output change.
            </p>
          </footer>
        </>
      )}
    </main>
  );
}
