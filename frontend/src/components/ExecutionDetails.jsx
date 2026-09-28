import { formatParams } from "../utils/format";

// The real backend execution trace (response.execution) -- the auditable
// "task, tools, key params" record CLAUDE.md's agentic-orchestration
// requirement asks for. Lives inside the "Execution details" tab of
// AnalysisResult.jsx, which already gates visibility, so this is just the
// table (the confidence recap that used to live here is on the Summary tab).
export default function ExecutionDetails({ execution = [] }) {
  if (execution.length === 0) return null;

  return (
    <div className="execution-details">
      <table className="execution-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Step</th>
            <th>Tool</th>
            <th>Method</th>
            <th>Key params</th>
            <th>Output summary</th>
            <th>ms</th>
          </tr>
        </thead>
        <tbody>
          {execution.map((step) => (
            <tr key={step.step} className={step.fallback ? "execution-row-fallback" : ""}>
              <td>{step.step}</td>
              <td>
                {step.name}
                {step.fallback && <span className="fallback-badge">FALLBACK</span>}
              </td>
              <td>{step.tool}</td>
              <td>{step.method}</td>
              <td>{formatParams(step.params)}</td>
              <td>{step.output_summary || "—"}</td>
              <td>{step.ms}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
