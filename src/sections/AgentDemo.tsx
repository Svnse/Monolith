import { useState, useEffect, useRef } from 'react';
import { Check, X } from 'lucide-react';

interface DemoStep {
  type: 'user' | 'agent' | 'tool' | 'approval' | 'system';
  role: string;
  msg: string;
  sub?: string;
}

const demoSteps: DemoStep[] = [
  { type: 'user', role: 'USER', msg: 'Refactor auth.py to use token hashing instead of plaintext passwords' },
  { type: 'agent', role: 'AGENT', msg: 'Reading goal. Checking project structure first.', sub: 'intent: explore' },
  { type: 'tool', role: 'TOOL', msg: "list_dir(path='.')", sub: '-> auth.py config.py models.py tests/' },
  { type: 'tool', role: 'TOOL', msg: "read_file(path='auth.py')", sub: '-> 87 lines read' },
  { type: 'agent', role: 'AGENT', msg: 'Found plaintext comparison on line 42. Planning patch.', sub: 'intent: plan' },
  { type: 'tool', role: 'TOOL', msg: "grep_search(pattern='password', path='auth.py')", sub: '-> 3 matches: L42, L61, L78' },
  {
    type: 'agent',
    role: 'AGENT',
    msg: 'Strategy: replace direct comparison with bcrypt.checkpw(). Import bcrypt and hash at register.',
    sub: 'intent: execute',
  },
  { type: 'approval', role: 'APPROVAL', msg: 'Apply patch to auth.py?', sub: 'write_file scope requires approval' },
  { type: 'agent', role: 'AGENT', msg: 'Patch applied. Running tests to verify.', sub: 'intent: verify' },
  { type: 'tool', role: 'TOOL', msg: "run_cmd(cmd='pytest tests/test_auth.py -q')", sub: '-> 6 passed in 0.4s' },
  { type: 'agent', role: 'AGENT', msg: 'All tests pass. auth.py now uses bcrypt hashing. Done.', sub: 'finish: true' },
  { type: 'system', role: 'SYSTEM', msg: 'Run complete -> 6 cycles | 5 tool calls | 0 walls hit' },
];

export default function AgentDemo() {
  const [visibleLines, setVisibleLines] = useState<number>(1);
  const [isComplete, setIsComplete] = useState(false);
  const [approved, setApproved] = useState(false);
  const terminalRef = useRef<HTMLDivElement>(null);
  const sectionRef = useRef<HTMLElement>(null);
  const hasStarted = useRef(false);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [visibleLines]);

  useEffect(() => {
    const section = sectionRef.current;
    if (!section) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && !hasStarted.current) {
            hasStarted.current = true;
            startDemo();
          }
        });
      },
      { threshold: 0.3 }
    );

    observer.observe(section);
    return () => observer.disconnect();
  }, []);

  const startDemo = () => {
    let currentLine = 1;

    const showNextLine = () => {
      if (currentLine >= demoSteps.length) {
        setIsComplete(true);
        return;
      }

      const step = demoSteps[currentLine];
      const delay = step.type === 'approval' ? 2000 : 800;

      if (step.type === 'approval') {
        setTimeout(() => setApproved(true), 1500);
      }

      setTimeout(() => {
        currentLine++;
        setVisibleLines(currentLine);
        showNextLine();
      }, delay);
    };

    showNextLine();
  };

  const visibleSteps = demoSteps.slice(0, visibleLines);

  return (
    <section ref={sectionRef} className="py-[var(--section-padding)] px-6 lg:px-8">
      <div className="max-w-5xl mx-auto">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          <div className="reveal">
            <div className="inline-flex items-center gap-2 px-3 py-1 mb-6 text-xs font-medium text-[var(--success)] bg-[var(--success)]/10 border border-[var(--success)]/30 rounded-full">
              <span className="w-1.5 h-1.5 bg-[var(--success)] rounded-full animate-pulse" />
              Coding Agent
            </div>

            <h2 className="heading-md mb-5">Observe each cycle</h2>
            <p className="body-md mb-8">
              The runtime narrates intent, tool call, and result before moving to the next step. Stop, redirect,
              or approve at any point.
            </p>

            <div className="space-y-3">
              {['Cycle-by-cycle trace', 'Structured tool output', 'Approval before writes', 'Budget and stall walls'].map(
                (item, i) => (
                  <div key={i} className="flex items-center gap-3">
                    <span className="w-1.5 h-1.5 rounded-full bg-[var(--ethereal)]" />
                    <span className="text-sm text-[var(--text-secondary)]">{item}</span>
                  </div>
                )
              )}
            </div>
          </div>

          <div className="reveal reveal-delay-2">
            <div className="terminal">
              <div className="terminal-header">
                <div className="code-dots">
                  <span />
                  <span />
                  <span />
                </div>
                <span className="code-title">CODE - Monolith Agent Loop</span>
                <div className="terminal-status">{isComplete ? 'COMPLETE' : 'RUNNING'}</div>
              </div>

              <div ref={terminalRef} className="terminal-body">
                {visibleSteps.map((step, i) => (
                  <div key={i} className="terminal-line">
                    <span className={`terminal-role ${step.type}`}>{step.role}</span>
                    <div className="flex-1">
                      <span className="terminal-msg">{step.msg}</span>
                      {step.sub && <span className="terminal-sub">{step.sub}</span>}
                      {step.type === 'approval' && (
                        <div className="flex gap-2 mt-2">
                          <span
                            className={`inline-flex items-center gap-1 px-2 py-1 text-xs rounded border ${
                              approved
                                ? 'bg-[var(--ethereal)]/20 border-[var(--ethereal)] text-[var(--ethereal)]'
                                : 'border-[var(--border-soft)] text-[var(--text-tertiary)]'
                            }`}
                          >
                            {approved ? <Check className="w-3 h-3" /> : null}
                            {approved ? 'Approved' : 'Approve'}
                          </span>
                          <span className="inline-flex items-center gap-1 px-2 py-1 text-xs rounded border border-[var(--border-soft)] text-[var(--text-tertiary)]">
                            {!approved && <X className="w-3 h-3" />}
                            Deny
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
