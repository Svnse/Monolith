import { useEffect, useRef, useState, useCallback } from 'react';
import { ArrowRight, Download } from 'lucide-react';

const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*';

function useTextScramble(text: string, delay = 0, trigger = true) {
  const [displayText, setDisplayText] = useState('');

  useEffect(() => {
    if (!trigger) return;

    const timeout = setTimeout(() => {
      let frame = 0;
      const totalFrames = 20;

      const interval = setInterval(() => {
        frame++;
        const progress = frame / totalFrames;

        const scrambled = text
          .split('')
          .map((char, i) => {
            if (char === ' ') return ' ';
            if (char === '\n') return '\n';
            const charProgress = i / text.length;
            if (progress > charProgress + 0.2) return char;
            if (progress < charProgress - 0.1) return '';
            return chars[Math.floor(Math.random() * chars.length)];
          })
          .join('');

        setDisplayText(scrambled);

        if (frame >= totalFrames) {
          setDisplayText(text);
          clearInterval(interval);
        }
      }, 25);
    }, delay);

    return () => clearTimeout(timeout);
  }, [text, delay, trigger]);

  return displayText;
}

const responses: Record<string, string> = {
  help: 'commands: what | features | architecture | version | github | download | sovereignty',
  chat: 'GGUF chat via llama-cpp-python. Local session, local data, local control.',
  code: 'CODE loop: infer -> parse -> wall check -> execute -> verify.',
  architecture: 'UI -> MonoKernel(Guard/Dock/Bridge) -> Engines. Signals only.',
  kernel: 'MonoKernel arbitrates execution timing. It does not execute workloads.',
  version: 'v0.28a | active experimental development',
  sovereignty: 'No forced cloud. No silent telemetry. You keep veto control.',
  github: 'github.com/Svnse/monolith',
  download: 'Clone repo, run bootstrap.py, choose your local model stack.',
  default: "Unknown command. Try 'help'.",
};

export default function Hero() {
  const [isLoaded, setIsLoaded] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const [reply, setReply] = useState('Authorship preserved. Automation constrained.');
  const inputRef = useRef<HTMLInputElement>(null);

  const titleDisplay = useTextScramble('Your models.\nYour machine. Your rules.', 350, isLoaded);
  const subDisplay = useTextScramble(
    'Monolith is a local execution workstation for LLM chat, coding loops, vision, audio, and relay flows. The kernel controls execution order. You control approvals, stop, and direction.',
    700,
    isLoaded
  );

  useEffect(() => {
    const timer = setTimeout(() => setIsLoaded(true), 180);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === '/' && document.activeElement?.tagName !== 'INPUT') {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleInputKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key !== 'Enter') return;
      const val = inputValue.toLowerCase().trim();
      if (!val) return;

      const cmd = val.replace(/^\.\//, '').replace(/^monolith\s+--?/, '');

      if (cmd === 'features' || cmd.includes('feature')) {
        document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' });
      } else if (cmd === 'download' || cmd.includes('download') || cmd === 'run') {
        document.getElementById('download')?.scrollIntoView({ behavior: 'smooth' });
      } else if (cmd === 'architecture' || cmd.includes('kernel') || cmd.includes('arch')) {
        document.getElementById('architecture')?.scrollIntoView({ behavior: 'smooth' });
      } else if (cmd === 'what') {
        document.getElementById('what')?.scrollIntoView({ behavior: 'smooth' });
      } else if (cmd === 'github' || cmd === 'repo' || cmd === 'source') {
        window.open('https://github.com/Svnse/monolith', '_blank');
      } else {
        setReply(responses[cmd] || responses.default);
      }

      setTimeout(() => setInputValue(''), 180);
    },
    [inputValue]
  );

  return (
    <section className="min-h-screen flex items-center justify-center px-6 pt-20 pb-16">
      <div
        className={`w-full max-w-2xl transition-all duration-1000 ${
          isLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
        }`}
      >
        <div className="glass p-8 md:p-12">
          <div className="flex items-center justify-center gap-4 mb-8">
            <span className="px-3 py-1 text-xs font-semibold tracking-wider text-[var(--warning)] bg-[var(--warning)]/10 border border-[var(--warning)]/30 rounded-full">
              EXPERIMENTAL
            </span>
            <span className="text-sm text-[var(--text-tertiary)] mono">v0.28a</span>
          </div>

          <h1 className="heading-xl text-center mb-6 whitespace-pre-line">
            {titleDisplay.split('\n').map((line, i) => (
              <span key={i} className={i === 1 ? 'text-[var(--ethereal)]' : ''}>
                {line}
                {i === 0 && '\n'}
              </span>
            ))}
          </h1>

          <p className="body-lg text-center max-w-xl mx-auto mb-10">{subDisplay}</p>

          <div
            className={`flex justify-center gap-8 md:gap-12 mb-10 transition-all duration-700 delay-500 ${
              isLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
            }`}
          >
            {[
              { value: '100%', label: 'Local' },
              { value: '0', label: 'Forced APIs' },
              { value: 'Hard', label: 'Control Gates' },
            ].map((stat, i) => (
              <div key={i} className="text-center">
                <div className="text-2xl md:text-3xl font-semibold text-[var(--ethereal)] mono mb-1">{stat.value}</div>
                <div className="text-xs text-[var(--text-tertiary)] uppercase tracking-widest">{stat.label}</div>
              </div>
            ))}
          </div>

          <div
            className={`relative mb-4 transition-all duration-700 delay-700 ${
              isLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
            }`}
          >
            <div className="flex items-center gap-3 px-5 py-4 bg-[var(--abyss)] border border-[var(--border-soft)] rounded-xl">
              <span className="text-[var(--success)] mono text-lg">$</span>
              <input
                ref={inputRef}
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleInputKeyDown}
                placeholder="monolith --help"
                className="flex-1 bg-transparent border-none outline-none text-[var(--text-primary)] mono text-sm placeholder:text-[var(--text-tertiary)]"
                spellCheck={false}
                autoComplete="off"
              />
            </div>
            <div
              className="absolute -inset-3 rounded-2xl pointer-events-none opacity-50"
              style={{
                background: 'radial-gradient(circle at 10% 50%, var(--ethereal-glow), transparent 45%)',
                filter: 'blur(12px)',
                animation: 'orbitGlow 7s linear infinite',
              }}
            />
          </div>

          <p
            className={`text-center text-sm text-[var(--text-secondary)] mb-10 transition-all duration-700 delay-900 ${
              isLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
            }`}
          >
            {reply}
          </p>

          <div
            className={`flex flex-col sm:flex-row items-center justify-center gap-4 transition-all duration-700 delay-1000 ${
              isLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
            }`}
          >
            <a href="#download" className="btn-primary w-full sm:w-auto">
              <Download className="w-4 h-4" />
              <span>Run Monolith</span>
              <span className="text-xs opacity-70 ml-1">Windows | macOS | Linux</span>
            </a>
            <a href="#what" className="btn-ghost w-full sm:w-auto">
              <span>See the model</span>
              <ArrowRight className="w-4 h-4" />
            </a>
          </div>
        </div>

        <div className="absolute inset-0 -z-10 glass opacity-50 translate-y-3 scale-[0.98]" aria-hidden="true" />
      </div>

      <style>{`
        @keyframes orbitGlow {
          0% { transform: rotate(0deg); opacity: 0.4; }
          50% { opacity: 0.6; }
          100% { transform: rotate(360deg); opacity: 0.4; }
        }
      `}</style>
    </section>
  );
}
