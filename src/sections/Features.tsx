import { MessageSquare, Bot, Image, Music, Cpu, Shield } from 'lucide-react';

const features = [
  {
    icon: MessageSquare,
    title: 'Local LLM Chat',
    description:
      'Inference via llama-cpp-python with GGUF models, persistent context, and command-driven workflows. No forced outbound calls.',
    meta: 'GGUF | llama-cpp-python | local sessions',
    featured: false,
  },
  {
    icon: Bot,
    title: 'Autonomous Coding Agent',
    description:
      'Execution loop with tool walls, event traces, and explicit approval pauses before sensitive operations touch disk.',
    meta: 'LoopRuntime | approvals | hard walls',
    featured: true,
  },
  {
    icon: Image,
    title: 'Vision Pipeline',
    description:
      'Stable Diffusion module support in subprocess isolation so image generation stays responsive and bounded.',
    meta: 'SD 1.5 | SDXL | Flux-ready paths',
    featured: false,
  },
  {
    icon: Music,
    title: 'Audio Pipeline',
    description:
      'Audio generation and speech modules execute out of process with clear lifecycle boundaries and runtime events.',
    meta: 'MusicGen | Whisper STT | TTS modules',
    featured: false,
  },
  {
    icon: Cpu,
    title: 'Kernel Arbitration',
    description:
      'MonoKernel resolves when execution happens and where signals route. Workloads run in engines, not in kernel control code.',
    meta: 'Guard | Dock | Bridge',
    featured: false,
  },
  {
    icon: Shield,
    title: 'Sovereignty Model',
    description:
      'Local-first by default with explicit control over approvals, stop behavior, and execution boundaries.',
    meta: 'offline-first | no silent telemetry',
    featured: false,
  },
];

export default function Features() {
  return (
    <section id="features" className="py-[var(--section-padding)] px-6 lg:px-8">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-16 reveal">
          <h2 className="heading-lg mb-4">Execution, not chatter</h2>
          <p className="text-lg text-[var(--text-secondary)]">
            Observe the loop. Gate the writes. Keep control at every layer.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5">
          {features.map((feature, i) => (
            <div
              key={i}
              className={`group p-7 rounded-2xl border transition-all duration-300 hover:-translate-y-1 reveal ${
                feature.featured
                  ? 'bg-[var(--ethereal-subtle)] border-[var(--ethereal-dim)] lg:col-span-1'
                  : 'glass hover:border-[var(--ethereal-dim)]'
              }`}
              style={{ transitionDelay: `${(i % 3) * 100}ms` }}
            >
              <div
                className={`w-10 h-10 rounded-lg flex items-center justify-center mb-5 ${
                  feature.featured ? 'bg-[var(--ethereal)]/20' : 'bg-[var(--surface)]'
                }`}
              >
                <feature.icon
                  className={`w-5 h-5 ${feature.featured ? 'text-[var(--ethereal)]' : 'text-[var(--text-secondary)]'}`}
                />
              </div>

              <h3 className="text-base font-medium mb-3 text-[var(--text-primary)]">{feature.title}</h3>
              <p className="body-sm mb-5">{feature.description}</p>

              <div className="pt-4 border-t border-[var(--border-subtle)]">
                <span className="text-xs mono text-[var(--text-tertiary)]">{feature.meta}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
