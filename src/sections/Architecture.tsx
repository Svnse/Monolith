import { Check } from 'lucide-react';

const layers = [
  {
    name: 'UI / Addons',
    items: ['Pages', 'Modules', 'Commands', 'Timeline'],
    isKernel: false,
  },
  {
    name: 'MonoKernel',
    items: ['Guard', 'Dock', 'Bridge'],
    isKernel: true,
  },
  {
    name: 'Engines',
    items: ['LLM', 'Loop', 'Vision', 'Audio', 'Relay'],
    isKernel: false,
  },
];

const kernelContracts = [
  'STOP dominance: interrupt active execution immediately, including child workloads',
  'Generation gating: stale output is blocked after stop/redirect transitions',
  'Loop walls: budget, stall, and repetition checks run every cycle',
  'Approval gates: sensitive tool scopes require explicit operator approval',
  'Truthful READY: runtime state is never reported as ready while still executing',
];

export default function Architecture() {
  return (
    <section id="architecture" className="py-[var(--section-padding)] px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-16 reveal">
          <h2 className="heading-lg mb-4">Kernel Contract</h2>
          <p className="text-lg text-[var(--text-secondary)]">The kernel decides when and where, never what.</p>
        </div>

        <div className="flex flex-col items-center gap-4 mb-12">
          {layers.map((layer, i) => (
            <div key={i} className="w-full">
              <div
                className={`flex flex-col items-center gap-4 p-6 rounded-2xl border reveal ${
                  layer.isKernel ? 'bg-[var(--ethereal-subtle)] border-[var(--ethereal-dim)]' : 'glass'
                }`}
                style={{ transitionDelay: `${i * 150}ms` }}
              >
                <span
                  className={`text-xs uppercase tracking-widest ${
                    layer.isKernel ? 'text-[var(--ethereal)]' : 'text-[var(--text-tertiary)]'
                  }`}
                >
                  {layer.name}
                </span>
                <div className="flex flex-wrap justify-center gap-3">
                  {layer.items.map((item, j) => (
                    <span
                      key={j}
                      className={`px-4 py-2 rounded-lg text-sm mono ${
                        layer.isKernel
                          ? 'bg-[var(--ethereal)]/15 text-[var(--ethereal)] border border-[var(--ethereal)]/30'
                          : 'bg-[var(--surface)] text-[var(--text-secondary)] border border-[var(--border-soft)]'
                      }`}
                    >
                      {item}
                    </span>
                  ))}
                </div>
              </div>

              {i < layers.length - 1 && (
                <div className="flex justify-center py-3 reveal" style={{ transitionDelay: `${(i + 1) * 150}ms` }}>
                  <span className="text-[var(--text-tertiary)] text-lg opacity-50">v</span>
                </div>
              )}
            </div>
          ))}
        </div>

        <div className="text-center mb-12 reveal">
          <span className="text-xs mono text-[var(--text-tertiary)]">signals only</span>
        </div>

        <div className="grid sm:grid-cols-2 gap-3 max-w-3xl mx-auto">
          {kernelContracts.map((feature, i) => (
            <div
              key={i}
              className="flex items-start gap-3 p-4 glass rounded-xl reveal"
              style={{ transitionDelay: `${(i + 4) * 100}ms` }}
            >
              <Check className="w-4 h-4 text-[var(--success)] flex-shrink-0 mt-0.5" />
              <span className="text-sm text-[var(--text-secondary)]">{feature}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
