import { useState } from 'react';
import { ChevronDown } from 'lucide-react';

const versions = [
  {
    version: 'V1',
    label: 'The First Script',
    date: 'Dec 2025',
    framework: 'tkinter',
    description:
      'One file. Raw tkinter widgets. Gold-on-black palette. A file system panel and a chat surface wired together directly — no state layer, no separation. The original monolith.',
    traits: ['Single-file script', 'tkinter + raw widgets', 'Gold accent (#d4af37)', 'File browser + basic chat'],
    maturity: 1,
  },
  {
    version: 'V1.5',
    label: 'Expanding the Script',
    date: 'Dec 2025',
    framework: 'tkinter',
    description:
      'The single file grew. More panels, more callbacks, more patching. tkinter was hitting its ceiling. The need for something different was becoming clear.',
    traits: ['Expanded panels', 'More UI surface area', 'Still single-file', 'Tkinter at its limits'],
    maturity: 2,
  },
  {
    version: 'V2',
    label: 'First Structure',
    date: 'Dec 16, 2025',
    framework: 'PySide6',
    description:
      'Migrated to PySide6 the same day V1.5 was finalized. AppState and LLMEngine emerged as distinct objects. Config was extracted. The idea of "model" and "UI" as separate concerns appeared for the first time.',
    traits: ['PySide6 migration', 'AppState + LLMEngine introduced', 'Constants extracted', 'Settings page'],
    maturity: 3,
  },
  {
    version: 'V2.5',
    label: 'Components Form',
    date: 'Dec 16, 2025',
    framework: 'PySide6',
    description:
      'Databank module added with QFileSystemModel and drag-drop support. UI components became proper classes. Still a combined script, but the seams between modules were becoming visible.',
    traits: ['Databank + file tree (drag-drop)', 'Component classes formalized', 'Gold accent system refined', 'Combined but structured'],
    maturity: 3,
  },
  {
    version: 'V3',
    label: 'Pre-Kernel',
    date: 'Dec 16, 2025',
    framework: 'PySide6',
    description:
      'Architecture began stratifying into layers. Signals replaced direct calls in key paths. UI and engine logic were separating. The gap where a routing arbitration layer should go became obvious.',
    traits: ['Layer stratification', 'Signal-based decoupling', 'Multi-module UI pages', 'Pre-kernel routing patterns'],
    maturity: 4,
  },
  {
    version: 'v0.28a',
    label: 'Local AI OS',
    date: 'Feb 2026',
    framework: 'MonoKernel',
    description:
      'Full kernel-routed architecture. UI requests, MonoKernel arbitrates, Engines execute — signals only. CODE agent loop with tool walls, approval gates, and effect journaling. Vision, Audio, and Relay modules in isolated processes. STOP dominance.',
    traits: ['UI → MonoKernel → Engines', 'CODE agent loop + tool walls', 'Vision / Audio / Relay engines', 'STOP dominance + approval gates'],
    maturity: 5,
    isCurrent: true,
  },
];

function MaturityBar({ level, isCurrent }: { level: number; isCurrent?: boolean }) {
  const max = 5;
  return (
    <div className="flex gap-1 mt-4">
      {Array.from({ length: max }).map((_, i) => (
        <div
          key={i}
          className="h-0.5 flex-1 rounded-full transition-all"
          style={{
            background:
              i < level
                ? isCurrent
                  ? 'var(--ethereal)'
                  : `rgba(140, 166, 209, ${0.3 + (i / max) * 0.45})`
                : 'var(--border-subtle)',
          }}
        />
      ))}
    </div>
  );
}

export default function Genesis() {
  const [open, setOpen] = useState(false);

  return (
    <section id="genesis" className="py-8 px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">

        {/* Toggle button */}
        <div className="flex justify-center reveal">
          <button
            onClick={() => setOpen((v) => !v)}
            className="group flex items-center gap-3 px-5 py-3 rounded-xl border border-[var(--border-soft)] hover:border-[var(--ethereal-dim)] bg-[var(--ethereal-subtle)] hover:bg-[var(--ethereal-subtle)] transition-all duration-300"
            aria-expanded={open}
          >
            <span className="text-xs mono text-[var(--ethereal)] opacity-70 group-hover:opacity-100 transition-opacity">
              {open ? '—' : '$'}
            </span>
            <span className="text-sm text-[var(--text-secondary)] group-hover:text-[var(--text-primary)] transition-colors">
              {open ? 'Hide origin history' : 'Show origin history'}
            </span>
            <ChevronDown
              className="w-4 h-4 text-[var(--text-tertiary)] transition-transform duration-300"
              style={{ transform: open ? 'rotate(180deg)' : 'rotate(0deg)' }}
            />
          </button>
        </div>

        {/* Collapsible content — grid trick for smooth height transition */}
        <div
          className="grid transition-all duration-500 ease-in-out"
          style={{
            gridTemplateRows: open ? '1fr' : '0fr',
            opacity: open ? 1 : 0,
            transitionProperty: 'grid-template-rows, opacity',
          }}
        >
          <div className="overflow-hidden">
            <div className="pt-12 pb-4">

              {/* Section header */}
              <div className="text-center mb-14">
                <div className="flex items-center justify-center gap-3 mb-4">
                  <span className="h-px w-12 bg-gradient-to-r from-transparent to-[var(--border-medium)]" />
                  <span className="text-xs mono text-[var(--text-tertiary)] tracking-widest uppercase">Origin</span>
                  <span className="h-px w-12 bg-gradient-to-l from-transparent to-[var(--border-medium)]" />
                </div>
                <h2 className="heading-lg mb-3">Genesis</h2>
                <p className="text-[var(--text-secondary)]">
                  From a glued-together script to a kernel-routed OS.
                </p>
                <p className="text-xs mono text-[var(--text-tertiary)] mt-2">Dec 2025 → Feb 2026</p>
              </div>

              {/* Timeline */}
              <div className="relative">

                {/* Spine */}
                <div
                  className="absolute hidden md:block top-6 bottom-6 w-px left-[22px]"
                  style={{
                    background: 'linear-gradient(to bottom, transparent, var(--border-soft) 15%, var(--border-soft) 85%, transparent)',
                  }}
                />

                <div className="space-y-5">
                  {versions.map((v, i) => (
                    <div key={i} className="relative flex gap-6 md:pl-14">

                      {/* Timeline dot */}
                      <div
                        className="absolute hidden md:flex left-[14px] top-[22px] w-[18px] h-[18px] rounded-full border items-center justify-center"
                        style={{
                          borderColor: v.isCurrent ? 'var(--ethereal)' : 'var(--border-medium)',
                          background: v.isCurrent ? 'rgba(140,166,209,0.12)' : 'var(--void)',
                          zIndex: 1,
                        }}
                      >
                        {v.isCurrent && (
                          <div className="w-1.5 h-1.5 rounded-full bg-[var(--ethereal)]" style={{ animation: 'pulse 2s ease-in-out infinite' }} />
                        )}
                      </div>

                      {/* Card */}
                      <div
                        className={`flex-1 glass p-5 rounded-2xl transition-all duration-300 hover:-translate-y-0.5 ${
                          v.isCurrent
                            ? 'border-[var(--ethereal-dim)]'
                            : 'border-[var(--border-soft)] hover:border-[var(--border-medium)]'
                        }`}
                      >
                        {/* Header row */}
                        <div className="flex flex-wrap items-start justify-between gap-2 mb-3">
                          <div className="flex items-center gap-3">
                            <span className="text-base font-semibold mono text-[var(--text-primary)]">{v.version}</span>
                            {v.isCurrent && (
                              <span className="px-2 py-0.5 text-xs font-semibold tracking-wider text-[var(--warning)] bg-[var(--warning)]/10 border border-[var(--warning)]/30 rounded-full">
                                CURRENT
                              </span>
                            )}
                            <span
                              className="px-2 py-0.5 text-xs mono rounded-md"
                              style={{
                                background: v.isCurrent ? 'var(--ethereal-subtle)' : 'var(--surface)',
                                color: v.isCurrent ? 'var(--ethereal)' : 'var(--text-tertiary)',
                                border: `1px solid ${v.isCurrent ? 'var(--ethereal-dim)' : 'var(--border-subtle)'}`,
                              }}
                            >
                              {v.framework}
                            </span>
                          </div>
                          <div className="text-right">
                            <span className="text-xs mono text-[var(--text-tertiary)]">{v.date}</span>
                            <div className="text-xs text-[var(--text-tertiary)] mt-0.5">{v.label}</div>
                          </div>
                        </div>

                        {/* Description */}
                        <p className="text-sm text-[var(--text-secondary)] leading-relaxed mb-3">{v.description}</p>

                        {/* Traits */}
                        <div className="flex flex-wrap gap-1.5">
                          {v.traits.map((t, j) => (
                            <span
                              key={j}
                              className="px-2.5 py-1 text-xs mono rounded-lg"
                              style={{
                                background: v.isCurrent ? 'var(--ethereal-subtle)' : 'rgba(255,255,255,0.025)',
                                color: v.isCurrent ? 'var(--ethereal)' : 'var(--text-tertiary)',
                                border: `1px solid ${v.isCurrent ? 'rgba(140,166,209,0.2)' : 'var(--border-subtle)'}`,
                              }}
                            >
                              {t}
                            </span>
                          ))}
                        </div>

                        <MaturityBar level={v.maturity} isCurrent={v.isCurrent} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

            </div>
          </div>
        </div>

      </div>
    </section>
  );
}
