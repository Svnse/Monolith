import { MessageSquare, Code2, Wand2 } from 'lucide-react';

const cards = [
  {
    num: '01',
    title: 'Chat',
    description:
      'Run GGUF models locally with persistent sessions, slash commands, and file context. Switch models without changing your workflow contract.',
    icon: MessageSquare,
  },
  {
    num: '02',
    title: 'Code',
    description:
      'Goal-seeking runtime with explicit walls, effect traces, and approval gates before writes. Every cycle is observable, stoppable, and reviewable.',
    icon: Code2,
  },
  {
    num: '03',
    title: 'Create',
    description:
      'Image and audio modules run in isolated processes so generation workloads do not lock the UI or compromise core control flow.',
    icon: Wand2,
  },
];

export default function WhatIs() {
  return (
    <section id="what" className="py-[var(--section-padding)] px-6 lg:px-8">
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-16 reveal">
          <h2 className="heading-lg mb-4">What is Monolith</h2>
          <p className="text-lg text-[var(--text-secondary)]">A local workstation for AI execution under explicit control.</p>
        </div>

        <div className="max-w-3xl mx-auto text-center mb-16 reveal reveal-delay-1">
          <p className="body-lg leading-relaxed">
            Monolith is kernel-routed infrastructure: the UI requests, the kernel arbitrates, engines execute.
            That separation keeps behavior inspectable and stops hidden side effects.
          </p>
          <p className="body-lg mt-6 leading-relaxed">
            It is not a cloud wrapper and not a chatbot shell. It is a local runtime where you can inspect,
            veto, and redirect execution without losing state.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-5">
          {cards.map((card, i) => (
            <div
              key={i}
              className="group glass p-7 hover:border-[var(--ethereal-dim)] transition-all duration-300 hover:-translate-y-1 reveal"
              style={{ transitionDelay: `${(i + 2) * 100}ms` }}
            >
              <div className="flex items-center justify-between mb-5">
                <span className="text-xs mono text-[var(--ethereal)] opacity-60">{card.num}</span>
                <card.icon className="w-5 h-5 text-[var(--ethereal)] opacity-60 group-hover:opacity-100 transition-opacity" />
              </div>

              <h3 className="text-lg font-medium mb-3 text-[var(--text-primary)]">{card.title}</h3>
              <p className="body-sm">{card.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
