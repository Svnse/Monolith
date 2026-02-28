import { Github, ExternalLink } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="border-t border-[var(--border-subtle)]">
      <div className="max-w-6xl mx-auto px-6 lg:px-8 py-12">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <span className="w-2.5 h-2.5 rounded-full bg-[var(--ethereal)]" />
            <span className="text-[var(--text-primary)] font-medium tracking-wide">Monolith</span>
            <span className="text-xs text-[var(--text-tertiary)] mono">by Eryndel</span>
          </div>

          <div className="flex items-center gap-6">
            <a
              href="https://github.com/Svnse/monolith"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
            >
              <Github className="w-4 h-4" />
              GitHub
            </a>
            <a
              href="https://eryndel.us"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
            >
              <ExternalLink className="w-4 h-4" />
              Eryndel
            </a>
          </div>
        </div>

        <div className="mt-8 pt-8 border-t border-[var(--border-subtle)] flex flex-col sm:flex-row items-center justify-between gap-4">
          <span className="text-xs text-[var(--text-tertiary)]">Kernel authority. Operator control.</span>
          <span className="text-xs text-[var(--text-tertiary)]">(c) 2026 Monolith</span>
        </div>
      </div>
    </footer>
  );
}
