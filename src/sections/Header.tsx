import { useState, useEffect } from 'react';
import { Github } from 'lucide-react';

interface HeaderProps {
  scrollY: number;
}

const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';

function useScrambleText(originalText: string) {
  const [displayText, setDisplayText] = useState(originalText);
  const [isHovering, setIsHovering] = useState(false);

  useEffect(() => {
    if (!isHovering) {
      setDisplayText(originalText);
      return;
    }

    let frame = 0;
    const totalFrames = 12;
    const interval = setInterval(() => {
      frame++;
      const progress = frame / totalFrames;

      const scrambled = originalText
        .split('')
        .map((char, i) => {
          if (char === ' ') return ' ';
          const charProgress = i / originalText.length;
          if (progress > charProgress + 0.3) return char;
          if (progress < charProgress) return char;
          return chars[Math.floor(Math.random() * chars.length)];
        })
        .join('');

      setDisplayText(scrambled);

      if (frame >= totalFrames) {
        setDisplayText(originalText);
        clearInterval(interval);
      }
    }, 30);

    return () => clearInterval(interval);
  }, [isHovering, originalText]);

  return { displayText, setIsHovering };
}

function ScrambleLink({ href, children }: { href: string; children: string }) {
  const { displayText, setIsHovering } = useScrambleText(children);

  return (
    <a
      href={href}
      className="text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors duration-200"
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => setIsHovering(false)}
    >
      {displayText}
    </a>
  );
}

export default function Header({ scrollY }: HeaderProps) {
  const isScrolled = scrollY > 50;

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
        isScrolled
          ? 'bg-[var(--void)]/80 backdrop-blur-xl border-b border-[var(--border-subtle)]'
          : 'bg-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 lg:h-20">
          <a href="#" className="flex items-center gap-3 group">
            <span className="w-2.5 h-2.5 rounded-full bg-[var(--ethereal)] transition-transform duration-300 group-hover:scale-125" />
            <span className="text-[var(--text-primary)] font-medium tracking-wide">Monolith</span>
            <span className="text-xs text-[var(--text-tertiary)] mono hidden sm:inline">by Eryndel</span>
          </a>

          <nav className="hidden md:flex items-center gap-8">
            <ScrambleLink href="#what">What</ScrambleLink>
            <ScrambleLink href="#features">Features</ScrambleLink>
            <ScrambleLink href="#architecture">Kernel</ScrambleLink>
            <ScrambleLink href="#download">Run</ScrambleLink>
          </nav>

          <a
            href="https://github.com/Svnse/monolith"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-[var(--text-primary)] border border-[var(--border-medium)] rounded-lg hover:bg-[var(--surface)] hover:border-[var(--ethereal-dim)] transition-all duration-200"
          >
            <Github className="w-4 h-4" />
            <span className="hidden sm:inline">GitHub</span>
          </a>
        </div>
      </div>
    </header>
  );
}
