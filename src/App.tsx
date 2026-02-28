import { useEffect, useRef, useState } from 'react';
import Header from './sections/Header';
import Hero from './sections/Hero';
import WhatIs from './sections/WhatIs';
import Features from './sections/Features';
import AgentDemo from './sections/AgentDemo';
import Architecture from './sections/Architecture';
import Genesis from './sections/Genesis';
import CodePreview from './sections/CodePreview';
import Download from './sections/Download';
import Footer from './sections/Footer';
import AmbientBackground from './components/AmbientBackground';

function App() {
  const [scrollY, setScrollY] = useState(0);
  const mainRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleScroll = () => {
      setScrollY(window.scrollY);
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Scroll reveal observer
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('visible');
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.08, rootMargin: '0px 0px -50px 0px' }
    );

    document.querySelectorAll('.reveal').forEach((el) => {
      observer.observe(el);
    });

    return () => observer.disconnect();
  }, []);

  return (
    <div ref={mainRef} className="relative min-h-screen">
      <AmbientBackground scrollY={scrollY} />
      <Header scrollY={scrollY} />
      
      <main className="relative z-10">
        <Hero />
        <WhatIs />
        <Features />
        <AgentDemo />
        <Architecture />
        <Genesis />
        <CodePreview />
        <Download />
      </main>
      
      <Footer />
    </div>
  );
}

export default App;
