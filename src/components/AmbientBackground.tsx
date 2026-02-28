interface AmbientBackgroundProps {
  scrollY: number;
}

export default function AmbientBackground({ scrollY }: AmbientBackgroundProps) {
  const parallaxOffset = scrollY * 0.15;

  return (
    <div className="ambient-bg">
      {/* Nebula effects */}
      <div 
        className="nebula-core"
        style={{ transform: `translateX(-50%) translateY(${parallaxOffset * 0.5}px)` }}
      />
      <div 
        className="nebula-drift"
        style={{ transform: `translateY(${parallaxOffset * 0.3}px)` }}
      />
      
      {/* Star field */}
      <div className="star-field" />
      
      {/* Grain overlay */}
      <div className="grain-overlay" />
      
      {/* Subtle grid */}
      <div 
        className="absolute inset-0 opacity-[0.02]"
        style={{
          backgroundImage: `
            linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)
          `,
          backgroundSize: '60px 60px'
        }}
      />
    </div>
  );
}
