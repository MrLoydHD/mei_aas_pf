import { useRef, useEffect, useState } from 'react';

// Sample domains - mix of legit-looking and DGA-like
const LEGIT_DOMAINS = [
  'google.com', 'facebook.com', 'amazon.com', 'microsoft.com', 'apple.com',
  'github.com', 'stackoverflow.com', 'reddit.com', 'twitter.com', 'linkedin.com',
  'youtube.com', 'netflix.com', 'spotify.com', 'dropbox.com', 'slack.com',
  'zoom.us', 'notion.so', 'figma.com', 'vercel.app', 'cloudflare.com',
  'api.stripe.com', 'auth.google.com', 'login.microsoft.com', 'cdn.jsdelivr.net',
];

const DGA_DOMAINS = [
  'xk7mn2p.com', 'qw3rt8y.net', 'zx9cv2b.org', 'lk4jh7g.com', 'mn8bv3c.net',
  'p0o9i8u.com', 'a1s2d3f.org', 'gh5jk6l.net', 'qazwsx.com', 'edcrfv.org',
  'tgbyhn.net', 'ujmik.com', 'olpzaq.org', 'wsxedc.net', 'rfvtgb.com',
  'yhnujm.org', 'ik8ol9p.net', 'zaq1xsw.com', '2wsx3ed.org', 'c4rfv5t.net',
  'gb6yhn7.com', 'uj8mik9.org', 'ol0pzaq.net', '1qaz2ws.com', '3edc4rf.org',
];

const generateRandomDGA = () => {
  const chars = 'abcdefghijklmnopqrstuvwxyz0123456789';
  const length = 6 + Math.floor(Math.random() * 8);
  let domain = '';
  for (let i = 0; i < length; i++) {
    domain += chars[Math.floor(Math.random() * chars.length)];
  }
  const tlds = ['.com', '.net', '.org', '.io', '.xyz', '.info'];
  return domain + tlds[Math.floor(Math.random() * tlds.length)];
};

const getRandomDomain = () => {
  const rand = Math.random();
  if (rand < 0.4) {
    return LEGIT_DOMAINS[Math.floor(Math.random() * LEGIT_DOMAINS.length)];
  } else if (rand < 0.7) {
    return DGA_DOMAINS[Math.floor(Math.random() * DGA_DOMAINS.length)];
  } else {
    return generateRandomDGA();
  }
};

interface LetterGlitchProps {
  glitchSpeed?: number;
  outerVignette?: boolean;
  smooth?: boolean;
}

const LetterGlitch = ({
  glitchSpeed = 100,
  outerVignette = true,
  smooth = true,
}: LetterGlitchProps) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationRef = useRef<number | null>(null);
  const lines = useRef<
    {
      text: string;
      color: string;
      targetColor: string;
      colorProgress: number;
    }[]
  >([]);
  const gridInfo = useRef({ rows: 0, charsPerRow: 0 });
  const context = useRef<CanvasRenderingContext2D | null>(null);
  const lastGlitchTime = useRef(Date.now());

  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    const checkDarkMode = () => {
      setIsDark(document.documentElement.classList.contains('dark'));
    };
    checkDarkMode();
    const observer = new MutationObserver(checkDarkMode);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    return () => observer.disconnect();
  }, []);

  // Very subtle colors
  const getGlitchColors = () => {
    if (isDark) {
      return [
        'rgba(34, 197, 94, 0.08)',   // green - very faint
        'rgba(239, 68, 68, 0.06)',   // red - very faint
        'rgba(148, 163, 184, 0.05)', // slate - barely visible
        'rgba(59, 130, 246, 0.07)',  // blue - very faint
      ];
    } else {
      return [
        'rgba(34, 197, 94, 0.12)',   // green
        'rgba(239, 68, 68, 0.10)',   // red
        'rgba(100, 116, 139, 0.08)', // slate
        'rgba(59, 130, 246, 0.10)', // blue
      ];
    }
  };

  const fontSize = 13;
  const lineHeight = 20;

  const getRandomColor = () => {
    const colors = getGlitchColors();
    return colors[Math.floor(Math.random() * colors.length)];
  };

  const generateLine = (charsPerRow: number) => {
    let line = '';
    while (line.length < charsPerRow) {
      const domain = getRandomDomain();
      const spacing = '   '; // space between domains
      if (line.length + domain.length + spacing.length <= charsPerRow) {
        line += domain + spacing;
      } else {
        // Fill remaining with partial or spaces
        const remaining = charsPerRow - line.length;
        if (remaining > 3) {
          line += getRandomDomain().slice(0, remaining);
        } else {
          line += ' '.repeat(remaining);
        }
      }
    }
    return line.slice(0, charsPerRow);
  };

  const parseColor = (color: string) => {
    const rgbaMatch = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/);
    if (rgbaMatch) {
      return {
        r: parseInt(rgbaMatch[1]),
        g: parseInt(rgbaMatch[2]),
        b: parseInt(rgbaMatch[3]),
        a: rgbaMatch[4] ? parseFloat(rgbaMatch[4]) : 1
      };
    }
    return { r: 100, g: 100, b: 100, a: 0.1 };
  };

  const interpolateColor = (
    start: { r: number; g: number; b: number; a: number },
    end: { r: number; g: number; b: number; a: number },
    factor: number
  ) => {
    const result = {
      r: Math.round(start.r + (end.r - start.r) * factor),
      g: Math.round(start.g + (end.g - start.g) * factor),
      b: Math.round(start.b + (end.b - start.b) * factor),
      a: start.a + (end.a - start.a) * factor
    };
    return `rgba(${result.r}, ${result.g}, ${result.b}, ${result.a})`;
  };

  const initializeLines = (width: number, height: number) => {
    const charWidth = fontSize * 0.6;
    const charsPerRow = Math.ceil(width / charWidth) + 10;
    const rows = Math.ceil(height / lineHeight) + 2;

    gridInfo.current = { rows, charsPerRow };

    lines.current = Array.from({ length: rows }, () => ({
      text: generateLine(charsPerRow),
      color: getRandomColor(),
      targetColor: getRandomColor(),
      colorProgress: 1
    }));
  };

  const resizeCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement;
    if (!parent) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = parent.getBoundingClientRect();

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;

    if (context.current) {
      context.current.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    initializeLines(rect.width, rect.height);
    drawLines();
  };

  const drawLines = () => {
    if (!context.current || lines.current.length === 0) return;
    const ctx = context.current;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const { width, height } = canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, width, height);
    ctx.font = `${fontSize}px monospace`;
    ctx.textBaseline = 'top';

    lines.current.forEach((line, index) => {
      const y = index * lineHeight;
      ctx.fillStyle = line.color;
      ctx.fillText(line.text, 0, y);
    });
  };

  const updateLines = () => {
    if (!lines.current || lines.current.length === 0) return;

    // Update 1-2 random lines at a time
    const updateCount = Math.max(1, Math.floor(Math.random() * 3));

    for (let i = 0; i < updateCount; i++) {
      const index = Math.floor(Math.random() * lines.current.length);
      if (!lines.current[index]) continue;

      lines.current[index].text = generateLine(gridInfo.current.charsPerRow);
      lines.current[index].targetColor = getRandomColor();

      if (!smooth) {
        lines.current[index].color = lines.current[index].targetColor;
        lines.current[index].colorProgress = 1;
      } else {
        lines.current[index].colorProgress = 0;
      }
    }
  };

  const handleSmoothTransitions = () => {
    let needsRedraw = false;
    lines.current.forEach(line => {
      if (line.colorProgress < 1) {
        line.colorProgress += 0.03;
        if (line.colorProgress > 1) line.colorProgress = 1;

        const startRgb = parseColor(line.color);
        const endRgb = parseColor(line.targetColor);
        line.color = interpolateColor(startRgb, endRgb, line.colorProgress);
        needsRedraw = true;
      }
    });

    if (needsRedraw) {
      drawLines();
    }
  };

  const animate = () => {
    const now = Date.now();
    if (now - lastGlitchTime.current >= glitchSpeed) {
      updateLines();
      drawLines();
      lastGlitchTime.current = now;
    }

    if (smooth) {
      handleSmoothTransitions();
    }

    animationRef.current = requestAnimationFrame(animate);
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    context.current = canvas.getContext('2d');
    resizeCanvas();
    animate();

    let resizeTimeout: NodeJS.Timeout;
    const handleResize = () => {
      clearTimeout(resizeTimeout);
      resizeTimeout = setTimeout(() => {
        cancelAnimationFrame(animationRef.current as number);
        resizeCanvas();
        animate();
      }, 100);
    };

    window.addEventListener('resize', handleResize);
    return () => {
      cancelAnimationFrame(animationRef.current!);
      window.removeEventListener('resize', handleResize);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [glitchSpeed, smooth, isDark]);

  useEffect(() => {
    if (lines.current.length > 0) {
      lines.current.forEach(line => {
        line.targetColor = getRandomColor();
        line.colorProgress = 0;
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isDark]);

  return (
    <div className="fixed inset-0 -z-10 overflow-hidden">
      <canvas ref={canvasRef} className="block w-full h-full" />
      {outerVignette && (
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background: isDark
              ? 'radial-gradient(circle, transparent 30%, hsl(var(--background)) 90%)'
              : 'radial-gradient(circle, transparent 30%, hsl(var(--background)) 90%)'
          }}
        />
      )}
    </div>
  );
};

export default LetterGlitch;
