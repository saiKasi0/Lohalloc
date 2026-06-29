/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Advanced Hardware Terminal palette
        canvas: '#0A0A0A',       // true black void
        ink: '#E5E0D8',          // architectural tan/parchment
        'ink-muted': '#8A857D',  // dimmed tan for secondary text
        'ink-faint': '#3A3733',  // very dim tan for subtle borders
        heat: '#FF2E2E',         // glowing crimson hotspots
        'heat-dim': '#7A1717',   // dimmed crimson
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
      },
      borderRadius: {
        // Force hard edges everywhere
        none: '0',
      },
      boxShadow: {
        // Crimson glow used sparingly for hotspots only
        'heat-glow': '0 0 4px #FF2E2E, 0 0 16px rgba(255, 46, 46, 0.4)',
        'heat-glow-sm': '0 0 2px #FF2E2E, 0 0 8px rgba(255, 46, 46, 0.3)',
      },
    },
  },
  plugins: [],
};