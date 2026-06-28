/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        'lohalloc-dark': '#0f0f1a',
        'lohalloc-panel': '#1a1a2e',
        'lohalloc-accent': '#6366f1',
      },
    },
  },
  plugins: [],
};