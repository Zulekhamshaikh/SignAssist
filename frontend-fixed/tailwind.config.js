/** @type {import('tailwindcss').Config} */
export default {
  content: [
    // This array tells Tailwind to scan all JavaScript, JSX, TS, and TSX files
    // inside the 'src' directory and all its subfolders.
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}