import { Outlet } from 'react-router-dom';
import { Navbar } from './navbar';
import { StatusIndicator } from './StatusIndicator';
import LetterGlitch from '@/components/LetterGlitch';

export function Layout() {
  return (
    <div>
      <LetterGlitch glitchSpeed={80} outerVignette={true} smooth={true} />
      <Navbar />
      <main className="max-w-7xl mx-auto px-4 sm:px-6  py-8 relative">
        <Outlet />
      </main>
      <StatusIndicator />
    </div>
  );
}
