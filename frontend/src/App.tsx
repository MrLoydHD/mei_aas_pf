import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Shield, BarChart3, Search, Settings, Activity, Moon, Sun } from 'lucide-react';
import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import Dashboard from './pages/Dashboard';
import Scanner from './pages/Scanner';
import Models from './pages/Models';

function ThemeToggle() {
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const shouldBeDark = stored === 'dark' || (!stored && prefersDark);
    setIsDark(shouldBeDark);
    document.documentElement.classList.toggle('dark', shouldBeDark);
  }, []);

  const toggleTheme = () => {
    const newValue = !isDark;
    setIsDark(newValue);
    document.documentElement.classList.toggle('dark', newValue);
    localStorage.setItem('theme', newValue ? 'dark' : 'light');
  };

  return (
    <Button variant="ghost" size="icon" onClick={toggleTheme}>
      {isDark ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
    </Button>
  );
}

function Navigation() {
  const location = useLocation();

  const isActive = (path: string) => location.pathname === path;

  const navItems = [
    { path: '/', icon: BarChart3, label: 'Dashboard' },
    { path: '/scanner', icon: Search, label: 'Scanner' },
    { path: '/models', icon: Settings, label: 'Models' },
  ];

  return (
    <nav className="bg-card border-b border-border">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <Shield className="h-8 w-8 text-primary" />
            <span className="ml-2 text-xl font-bold text-foreground">DGA Detector</span>
          </div>

          <div className="flex items-center space-x-2">
            {navItems.map(({ path, icon: Icon, label }) => (
              <Link key={path} to={path}>
                <Button
                  variant={isActive(path) ? 'default' : 'ghost'}
                  className="flex items-center gap-1"
                >
                  <Icon className="h-4 w-4" />
                  {label}
                </Button>
              </Link>
            ))}
            <ThemeToggle />
          </div>
        </div>
      </div>
    </nav>
  );
}

function StatusIndicator() {
  return (
    <div className="fixed bottom-4 right-4 flex items-center bg-card rounded-full shadow-lg px-4 py-2 border border-border">
      <Activity className="h-4 w-4 text-success-500 mr-2 animate-pulse" />
      <span className="text-sm text-muted-foreground">API Connected</span>
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-background">
        <Navigation />
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/scanner" element={<Scanner />} />
            <Route path="/models" element={<Models />} />
          </Routes>
        </main>
        <StatusIndicator />
      </div>
    </BrowserRouter>
  );
}

export default App;
