import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { BarChart3, Search, Settings } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ThemeToggle } from '@/components/ThemeToggle';
import { GoogleSignIn } from '@/components/GoogleSignIn';

const navItems = [
  { path: '/dashboard', icon: BarChart3, label: 'Dashboard' },
  { path: '/scanner', icon: Search, label: 'Scanner' },
  { path: '/models', icon: Settings, label: 'Models' },
];

export function Navbar() {
  const location = useLocation();

  const isActive = (path: string) => location.pathname === path;

  return (
    <motion.nav
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.4, ease: 'easeOut' }}
      className="bg-card/90 backdrop-blur-sm border-b border-border relative z-10"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center py-2">
          {/* Logo - Left */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4, delay: 0.1 }}
          >
            <Link to="/" className="flex items-center">
              <img src="/logo.png" alt="DGA Detector" className="h-12" />
            </Link>
          </motion.div>

          {/* Nav Items - Center */}
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.2 }}
            className="flex items-center space-x-1"
          >
            {navItems.map(({ path, icon: Icon, label }, index) => (
              <motion.div
                key={path}
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: 0.2 + index * 0.1 }}
              >
                <Link to={path}>
                  <Button
                    variant={isActive(path) ? 'default' : 'ghost'}
                    className="flex items-center gap-2"
                  >
                    <Icon className="h-4 w-4" />
                    {label}
                  </Button>
                </Link>
              </motion.div>
            ))}
          </motion.div>

          {/* Auth & Theme - Right */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4, delay: 0.3 }}
            className="flex items-center space-x-2"
          >
            <ThemeToggle />
            <div className="border-l border-border pl-2 ml-2">
              <GoogleSignIn />
            </div>
          </motion.div>
        </div>
      </div>
    </motion.nav>
  );
}
