import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Shield, BarChart3, Search, Settings, Activity } from 'lucide-react';
import Dashboard from './pages/Dashboard';
import Scanner from './pages/Scanner';
import Models from './pages/Models';

function Navigation() {
  const location = useLocation();

  const isActive = (path: string) => location.pathname === path;

  const navItems = [
    { path: '/', icon: BarChart3, label: 'Dashboard' },
    { path: '/scanner', icon: Search, label: 'Scanner' },
    { path: '/models', icon: Settings, label: 'Models' },
  ];

  return (
    <nav className="bg-white shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <Shield className="h-8 w-8 text-primary-600" />
            <span className="ml-2 text-xl font-bold text-gray-900">DGA Detector</span>
          </div>

          <div className="flex items-center space-x-4">
            {navItems.map(({ path, icon: Icon, label }) => (
              <Link
                key={path}
                to={path}
                className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive(path)
                    ? 'bg-primary-100 text-primary-700'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                <Icon className="h-5 w-5 mr-1" />
                {label}
              </Link>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
}

function StatusIndicator() {
  return (
    <div className="fixed bottom-4 right-4 flex items-center bg-white rounded-full shadow-lg px-4 py-2">
      <Activity className="h-4 w-4 text-success-500 mr-2 animate-pulse" />
      <span className="text-sm text-gray-600">API Connected</span>
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50">
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
