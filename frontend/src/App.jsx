import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import { LayoutDashboard, Radio, FolderArchive, ShieldAlert, Sun, Moon } from 'lucide-react';

import Dashboard from './pages/Dashboard';
import LivePrediction from './pages/LivePrediction';
import EnsembleDefense from './pages/EnsembleDefense';
import BatchAnalysis from './pages/BatchAnalysis';
import { getSystemStatus } from './services/api';

const Sidebar = ({ systemStatus, theme, onToggleTheme }) => {
  return (
    <div className="sidebar" style={{ background: 'var(--sidebar-bg)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '2rem' }}>
        <h2 style={{ margin: 0 }}>NIDS-DL</h2>
      </div>

      <nav style={{ flex: 1 }}>
        <NavLink to="/" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          <LayoutDashboard size={20} />
          Dashboard
        </NavLink>
        <NavLink to="/live" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          <Radio size={20} />
          Live Prediction
        </NavLink>
        <NavLink to="/ensemble" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          <ShieldAlert size={20} />
          Ensemble Defense
        </NavLink>
        <NavLink to="/batch" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          <FolderArchive size={20} />
          Batch Analysis
        </NavLink>
      </nav>

      <div className="glass-panel" style={{ padding: '1rem', marginTop: 'auto' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
          <h4 style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: 0, textTransform: 'uppercase' }}>System Status</h4>
          {/* Theme toggle button */}
          <button
            onClick={onToggleTheme}
            title={theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '5px',
              padding: '4px 10px',
              borderRadius: '20px',
              border: '1px solid var(--glass-border)',
              background: 'var(--glass-bg)',
              color: 'var(--text-secondary)',
              cursor: 'pointer',
              fontSize: '0.75rem',
              fontWeight: 500,
              fontFamily: 'inherit',
              transition: 'all 0.2s ease',
            }}
            onMouseEnter={e => { e.currentTarget.style.color = 'var(--text-primary)'; e.currentTarget.style.borderColor = 'var(--primary-color)'; }}
            onMouseLeave={e => { e.currentTarget.style.color = 'var(--text-secondary)'; e.currentTarget.style.borderColor = 'var(--glass-border)'; }}
          >
            {theme === 'dark'
              ? <><Sun size={13} /> Light</>
              : <><Moon size={13} /> Dark</>
            }
          </button>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
          <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: systemStatus ? '#10b981' : '#f59e0b' }} />
          <span style={{ fontSize: '0.9rem' }}>{systemStatus ? systemStatus.status : 'Connecting...'}</span>
        </div>
        {systemStatus && (
          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            Device: <span style={{ color: 'var(--text-primary)' }}>{systemStatus.device}</span>
          </div>
        )}
      </div>
    </div>
  );
};

function App() {
  const [systemStatus, setSystemStatus] = useState(null);
  const [theme, setTheme] = useState(() => localStorage.getItem('nids-theme') || 'dark');

  // Apply theme to document root
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('nids-theme', theme);
  }, [theme]);

  const toggleTheme = () => setTheme(t => t === 'dark' ? 'light' : 'dark');

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const data = await getSystemStatus();
        setSystemStatus(data);
      } catch (error) {
        console.error("Failed to fetch system status:", error);
      }
    };
    fetchStatus();
  }, []);

  return (
    <Router>
      <div className="app-container">
        <Sidebar systemStatus={systemStatus} theme={theme} onToggleTheme={toggleTheme} />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard systemStatus={systemStatus} />} />
            <Route path="/live" element={<LivePrediction systemStatus={systemStatus} />} />
            <Route path="/ensemble" element={<EnsembleDefense systemStatus={systemStatus} />} />
            <Route path="/batch" element={<BatchAnalysis systemStatus={systemStatus} />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;

