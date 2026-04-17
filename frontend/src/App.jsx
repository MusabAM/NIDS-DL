import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import { LayoutDashboard, Radio, FolderArchive, ShieldAlert, Sun, Moon, Cpu, Zap } from 'lucide-react';

import Dashboard from './pages/Dashboard';
import LivePrediction from './pages/LivePrediction';
import EnsembleDefense from './pages/EnsembleDefense';
import BatchAnalysis from './pages/BatchAnalysis';
import { getSystemStatus, setDevice } from './services/api';

const Sidebar = ({ systemStatus, theme, onToggleTheme, onToggleDevice }) => {
  const isGpu = systemStatus?.device === 'CUDA';
  const cudaAvailable = systemStatus?.cuda_available;

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

        {/* ── Header row: title + theme toggle ────────────────── */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
          <h4 style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: 0, textTransform: 'uppercase' }}>
            System Status
          </h4>
          <button
            id="theme-toggle-btn"
            onClick={onToggleTheme}
            title={theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            style={{
              display: 'flex', alignItems: 'center', gap: '5px',
              padding: '4px 10px', borderRadius: '20px',
              border: '1px solid var(--glass-border)',
              background: 'var(--glass-bg)',
              color: 'var(--text-secondary)',
              cursor: 'pointer', fontSize: '0.75rem', fontWeight: 500,
              fontFamily: 'inherit', transition: 'all 0.2s ease',
            }}
            onMouseEnter={e => { e.currentTarget.style.color = 'var(--text-primary)'; e.currentTarget.style.borderColor = 'var(--primary-color)'; }}
            onMouseLeave={e => { e.currentTarget.style.color = 'var(--text-secondary)'; e.currentTarget.style.borderColor = 'var(--glass-border)'; }}
          >
            {theme === 'dark' ? <><Sun size={13} /> Light</> : <><Moon size={13} /> Dark</>}
          </button>
        </div>

        {/* ── Status dot + label ──────────────────────────────── */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
          <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: systemStatus ? '#10b981' : '#f59e0b', flexShrink: 0 }} />
          <span style={{ fontSize: '0.9rem' }}>{systemStatus ? systemStatus.status : 'Connecting...'}</span>
        </div>

        {/* ── Active device label ─────────────────────────────── */}
        {systemStatus && (
          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
            Device:{' '}
            <span style={{ color: isGpu ? 'var(--secondary-color)' : 'var(--text-primary)', fontWeight: 600 }}>
              {systemStatus.device}
            </span>
          </div>
        )}

        {/* ── GPU / CPU toggle button ─────────────────────────── */}
        <button
          id="device-toggle-btn"
          onClick={onToggleDevice}
          disabled={!cudaAvailable}
          title={
            !cudaAvailable
              ? 'CUDA not available — restart backend with venv python to enable GPU'
              : isGpu
                ? 'Switch inference to CPU'
                : 'Switch inference to GPU (CUDA)'
          }
          style={{
            width: '100%',
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '7px',
            padding: '8px 12px', borderRadius: '8px',
            border: `1px solid ${isGpu ? 'rgba(16,185,129,0.3)' : 'rgba(59,130,246,0.3)'}`,
            background: isGpu ? 'rgba(16,185,129,0.08)' : 'rgba(59,130,246,0.08)',
            color: isGpu ? 'var(--secondary-color)' : 'var(--primary-color)',
            cursor: cudaAvailable ? 'pointer' : 'not-allowed',
            opacity: cudaAvailable ? 1 : 0.4,
            fontSize: '0.8rem', fontWeight: 600,
            fontFamily: 'inherit', transition: 'all 0.25s ease',
          }}
        >
          {isGpu
            ? <><Zap size={14} fill="currentColor" /> GPU Active &mdash; Switch to CPU</>
            : <><Cpu size={14} /> CPU Active &mdash; Switch to GPU</>
          }
        </button>

        {/* ── Hint when CUDA not available ───────────────────── */}
        {systemStatus && !cudaAvailable && (
          <p style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', margin: '6px 0 0 0', textAlign: 'center', opacity: 0.65, lineHeight: 1.4 }}>
            Restart the backend with<br />
            <code style={{ background: 'var(--code-bg)', padding: '1px 4px', borderRadius: '3px', color: 'var(--text-primary)' }}>venv\Scripts\python</code><br />
            to enable CUDA.
          </p>
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

  const toggleDevice = async () => {
    if (!systemStatus?.cuda_available) return;
    const next = systemStatus.device === 'CUDA' ? 'cpu' : 'cuda';
    try {
      const result = await setDevice(next);
      setSystemStatus(prev => ({ ...prev, device: result.device }));
    } catch (err) {
      console.error('Failed to toggle device:', err);
    }
  };

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
        <Sidebar
          systemStatus={systemStatus}
          theme={theme}
          onToggleTheme={toggleTheme}
          onToggleDevice={toggleDevice}
        />
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
