import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import { LayoutDashboard, Radio, FolderArchive, ShieldAlert } from 'lucide-react';

import Dashboard from './pages/Dashboard';
import LivePrediction from './pages/LivePrediction';
import BatchAnalysis from './pages/BatchAnalysis';
import { getSystemStatus } from './services/api';

const Sidebar = ({ systemStatus }) => {
  return (
    <div className="sidebar">
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '2rem' }}>
        <ShieldAlert size={32} color="var(--primary-color)" />
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
        <NavLink to="/batch" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          <FolderArchive size={20} />
          Batch Analysis
        </NavLink>
      </nav>

      <div className="glass-panel" style={{ padding: '1rem', marginTop: 'auto' }}>
        <h4 style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.5rem', textTransform: 'uppercase' }}>System Status</h4>
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
        <Sidebar systemStatus={systemStatus} />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard systemStatus={systemStatus} />} />
            <Route path="/live" element={<LivePrediction systemStatus={systemStatus} />} />
            <Route path="/batch" element={<BatchAnalysis systemStatus={systemStatus} />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
