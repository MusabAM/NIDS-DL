import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Activity, ShieldCheck, Target } from 'lucide-react';

const Dashboard = ({ systemStatus }) => {
    const chartData = [
        { name: 'CNN', accuracy: 96.43, f1: 0.96 },
        { name: 'LSTM', accuracy: 95.90, f1: 0.96 },
        { name: 'Transformer', accuracy: 96.05, f1: 0.96 },
        { name: 'Autoencoder', accuracy: 95.0, f1: 0.95 },
    ];

    return (
        <div className="dashboard-page fade-in">
            <div style={{ marginBottom: '2rem' }}>
                <h1>Network Intrusion Detection System</h1>
                <p>Deep Learning Optimized Security Monitoring</p>
            </div>

            <div className="grid-cards">
                <div className="glass-panel" style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem' }}>
                    <div style={{ background: 'rgba(59, 130, 246, 0.15)', padding: '12px', borderRadius: '12px' }}>
                        <Activity color="#3b82f6" size={32} />
                    </div>
                    <div>
                        <h4 style={{ color: 'var(--text-secondary)', marginBottom: '4px' }}>System Status</h4>
                        <div style={{ fontSize: '1.8rem', fontWeight: 'bold', color: 'var(--text-primary)' }}>
                            {systemStatus ? 'Active' : 'Offline'}
                        </div>
                        <p style={{ margin: 0, fontSize: '0.85rem' }}>Monitoring enabled</p>
                    </div>
                </div>

                <div className="glass-panel" style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem' }}>
                    <div style={{ background: 'rgba(16, 185, 129, 0.15)', padding: '12px', borderRadius: '12px' }}>
                        <Target color="#10b981" size={32} />
                    </div>
                    <div>
                        <h4 style={{ color: 'var(--text-secondary)', marginBottom: '4px' }}>Best Model Accuracy</h4>
                        <div style={{ fontSize: '1.8rem', fontWeight: 'bold', color: 'var(--text-primary)' }}>
                            96.43%
                        </div>
                        <p style={{ margin: 0, fontSize: '0.85rem' }}>CNN on CICIDS2018</p>
                    </div>
                </div>

                <div className="glass-panel" style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem' }}>
                    <div style={{ background: 'rgba(139, 92, 246, 0.15)', padding: '12px', borderRadius: '12px' }}>
                        <ShieldCheck color="#8b5cf6" size={32} />
                    </div>
                    <div>
                        <h4 style={{ color: 'var(--text-secondary)', marginBottom: '4px' }}>Supported Attacks</h4>
                        <div style={{ fontSize: '1.8rem', fontWeight: 'bold', color: 'var(--text-primary)' }}>
                            39+
                        </div>
                        <p style={{ margin: 0, fontSize: '0.85rem' }}>DoS, Probe, R2L, U2R</p>
                    </div>
                </div>
            </div>

            <div className="glass-panel" style={{ marginBottom: '2rem' }}>
                <h3 style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    Model Performance Overview
                </h3>
                <div style={{ width: '100%', height: 350 }}>
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                            data={chartData}
                            margin={{ top: 20, right: 30, left: 0, bottom: 5 }}
                        >
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                            <XAxis dataKey="name" stroke="var(--text-secondary)" tick={{ fill: 'var(--text-secondary)' }} />
                            <YAxis stroke="var(--text-secondary)" tick={{ fill: 'var(--text-secondary)' }} />
                            <Tooltip
                                contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px' }}
                                itemStyle={{ color: '#fff' }}
                            />
                            <Legend wrapperStyle={{ paddingTop: '20px' }} />
                            <Bar dataKey="accuracy" name="Accuracy (%)" fill="var(--primary-color)" radius={[4, 4, 0, 0]} barSize={40} />
                            <Bar dataKey="f1" name="F1-Score" fill="var(--accent-color)" radius={[4, 4, 0, 0]} barSize={40} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            <div className="glass-panel" style={{ borderLeft: '4px solid var(--primary-color)' }}>
                <h3>Recent Alerts</h3>
                <p style={{ color: 'var(--text-primary)', margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span className="badge badge-success">Safe</span> No recent high-severity alerts detected in live traffic stream.
                </p>
            </div>
        </div>
    );
};

export default Dashboard;
