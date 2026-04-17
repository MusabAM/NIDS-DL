import React, { useState } from 'react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip } from 'recharts';
import { Activity, ShieldCheck, Target, Layers } from 'lucide-react';

const Dashboard = ({ systemStatus }) => {
    const [selectedModel, setSelectedModel] = useState('CNN');

    // Model Performance Data (normalized 0-100 for radar chart scaling)
    const modelMetrics = {
        'CNN': [
            { metric: 'Accuracy', value: 96.43, fullMark: 100 },
            { metric: 'Precision', value: 96.12, fullMark: 100 },
            { metric: 'Recall', value: 96.85, fullMark: 100 },
            { metric: 'F1-Score', value: 96.48, fullMark: 100 },
            { metric: 'ROC AUC', value: 99.10, fullMark: 100 },
        ],
        'LSTM': [
            { metric: 'Accuracy', value: 95.90, fullMark: 100 },
            { metric: 'Precision', value: 95.50, fullMark: 100 },
            { metric: 'Recall', value: 96.20, fullMark: 100 },
            { metric: 'F1-Score', value: 95.85, fullMark: 100 },
            { metric: 'ROC AUC', value: 98.80, fullMark: 100 },
        ],
        'Transformer': [
            { metric: 'Accuracy', value: 96.05, fullMark: 100 },
            { metric: 'Precision', value: 96.00, fullMark: 100 },
            { metric: 'Recall', value: 96.10, fullMark: 100 },
            { metric: 'F1-Score', value: 96.05, fullMark: 100 },
            { metric: 'ROC AUC', value: 98.95, fullMark: 100 },
        ],
        'Autoencoder': [
            { metric: 'Accuracy', value: 95.00, fullMark: 100 },
            { metric: 'Precision', value: 94.20, fullMark: 100 },
            { metric: 'Recall', value: 95.80, fullMark: 100 },
            { metric: 'F1-Score', value: 95.00, fullMark: 100 },
            { metric: 'ROC AUC', value: 97.50, fullMark: 100 },
        ]
    };

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
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                    <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px', margin: 0 }}>
                        <Layers size={20} color="var(--primary-color)" />
                        Model Performance Metrics
                    </h3>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 500 }}>Select Model:</span>
                        <select
                            className="form-control"
                            style={{ width: '180px', padding: '0.4rem 0.8rem', background: 'var(--input-bg)', color: 'var(--input-color)' }}
                            value={selectedModel}
                            onChange={(e) => setSelectedModel(e.target.value)}
                        >
                            {Object.keys(modelMetrics).map(model => (
                                <option key={model} value={model}>{model}</option>
                            ))}
                        </select>
                    </div>
                </div>

                <div style={{ width: '100%', height: 400, background: 'var(--chart-bg)', borderRadius: '12px', padding: '1rem', border: '1px solid var(--chart-border)' }}>
                    <ResponsiveContainer width="100%" height="100%">
                        <RadarChart cx="50%" cy="50%" outerRadius="75%" data={modelMetrics[selectedModel]}>
                            <PolarGrid stroke="var(--chart-grid)" />
                            <PolarAngleAxis
                                dataKey="metric"
                                tick={{ fill: 'var(--text-secondary)', fontSize: 13, fontWeight: 500 }}
                            />
                            <PolarRadiusAxis
                                angle={90}
                                domain={[80, 100]}
                                stroke="var(--chart-grid)"
                                tick={{ fill: 'var(--chart-tick)', fontSize: 11 }}
                                tickCount={5}
                            />
                            <Tooltip
                                contentStyle={{ backgroundColor: 'var(--glass-bg)', borderColor: 'rgba(59, 130, 246, 0.3)', borderRadius: '8px', boxShadow: '0 4px 12px rgba(0,0,0,0.2)' }}
                                itemStyle={{ color: 'var(--text-primary)', fontWeight: 'bold' }}
                                formatter={(value) => [`${value}%`]}
                            />
                            <Radar
                                name={selectedModel}
                                dataKey="value"
                                stroke="var(--primary-color)"
                                fill="var(--primary-color)"
                                fillOpacity={0.4}
                                strokeWidth={2}
                                dot={{ r: 4, fill: "var(--accent-color)" }}
                                activeDot={{ r: 6, fill: "var(--text-primary)", stroke: "var(--primary-color)", strokeWidth: 2 }}
                            />
                        </RadarChart>
                    </ResponsiveContainer>
                </div>

                <div style={{ marginTop: '1rem', display: 'flex', justifyContent: 'center', gap: '2rem' }}>
                    <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '1px' }}>Accuracy</div>
                        <div style={{ fontSize: '1.25rem', fontWeight: 'bold', color: 'var(--text-primary)' }}>{modelMetrics[selectedModel][0].value}%</div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '1px' }}>F1-Score</div>
                        <div style={{ fontSize: '1.25rem', fontWeight: 'bold', color: 'var(--text-primary)' }}>{modelMetrics[selectedModel][3].value}%</div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '1px' }}>ROC-AUC</div>
                        <div style={{ fontSize: '1.25rem', fontWeight: 'bold', color: 'var(--text-primary)' }}>{modelMetrics[selectedModel][4].value}%</div>
                    </div>
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
