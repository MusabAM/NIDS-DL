import React, { useState, useEffect } from 'react';
import { predictLive, getLiveHistory } from '../services/api';
import { AlertTriangle, CheckCircle, Loader2, Play, Square, Activity } from 'lucide-react';

const LivePrediction = ({ systemStatus }) => {
    const [dataset] = useState('CICIDS2018'); // Locked to CICIDS2018
    const [modelType, setModelType] = useState('CNN');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [mode, setMode] = useState('manual');

    // Streaming History State
    const [isStreaming, setIsStreaming] = useState(false);
    const [streamHistory, setStreamHistory] = useState([]);

    // CICIDS2018 features state
    const [features, setFeatures] = useState({
        'Flow Duration': 0,
        'Tot Fwd Pkts': 1,
        'Tot Bwd Pkts': 0,
        'Fwd Pkt Len Mean': 100.0,
        'Bwd Pkt Len Mean': 0.0,
        'Flow Byts/s': 1000.0,
        'Flow Pkts/s': 10.0,
        'Flow IAT Mean': 0.0,
        'Fwd IAT Mean': 0.0,
        'Bwd IAT Mean': 0.0,
        'Pkt Len Mean': 50.0,
        'Down/Up Ratio': 0
    });

    const availableModels = systemStatus?.models?.[dataset] || ['CNN', 'LSTM', 'Transformer', 'Autoencoder'];

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setFeatures(prev => ({ ...prev, [name]: Number(value) }));
    };

    const submitPrediction = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setResult(null);

        try {
            const data = await predictLive(dataset, modelType, features);
            setResult(data);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to connect to backend.');
        } finally {
            setLoading(false);
        }
    };

    // Auto-stream effect
    useEffect(() => {
        let interval;
        if (isStreaming) {
            interval = setInterval(async () => {
                try {
                    const data = await getLiveHistory();
                    if (data && data.history) {
                        setStreamHistory(data.history);
                    }
                } catch (e) {
                    console.error("Stream polling error:", e);
                }
            }, 1000);
        }
        return () => clearInterval(interval);
    }, [isStreaming]);

    return (
        <div className="live-prediction-page fade-in">
            <div style={{ marginBottom: '2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                    <h2>Live Traffic Analysis</h2>
                    <p>Enter network flow parameters or listen to live network interfaces.</p>
                </div>
                <div style={{ display: 'flex', gap: '0.5rem', background: 'rgba(0,0,0,0.2)', padding: '0.5rem', borderRadius: '8px' }}>
                    <button
                        className={`btn ${mode === 'manual' ? 'btn-primary' : ''}`}
                        style={{ padding: '0.5rem 1rem', background: mode !== 'manual' ? 'transparent' : '' }}
                        onClick={() => { setMode('manual'); setIsStreaming(false); }}
                    >
                        Manual Entry
                    </button>
                    <button
                        className={`btn ${mode === 'stream' ? 'btn-primary' : ''}`}
                        style={{ padding: '0.5rem 1rem', background: mode !== 'stream' ? 'transparent' : '' }}
                        onClick={() => setMode('stream')}
                    >
                        Live Stream
                    </button>
                </div>
            </div>

            <div className="glass-panel" style={{ marginBottom: '2rem' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                    <div className="form-group">
                        <label className="form-label">Dataset Mode</label>
                        <select className="form-control" value={dataset} disabled>
                            <option value="CICIDS2018">CICIDS2018</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label className="form-label">Active Analysis Model</label>
                        <select className="form-control" value={modelType} onChange={(e) => setModelType(e.target.value)}>
                            {availableModels.map(m => <option key={m} value={m}>{m}</option>)}
                        </select>
                    </div>
                </div>
            </div>

            {mode === 'manual' ? (
                <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 1fr)', gap: '2rem' }}>
                    <form className="glass-panel" onSubmit={submitPrediction}>
                        <h3 style={{ marginBottom: '1.5rem' }}>Flow Characteristics (CICIDS2018)</h3>

                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem' }}>
                            <div className="form-group">
                                <label className="form-label">Flow Duration</label>
                                <input type="number" name="Flow Duration" className="form-control" value={features['Flow Duration']} onChange={handleInputChange} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Tot Fwd Pkts</label>
                                <input type="number" name="Tot Fwd Pkts" className="form-control" value={features['Tot Fwd Pkts']} onChange={handleInputChange} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Tot Bwd Pkts</label>
                                <input type="number" name="Tot Bwd Pkts" className="form-control" value={features['Tot Bwd Pkts']} onChange={handleInputChange} />
                            </div>

                            <div className="form-group">
                                <label className="form-label">Fwd Pkt Len Mean</label>
                                <input type="number" step="0.1" name="Fwd Pkt Len Mean" className="form-control" value={features['Fwd Pkt Len Mean']} onChange={handleInputChange} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Bwd Pkt Len Mean</label>
                                <input type="number" step="0.1" name="Bwd Pkt Len Mean" className="form-control" value={features['Bwd Pkt Len Mean']} onChange={handleInputChange} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Pkt Len Mean</label>
                                <input type="number" step="0.1" name="Pkt Len Mean" className="form-control" value={features['Pkt Len Mean']} onChange={handleInputChange} />
                            </div>

                            <div className="form-group">
                                <label className="form-label">Flow Byts/s</label>
                                <input type="number" step="0.1" name="Flow Byts/s" className="form-control" value={features['Flow Byts/s']} onChange={handleInputChange} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Flow Pkts/s</label>
                                <input type="number" step="0.1" name="Flow Pkts/s" className="form-control" value={features['Flow Pkts/s']} onChange={handleInputChange} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Down/Up Ratio</label>
                                <input type="number" step="0.1" name="Down/Up Ratio" className="form-control" value={features['Down/Up Ratio']} onChange={handleInputChange} />
                            </div>

                            <div className="form-group">
                                <label className="form-label">Flow IAT Mean</label>
                                <input type="number" step="0.1" name="Flow IAT Mean" className="form-control" value={features['Flow IAT Mean']} onChange={handleInputChange} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Fwd IAT Mean</label>
                                <input type="number" step="0.1" name="Fwd IAT Mean" className="form-control" value={features['Fwd IAT Mean']} onChange={handleInputChange} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Bwd IAT Mean</label>
                                <input type="number" step="0.1" name="Bwd IAT Mean" className="form-control" value={features['Bwd IAT Mean']} onChange={handleInputChange} />
                            </div>
                        </div>

                        <button type="submit" className="btn btn-primary" style={{ marginTop: '1.5rem', width: '100%' }} disabled={loading}>
                            {loading ? <><Loader2 size={18} className="animate-spin" /> Analyzing...</> : 'Analyze Traffic'}
                        </button>
                    </form>

                    <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column' }}>
                        <h3 style={{ marginBottom: '1.5rem' }}>Analysis Result</h3>

                        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', textAlign: 'center' }}>
                            {!result && !error && !loading && (
                                <p style={{ color: 'var(--text-secondary)' }}>Awaiting parameters for analysis...</p>
                            )}

                            {loading && (
                                <div style={{ padding: '2rem' }}>
                                    <Loader2 size={48} color="var(--primary-color)" className="animate-spin mb-4" />
                                    <p>Running Inference Model...</p>
                                </div>
                            )}

                            {error && (
                                <div style={{ color: 'var(--danger-color)', padding: '1rem', background: 'rgba(239,68,68,0.1)', borderRadius: '8px' }}>
                                    <AlertTriangle size={32} style={{ marginBottom: '8px' }} />
                                    <p style={{ margin: 0, color: 'inherit' }}>{error}</p>
                                </div>
                            )}

                            {result && !loading && (
                                <div className="fade-in" style={{ width: '100%' }}>
                                    {result.prediction === 'Attack' ? (
                                        <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', padding: '2rem 1rem', borderRadius: '12px' }}>
                                            <AlertTriangle size={64} color="var(--danger-color)" style={{ marginBottom: '1rem' }} />
                                            <h2 style={{ color: 'var(--danger-color)', marginBottom: '0.5rem' }}>Threat Detected</h2>
                                            <p style={{ margin: 0, fontWeight: 600 }}>ATTACK</p>
                                        </div>
                                    ) : (
                                        <div style={{ background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.3)', padding: '2rem 1rem', borderRadius: '12px' }}>
                                            <CheckCircle size={64} color="var(--secondary-color)" style={{ marginBottom: '1rem' }} />
                                            <h2 style={{ color: 'var(--secondary-color)', marginBottom: '0.5rem' }}>Traffic Status</h2>
                                            <p style={{ margin: 0, fontWeight: 600 }}>NORMAL</p>
                                        </div>
                                    )}

                                    <div style={{ marginTop: '2rem', padding: '1rem', background: 'rgba(0,0,0,0.2)', borderRadius: '8px' }}>
                                        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>
                                            {result.metric_type}
                                        </div>
                                        <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
                                            {result.metric_type === 'Confidence' ? `${(result.confidence * 100).toFixed(2)}%` : result.confidence.toFixed(4)}
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            ) : (
                <div className="glass-panel fade-in" style={{ minHeight: '500px', display: 'flex', flexDirection: 'column' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '1rem' }}>
                        <div>
                            <h3 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
                                <Activity size={20} color="var(--primary-color)" />
                                Real-Time Sniffer Stream
                            </h3>
                            <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                                Run <code style={{ color: '#fff', background: 'rgba(0,0,0,0.3)', padding: '2px 4px', borderRadius: '4px' }}>venv/Scripts/python scripts/live_sniffer.py</code> internally.
                            </p>
                        </div>
                        <button
                            className={`btn ${isStreaming ? 'btn-danger' : 'btn-success'}`}
                            style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
                            onClick={() => setIsStreaming(!isStreaming)}
                        >
                            {isStreaming ? <><Square size={16} fill="currentColor" /> Stop Stream</> : <><Play size={16} fill="currentColor" /> Connect Stream</>}
                        </button>
                    </div>

                    <div style={{ flex: 1, background: 'rgba(0,0,0,0.3)', borderRadius: '8px', padding: '1rem', overflowY: 'auto', maxHeight: '500px' }}>
                        {!isStreaming && streamHistory.length === 0 && (
                            <div style={{ height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)' }}>
                                Stream is paused. Click Connect to begin polling.
                            </div>
                        )}

                        {isStreaming && streamHistory.length === 0 && (
                            <div style={{ height: '300px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)' }}>
                                <Loader2 size={32} className="animate-spin" style={{ marginBottom: '1rem', color: 'var(--primary-color)' }} />
                                Listening for live packets on {modelType}...
                            </div>
                        )}

                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                            {streamHistory.map((item, idx) => (
                                <div key={idx} className="fade-in" style={{
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    alignItems: 'center',
                                    padding: '1rem',
                                    background: item.prediction === 'Attack' ? 'rgba(239,68,68,0.1)' : 'rgba(16,185,129,0.1)',
                                    borderLeft: `4px solid ${item.prediction === 'Attack' ? 'var(--danger-color)' : 'var(--secondary-color)'}`,
                                    borderRadius: '4px'
                                }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                                        {item.prediction === 'Attack' ? <AlertTriangle color="var(--danger-color)" size={20} /> : <CheckCircle color="var(--secondary-color)" size={20} />}
                                        <div>
                                            <div style={{ fontWeight: 'bold', color: item.prediction === 'Attack' ? 'var(--danger-color)' : 'var(--secondary-color)' }}>
                                                {item.prediction}
                                            </div>
                                            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                                                Flow Received @ {item.timestamp}
                                            </div>
                                        </div>
                                    </div>
                                    <div style={{ textAlign: 'right' }}>
                                        <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{item.metric_type}</div>
                                        <div style={{ fontWeight: 'bold' }}>
                                            {item.metric_type === 'Confidence' ? `${(item.confidence * 100).toFixed(2)}%` : item.confidence.toFixed(4)}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default LivePrediction;
