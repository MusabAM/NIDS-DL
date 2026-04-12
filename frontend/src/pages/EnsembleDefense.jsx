import React, { useState, useEffect } from 'react';
import { predictLive, getLiveHistory } from '../services/api';
import { AlertTriangle, CheckCircle, Loader2, Play, Square, Activity } from 'lucide-react';

// Feature templates per dataset
const DATASET_FEATURES = {
    'CICIDS2018': {
        'Flow Duration': 0,
        'Tot Fwd Pkts': 1,
        'Tot Bwd Pkts': 0,
        'Fwd Pkt Len Max': 100.0,
        'Bwd Pkt Len Max': 0.0,
        'Flow Byts/s': 1000.0,
        'Flow Pkts/s': 10.0,
        'Flow IAT Mean': 0.0,
        'Fwd IAT Mean': 0.0,
        'Bwd IAT Mean': 0.0,
        'Pkt Len Mean': 50.0,
        'Down/Up Ratio': 0,
    },
    'NSL-KDD': {
        duration: 0,
        src_bytes: 0,
        dst_bytes: 0,
        land: 0,
        wrong_fragment: 0,
        urgent: 0,
        hot: 0,
        num_failed_logins: 0,
        logged_in: 0,
        count: 1,
        srv_count: 1,
        serror_rate: 0.0,
        rerror_rate: 0.0,
        same_srv_rate: 1.0,
        diff_srv_rate: 0.0,
        dst_host_count: 255,
        dst_host_srv_count: 255,
        dst_host_same_srv_rate: 1.0,
        dst_host_diff_srv_rate: 0.0,
        dst_host_serror_rate: 0.0,
    },
    'UNSW-NB15': {
        dur: 0.0,
        spkts: 1,
        dpkts: 0,
        sbytes: 100,
        dbytes: 0,
        rate: 10.0,
        sttl: 254,
        dttl: 0,
        sload: 1000.0,
        dload: 0.0,
        sloss: 0,
        dloss: 0,
    },
    'CICIDS2017': {
        'Flow Duration': 0.0,
        'Total Fwd Packets': 1,
        'Total Backward Packets': 0,
        'Total Length of Fwd Packets': 100,
        'Total Length of Bwd Packets': 0,
        'Fwd Packet Length Max': 50,
        'Bwd Packet Length Max': 0,
        'Flow Bytes/s': 10.0,
        'Flow Packets/s': 1.0,
        'Fwd IAT Mean': 0.0,
        'Bwd IAT Mean': 0.0,
        'Init_Win_bytes_forward': 255,
        'Init_Win_bytes_backward': 255,
        'act_data_pkt_fwd': 1,
        'min_seg_size_forward': 20,
        'SYN Flag Count': 0,
        'ACK Flag Count': 1,
        'PSH Flag Count': 0,
    },
};

const EnsembleDefense = ({ systemStatus }) => {
    const [dataset, setDataset] = useState('CICIDS2018');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [mode, setMode] = useState('manual');
    const [ensembleMode, setEnsembleMode] = useState('Ensemble');

    // Streaming History State
    const [isStreaming, setIsStreaming] = useState(false);
    const [streamHistory, setStreamHistory] = useState([]);

    // Dynamic features state
    const [features, setFeatures] = useState(DATASET_FEATURES['CICIDS2018']);

    // Swap features when dataset changes
    useEffect(() => {
        setFeatures({ ...DATASET_FEATURES[dataset] });
        setResult(null);
        setError(null);
    }, [dataset]);

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        const parsedValue = (name === 'protocol_type' || name === 'service' || name === 'flag') ? value : Number(value);
        setFeatures(prev => ({ ...prev, [name]: parsedValue }));
    };

    const submitPrediction = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setResult(null);

        try {
            const data = await predictLive(dataset, ensembleMode, features);
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
                        const ensembleHistory = data.history.filter(h => h.isEnsemble);
                        setStreamHistory(ensembleHistory);
                    }
                } catch (e) {
                    console.error("Stream polling error:", e);
                }
            }, 1000);
        }
        return () => clearInterval(interval);
    }, [isStreaming]);

    const featureEntries = Object.keys(features);
    const numCols = featureEntries.length > 12 ? 4 : 3;

    return (
        <div className="live-prediction-page fade-in">
            <div style={{ marginBottom: '2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                    <h2>Ensemble Defense</h2>
                    <p>Live, concurrent multi-model analysis leveraging supervised &amp; unsupervised checks.</p>
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
                        <select className="form-control" value={dataset} onChange={(e) => setDataset(e.target.value)}>
                            <option value="CICIDS2018">CICIDS2018</option>
                            <option value="CICIDS2017">CICIDS2017</option>
                            <option value="NSL-KDD">NSL-KDD</option>
                            <option value="UNSW-NB15">UNSW-NB15</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label className="form-label">Active Analysis Model</label>
                        <select className="form-control" value={ensembleMode} onChange={(e) => setEnsembleMode(e.target.value)}>
                            <option value="Ensemble">🛡️ Full Ensemble (Phase 1 + 2 - Recommended)</option>
                            <option value="Ensemble_Phase1">Supervised Ensemble (Phase 1 Only - Fast)</option>
                        </select>
                    </div>
                </div>
            </div>

            {mode === 'manual' ? (
                <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 1fr)', gap: '2rem' }}>
                    <form className="glass-panel" onSubmit={submitPrediction}>
                        <h3 style={{ marginBottom: '1.5rem' }}>
                            Flow Characteristics ({dataset})
                        </h3>

                        <div style={{ display: 'grid', gridTemplateColumns: `repeat(${numCols}, 1fr)`, gap: '1rem' }}>
                            {featureEntries.map((key) => (
                                <div className="form-group" key={key}>
                                    <label className="form-label" style={{ fontSize: '0.8rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }} title={key}>{key}</label>
                                    {(key === 'protocol_type' || key === 'service' || key === 'flag') ? (
                                        <input
                                            type="text"
                                            name={key}
                                            className="form-control"
                                            value={features[key]}
                                            onChange={handleInputChange}
                                        />
                                    ) : (
                                        <input
                                            type="number"
                                            step="any"
                                            name={key}
                                            className="form-control"
                                            value={features[key]}
                                            onChange={handleInputChange}
                                        />
                                    )}
                                </div>
                            ))}
                        </div>

                        <button type="submit" className="btn btn-primary" style={{ marginTop: '1.5rem', width: '100%' }} disabled={loading}>
                            {loading ? <><Loader2 size={18} className="animate-spin" /> Analyzing...</> : 'Analyze Traffic'}
                        </button>
                    </form>

                    <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column' }}>
                        <h3 style={{ marginBottom: '1.5rem' }}>Ensemble Result</h3>

                        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', textAlign: 'center' }}>
                            {!result && !error && !loading && (
                                <p style={{ color: 'var(--text-secondary)' }}>Awaiting parameters for analysis...</p>
                            )}

                            {loading && (
                                <div style={{ padding: '2rem' }}>
                                    <Loader2 size={48} color="var(--primary-color)" className="animate-spin mb-4" />
                                    <p>Running Inference Models...</p>
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
                                    {result.finalPrediction === 'Attack' ? (
                                        <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', padding: '1.5rem 1rem', borderRadius: '12px', marginBottom: '1rem' }}>
                                            <AlertTriangle size={48} color="var(--danger-color)" style={{ marginBottom: '0.5rem' }} />
                                            <h3 style={{ color: 'var(--danger-color)', margin: 0 }}>Threat Detected (Ensemble)</h3>
                                        </div>
                                    ) : result.zeroDayPossible ? (
                                        <div style={{ background: 'rgba(234, 179, 8, 0.1)', border: '1px solid rgba(234, 179, 8, 0.3)', padding: '1.5rem 1rem', borderRadius: '12px', marginBottom: '1rem' }}>
                                            <AlertTriangle size={48} color="#eab308" style={{ marginBottom: '0.5rem' }} />
                                            <h3 style={{ color: '#eab308', margin: 0 }}>Possible Zero-Day</h3>
                                            <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Phase 1 Normal, but Autoencoder flagged anomalous trace.</p>
                                        </div>
                                    ) : (
                                        <div style={{ background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.3)', padding: '1.5rem 1rem', borderRadius: '12px', marginBottom: '1rem' }}>
                                            <CheckCircle size={48} color="var(--secondary-color)" style={{ marginBottom: '0.5rem' }} />
                                            <h3 style={{ color: 'var(--secondary-color)', margin: 0 }}>Traffic Status Normal</h3>
                                            <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Passed both Supervised and Unsupervised checks.</p>
                                        </div>
                                    )}

                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', textAlign: 'left' }}>
                                        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '0.5rem' }}>Ensemble Details</div>
                                        {result.phase1?.map((r, i) => (
                                            <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.75rem', background: 'rgba(0,0,0,0.2)', borderRadius: '6px', borderLeft: r.prediction === 'Attack' ? '3px solid var(--danger-color)' : '3px solid var(--secondary-color)' }}>
                                                <span>{r.model || ['CNN', 'LSTM', 'Transformer'][i]}</span>
                                                <span style={{ color: r.prediction === 'Attack' ? 'var(--danger-color)' : 'var(--secondary-color)', fontWeight: 600 }}>{r.prediction.toUpperCase()}</span>
                                            </div>
                                        ))}
                                        {result.phase2 && (
                                            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.75rem', background: 'rgba(0,0,0,0.2)', borderRadius: '6px', borderLeft: result.phase2.prediction === 'Attack' ? '3px solid #eab308' : '3px solid var(--secondary-color)', marginTop: '0.5rem' }}>
                                                <span>Autoencoder</span>
                                                <span style={{ color: result.phase2.prediction === 'Attack' ? '#eab308' : 'var(--secondary-color)', fontWeight: 600 }}>{result.phase2.prediction === 'Attack' ? 'ANOMALY' : 'NORMAL'}</span>
                                            </div>
                                        )}
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
                                Real-Time Sniffer Stream (Ensemble)
                            </h3>
                            <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                                Run <code style={{ color: '#fff', background: 'rgba(0,0,0,0.3)', padding: '2px 4px', borderRadius: '4px' }}>venv/Scripts/python scripts/live_sniffer.py --model Ensemble_Phase1</code> internally.
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
                                Listening for live Ensemble packets...
                            </div>
                        )}

                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                            {streamHistory.map((item, idx) => (
                                <div key={idx} className="fade-in" style={{
                                    display: 'flex',
                                    flexDirection: 'column',
                                    gap: '0.5rem',
                                    padding: '1rem',
                                    background: item.zeroDayPossible ? 'rgba(234, 179, 8, 0.1)' : (item.prediction === 'Attack' ? 'rgba(239,68,68,0.1)' : 'rgba(16,185,129,0.1)'),
                                    borderLeft: `4px solid ${item.zeroDayPossible ? '#eab308' : (item.prediction === 'Attack' ? 'var(--danger-color)' : 'var(--secondary-color)')}`,
                                    borderRadius: '4px'
                                }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                                            {item.zeroDayPossible ? <AlertTriangle color="#eab308" size={20} /> : (item.prediction === 'Attack' ? <AlertTriangle color="var(--danger-color)" size={20} /> : <CheckCircle color="var(--secondary-color)" size={20} />)}
                                            <div>
                                                <div style={{ fontWeight: 'bold', color: item.zeroDayPossible ? '#eab308' : (item.prediction === 'Attack' ? 'var(--danger-color)' : 'var(--secondary-color)') }}>
                                                    {item.zeroDayPossible ? 'AE: POSSIBLE ZERO-DAY' : (item.prediction === 'Attack' ? 'ATTACK' : 'NORMAL')}
                                                </div>
                                                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                                                    Flow Received @ {item.timestamp}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
                                        {item.phase1?.map((r, i) => (
                                            <span key={i} style={{ fontSize: '0.75rem', padding: '2px 6px', borderRadius: '4px', background: 'rgba(0,0,0,0.3)', color: r.prediction === 'Attack' ? 'var(--danger-color)' : 'var(--secondary-color)' }}>
                                                {r.model || ['CNN', 'LSTM', 'Transformer'][i]}: {r.prediction.toUpperCase()}
                                            </span>
                                        ))}
                                        {item.phase2 && (
                                            <span style={{ fontSize: '0.75rem', padding: '2px 6px', borderRadius: '4px', background: 'rgba(0,0,0,0.3)', color: item.phase2.prediction === 'Attack' ? '#eab308' : 'var(--secondary-color)' }}>
                                                AE: {item.phase2.prediction === 'Attack' ? 'ANOMALY' : 'NORMAL'}
                                            </span>
                                        )}
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

export default EnsembleDefense;
