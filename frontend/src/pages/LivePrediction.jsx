import React, { useState, useEffect } from 'react';
import { predictLive, getLiveHistory, startSniffer, stopSniffer } from '../services/api';
import { AlertTriangle, CheckCircle, Loader2, Play, Square, Activity, Code, X } from 'lucide-react';

// Feature templates per dataset (curated key features)
const DATASET_FEATURES = {
    'CICIDS2018': {
        'Flow Duration': 0, 'Tot Fwd Pkts': 1, 'Tot Bwd Pkts': 0, 'TotLen Fwd Pkts': 0, 'TotLen Bwd Pkts': 0,
        'Fwd Pkt Len Max': 0, 'Fwd Pkt Len Min': 0, 'Fwd Pkt Len Mean': 0, 'Fwd Pkt Len Std': 0,
        'Bwd Pkt Len Max': 0, 'Bwd Pkt Len Min': 0, 'Bwd Pkt Len Mean': 0, 'Bwd Pkt Len Std': 0,
        'Flow Byts/s': 0, 'Flow Pkts/s': 0, 'Flow IAT Mean': 0, 'Flow IAT Std': 0, 'Flow IAT Max': 0, 'Flow IAT Min': 0,
        'Fwd IAT Tot': 0, 'Fwd IAT Mean': 0, 'Fwd IAT Std': 0, 'Fwd IAT Max': 0, 'Fwd IAT Min': 0,
        'Bwd IAT Tot': 0, 'Bwd IAT Mean': 0, 'Bwd IAT Std': 0, 'Bwd IAT Max': 0, 'Bwd IAT Min': 0,
        'Fwd PSH Flags': 0, 'Bwd PSH Flags': 0, 'Fwd URG Flags': 0, 'Bwd URG Flags': 0,
        'Fwd Header Len': 0, 'Bwd Header Len': 0, 'Fwd Pkts/s': 0, 'Bwd Pkts/s': 0,
        'Pkt Len Min': 0, 'Pkt Len Max': 0, 'Pkt Len Mean': 0, 'Pkt Len Std': 0, 'Pkt Len Var': 0,
        'FIN Flag Cnt': 0, 'SYN Flag Cnt': 0, 'RST Flag Cnt': 0, 'PSH Flag Cnt': 0, 'ACK Flag Cnt': 0,
        'URG Flag Cnt': 0, 'CWE Flag Count': 0, 'ECE Flag Cnt': 0, 'Down/Up Ratio': 0, 'Pkt Size Avg': 0,
        'Fwd Seg Size Avg': 0, 'Bwd Seg Size Avg': 0, 'Fwd Byts/b Avg': 0, 'Fwd Pkts/b Avg': 0, 'Fwd Blk Rate Avg': 0,
        'Bwd Byts/b Avg': 0, 'Bwd Pkts/b Avg': 0, 'Bwd Blk Rate Avg': 0,
        'Subflow Fwd Pkts': 0, 'Subflow Fwd Byts': 0, 'Subflow Bwd Pkts': 0, 'Subflow Bwd Byts': 0,
        'Init Fwd Win Byts': 0, 'Init Bwd Win Byts': 0, 'Fwd Act Data Pkts': 0, 'Fwd Seg Size Min': 0,
        'Active Mean': 0, 'Active Std': 0, 'Active Max': 0, 'Active Min': 0,
        'Idle Mean': 0, 'Idle Std': 0, 'Idle Max': 0, 'Idle Min': 0
    },
    'NSL-KDD': {
        duration: 0, protocol_type: 'tcp', service: 'http', flag: 'SF', src_bytes: 0, dst_bytes: 0,
        land: 0, wrong_fragment: 0, urgent: 0, hot: 0, num_failed_logins: 0, logged_in: 0,
        num_compromised: 0, root_shell: 0, su_attempted: 0, num_root: 0, num_file_creations: 0,
        num_shells: 0, num_access_files: 0, num_outbound_cmds: 0, is_host_login: 0, is_guest_login: 0,
        count: 0, srv_count: 0, serror_rate: 0, srv_serror_rate: 0, rerror_rate: 0, srv_rerror_rate: 0,
        same_srv_rate: 0, diff_srv_rate: 0, srv_diff_host_rate: 0, dst_host_count: 0, dst_host_srv_count: 0,
        dst_host_same_srv_rate: 0, dst_host_diff_srv_rate: 0, dst_host_same_src_port_rate: 0,
        dst_host_srv_diff_host_rate: 0, dst_host_serror_rate: 0, dst_host_srv_serror_rate: 0,
        dst_host_rerror_rate: 0, dst_host_srv_rerror_rate: 0
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

const LivePrediction = ({ systemStatus }) => {
    const [dataset, setDataset] = useState('CICIDS2018');
    const [modelType, setModelType] = useState('CNN');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [mode, setMode] = useState('manual');
    const [showExtended, setShowExtended] = useState(false);
    const [isJsonModalOpen, setIsJsonModalOpen] = useState(false);
    const [jsonInput, setJsonInput] = useState('');

    // Streaming History State
    const [isStreaming, setIsStreaming] = useState(false);
    const [streamHistory, setStreamHistory] = useState([]);

    // Dynamic features state — resets when dataset changes
    const [features, setFeatures] = useState(DATASET_FEATURES['CICIDS2018']);

    const availableModels = systemStatus?.models?.[dataset] || ['CNN', 'LSTM', 'Transformer', 'Autoencoder', 'VQC'];

    // Swap features template when dataset changes
    useEffect(() => {
        setFeatures({ ...DATASET_FEATURES[dataset] });
        setResult(null);
        setError(null);
        // Reset to CNN if the current model is not available for the new dataset
        const models = systemStatus?.models?.[dataset] || ['CNN', 'LSTM', 'Transformer', 'Autoencoder'];
        if (!models.includes(modelType)) {
            setModelType('CNN');
        }
    }, [dataset]);

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        // Check if value is numeric or string (for categorical fields)
        const parsedValue = (name === 'protocol_type' || name === 'service' || name === 'flag') ? value : Number(value);
        setFeatures(prev => ({ ...prev, [name]: parsedValue }));
    };

    const handleImportJson = () => {
        try {
            const parsed = JSON.parse(jsonInput);
            const newFeatures = { ...features };
            let count = 0;

            Object.keys(parsed).forEach(key => {
                if (key in newFeatures) {
                    newFeatures[key] = parsed[key];
                    count++;
                }
            });

            setFeatures(newFeatures);
            setIsJsonModalOpen(false);
            setJsonInput('');
            // Optional: alert(`Successfully imported ${count} features.`);
        } catch (e) {
            alert('Invalid JSON format. Please check your input.');
        }
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
            startSniffer(modelType).catch(e => console.error("Failed to start sniffer:", e));
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
        } else {
            stopSniffer().catch(e => console.error("Failed to stop sniffer:", e));
        }
        return () => clearInterval(interval);
    }, [isStreaming, modelType]);

    // Cleanup sniffer on component unmount
    useEffect(() => {
        return () => {
            stopSniffer().catch(e => console.error("Failed to cleanup sniffer:", e));
        };
    }, []);

    const featureEntries = Object.keys(features);
    const visibleFeatures = showExtended ? featureEntries : featureEntries.slice(0, 12);
    const numCols = visibleFeatures.length > 12 ? 4 : 3;
    const hasMore = featureEntries.length > 12;

    return (
        <div className="live-prediction-page fade-in">
            <div style={{ marginBottom: '2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                    <h2>Live Traffic Analysis</h2>
                    <p>Enter network flow parameters or listen to live network interfaces.</p>
                </div>
                <div style={{ display: 'flex', gap: '0.5rem', background: 'var(--nav-hover-bg)', padding: '0.5rem', borderRadius: '8px', border: '1px solid var(--glass-border)' }}>
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
                        <select className="form-control" value={modelType} onChange={(e) => setModelType(e.target.value)}>
                            {availableModels.map(m => <option key={m} value={m}>{m}</option>)}
                        </select>
                    </div>
                </div>
            </div>

            {mode === 'manual' ? (
                <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 1fr)', gap: '2rem' }}>
                    <form className="glass-panel" onSubmit={submitPrediction} style={{ position: 'relative' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                            <h3 style={{ margin: 0 }}>
                                Flow Characteristics ({dataset})
                            </h3>
                            <button
                                type="button"
                                className="btn"
                                style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '6px',
                                    fontSize: '0.8rem',
                                    padding: '4px 10px',
                                    background: 'var(--nav-hover-bg)',
                                    border: '1px solid var(--glass-border)'
                                }}
                                onClick={() => setIsJsonModalOpen(true)}
                            >
                                <Code size={14} /> Import JSON
                            </button>
                        </div>

                        <div style={{ display: 'grid', gridTemplateColumns: `repeat(${numCols}, 1fr)`, gap: '1rem' }}>
                            {visibleFeatures.map((key) => (
                                <div className="form-group" key={key}>
                                    <label className="form-label" style={{ fontSize: '0.8rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }} title={key}>{key}</label>
                                    {(key === 'protocol_type' || key === 'service' || key === 'flag' || key === 'proto' || key === 'state' || key === 'service_unsw') ? (
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

                        {hasMore && (
                            <div style={{ marginTop: '1.5rem', textAlign: 'center' }}>
                                <button
                                    type="button"
                                    className="btn"
                                    style={{ fontSize: '0.85rem', background: 'transparent', color: 'var(--primary-color)', border: '1px solid var(--primary-color)' }}
                                    onClick={() => setShowExtended(!showExtended)}
                                >
                                    {showExtended ? 'Hide Extended Features' : `Show All ${featureEntries.length} Features`}
                                </button>
                            </div>
                        )}

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
                                        <div style={{ background: 'var(--stream-entry-attack)', border: '1px solid rgba(239, 68, 68, 0.3)', padding: '2rem 1rem', borderRadius: '12px' }}>
                                            <AlertTriangle size={64} color="var(--danger-color)" style={{ marginBottom: '1rem' }} />
                                            <h2 style={{ color: 'var(--danger-color)', marginBottom: '0.5rem' }}>Threat Detected</h2>
                                            <p style={{ margin: 0, fontWeight: 600 }}>ATTACK</p>
                                        </div>
                                    ) : (
                                        <div style={{ background: 'var(--stream-entry-normal)', border: '1px solid rgba(16, 185, 129, 0.3)', padding: '2rem 1rem', borderRadius: '12px' }}>
                                            <CheckCircle size={64} color="var(--secondary-color)" style={{ marginBottom: '1rem' }} />
                                            <h2 style={{ color: 'var(--secondary-color)', marginBottom: '0.5rem' }}>Traffic Status</h2>
                                            <p style={{ margin: 0, fontWeight: 600 }}>NORMAL</p>
                                        </div>
                                    )}

                                    <div style={{ marginTop: '2rem', padding: '1rem', background: 'var(--nav-hover-bg)', borderRadius: '8px', border: '1px solid var(--glass-border)' }}>
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
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', borderBottom: '1px solid var(--glass-border)', paddingBottom: '1rem' }}>
                        <div>
                            <h3 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
                                <Activity size={20} color="var(--primary-color)" />
                                Real-Time Sniffer Stream
                            </h3>
                            <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                                Run <code style={{ color: 'var(--text-primary)', background: 'var(--code-bg)', padding: '2px 4px', borderRadius: '4px' }}>venv/Scripts/python scripts/live_sniffer.py</code> internally.
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

                    <div style={{ flex: 1, background: 'var(--stream-bg)', borderRadius: '8px', padding: '1rem', overflowY: 'auto', maxHeight: '500px', border: '1px solid var(--glass-border)' }}>
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
                                    background: item.prediction === 'Attack' ? 'var(--stream-entry-attack)' : 'var(--stream-entry-normal)',
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

            {/* JSON Import Modal */}
            {isJsonModalOpen && (
                <div style={{
                    position: 'fixed',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    background: 'rgba(0,0,0,0.6)',
                    backdropFilter: 'blur(4px)',
                    zIndex: 1000,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: '1rem'
                }}>
                    <div className="glass-panel fade-in" style={{ width: '100%', maxWidth: '500px', padding: '1.5rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                            <h3 style={{ margin: 0 }}>Import Feature JSON</h3>
                            <button className="btn-icon" onClick={() => setIsJsonModalOpen(false)}>
                                <X size={20} />
                            </button>
                        </div>
                        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                            Paste a JSON object with key-value pairs matching the feature names for {dataset}.
                        </p>
                        <textarea
                            className="form-control"
                            style={{
                                width: '100%',
                                minHeight: '200px',
                                fontFamily: 'monospace',
                                fontSize: '0.8rem',
                                background: 'rgba(0,0,0,0.2)',
                                marginBottom: '1.5rem'
                            }}
                            placeholder={`{\n  "Flow Duration": 150000,\n  "Tot Fwd Pkts": 10,\n  ...\n}`}
                            value={jsonInput}
                            onChange={(e) => setJsonInput(e.target.value)}
                        />
                        <div style={{ display: 'flex', gap: '1rem' }}>
                            <button className="btn" style={{ flex: 1 }} onClick={() => setIsJsonModalOpen(false)}>Cancel</button>
                            <button className="btn btn-primary" style={{ flex: 2 }} onClick={handleImportJson}>Apply Features</button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default LivePrediction;
