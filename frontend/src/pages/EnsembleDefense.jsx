import React, { useState, useEffect } from 'react';
import { predictLive, getLiveHistory, startSniffer, stopSniffer } from '../services/api';
import { AlertTriangle, CheckCircle, Loader2, Play, Square, Activity, Radio, Code, X } from 'lucide-react';

// Feature templates per dataset
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

const EnsembleDefense = ({ systemStatus }) => {
    const [dataset, setDataset] = useState('CICIDS2018');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [mode, setMode] = useState('manual');
    const [ensembleMode, setEnsembleMode] = useState('Ensemble');
    const [showExtended, setShowExtended] = useState(false);
    const [isJsonModalOpen, setIsJsonModalOpen] = useState(false);
    const [jsonInput, setJsonInput] = useState('');

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
            const data = await predictLive(dataset, ensembleMode, features);
            setResult(data);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to connect to backend.');
        } finally {
            setLoading(false);
        }
    };

    // Auto-stream effect: start/stop sniffer and poll history
    useEffect(() => {
        let interval;
        if (isStreaming) {
            // Start the backend sniffer process with current dataset + model
            startSniffer(ensembleMode, dataset).catch((e) =>
                console.error('Failed to start sniffer:', e)
            );
            interval = setInterval(async () => {
                try {
                    const data = await getLiveHistory();
                    if (data && data.history) {
                        // Filter by the currently selected dataset AND model_type
                        const filtered = data.history.filter(
                            (h) => h.dataset_name === dataset && h.model_type === ensembleMode
                        );
                        setStreamHistory(filtered);
                    }
                } catch (e) {
                    console.error('Stream polling error:', e);
                }
            }, 1000);
        } else {
            // Stop the sniffer when the user clicks Stop
            stopSniffer().catch(() => {});
        }
        return () => clearInterval(interval);
    }, [isStreaming, dataset, ensembleMode]);

    const featureEntries = Object.keys(features);
    const visibleFeatures = showExtended ? featureEntries : featureEntries.slice(0, 12);
    const numCols = visibleFeatures.length > 12 ? 4 : 3;
    const hasMore = featureEntries.length > 12;

    return (
        <div className="live-prediction-page fade-in">
            <div style={{ marginBottom: '2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                    <h2>Ensemble Defense</h2>
                    <p>Live, concurrent multi-model analysis leveraging supervised &amp; unsupervised checks.</p>
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
                        <select className="form-control" value={ensembleMode} onChange={(e) => setEnsembleMode(e.target.value)}>
                            <option value="Ensemble">Full Ensemble (All Models - Recommended)</option>
                            <option value="Ensemble_Phase1">Phase 1 Only (CNN + LSTM + Transformer)</option>
                            <option value="Ensemble_AE">Phase 1 + Autoencoder (No VQC)</option>
                            <option value="Ensemble_VQC">Phase 1 + VQC (No Autoencoder)</option>
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
                                        <div style={{ background: 'var(--stream-entry-attack)', border: '1px solid rgba(239, 68, 68, 0.3)', padding: '1.5rem 1rem', borderRadius: '12px', marginBottom: '1rem' }}>
                                            <AlertTriangle size={48} color="var(--danger-color)" style={{ marginBottom: '0.5rem' }} />
                                            <h3 style={{ color: 'var(--danger-color)', margin: 0 }}>Threat Detected (Ensemble)</h3>
                                        </div>
                                    ) : result.zeroDayPossible ? (
                                        <div style={{ background: 'var(--stream-entry-zero)', border: '1px solid rgba(234, 179, 8, 0.3)', padding: '1.5rem 1rem', borderRadius: '12px', marginBottom: '1rem' }}>
                                            <AlertTriangle size={48} color="#eab308" style={{ marginBottom: '0.5rem' }} />
                                            <h3 style={{ color: '#eab308', margin: 0 }}>Possible Zero-Day</h3>
                                            <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Phase 1 Normal, but Phase 2 flagged anomalous trace.</p>
                                        </div>
                                    ) : (
                                        <div style={{ background: 'var(--stream-entry-normal)', border: '1px solid rgba(16, 185, 129, 0.3)', padding: '1.5rem 1rem', borderRadius: '12px', marginBottom: '1rem' }}>
                                            <CheckCircle size={48} color="var(--secondary-color)" style={{ marginBottom: '0.5rem' }} />
                                            <h3 style={{ color: 'var(--secondary-color)', margin: 0 }}>Traffic Status Normal</h3>
                                            <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Passed both Supervised and Unsupervised checks.</p>
                                        </div>
                                    )}

                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', textAlign: 'left' }}>
                                        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '0.5rem' }}>Ensemble Details</div>
                                        {result.phase1?.map((r, i) => (
                                            <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.75rem', background: 'var(--nav-hover-bg)', borderRadius: '6px', borderLeft: r.prediction === 'Attack' ? '3px solid var(--danger-color)' : '3px solid var(--secondary-color)' }}>
                                                <span>{r.model || ['CNN', 'LSTM', 'Transformer'][i]}</span>
                                                <span style={{ color: r.prediction === 'Attack' ? 'var(--danger-color)' : 'var(--secondary-color)', fontWeight: 600 }}>{r.prediction.toUpperCase()}</span>
                                            </div>
                                        ))}
                                        {result.phase2 && result.phase2.map((p2, i) => (
                                            <div key={`p2-${i}`} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.75rem', background: 'var(--nav-hover-bg)', borderRadius: '6px', borderLeft: p2.prediction === 'Attack' ? '3px solid #eab308' : '3px solid var(--secondary-color)', marginTop: '0.5rem' }}>
                                                <span>{p2.model} <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>(Phase 2)</span></span>
                                                <span style={{ color: p2.prediction === 'Attack' ? '#eab308' : 'var(--secondary-color)', fontWeight: 600 }}>{p2.prediction === 'Attack' ? 'ANOMALY' : 'NORMAL'}</span>
                                            </div>
                                        ))}
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
                            <div style={{ marginTop: '6px', display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                                <span style={{ fontSize: '0.78rem', padding: '2px 8px', borderRadius: '4px', background: 'rgba(99,102,241,0.15)', color: 'var(--primary-color)', border: '1px solid rgba(99,102,241,0.3)' }}>
                                    Dataset: {dataset}
                                </span>
                                <span style={{ fontSize: '0.78rem', padding: '2px 8px', borderRadius: '4px', background: 'rgba(16,185,129,0.1)', color: 'var(--secondary-color)', border: '1px solid rgba(16,185,129,0.2)' }}>
                                    Model: {ensembleMode}
                                </span>
                                {isStreaming && (
                                    <span style={{ fontSize: '0.78rem', padding: '2px 8px', borderRadius: '4px', background: 'rgba(239,68,68,0.1)', color: 'var(--danger-color)', border: '1px solid rgba(239,68,68,0.2)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                                        <Radio size={10} /> Live
                                    </span>
                                )}
                            </div>
                            <p style={{ margin: '6px 0 0 0', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                                Packets are captured from the selected interface, features extracted, then submitted to the{' '}
                                <code style={{ color: 'var(--text-primary)', background: 'var(--code-bg)', padding: '1px 4px', borderRadius: '3px' }}>{ensembleMode}</code>{' '}
                                model using the{' '}
                                <code style={{ color: 'var(--text-primary)', background: 'var(--code-bg)', padding: '1px 4px', borderRadius: '3px' }}>{dataset}</code>{' '}
                                feature schema.
                            </p>
                        </div>
                        <button
                            className={`btn ${isStreaming ? 'btn-danger' : 'btn-success'}`}
                            style={{ display: 'flex', alignItems: 'center', gap: '8px', flexShrink: 0 }}
                            onClick={() => setIsStreaming(!isStreaming)}
                        >
                            {isStreaming ? <><Square size={16} fill="currentColor" /> Stop Stream</> : <><Play size={16} fill="currentColor" /> Connect Stream</>}
                        </button>
                    </div>

                    <div style={{ flex: 1, background: 'var(--stream-bg)', borderRadius: '8px', padding: '1rem', overflowY: 'auto', maxHeight: '500px', border: '1px solid var(--glass-border)' }}>
                        {!isStreaming && streamHistory.length === 0 && (
                            <div style={{ height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)' }}>
                                Stream is paused. Click Connect to begin capturing live packets.
                            </div>
                        )}

                        {isStreaming && streamHistory.length === 0 && (
                            <div style={{ height: '300px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)' }}>
                                <Loader2 size={32} className="animate-spin" style={{ marginBottom: '1rem', color: 'var(--primary-color)' }} />
                                Listening on {dataset} with {ensembleMode}... Generate some traffic!
                            </div>
                        )}

                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                            {streamHistory.map((item, idx) => (
                                <div key={idx} className="fade-in" style={{
                                    display: 'flex',
                                    flexDirection: 'column',
                                    gap: '0.5rem',
                                    padding: '1rem',
                                    background: item.zeroDayPossible ? 'var(--stream-entry-zero)' : (item.prediction === 'Attack' ? 'var(--stream-entry-attack)' : 'var(--stream-entry-normal)'),
                                    borderLeft: `4px solid ${item.zeroDayPossible ? '#eab308' : (item.prediction === 'Attack' ? 'var(--danger-color)' : 'var(--secondary-color)')}`,
                                    borderRadius: '4px'
                                }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                                            {item.zeroDayPossible
                                                ? <AlertTriangle color="#eab308" size={20} />
                                                : item.prediction === 'Attack'
                                                    ? <AlertTriangle color="var(--danger-color)" size={20} />
                                                    : <CheckCircle color="var(--secondary-color)" size={20} />
                                            }
                                            <div>
                                                <div style={{ fontWeight: 'bold', color: item.zeroDayPossible ? '#eab308' : (item.prediction === 'Attack' ? 'var(--danger-color)' : 'var(--secondary-color)') }}>
                                                    {item.zeroDayPossible
                                                        ? 'POSSIBLE ZERO-DAY'
                                                        : item.prediction === 'Attack' ? 'ATTACK' : 'NORMAL'
                                                    }
                                                </div>
                                                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                                                    Flow @ {item.timestamp}
                                                    {item.metric_type && (
                                                        <span style={{ marginLeft: '8px', opacity: 0.7 }}>— {item.metric_type}</span>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Phase 1 model badges (ensemble modes) */}
                                    {item.phase1 && item.phase1.length > 0 && (
                                        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginTop: '0.25rem' }}>
                                            {item.phase1.map((r, i) => (
                                                <span key={i} style={{ fontSize: '0.75rem', padding: '2px 6px', borderRadius: '4px', background: 'var(--code-bg)', color: r.prediction === 'Attack' ? 'var(--danger-color)' : 'var(--secondary-color)' }}>
                                                    {r.model || ['CNN', 'LSTM', 'Transformer'][i]}: {r.prediction.toUpperCase()}
                                                </span>
                                            ))}
                                            {item.phase2 && item.phase2.map((p2, pi) => (
                                                <span key={`stream-p2-${pi}`} style={{ fontSize: '0.75rem', padding: '2px 6px', borderRadius: '4px', background: 'var(--nav-hover-bg)', border: '1px solid rgba(234,179,8,0.3)', color: p2.prediction === 'Attack' ? '#eab308' : 'var(--secondary-color)' }}>
                                                    {p2.model} (P2): {p2.prediction === 'Attack' ? 'ANOMALY' : 'NORMAL'}
                                                </span>
                                            ))}
                                        </div>
                                    )}

                                    {/* Standalone model: show confidence bar */}
                                    {!item.phase1 && (
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.25rem' }}>
                                            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                                                {item.metric_type === 'Reconstruction Error' ? 'Recon. Error' : 'Confidence'}
                                            </span>
                                            <div style={{ flex: 1, height: '4px', background: 'var(--glass-border)', borderRadius: '2px', overflow: 'hidden' }}>
                                                <div style={{
                                                    width: `${Math.min(item.confidence * 100, 100)}%`,
                                                    height: '100%',
                                                    background: item.prediction === 'Attack' ? 'var(--danger-color)' : 'var(--secondary-color)',
                                                    borderRadius: '2px',
                                                    transition: 'width 0.3s ease'
                                                }} />
                                            </div>
                                            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', minWidth: '44px', textAlign: 'right' }}>
                                                {item.confidence <= 1
                                                    ? `${(item.confidence * 100).toFixed(1)}%`
                                                    : item.confidence.toFixed(4)
                                                }
                                            </span>
                                        </div>
                                    )}
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
                            <button className="btn-icon" onClick={() => setIsJsonModalOpen(false)} style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: 'var(--text-primary)' }}>
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

export default EnsembleDefense;
