import React, { useState } from 'react';
import { predictLive } from '../services/api';
import { AlertTriangle, CheckCircle, Loader2 } from 'lucide-react';

const LivePrediction = ({ systemStatus }) => {
    const [dataset, setDataset] = useState('NSL-KDD');
    const [modelType, setModelType] = useState('CNN');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    // NSL-KDD features state
    const [features, setFeatures] = useState({
        duration: 0,
        protocol_type: 'tcp',
        service: 'http',
        flag: 'SF',
        src_bytes: 100,
        dst_bytes: 0,
        count: 1,
        serror_rate: 0,
        rerror_rate: 0,
        same_srv_rate: 1.0,
    });

    const availableModels = systemStatus?.models?.[dataset] || ['CNN'];

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setFeatures(prev => ({
            ...prev,
            [name]: ['protocol_type', 'service', 'flag'].includes(name) ? value : Number(value)
        }));
    };

    const submitPrediction = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setResult(null);

        // Build the full 41-feature dictionary mimicking the Streamlit app logic
        const fullFeatures = {
            ...features,
            land: 0, wrong_fragment: 0, urgent: 0, hot: 0,
            num_failed_logins: 0, logged_in: 1, num_compromised: 0,
            root_shell: 0, su_attempted: 0, num_root: 0, num_file_creations: 0,
            num_shells: 0, num_access_files: 0, num_outbound_cmds: 0,
            is_host_login: 0, is_guest_login: 0, srv_count: features.count,
            srv_serror_rate: features.serror_rate, srv_rerror_rate: features.rerror_rate,
            diff_srv_rate: 0.0, srv_diff_host_rate: 0.0, dst_host_count: 1,
            dst_host_srv_count: 1, dst_host_same_srv_rate: 1.0, dst_host_diff_srv_rate: 0.0,
            dst_host_same_src_port_rate: 0.0, dst_host_srv_diff_host_rate: 0.0,
            dst_host_serror_rate: features.serror_rate, dst_host_srv_serror_rate: features.serror_rate,
            dst_host_rerror_rate: features.rerror_rate, dst_host_srv_rerror_rate: features.rerror_rate,
        };

        try {
            const data = await predictLive(dataset, modelType, fullFeatures);
            setResult(data);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to connect to backend.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="live-prediction-page fade-in">
            <div style={{ marginBottom: '2rem' }}>
                <h2>Live Traffic Analysis</h2>
                <p>Enter network flow parameters to classify traffic instantly.</p>
            </div>

            <div className="glass-panel" style={{ marginBottom: '2rem' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                    <div className="form-group">
                        <label className="form-label">Dataset Mode</label>
                        <select className="form-control" value={dataset} onChange={(e) => setDataset(e.target.value)}>
                            <option value="NSL-KDD">NSL-KDD</option>
                            {/* Only implementing NSL-KDD form for simplicity in migration, as per Streamlit */}
                        </select>
                    </div>
                    <div className="form-group">
                        <label className="form-label">Model Type</label>
                        <select className="form-control" value={modelType} onChange={(e) => setModelType(e.target.value)}>
                            {availableModels.map(m => <option key={m} value={m}>{m}</option>)}
                        </select>
                    </div>
                </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 1fr)', gap: '2rem' }}>
                <form className="glass-panel" onSubmit={submitPrediction}>
                    <h3 style={{ marginBottom: '1.5rem' }}>Flow Characteristics</h3>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                        <div className="form-group">
                            <label className="form-label">Duration</label>
                            <input type="number" name="duration" className="form-control" value={features.duration} onChange={handleInputChange} />
                        </div>

                        <div className="form-group">
                            <label className="form-label">Protocol</label>
                            <select name="protocol_type" className="form-control" value={features.protocol_type} onChange={handleInputChange}>
                                <option value="tcp">tcp</option>
                                <option value="udp">udp</option>
                                <option value="icmp">icmp</option>
                            </select>
                        </div>

                        <div className="form-group">
                            <label className="form-label">Source Bytes</label>
                            <input type="number" name="src_bytes" className="form-control" value={features.src_bytes} onChange={handleInputChange} />
                        </div>

                        <div className="form-group">
                            <label className="form-label">Destination Bytes</label>
                            <input type="number" name="dst_bytes" className="form-control" value={features.dst_bytes} onChange={handleInputChange} />
                        </div>

                        <div className="form-group">
                            <label className="form-label">Service</label>
                            <select name="service" className="form-control" value={features.service} onChange={handleInputChange}>
                                <option value="http">http</option>
                                <option value="private">private</option>
                                <option value="ftp_data">ftp_data</option>
                                <option value="smtp">smtp</option>
                                <option value="other">other</option>
                            </select>
                        </div>

                        <div className="form-group">
                            <label className="form-label">Flag</label>
                            <select name="flag" className="form-control" value={features.flag} onChange={handleInputChange}>
                                <option value="SF">SF</option>
                                <option value="S0">S0</option>
                                <option value="REJ">REJ</option>
                                <option value="RSTR">RSTR</option>
                            </select>
                        </div>
                    </div>

                    <div style={{ marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
                        <h4 style={{ marginBottom: '1rem' }}>Advanced Features</h4>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
                            <div className="form-group">
                                <label className="form-label">Count</label>
                                <input type="number" name="count" className="form-control" value={features.count} onChange={handleInputChange} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">SYN Error Rate</label>
                                <input type="number" step="0.1" max="1" min="0" name="serror_rate" className="form-control" value={features.serror_rate} onChange={handleInputChange} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Same Srv Rate</label>
                                <input type="number" step="0.1" max="1" min="0" name="same_srv_rate" className="form-control" value={features.same_srv_rate} onChange={handleInputChange} />
                            </div>
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
        </div>
    );
};

export default LivePrediction;
