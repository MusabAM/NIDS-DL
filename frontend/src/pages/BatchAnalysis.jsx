import React, { useState } from 'react';
import { predictBatch } from '../services/api';
import { UploadCloud, FileText, Loader2, AlertCircle } from 'lucide-react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts';

const BatchAnalysis = ({ systemStatus }) => {
    const [dataset, setDataset] = useState('CICIDS2018');
    const [modelType, setModelType] = useState('CNN');
    const [file, setFile] = useState(null);

    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const availableModels = systemStatus?.models?.[dataset] || ['CNN'];

    const handleFileChange = (e) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
        }
    };

    const handleDragOver = (e) => {
        e.preventDefault();
    };

    const handleDrop = (e) => {
        e.preventDefault();
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setFile(e.dataTransfer.files[0]);
        }
    };

    const submitBatch = async () => {
        if (!file) return;
        setLoading(true);
        setError(null);
        setResult(null);

        try {
            const data = await predictBatch(dataset, modelType, file);
            setResult(data);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to process batch file.');
        } finally {
            setLoading(false);
        }
    };

    const COLORS = ['#10b981', '#ef4444'];
    const pieData = result ? [
        { name: 'Normal Traffic', value: result.normal },
        { name: 'Attack Traffic', value: result.attacks }
    ] : [];

    return (
        <div className="batch-analysis-page fade-in">
            <div style={{ marginBottom: '2rem' }}>
                <h2>Batch File Analysis</h2>
                <p>Upload CSV files containing multiple network flows for bulk prediction.</p>
            </div>

            <div className="glass-panel" style={{ marginBottom: '2rem' }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
                    <div className="form-group">
                        <label className="form-label">Dataset Mode</label>
                        <select className="form-control" value={dataset} disabled>
                            <option value="CICIDS2018">CICIDS2018</option>
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

            <div style={{ display: 'grid', gridTemplateColumns: result ? '1fr 1fr' : '1fr', gap: '2rem' }}>

                {/* Upload Column */}
                <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column' }}>
                    <h3 style={{ marginBottom: '1rem' }}>Upload PCAP/CSV</h3>

                    <div
                        onDragOver={handleDragOver}
                        onDrop={handleDrop}
                        style={{
                            flex: 1,
                            border: '2px dashed var(--glass-border)',
                            borderRadius: '12px',
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            justifyContent: 'center',
                            padding: '3rem 2rem',
                            textAlign: 'center',
                            background: 'rgba(255,255,255,0.02)',
                            cursor: 'pointer',
                            transition: 'all 0.2s',
                            minHeight: '250px'
                        }}
                        onClick={() => document.getElementById('fileUpload').click()}
                    >
                        <input type="file" id="fileUpload" hidden accept=".csv,.txt" onChange={handleFileChange} />

                        {file ? (
                            <>
                                <FileText size={48} color="var(--primary-color)" style={{ marginBottom: '1rem' }} />
                                <h4 style={{ marginBottom: '0.5rem' }}>{file.name}</h4>
                                <p style={{ fontSize: '0.85rem' }}>{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                            </>
                        ) : (
                            <>
                                <UploadCloud size={48} color="var(--text-secondary)" style={{ marginBottom: '1rem' }} />
                                <h4 style={{ marginBottom: '0.5rem' }}>Drag & drop file here</h4>
                                <p style={{ fontSize: '0.85rem' }}>or click to browse (.csv format)</p>
                            </>
                        )}
                    </div>

                    <button
                        className="btn btn-primary"
                        style={{ marginTop: '1.5rem', width: '100%', padding: '12px' }}
                        disabled={!file || loading}
                        onClick={submitBatch}
                    >
                        {loading ? <><Loader2 size={18} className="animate-spin" /> Processing Batch...</> : 'Run Predictions'}
                    </button>

                    {error && (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--danger-color)', marginTop: '1rem', padding: '12px', background: 'rgba(239,68,68,0.1)', borderRadius: '8px' }}>
                            <AlertCircle size={20} />
                            <span style={{ fontSize: '0.9rem' }}>{error}</span>
                        </div>
                    )}
                </div>

                {/* Results Column */}
                {result && (
                    <div className="glass-panel fade-in">
                        <h3 style={{ marginBottom: '1.5rem' }}>Batch Summary</h3>

                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.5rem' }}>
                            <div style={{ padding: '1rem', background: 'rgba(239,68,68,0.1)', borderRadius: '8px', borderLeft: '3px solid var(--danger-color)' }}>
                                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '4px' }}>Attacks Found</div>
                                <div style={{ fontSize: '1.8rem', fontWeight: 'bold', color: 'var(--text-primary)' }}>{result.attacks}</div>
                            </div>

                            <div style={{ padding: '1rem', background: 'rgba(16,185,129,0.1)', borderRadius: '8px', borderLeft: '3px solid var(--secondary-color)' }}>
                                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '4px' }}>Normal Flows</div>
                                <div style={{ fontSize: '1.8rem', fontWeight: 'bold', color: 'var(--text-primary)' }}>{result.normal}</div>
                            </div>
                        </div>

                        <div style={{ width: '100%', height: '220px' }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <PieChart>
                                    <Pie
                                        data={pieData}
                                        cx="50%"
                                        cy="50%"
                                        innerRadius={60}
                                        outerRadius={80}
                                        paddingAngle={5}
                                        dataKey="value"
                                        stroke="none"
                                    >
                                        {pieData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                        ))}
                                    </Pie>
                                    <Tooltip
                                        contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px' }}
                                        itemStyle={{ color: '#fff' }}
                                    />
                                    <Legend />
                                </PieChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                )}
            </div>

            {result && result.results && result.results.length > 0 && (
                <div className="glass-panel fade-in" style={{ marginTop: '2rem' }}>
                    <h3 style={{ marginBottom: '1.5rem' }}>Flow Details (Top 100)</h3>

                    <div style={{ overflowX: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
                            <thead>
                                <tr style={{ borderBottom: '1px solid var(--glass-border)', textAlign: 'left' }}>
                                    <th style={{ padding: '12px 16px', color: 'var(--text-secondary)' }}>No.</th>
                                    <th style={{ padding: '12px 16px', color: 'var(--text-secondary)' }}>Prediction</th>
                                    <th style={{ padding: '12px 16px', color: 'var(--text-secondary)' }}>Confidence</th>
                                    <th style={{ padding: '12px 16px', color: 'var(--text-secondary)' }}>Source Bytes</th>
                                    <th style={{ padding: '12px 16px', color: 'var(--text-secondary)' }}>Dest Bytes</th>
                                    <th style={{ padding: '12px 16px', color: 'var(--text-secondary)' }}>Protocol</th>
                                </tr>
                            </thead>
                            <tbody>
                                {result.results.slice(0, 10).map((row, idx) => (
                                    <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                                        <td style={{ padding: '12px 16px' }}>{idx + 1}</td>
                                        <td style={{ padding: '12px 16px' }}>
                                            <span className={`badge ${row.Prediction === 'Attack' ? 'badge-danger' : 'badge-success'}`}>
                                                {row.Prediction}
                                            </span>
                                        </td>
                                        <td style={{ padding: '12px 16px' }}>
                                            {typeof row.Attack_Probability === 'number' && row.Attack_Probability <= 1
                                                ? `${(row.Attack_Probability * 100).toFixed(2)}%`
                                                : (row.Attack_Probability || 0).toFixed(4)}
                                        </td>
                                        <td style={{ padding: '12px 16px' }}>{row.src_bytes !== undefined ? row.src_bytes : row['Flow Byts/s'] || '-'}</td>
                                        <td style={{ padding: '12px 16px' }}>{row.dst_bytes !== undefined ? row.dst_bytes : row['Down/Up Ratio'] || '-'}</td>
                                        <td style={{ padding: '12px 16px' }}>{row.protocol_type || '-'}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                        {result.results.length > 10 && (
                            <div style={{ padding: '12px 16px', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                                Showing 10 of {result.total} records...
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default BatchAnalysis;
