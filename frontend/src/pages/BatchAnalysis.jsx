import React, { useState } from 'react';
import { predictBatch } from '../services/api';
import { UploadCloud, FileText, Loader2, AlertCircle } from 'lucide-react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts';

const DATASET_OPTIONS = ['CICIDS2018', 'CICIDS2017', 'NSL-KDD', 'UNSW-NB15'];

// Column display map per dataset for the results table
const DATASET_TABLE_COLS = {
    'CICIDS2018': [
        { label: 'Source Bytes', key: 'Flow Byts/s' },
        { label: 'Dest Bytes', key: 'Down/Up Ratio' },
        { label: 'Protocol', key: 'Protocol' },
    ],
    'NSL-KDD': [
        { label: 'Src Bytes', key: 'src_bytes' },
        { label: 'Dst Bytes', key: 'dst_bytes' },
        { label: 'Protocol', key: 'protocol_type' },
    ],
    'UNSW-NB15': [
        { label: 'Src Bytes', key: 'sbytes' },
        { label: 'Dst Bytes', key: 'dbytes' },
        { label: 'Protocol', key: 'proto' },
    ],
    'CICIDS2017': [
        { label: 'Src Packets', key: 'Total Fwd Packets' },
        { label: 'Dst Packets', key: 'Total Backward Packets' },
        { label: 'Flow Duration', key: 'Flow Duration' },
    ],
};

const BatchAnalysis = ({ systemStatus }) => {
    const [dataset, setDataset] = useState('CICIDS2018');
    const [modelType, setModelType] = useState('CNN');
    const [file, setFile] = useState(null);

    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const availableModels = systemStatus?.models?.[dataset]
        ? [...new Set([...systemStatus.models[dataset], 'Ensemble', 'Ensemble_Phase1'])]
        : ['CNN', 'LSTM', 'Transformer', 'Autoencoder', 'Ensemble', 'Ensemble_Phase1'];

    // Reset model and result when dataset changes
    const handleDatasetChange = (e) => {
        setDataset(e.target.value);
        setModelType('CNN');
        setResult(null);
        setError(null);
        setFile(null);
    };

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

    const tableCols = DATASET_TABLE_COLS[dataset] || DATASET_TABLE_COLS['CICIDS2018'];

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
                        <select className="form-control" value={dataset} onChange={handleDatasetChange}>
                            {DATASET_OPTIONS.map(d => (
                                <option key={d} value={d}>{d}</option>
                            ))}
                        </select>
                    </div>

                    <div className="form-group">
                        <label className="form-label">Model Type</label>
                        <select className="form-control" value={modelType} onChange={(e) => setModelType(e.target.value)}>
                            {availableModels.map(m => <option key={m} value={m}>{m}</option>)}
                        </select>
                    </div>
                </div>

                {dataset === 'NSL-KDD' && (
                    <div style={{ marginTop: '0.75rem', padding: '0.75rem 1rem', background: 'rgba(59,130,246,0.08)', border: '1px solid rgba(59,130,246,0.2)', borderRadius: '8px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                        <strong style={{ color: 'var(--primary-color)' }}>NSL-KDD format:</strong>&nbsp;
                        Upload a <code>.txt</code> or <code>.csv</code> file with the standard NSL-KDD 41-feature format (with or without headers). Categorical columns (<em>protocol_type</em>, <em>service</em>, <em>flag</em>) will be one-hot encoded automatically.
                    </div>
                )}
                {dataset === 'CICIDS2018' && (
                    <div style={{ marginTop: '0.75rem', padding: '0.75rem 1rem', background: 'rgba(16,185,129,0.08)', border: '1px solid rgba(16,185,129,0.2)', borderRadius: '8px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                        <strong style={{ color: 'var(--secondary-color)' }}>CICIDS2018 format:</strong>&nbsp;
                        Upload a <code>.csv</code> file exported from CICFlowMeter with labeled headers. Metadata columns will be dropped automatically.
                    </div>
                )}
                {dataset === 'UNSW-NB15' && (
                    <div style={{ marginTop: '0.75rem', padding: '0.75rem 1rem', background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.2)', borderRadius: '8px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                        <strong style={{ color: '#f59e0b' }}>UNSW-NB15 format:</strong>&nbsp;
                        Upload a <code>.csv</code> file with standard UNSW-NB15 features.
                    </div>
                )}
                {dataset === 'CICIDS2017' && (
                    <div style={{ marginTop: '0.75rem', padding: '0.75rem 1rem', background: 'rgba(59,130,246,0.08)', border: '1px solid rgba(59,130,246,0.2)', borderRadius: '8px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                        <strong style={{ color: '#3b82f6' }}>CICIDS2017 format:</strong>&nbsp;
                        Upload a <code>.csv</code> file with standard CICIDS2017 features.
                    </div>
                )}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: result ? '1fr 1fr' : '1fr', gap: '2rem' }}>

                {/* Upload Column */}
                <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column' }}>
                    <h3 style={{ marginBottom: '1rem' }}>Upload {dataset === 'NSL-KDD' ? 'TXT/CSV' : 'PCAP/CSV'}</h3>

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
                                <h4 style={{ marginBottom: '0.5rem' }}>Drag &amp; drop file here</h4>
                                <p style={{ fontSize: '0.85rem' }}>or click to browse (.csv / .txt format)</p>
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
                                    {tableCols.map(col => (
                                        <th key={col.key} style={{ padding: '12px 16px', color: 'var(--text-secondary)' }}>{col.label}</th>
                                    ))}
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
                                        {tableCols.map(col => (
                                            <td key={col.key} style={{ padding: '12px 16px' }}>{row[col.key] ?? '-'}</td>
                                        ))}
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
