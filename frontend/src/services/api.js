import axios from 'axios';

const API_URL = 'http://localhost:8000/api';

const api = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const getSystemStatus = async () => {
    const response = await api.get('/status');
    return response.data;
};

export const predictLive = async (dataset_name, model_type, features) => {
    const response = await api.post('/predict/live', {
        dataset_name,
        model_type,
        features,
    });
    return response.data;
};

export const predictBatch = async (dataset_name, model_type, file) => {
    const formData = new FormData();
    formData.append('dataset_name', dataset_name);
    formData.append('model_type', model_type);
    formData.append('file', file);

    const response = await api.post('/predict/batch', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });
    return response.data;
};

export const getLiveHistory = async () => {
    const response = await api.get('/predict/history');
    return response.data;
};

export const startSniffer = async (model_type, dataset = 'CICIDS2018') => {
    const response = await api.post('/sniffer/start', { model: model_type, dataset });
    return response.data;
};

export const stopSniffer = async () => {
    const response = await api.post('/sniffer/stop');
    return response.data;
};

export const getSnifferStatus = async () => {
    const response = await api.get('/sniffer/status');
    return response.data;
};

export const setDevice = async (deviceType) => {
    const response = await api.post('/device', { device: deviceType });
    return response.data;
};
