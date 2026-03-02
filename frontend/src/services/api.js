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
