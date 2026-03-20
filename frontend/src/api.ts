import axios from 'axios';

const apiHost = typeof window !== 'undefined' ? window.location.hostname : '127.0.0.1';

const api = axios.create({
  baseURL: `http://${apiHost}:8010`,
  headers: {
    'Content-Type': 'application/json',
  },
});

let authToken: string | null = null;

api.interceptors.request.use((config) => {
  if (authToken) {
    config.headers.Authorization = `Bearer ${authToken}`;
  }
  return config;
});

export interface Metadata {
  task: string;
  selected_model: string;
  decision_threshold?: number;
  selected_hyperparameters?: {
    n_top_genes: number;
    n_qubits: number;
    n_layers: number;
    imbalance_strategy: string;
  };
  dataset_summary?: {
    source: string;
    final_labeled_cohort: {
      samples: number;
      dead: number;
      alive: number;
      dead_rate: number;
    };
    split_summary: {
      train_samples: number;
      holdout_samples: number;
      train_dead: number;
      train_alive: number;
      holdout_dead: number;
      holdout_alive: number;
    };
  };
  risk_band_cutoffs?: {
    low_upper: number;
    high_lower: number;
  };
  explainability?: {
    local_method: string;
    global_method: string;
    baseline_reference: string;
    top_global_features: GlobalFeatureImportance[];
  };
  holdout_metrics: {
    auc: number;
    pr_auc?: number;
    accuracy: number;
    balanced_accuracy?: number;
    f1: number;
    precision: number;
    recall: number;
    specificity?: number;
    brier: number;
  };
  selected_genes: string[];
}

export interface SamplePatient {
  patient_index: number;
  age: number;
  gender: 'male' | 'female';
  genes: Record<string, number>;
  actual_status: 'Alive' | 'Dead';
  predicted_status?: 'Alive' | 'Dead';
  mortality_probability: number;
}

export interface PredictionResponse {
  prediction: 'Alive' | 'Dead';
  mortality_probability: number;
  alive_probability: number;
  model_name: string;
  decision_threshold: number;
}

export interface FeatureContribution {
  feature: string;
  patient_value: number;
  reference_value: number;
  attribution: number;
  absolute_attribution: number;
  direction: 'increases_risk' | 'reduces_risk';
}

export interface GlobalFeatureImportance {
  feature: string;
  mean_absolute_attribution: number;
  mean_signed_attribution: number;
  rank: number;
}

export interface ExplanationResponse {
  method: string;
  baseline_description: string;
  prediction: 'Alive' | 'Dead';
  mortality_probability: number;
  decision_threshold: number;
  top_risk_increasing: FeatureContribution[];
  top_risk_reducing: FeatureContribution[];
  global_top_features: GlobalFeatureImportance[];
}

export interface LoginResponse {
  access_token: string;
  token_type: 'bearer';
  username: string;
  display_name: string;
  expires_in_hours: number;
}

export interface CurrentUser {
  username: string;
  display_name: string;
}

export const setAuthToken = (token: string | null): void => {
  authToken = token;
};

export const login = async (payload: { username: string; password: string }): Promise<LoginResponse> => {
  const response = await api.post('/auth/login', payload);
  return response.data;
};

export const fetchCurrentUser = async (): Promise<CurrentUser> => {
  const response = await api.get('/auth/me');
  return response.data;
};

export const logout = async (): Promise<void> => {
  await api.post('/auth/logout');
};

export const fetchMetadata = async (): Promise<Metadata> => {
  const response = await api.get('/metadata');
  return response.data;
};

export const fetchSample = async (): Promise<SamplePatient> => {
  const response = await api.get('/samples/random');
  return response.data;
};

export const predict = async (payload: {
  age: number;
  gender: 'male' | 'female';
  genes: Record<string, number>;
}): Promise<PredictionResponse> => {
  const response = await api.post('/predict', payload);
  return response.data;
};

export const explain = async (payload: {
  age: number;
  gender: 'male' | 'female';
  genes: Record<string, number>;
}): Promise<ExplanationResponse> => {
  const response = await api.post('/explain', payload);
  return response.data;
};
