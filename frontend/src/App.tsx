import { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import {
  explain,
  fetchCurrentUser,
  fetchMetadata,
  fetchSample,
  login,
  logout,
  predict,
  setAuthToken,
  type ExplanationResponse,
  type CurrentUser,
  type LoginResponse,
  type Metadata,
  type PredictionResponse,
  type SamplePatient,
} from './api';

type Tab = 'assessment' | 'history' | 'validation' | 'docs' | 'about';
type RiskBand = 'low' | 'moderate' | 'high';

type RiskProfile = {
  band: RiskBand;
  badgeClass: 'routine' | 'watch' | 'critical';
  label: string;
  summary: string;
  action: string;
  survivalRange: string;
};

type HistoryItem = {
  id: string;
  timestamp: string;
  patientInfo: {
    age: number;
    gender: string;
  };
  result: PredictionResponse;
};

const RISK_PROFILES: Record<RiskBand, RiskProfile> = {
  low: {
    band: 'low',
    badgeClass: 'routine',
    label: 'Low Risk Profile',
    summary: 'Biomarker signature indicates favorable prognosis for GBM.',
    action: 'Continue standard clinical surveillance.',
    survivalRange: '>15 months',
  },
  moderate: {
    band: 'moderate',
    badgeClass: 'watch',
    label: 'Intermediate Risk Profile',
    summary: 'Prognostic indicators fall between defined low and high risk thresholds.',
    action: 'Review within Multi-Disciplinary Team (MDT) context.',
    survivalRange: '10-15 months',
  },
  high: {
    band: 'high',
    badgeClass: 'critical',
    label: 'High Risk Profile',
    summary: 'Molecular signature consistent with poor-prognosis glioblastoma cohorts.',
    action: 'Escalate senior clinical review and care-plan optimization.',
    survivalRange: '<10 months',
  },
};

function getRiskProfile(probability: number, metadata: Metadata | null): RiskProfile {
  const cutoffs = metadata?.risk_band_cutoffs;
  if (cutoffs) {
    if (probability <= cutoffs.low_upper) {
      return RISK_PROFILES.low;
    }
    if (probability >= cutoffs.high_lower) {
      return RISK_PROFILES.high;
    }
    return RISK_PROFILES.moderate;
  }

  if (probability >= 0.6) {
    return RISK_PROFILES.high;
  }
  if (probability >= 0.4) {
    return RISK_PROFILES.moderate;
  }
  return RISK_PROFILES.low;
}

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('assessment');
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [session, setSession] = useState<LoginResponse | null>(null);
  const [currentUser, setCurrentUser] = useState<CurrentUser | null>(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [loginLoading, setLoginLoading] = useState(false);
  const [loginUsername, setLoginUsername] = useState('oncologist');
  const [loginPassword, setLoginPassword] = useState('qubrain-demo-2026');
  const [loginError, setLoginError] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [metadataLoading, setMetadataLoading] = useState(true);
  const [age, setAge] = useState<string>('');
  const [gender, setGender] = useState<'male' | 'female'>('male');
  const [genes, setGenes] = useState<Record<string, string>>({});
  const [geneFilter, setGeneFilter] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null);
  const [explanationLoading, setExplanationLoading] = useState(false);
  const [loadedCase, setLoadedCase] = useState<SamplePatient | null>(null);
  const [showSignal, setShowSignal] = useState(false);
  const [showImport, setShowImport] = useState(false);
  const [importText, setImportText] = useState('');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const storedToken = localStorage.getItem('qubrain_token');
    const storedSession = localStorage.getItem('qubrain_session');

    if (!storedToken || !storedSession) {
      setAuthLoading(false);
      return;
    }

    try {
      const parsedSession = JSON.parse(storedSession) as LoginResponse;
      setAuthToken(storedToken);
      setSession(parsedSession);

      void fetchCurrentUser()
        .then((user) => {
          setCurrentUser(user);
          const savedHistory = localStorage.getItem(`qubrain_history_${user.username}`);
          if (savedHistory) {
            setHistory(JSON.parse(savedHistory));
          }
        })
        .catch(() => {
          setAuthToken(null);
          localStorage.removeItem('qubrain_token');
          localStorage.removeItem('qubrain_session');
          setSession(null);
          setCurrentUser(null);
        })
        .finally(() => {
          setAuthLoading(false);
        });
    } catch {
      localStorage.removeItem('qubrain_token');
      localStorage.removeItem('qubrain_session');
      setAuthLoading(false);
    }
  }, []);

  const saveToHistory = (res: PredictionResponse) => {
    if (!currentUser) {
      return;
    }

    const newItem: HistoryItem = {
      id: `PAT-${Math.floor(Math.random() * 900000) + 100000}`,
      timestamp: new Date().toLocaleString(),
      patientInfo: { age: Number(age), gender },
      result: res,
    };

    const updated = [newItem, ...history];
    setHistory(updated);
    localStorage.setItem(`qubrain_history_${currentUser.username}`, JSON.stringify(updated));
  };

  const deleteHistoryItem = (id: string) => {
    if (!currentUser) {
      return;
    }
    const updated = history.filter((item) => item.id !== id);
    setHistory(updated);
    localStorage.setItem(`qubrain_history_${currentUser.username}`, JSON.stringify(updated));
  };

  const loadMetadata = async (): Promise<Metadata | null> => {
    try {
      setMetadataLoading(true);
      const data = await fetchMetadata();
      setMetadata(data);
      setGenes((current) => {
        const nextEntries = data.selected_genes.map((gene) => [gene, current[gene] ?? ''] as const);
        return Object.fromEntries(nextEntries);
      });
      setError(null);
      return data;
    } catch {
      setError('System metadata could not be retrieved. Please check backend connectivity.');
      return null;
    } finally {
      setMetadataLoading(false);
    }
  };

  useEffect(() => {
    if (!session) {
      return;
    }
    void loadMetadata();
  }, [session]);

  const filteredGenes = useMemo(() => {
    if (!metadata) {
      return [];
    }
    const term = geneFilter.trim().toLowerCase();
    if (!term) {
      return metadata.selected_genes;
    }
    return metadata.selected_genes.filter((gene) => gene.toLowerCase().includes(term));
  }, [geneFilter, metadata]);

  const completedGenes = useMemo(() => {
    if (!metadata) {
      return 0;
    }
    return metadata.selected_genes.filter((gene) => genes[gene] !== '').length;
  }, [genes, metadata]);

  const updateGene = (gene: string, value: string) => {
    setGenes((current) => ({ ...current, [gene]: value }));
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result;
      if (typeof text === 'string') {
        processImportText(text);
      }
    };
    reader.readAsText(file);
    // Reset input
    event.target.value = '';
  };

  const handleBulkImport = () => {
    processImportText(importText);
  };

  const processImportText = (text: string) => {
    if (!metadata) return;
    try {
      let imported: Record<string, any> = {};
      const trimmed = text.trim();

      if (trimmed.startsWith('{')) {
        imported = JSON.parse(trimmed);
      } else {
        trimmed.split('\n').forEach((line: string) => {
          const parts = line.split(/[ ,:\t]+/).filter(Boolean);
          if (parts.length >= 2) {
            imported[parts[0].toUpperCase()] = parts[1];
          }
        });
      }

      setGenes((current) => {
        const next = { ...current };
        let count = 0;
        metadata.selected_genes.forEach((gene) => {
          if (imported[gene.toUpperCase()] !== undefined) {
            next[gene] = String(imported[gene.toUpperCase()]);
            count++;
          }
        });
        if (count > 0) {
          setShowImport(false);
          setImportText('');
          setError(null);
        } else {
          setError('No matching biomarkers found in the provided report.');
        }
        return next;
      });
    } catch {
      setError('Import failed. Please ensure the file is a valid CSV or JSON report.');
    }
  };

  const loadSample = async () => {
    try {
      setLoading(true);
      setError(null);
      const sample = await fetchSample();
      setLoadedCase(sample);
      setAge(String(Math.round(sample.age)));
      setGender(sample.gender);
      setGenes(Object.fromEntries(Object.entries(sample.genes).map(([gene, value]) => [gene, String(value)])));
      setResult(null);
      setExplanation(null);
      setShowSignal(false);
    } catch {
      setError('Failed to load an archived cohort case.');
    } finally {
      setLoading(false);
    }
  };

  const submit = async () => {
    if (!metadata) {
      return;
    }

    if (age === '') {
      setError('Patient age is required before running the assessment.');
      return;
    }

    const missing = metadata.selected_genes.filter((gene) => genes[gene] === '');
    if (missing.length > 0) {
      setError(`Biomarker panel incomplete. First missing markers: ${missing.slice(0, 3).join(', ')}`);
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const payload = {
        age: Number(age),
        gender,
        genes: Object.fromEntries(metadata.selected_genes.map((gene) => [gene, Number(genes[gene])])),
      };
      const response = await predict(payload);
      setResult(response);
      setExplanation(null);
      setExplanationLoading(true);
      setShowSignal(false);
      saveToHistory(response);
      void explain(payload)
        .then((explanationResponse) => {
          setExplanation(explanationResponse);
        })
        .catch(() => {
          setExplanation(null);
        })
        .finally(() => {
          setExplanationLoading(false);
        });
    } catch (requestError: unknown) {
      if (axios.isAxiosError(requestError)) {
        setError(requestError.response?.data?.detail ?? 'The clinical assessment could not be processed.');
      } else {
        setError('The clinical assessment could not be processed.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleLogin = async () => {
    try {
      setLoginLoading(true);
      setLoginError(null);
      const response = await login({ username: loginUsername.trim(), password: loginPassword });
      setAuthToken(response.access_token);
      localStorage.setItem('qubrain_token', response.access_token);
      localStorage.setItem('qubrain_session', JSON.stringify(response));
      setSession(response);

      const user = await fetchCurrentUser();
      setCurrentUser(user);
      const savedHistory = localStorage.getItem(`qubrain_history_${user.username}`);
      setHistory(savedHistory ? JSON.parse(savedHistory) : []);
    } catch (requestError: unknown) {
      if (axios.isAxiosError(requestError)) {
        setLoginError(requestError.response?.data?.detail ?? 'Sign-in failed.');
      } else {
        setLoginError('Sign-in failed.');
      }
    } finally {
      setLoginLoading(false);
      setAuthLoading(false);
    }
  };

  const handleLogout = async () => {
    try {
      await logout();
    } catch {
      // Best-effort sign-out for the simple session flow.
    } finally {
      setAuthToken(null);
      localStorage.removeItem('qubrain_token');
      localStorage.removeItem('qubrain_session');
      setSession(null);
      setCurrentUser(null);
      setMetadata(null);
      setHistory([]);
      setResult(null);
      setExplanation(null);
      setLoadedCase(null);
      setError(null);
    }
  };

  const riskProfile = result ? getRiskProfile(result.mortality_probability, metadata) : null;

  if (authLoading) {
    return (
      <div className="login-shell fade-in">
        <div className="login-card">
          <h1>QuBrain</h1>
          <p className="login-copy">Checking clinician session...</p>
        </div>
      </div>
    );
  }

  if (!session || !currentUser) {
    return (
      <div className="login-shell fade-in">
        <div className="login-card">
          <span className="kicker">Clinical Sign-In</span>
          <h1>QuBrain</h1>
          <p className="login-copy">Sign in to access the neuro-oncology assessment workspace.</p>

          <div className="form-field">
            <label>Username</label>
            <input value={loginUsername} onChange={(event) => setLoginUsername(event.target.value)} />
          </div>

          <div className="form-field">
            <label>Password</label>
            <input
              type="password"
              value={loginPassword}
              onChange={(event) => setLoginPassword(event.target.value)}
            />
          </div>

          <button className="btn btn-primary login-button" onClick={handleLogin} disabled={loginLoading}>
            {loginLoading ? 'Signing in...' : 'Sign In'}
          </button>

          {loginError && <div className="alert alert-error">{loginError}</div>}
        </div>
      </div>
    );
  }

  return (
    <div className="app-container fade-in">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <h2>QuBrain</h2>
          <p>Oncology Console</p>
        </div>
        <nav>
          <ul className="nav-list">
            <li className="nav-item">
              <button className={`nav-button ${activeTab === 'assessment' ? 'active' : ''}`} onClick={() => setActiveTab('assessment')}>
                <span>Assessment</span>
              </button>
            </li>
            <li className="nav-item">
              <button className={`nav-button ${activeTab === 'history' ? 'active' : ''}`} onClick={() => setActiveTab('history')}>
                <span>Patient History</span>
              </button>
            </li>
            <li className="nav-item">
              <button className={`nav-button ${activeTab === 'validation' ? 'active' : ''}`} onClick={() => setActiveTab('validation')}>
                <span>System Metrics</span>
              </button>
            </li>
            <li className="nav-item">
              <button className={`nav-button ${activeTab === 'docs' ? 'active' : ''}`} onClick={() => setActiveTab('docs')}>
                <span>Developer Docs</span>
              </button>
            </li>
            <li className="nav-item">
              <button className={`nav-button ${activeTab === 'about' ? 'active' : ''}`} onClick={() => setActiveTab('about')}>
                <span>About</span>
              </button>
            </li>
          </ul>
        </nav>
        <div className="sidebar-footer">
          <div>{currentUser.display_name}</div>
          <button className="sidebar-logout" onClick={handleLogout}>
            Sign Out
          </button>
        </div>
      </aside>

      <main className="main-content">
        {activeTab === 'assessment' && (
          <div className="page-assessment">
            <header className="page-header">
              <h1>Patient Assessment</h1>
            </header>

            <div className="result-container">
              <div className="left-col">
                <section className="panel">
                  <div className="section-title">
                    <h2>Patient Profile</h2>
                    <button className="btn btn-ghost" onClick={loadSample} disabled={loading}>
                      Load Archived Case
                    </button>
                  </div>

                  <div className="grid-2">
                    <div className="form-field">
                      <label>Age</label>
                      <input type="number" value={age} onChange={(event) => setAge(event.target.value)} placeholder="Years" />
                    </div>
                    <div className="form-field">
                      <label>Gender</label>
                      <select value={gender} onChange={(event) => setGender(event.target.value as 'male' | 'female')}>
                        <option value="male">Male</option>
                        <option value="female">Female</option>
                      </select>
                    </div>
                  </div>

                  <div className="case-context">
                    <div className="context-card">
                      <span>Source</span>
                      <strong>{loadedCase ? `Historical Case #${loadedCase.patient_index + 1}` : 'Manual Entry'}</strong>
                    </div>
                    <div className="context-card">
                      <span>Completion</span>
                      <strong>{metadata ? `${completedGenes}/${metadata.selected_genes.length} Verified` : '--'}</strong>
                    </div>
                  </div>

                  <div className="biomarker-section">
                    <div className="section-title">
                      <h3>Molecular Biomarkers</h3>
                      <div className="section-actions" style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                        <button className="btn btn-ghost" style={{ fontSize: '0.8rem' }} onClick={() => setShowImport(!showImport)}>
                          {showImport ? 'Close Import' : 'Import Report'}
                        </button>
                        <div className="search-bar">
                          <input
                            type="text"
                            placeholder="Search genes..."
                            value={geneFilter}
                            onChange={(event) => setGeneFilter(event.target.value)}
                            style={{ maxWidth: '140px', fontSize: '0.85rem' }}
                          />
                        </div>
                      </div>
                    </div>

                    {showImport && (
                      <div className="import-zone animate-fade-in">
                        <p className="import-hint">
                          Paste <code>Gene,Value</code> or <code>{"{ \"GENE\": VAL }"}</code> below to bulk populate the panel.
                        </p>
                        <textarea
                          className="import-textarea"
                          placeholder="Example:&#10;MGMT, 1.2&#10;EGFR, -0.5"
                          value={importText}
                          onChange={(e) => setImportText(e.target.value)}
                        />
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '12px' }}>
                          <div>
                            <input
                              type="file"
                              id="csv-upload"
                              style={{ display: 'none' }}
                              accept=".csv,.json,.txt"
                              onChange={handleFileUpload}
                            />
                            <button className="btn btn-ghost btn-sm" onClick={() => document.getElementById('csv-upload')?.click()}>
                              Upload CSV / JSON
                            </button>
                          </div>
                          <button className="btn btn-primary btn-sm" onClick={handleBulkImport}>
                            Apply Genomic Data
                          </button>
                        </div>
                      </div>
                    )}

                    <div className="biomarker-grid">
                      {metadataLoading ? (
                        <p>Initializing biomarker panel...</p>
                      ) : (
                        filteredGenes.map((gene) => (
                          <div key={gene} className="biomarker-card">
                            <div className="biomarker-header">
                              <span>{gene}</span>
                              <small>log2</small>
                            </div>
                            <input
                              type="number"
                              step="0.0001"
                              value={genes[gene] ?? ''}
                              onChange={(event) => updateGene(gene, event.target.value)}
                              placeholder="0.0"
                            />
                          </div>
                        ))
                      )}
                    </div>
                  </div>

                  <div className="assessment-actions">
                    <button className="btn btn-primary" onClick={submit} disabled={loading || !metadata}>
                      {loading ? 'Processing...' : 'Generate Clinical Assessment'}
                    </button>
                    <p>
                      Assessment incorporates the validated 50-gene signature and clinical demographics.
                    </p>
                  </div>

                  {error && <div className="alert alert-error">{error}</div>}
                </section>
              </div>

              <div className="right-col">
                {result && riskProfile ? (
                  <section className="panel risk-summary-card">
                    <div className="section-title">
                      <h2>Assessment Result</h2>
                    </div>

                    <div className="risk-summary-card-inner">
                      <div className="risk-gauge-section">
                        <div className={`risk-level-badge ${riskProfile.badgeClass}`} style={{ marginBottom: '24px' }}>
                          {riskProfile.label}
                        </div>

                        <div className={`survival-circle ${riskProfile.band}`}>
                          <div className="survival-circle-title">
                            Median Survival
                            <button
                              className="btn-info-toggle"
                              onClick={() => setShowSignal((current) => !current)}
                              title="Toggle Technical Signal"
                              type="button"
                            >
                              i
                            </button>
                          </div>
                          <div className="survival-circle-range">
                            {showSignal ? (
                              <span className="technical-signal">{(result.mortality_probability * 100).toFixed(1)}%</span>
                            ) : (
                              riskProfile.survivalRange
                            )}
                          </div>
                          <div className="survival-circle-unit">{showSignal ? 'analytical signal' : 'months'}</div>
                        </div>

                        <div style={{ marginTop: '24px', width: '100%' }}>
                          <button className="btn btn-ghost" onClick={() => window.print()} style={{ width: '100%' }}>
                            Print Formal Report
                          </button>
                        </div>
                      </div>

                      <div className="risk-info-section">
                        <div className="interpretation-card">
                          <h4>Interpretation</h4>
                          <p>{riskProfile.summary}</p>

                          <h4>Suggested Clinical Stance</h4>
                          <p>{riskProfile.action}</p>
                        </div>

                        <div className="result-footnote">
                          Survival context derived from TCGA-GBM cohort benchmarks. Operator: {currentUser.display_name}.
                        </div>
                      </div>
                    </div>
                  </section>
                ) : (
                  <section className="panel risk-summary-card empty-result">
                    <div className="section-title">
                      <h2>Assessment Result</h2>
                    </div>
                    <p className="empty-copy">
                      Complete the patient profile and biomarker panel to view the assessment result.
                    </p>
                  </section>
                )}

                {result && (
                  <section className="panel explanation-panel">
                    <div className="section-title">
                      <h2>Assessment Drivers</h2>
                    </div>

                    {explanationLoading ? (
                      <p className="explanation-copy">Computing feature-level explanation for this assessment...</p>
                    ) : explanation ? (
                      <>
                        <p className="explanation-copy">{explanation.baseline_description}</p>

                        <div className="explanation-grid">
                          <div className="explanation-column">
                            <h4>Risk-Increasing Factors</h4>
                            {explanation.top_risk_increasing.length === 0 ? (
                              <p className="explanation-empty">No dominant risk-increasing features were identified.</p>
                            ) : (
                              explanation.top_risk_increasing.map((item) => (
                                <div key={`up-${item.feature}`} className="contribution-card increase">
                                  <div className="contribution-header">
                                    <strong>{item.feature}</strong>
                                    <span>+{item.absolute_attribution.toFixed(3)}</span>
                                  </div>
                                  <div className="contribution-detail">
                                    patient {item.patient_value.toFixed(3)} | reference {item.reference_value.toFixed(3)}
                                  </div>
                                </div>
                              ))
                            )}
                          </div>

                          <div className="explanation-column">
                            <h4>Risk-Reducing Factors</h4>
                            {explanation.top_risk_reducing.length === 0 ? (
                              <p className="explanation-empty">No dominant risk-reducing features were identified.</p>
                            ) : (
                              explanation.top_risk_reducing.map((item) => (
                                <div key={`down-${item.feature}`} className="contribution-card reduce">
                                  <div className="contribution-header">
                                    <strong>{item.feature}</strong>
                                    <span>-{item.absolute_attribution.toFixed(3)}</span>
                                  </div>
                                  <div className="contribution-detail">
                                    patient {item.patient_value.toFixed(3)} | reference {item.reference_value.toFixed(3)}
                                  </div>
                                </div>
                              ))
                            )}
                          </div>
                        </div>

                        <div className="result-footnote">
                          Explainability method: {explanation.method}. These values indicate model contribution relative to a cohort reference profile and are not causal effects.
                        </div>
                      </>
                    ) : (
                      <p className="explanation-copy">Feature-level explanation is not available for this assessment.</p>
                    )}
                  </section>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'history' && (
          <div className="page-history">
            <header className="page-header">
              <h1>Assessment History</h1>
            </header>

            <div className="history-list">
              {history.length === 0 ? (
                <section className="panel" style={{ padding: '80px', textAlign: 'center' }}>
                  <p style={{ color: 'var(--text-muted)', fontSize: '1.1rem' }}>
                    No historical assessments found. Completed reports are saved automatically in this browser session.
                  </p>
                </section>
              ) : (
                history.map((item) => {
                  const profile = getRiskProfile(item.result.mortality_probability, metadata);
                  return (
                    <div key={item.id} className="history-item">
                      <div className="history-item-info">
                        <div className="history-headline">
                          <strong>{item.id}</strong>
                          <span className={`risk-level-badge ${profile.badgeClass}`}>{profile.label}</span>
                        </div>
                        <div className="history-meta">
                          {item.timestamp} | {item.patientInfo.age}Y, {item.patientInfo.gender.toUpperCase()}
                        </div>
                        <div className="history-summary">Expected range: {profile.survivalRange}</div>
                      </div>

                      <div className="history-item-meta">
                        <div className="history-label-box">
                          <div className="history-label-value">{item.result.prediction}</div>
                          <div className="history-label-caption">Assessment label</div>
                        </div>
                        <button className="btn btn-ghost history-delete" onClick={() => deleteHistoryItem(item.id)}>
                          Delete
                        </button>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        )}

        {activeTab === 'validation' && (
          <div className="page-validation">
            <header className="page-header">
              <span className="kicker">System Validation</span>
              <h1>Model Credibility & Benchmarks</h1>
            </header>

            <div className="grid-2 validation-panels">
              <section className="panel validation-panel">
                <h3>Classification Performance</h3>
                <div className="metric-list">
                  <div className="metric-row">
                    <span>Holdout AUC-ROC</span>
                    <strong>{metadata?.holdout_metrics.auc.toFixed(3) ?? '--'}</strong>
                  </div>
                  <div className="metric-row">
                    <span>Balanced Accuracy</span>
                    <strong>{metadata?.holdout_metrics.balanced_accuracy?.toFixed(3) ?? '--'}</strong>
                  </div>
                  <div className="metric-row">
                    <span>Specificity</span>
                    <strong>{metadata?.holdout_metrics.specificity?.toFixed(3) ?? '--'}</strong>
                  </div>
                  <div className="metric-row">
                    <span>PR AUC</span>
                    <strong>{metadata?.holdout_metrics.pr_auc?.toFixed(3) ?? '--'}</strong>
                  </div>
                </div>
              </section>

              <section className="panel validation-panel">
                <h3>Analytical Parameters</h3>
                <div className="metric-list">
                  <div className="metric-row">
                    <span>Biomarker panel</span>
                    <strong>{metadata?.selected_genes.length ?? '--'} genes</strong>
                  </div>
                  <div className="metric-row">
                    <span>Quantum circuit</span>
                    <strong>{metadata?.selected_hyperparameters?.n_qubits ?? '--'} qubits</strong>
                  </div>
                  <div className="metric-row">
                    <span>Imbalance strategy</span>
                    <strong>{metadata?.selected_hyperparameters?.imbalance_strategy ?? '--'}</strong>
                  </div>
                  <div className="metric-row">
                    <span>Validation cohort</span>
                    <strong>{metadata?.dataset_summary?.split_summary.holdout_samples ?? '--'} patients</strong>
                  </div>
                </div>
              </section>
            </div>

            <section className="panel">
              <h3 style={{ marginBottom: '16px' }}>Clinical interpretation note</h3>
              <p className="validation-note">
                The application displays literature-based survival ranges by risk band rather than raw predicted percentages. These ranges are population-level summaries from published glioblastoma studies and should not be interpreted as patient-specific survival forecasts.
              </p>
            </section>

            <section className="panel">
              <h3 style={{ marginBottom: '16px' }}>Global biomarker drivers</h3>
              {metadata?.explainability?.top_global_features?.length ? (
                <div className="global-driver-list">
                  {metadata.explainability.top_global_features.map((item) => (
                    <div key={item.feature} className="global-driver-row">
                      <span>{item.rank}. {item.feature}</span>
                      <strong>{item.mean_absolute_attribution.toFixed(3)}</strong>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="validation-note">Global explainability summary has not been generated yet.</p>
              )}
            </section>
          </div>
        )}

        {activeTab === 'docs' && (
          <div className="page-docs">
            <header className="page-header">
              <span className="kicker">Technical Resources</span>
              <h1>Developer Documentation</h1>
            </header>

            <div className="docs-content">
              <section className="panel">
                <h3>System Architecture</h3>
                <p>QuBrain is built as a hybrid quantum-classical system. It utilizes a classical feature extractor (PyTorch) combined with a Variational Quantum Circuit (VQC) implemented via PennyLane.</p>
                <div className="code-block-mock">
                  <pre>
                    {`
[Input: Genomic + Clinical] -> [Classical Dense Layer] 
                             -> [Angle Embedding] 
                             -> [Strongly Entangling VQC] 
                             -> [Measurement] 
                             -> [Probabilistic Risk Output]
                    `}
                  </pre>
                </div>
              </section>

              <div className="grid-2">
                <section className="panel">
                  <h3>Backend API (FastAPI)</h3>
                  <ul className="docs-list">
                    <li><code>POST /predict</code>: Core assessment engine</li>
                    <li><code>GET /metadata</code>: Model parameters & cutoffs</li>
                    <li><code>GET /samples/random</code>: Access test cohort</li>
                    <li><code>POST /auth/login</code>: Session initiation</li>
                  </ul>
                </section>

                <section className="panel">
                  <h3>Training Protocol</h3>
                  <ul className="docs-list">
                    <li><strong>Data:</strong> TCGA-GBM Genomic Cohort</li>
                    <li><strong>Input:</strong> Age, Gender, Top 50 Genes</li>
                    <li><strong>Validation:</strong> Stratified 80/20 Holdout</li>
                    <li><strong>Balancing:</strong> Class-Weighting</li>
                  </ul>
                </section>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'about' && (
          <div className="page-about">
            <header className="page-header">
              <span className="kicker">Project Context</span>
              <h1>About QuBrain</h1>
            </header>

            <section className="panel about-panel">
              <div style={{ maxWidth: '800px', margin: '0 auto', textAlign: 'center' }}>
                <p style={{ fontSize: '1.2rem', color: 'var(--brand-secondary)', fontWeight: 500, marginBottom: '32px' }}>
                  A next-generation neuro-oncology console bridging the gap between high-dimensional genomic research and clinical risk stratification.
                </p>
                <p>
                  Glioblastoma Multiforme (GBM) remains one of the most challenging primary brain tumors to prognosis. QuBrain leverages 
                  emergent hybrid quantum-classical machine learning architectures to identify subtle patterns in transcriptomic data 
                  that correlate with mortality outcomes.
                </p>
                
                <div style={{ margin: '48px 0', padding: '32px', borderTop: '1px solid var(--border-color)', borderBottom: '1px solid var(--border-color)' }}>
                  <h4>Clinical Research Disclaimer</h4>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                    This system is a clinical innovation prototype. The survival benchmarks (e.g., &gt;15 months, &lt;10 months) are based on historical 
                    cohort data and literature. The provided mortality probability ("Analytical Signal") is for research use and must be 
                    interpreted by a neuro-oncology specialist alongside longitudinal imaging, functional status, and standard pathology.
                  </p>
                </div>
                
                <p style={{ fontSize: '0.8rem', opacity: 0.6 }}>
                  v2.4.0 Technical Release &bull; Hybrid Quantum-Classical Pipeline &bull; TCGA Derived
                </p>
              </div>
            </section>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
