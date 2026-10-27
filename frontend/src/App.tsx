import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import type { HealthResponse, ModelInfoResponse, TranslateResponse, HistoryItem } from './types';

const SAMPLE_SENTENCES = [
  'I am a student.',
  'I want to learn English and find a good job.',
  'This is an interesting project.'
];

export const App: React.FC = () => {
  // Input & Output states
  const [inputText, setInputText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [isTranslating, setIsTranslating] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // System & Model metadata states
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [isConnecting, setIsConnecting] = useState(true);

  // Session translation history (max 5 items)
  const [history, setHistory] = useState<HistoryItem[]>([]);

  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Fetch initial health and model metadata
  useEffect(() => {
    const fetchSystemInfo = async () => {
      try {
        setIsConnecting(true);
        const [healthRes, infoRes] = await Promise.all([
          fetch('/api/health'),
          fetch('/api/model-info')
        ]);

        if (healthRes.ok) {
          const healthData: HealthResponse = await healthRes.json();
          setHealth(healthData);
        } else {
          setHealth({
            status: 'degraded',
            model_ready: false,
            message: 'Không thể kết nối dịch vụ backend.',
            missing_files: []
          });
        }

        if (infoRes.ok) {
          const infoData: ModelInfoResponse = await infoRes.json();
          setModelInfo(infoData);
        }
      } catch (err) {
        console.error('Lỗi khi tải thông tin hệ thống:', err);
        setHealth({
          status: 'degraded',
          model_ready: false,
          message: 'Không thể kết nối đến máy chủ backend (127.0.0.1:8000).',
          missing_files: []
        });
      } finally {
        setIsConnecting(false);
      }
    };

    fetchSystemInfo();
  }, []);

  // Compute word count
  const words = inputText.trim() ? inputText.trim().split(/\s+/) : [];
  const wordCount = words.length;
  const maxWords = modelInfo?.limits.max_words ?? 40;

  // Handle translation submission
  const handleTranslate = async () => {
    const trimmed = inputText.trim();
    if (!trimmed) {
      setErrorMessage('Vui lòng nhập một câu tiếng Anh cần dịch.');
      textareaRef.current?.focus();
      return;
    }

    if (wordCount > maxWords) {
      setErrorMessage(`Câu chứa ${wordCount} từ, vượt quá giới hạn cho phép (${maxWords} từ). Vui lòng rút gọn nội dung.`);
      return;
    }

    if (!health?.model_ready) {
      setErrorMessage('Mô hình dịch hiện chưa sẵn sàng trên máy chủ backend.');
      return;
    }

    setIsTranslating(true);
    setErrorMessage(null);
    setCopied(false);

    try {
      const response = await fetch('/api/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: trimmed }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Đã có lỗi xảy ra khi thực hiện dịch câu.');
      }

      const resData = data as TranslateResponse;
      setTranslatedText(resData.translated_text);
      setLatencyMs(resData.latency_ms);

      // Add to session history (max 5 items, latest first)
      const newItem: HistoryItem = {
        id: Date.now().toString(),
        source: resData.source_text,
        translation: resData.translated_text,
        latency_ms: resData.latency_ms,
        wordCount: resData.source_word_count,
        timestamp: new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
      };

      setHistory(prev => [newItem, ...prev.filter(item => item.source !== resData.source_text)].slice(0, 5));
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Lỗi kết nối hoặc xử lý bản dịch.';
      setErrorMessage(msg);
      setTranslatedText('');
    } finally {
      setIsTranslating(false);
    }
  };

  // Keyboard shortcut Ctrl+Enter / Cmd+Enter
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      if (!isTranslating) {
        handleTranslate();
      }
    }
  };

  // Copy to clipboard
  const handleCopy = async () => {
    if (!translatedText) return;
    try {
      await navigator.clipboard.writeText(translatedText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Không thể sao chép vào bộ nhớ đệm:', err);
    }
  };

  // Clear input
  const handleClear = () => {
    setInputText('');
    setTranslatedText('');
    setErrorMessage(null);
    setLatencyMs(null);
    textareaRef.current?.focus();
  };

  // Load sample sentence
  const handleSelectSample = (sample: string) => {
    setInputText(sample);
    setErrorMessage(null);
    textareaRef.current?.focus();
  };

  // Restore history item
  const handleSelectHistory = (item: HistoryItem) => {
    setInputText(item.source);
    setTranslatedText(item.translation);
    setLatencyMs(item.latency_ms);
    setErrorMessage(null);
  };

  return (
    <div className="app-layout">
      {/* Navbar */}
      <header className="navbar">
        <div className="navbar-container">
          <div className="brand">
            <div className="brand-icon">EV</div>
            <span>EnVi Lab</span>
            <span className="brand-tag">NMT Anh → Việt</span>
          </div>

          <nav>
            <ul className="nav-links">
              <li><a href="#translator" className="nav-link">Trải nghiệm</a></li>
              <li><a href="#how-it-works" className="nav-link">Cách hoạt động</a></li>
              <li><a href="#results" className="nav-link">Kết quả</a></li>
              <li><a href="#about" className="nav-link">Giới thiệu</a></li>
            </ul>
          </nav>

          <div className="nav-status">
            {isConnecting ? (
              <>
                <span className="status-dot checking" />
                <span>Đang kiểm tra...</span>
              </>
            ) : health?.model_ready ? (
              <>
                <span className="status-dot online" />
                <span>Model sẵn sàng</span>
              </>
            ) : (
              <>
                <span className="status-dot offline" />
                <span>Chưa sẵn sàng</span>
              </>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {/* Model status warning if degraded */}
        {!isConnecting && (!health?.model_ready || (health.missing_files && health.missing_files.length > 0)) && (
          <div className="alert-box degraded">
            <strong>Cảnh báo mô hình:</strong>
            <div>
              {health?.message || 'Model chưa sẵn sàng để phục vụ suy luận.'}
              {health?.missing_files && health.missing_files.length > 0 && (
                <div>Thiếu các tệp sau: <code>{health.missing_files.join(', ')}</code>. Vui lòng kiểm tra thư mục artifacts.</div>
              )}
            </div>
          </div>
        )}

        {/* Hero Section */}
        <section className="hero-section" id="translator">
          <div className="hero-badge">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
            </svg>
            BiLSTM + Bahdanau Attention
          </div>
          <h1 className="hero-title">
            Hệ thống dịch máy nơ-ron <span className="hero-title-accent">Anh – Việt</span>
          </h1>
          <p className="hero-desc">
            Trải nghiệm mô hình Sequence-to-Sequence với cơ chế Attention được huấn luyện từ đầu trên tập ngữ liệu IWSLT'15 TED Talks.
          </p>
        </section>

        {/* Translator Tool */}
        <section className="translator-card">
          <div className="translator-grid">
            {/* Input Pane (English) */}
            <div className="pane input-pane">
              <div className="pane-header">
                <span className="pane-title">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10" />
                    <line x1="2" y1="12" x2="22" y2="12" />
                    <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
                  </svg>
                  Tiếng Anh (Nguồn)
                </span>
                <div className="pane-actions">
                  <span className={`word-badge ${wordCount > maxWords ? 'danger' : wordCount > maxWords - 5 ? 'warning' : ''}`}>
                    {wordCount} / {maxWords} từ
                  </span>
                  {inputText && (
                    <button type="button" onClick={handleClear} className="text-btn" title="Xóa nội dung nhập">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <line x1="18" y1="6" x2="6" y2="18" />
                        <line x1="6" y1="6" x2="18" y2="18" />
                      </svg>
                      Xóa
                    </button>
                  )}
                </div>
              </div>

              <textarea
                ref={textareaRef}
                className="input-textarea"
                placeholder="Nhập một câu tiếng Anh cần dịch (tối đa 40 từ)...&#10;Nhấn Ctrl + Enter để dịch nhanh."
                value={inputText}
                onChange={(e) => {
                  setInputText(e.target.value);
                  if (errorMessage) setErrorMessage(null);
                }}
                onKeyDown={handleKeyDown}
                rows={5}
                aria-label="Nhập câu tiếng Anh cần dịch"
              />

              {/* Sample Sentences */}
              <div className="examples-wrapper">
                <span className="examples-label">Câu ví dụ (nhấn để điền):</span>
                <div className="examples-list">
                  {SAMPLE_SENTENCES.map((sample, idx) => (
                    <button
                      key={idx}
                      type="button"
                      className="example-chip"
                      onClick={() => handleSelectSample(sample)}
                    >
                      {sample}
                    </button>
                  ))}
                </div>
              </div>

              {/* Input Footer */}
              <div className="pane-footer">
                <span className="shortcut-hint">
                  <kbd>Ctrl</kbd> + <kbd>Enter</kbd>
                </span>
                <button
                  type="button"
                  className="translate-btn"
                  onClick={handleTranslate}
                  disabled={isTranslating || !inputText.trim() || wordCount > maxWords || !health?.model_ready}
                >
                  {isTranslating ? (
                    <>
                      <span className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
                      <span>Đang dịch...</span>
                    </>
                  ) : (
                    <>
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <polygon points="5 3 19 12 5 21 5 3" />
                      </svg>
                      <span>Dịch câu</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Output Pane (Vietnamese) */}
            <div className="pane output-pane">
              <div className="pane-header">
                <span className="pane-title">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z" />
                    <line x1="4" y1="22" x2="4" y2="15" />
                  </svg>
                  Tiếng Việt (Đích)
                </span>
                <div className="pane-actions">
                  {latencyMs !== null && (
                    <span className="latency-meta" title="Thời gian suy luận thực tế trên CPU">
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="12" cy="12" r="10" />
                        <polyline points="12 6 12 12 16 14" />
                      </svg>
                      {latencyMs} ms
                    </span>
                  )}
                  {translatedText && (
                    <button
                      type="button"
                      onClick={handleCopy}
                      className={`text-btn ${copied ? 'copied' : ''}`}
                      title="Sao chép bản dịch"
                    >
                      {copied ? (
                        <>
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <polyline points="20 6 9 17 4 12" />
                          </svg>
                          Đã chép!
                        </>
                      ) : (
                        <>
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                          </svg>
                          Sao chép
                        </>
                      )}
                    </button>
                  )}
                </div>
              </div>

              <div className="output-body">
                {isTranslating ? (
                  <div className="output-loading">
                    <div className="spinner" />
                    <p>Mô hình đang phân tích Attention và sinh token...</p>
                  </div>
                ) : errorMessage ? (
                  <div className="alert-box danger">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="12" cy="12" r="10" />
                      <line x1="12" y1="8" x2="12" y2="12" />
                      <line x1="12" y1="16" x2="12.01" y2="16" />
                    </svg>
                    <div>{errorMessage}</div>
                  </div>
                ) : translatedText ? (
                  <div className="output-result">{translatedText}</div>
                ) : (
                  <div className="output-empty">
                    Bản dịch tiếng Việt sẽ hiển thị ở đây sau khi bạn nhấn "Dịch câu".
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>

        {/* Translation History (if any) */}
        {history.length > 0 && (
          <section className="history-section">
            <div className="history-header">
              <span className="history-title">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 8v4l3 3" />
                  <circle cx="12" cy="12" r="9" />
                </svg>
                Lịch sử phiên dịch gần đây ({history.length}/5)
              </span>
              <button
                type="button"
                className="text-btn"
                onClick={() => setHistory([])}
              >
                Xóa lịch sử
              </button>
            </div>
            <div className="history-list">
              {history.map((item) => (
                <div key={item.id} className="history-item">
                  <div className="history-content" onClick={() => handleSelectHistory(item)} title="Nhấn để tải lại câu này">
                    <div className="history-source">{item.source}</div>
                    <div className="history-target">{item.translation}</div>
                  </div>
                  <div className="history-meta">
                    <span>{item.latency_ms} ms</span>
                    <span>{item.timestamp}</span>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* How It Works Section */}
        <section className="section-block" id="how-it-works">
          <div className="section-header">
            <span className="section-tag">Kiến trúc kỹ thuật</span>
            <h2 className="section-title">Cách mô hình EnVi hoạt động</h2>
            <p className="section-subtitle">
              Mô hình dịch máy tuần tự (Sequence-to-Sequence) kết hợp cơ chế tập trung cộng tính (Additive Bahdanau Attention).
            </p>
          </div>

          {/* Pipeline Visual Flow */}
          <div className="pipeline-diagram">
            <div className="pipeline-flow">
              <div className="pipeline-node">
                <div className="node-title">1. Chuẩn hóa & Token hóa</div>
                <div className="node-desc">SentencePiece BPE (8.000 tokens)</div>
              </div>
              <span className="pipeline-arrow">➔</span>

              <div className="pipeline-node accent">
                <div className="node-title">2. BiLSTM Encoder</div>
                <div className="node-desc">2 chiều, 256 units mỗi chiều (tổng 512)</div>
              </div>
              <span className="pipeline-arrow">➔</span>

              <div className="pipeline-node accent">
                <div className="node-title">3. Bahdanau Attention</div>
                <div className="node-desc">Tìm trọng số tương quan (128 units)</div>
              </div>
              <span className="pipeline-arrow">➔</span>

              <div className="pipeline-node accent">
                <div className="node-title">4. LSTM Decoder</div>
                <div className="node-desc">Unidirectional 512 units + Dense Logits</div>
              </div>
              <span className="pipeline-arrow">➔</span>

              <div className="pipeline-node">
                <div className="node-title">5. Greedy Decoding</div>
                <div className="node-desc">Sinh từng token tiếng Việt đến khi gặp &lt;EOS&gt;</div>
              </div>
            </div>
          </div>

          {/* Architecture Details Grid */}
          <div className="arch-grid">
            <div className="arch-card">
              <div className="arch-card-header">
                <div className="arch-icon">01</div>
                <div className="arch-card-title">Bộ mã hóa (BiLSTM Encoder)</div>
              </div>
              <div className="arch-card-body">
                Sử dụng 2 mạng LSTM đọc câu tiếng Anh theo cả chiều xuôi (forward) và ngược (backward), giúp biểu diễn ngữ cảnh toàn diện của từng từ trong câu nguồn trước khi chuyển tiếp cho Decoder.
              </div>
            </div>

            <div className="arch-card">
              <div className="arch-card-header">
                <div className="arch-icon">02</div>
                <div className="arch-card-title">Cơ chế Bahdanau Attention</div>
              </div>
              <div className="arch-card-body">
                Tính điểm liên kết giữa trạng thái hiện tại của Decoder và mọi vị trí Encoder qua hàm cộng tính kích hoạt Tanh. Vector ngữ cảnh (context vector) kết hợp trực tiếp với đầu ra của Decoder để dự đoán token tiếp theo.
              </div>
            </div>

            <div className="arch-card">
              <div className="arch-card-header">
                <div className="arch-icon">03</div>
                <div className="arch-card-title">Bộ giải mã (LSTM Decoder)</div>
              </div>
              <div className="arch-card-body">
                Nhận trạng thái ẩn ban đầu được ghép nối từ Encoder. Tại mỗi bước suy luận, mô hình kết hợp embedding của token trước đó với vector Attention để chiếu về không gian 8.000 từ vựng tiếng Việt.
              </div>
            </div>

            <div className="arch-card">
              <div className="arch-card-header">
                <div className="arch-icon">04</div>
                <div className="arch-card-title">SentencePiece BPE</div>
              </div>
              <div className="arch-card-body">
                Phân tích câu thành các mẩu subword độc lập với khoảng trắng, xử lý hiệu quả hiện tượng từ chưa thấy trong từ điển (OOV - Out of Vocabulary) cho cả hai ngôn ngữ Anh và Việt.
              </div>
            </div>
          </div>
        </section>

        {/* Experimental Results Section */}
        <section className="section-block" id="results">
          <div className="section-header">
            <span className="section-tag">Đánh giá thực nghiệm</span>
            <h2 className="section-title">Kết quả kiểm thử mô hình</h2>
            <p className="section-subtitle">
              Số liệu đo lường chuẩn hóa bằng SacreBLEU trên tập kiểm thử 1.177 câu sau khi lọc sạch trùng lặp.
            </p>
          </div>

          {/* Metrics Summary Grid */}
          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-label">Điểm BLEU (Test)</div>
              <div className="metric-value">{modelInfo?.evaluation.bleu ?? '24.14'}</div>
              <div className="metric-sub">SacreBLEU 2.6.0 (13a tokenizer)</div>
            </div>

            <div className="metric-card">
              <div className="metric-label">Điểm chrF++ (Test)</div>
              <div className="metric-value">{modelInfo?.evaluation.chrf_pp ?? '43.05'}</div>
              <div className="metric-sub">Đánh giá n-gram ký tự và từ</div>
            </div>

            <div className="metric-card">
              <div className="metric-label">Tổng tham số</div>
              <div className="metric-value">9.17M</div>
              <div className="metric-sub">9.171.136 tham số có thể huấn luyện</div>
            </div>

            <div className="metric-card">
              <div className="metric-label">Cặp câu Train</div>
              <div className="metric-value">115.154</div>
              <div className="metric-sub">IWSLT'15 TED sau loại bỏ trùng lặp</div>
            </div>
          </div>

          {/* Detailed Breakdown & Training Curve */}
          <div className="results-split">
            <div className="results-table-card">
              <div className="card-title">
                <span>Điểm số theo độ dài câu nguồn</span>
                <span className="brand-tag">Greedy Decoding</span>
              </div>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Độ dài câu nguồn</th>
                    <th>Số câu (n)</th>
                    <th>BLEU</th>
                    <th>chrF++</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Ngắn (1 – 10 từ)</td>
                    <td className="num">{modelInfo?.evaluation.by_source_length.short_1_10.n ?? 376}</td>
                    <td className="num">{modelInfo?.evaluation.by_source_length.short_1_10.bleu.toFixed(2) ?? '26.21'}</td>
                    <td className="num">{modelInfo?.evaluation.by_source_length.short_1_10.chrf_pp.toFixed(2) ?? '43.66'}</td>
                  </tr>
                  <tr>
                    <td>Vừa (11 – 20 từ)</td>
                    <td className="num">{modelInfo?.evaluation.by_source_length.medium_11_20.n ?? 449}</td>
                    <td className="num">{modelInfo?.evaluation.by_source_length.medium_11_20.bleu.toFixed(2) ?? '25.19'}</td>
                    <td className="num">{modelInfo?.evaluation.by_source_length.medium_11_20.chrf_pp.toFixed(2) ?? '43.95'}</td>
                  </tr>
                  <tr>
                    <td>Dài (21 – 40 từ)</td>
                    <td className="num">{modelInfo?.evaluation.by_source_length.medium_21_40.n ?? 352}</td>
                    <td className="num">{modelInfo?.evaluation.by_source_length.medium_21_40.bleu.toFixed(2) ?? '22.84'}</td>
                    <td className="num">{modelInfo?.evaluation.by_source_length.medium_21_40.chrf_pp.toFixed(2) ?? '42.25'}</td>
                  </tr>
                </tbody>
              </table>

              <div style={{ marginTop: '1.25rem', fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                <strong>Ghi chú đánh giá:</strong> Điểm số đo lường trên tập test đã lọc của project (IWSLT'15), không phản ánh phần trăm câu dịch đúng tuyệt đối. Mô hình duy trì độ chính xác cao nhất ở nhóm câu ngắn và vừa.
              </div>
            </div>

            <div className="results-chart-card">
              <div className="card-title">
                <span>Đường cong học tập (Learning Curve)</span>
                <span className="brand-tag">Checkpoint Epoch 14</span>
              </div>
              <div className="chart-image-wrapper">
                <img
                  src="/learning_curve.png"
                  alt="Biểu đồ suy giảm hàm mất mát Token NLL trong quá trình huấn luyện"
                  className="chart-img"
                  loading="lazy"
                />
              </div>
              <p className="chart-caption">
                Biểu đồ đối chiếu Loss / NLL giữa tập huấn luyện và validation qua 19 epochs trên GPU Tesla T4. Trọng số tối ưu được lựa chọn tại epoch 14 (Validation Token NLL = 2.6487).
              </p>
            </div>
          </div>
        </section>

        {/* About & Tech Stack Section */}
        <section className="section-block" id="about">
          <div className="section-header">
            <span className="section-tag">Công nghệ & Trình bày CV</span>
            <h2 className="section-title">Giới thiệu Project</h2>
            <p className="section-subtitle">
              Project cá nhân được phát triển phục vụ ứng tuyển vị trí Thực tập sinh AI Engineer / Full-stack Developer.
            </p>
          </div>

          <div className="tech-grid">
            <div className="tech-card">
              <div className="tech-name">TensorFlow 2.21 & Keras 3</div>
              <div className="tech-role">Xây dựng kiến trúc BiLSTM, Bahdanau Attention tùy biến và tối ưu đồ thị tính toán</div>
            </div>
            <div className="tech-card">
              <div className="tech-name">SentencePiece 0.2.2</div>
              <div className="tech-role">Mô hình hóa ngôn ngữ con BPE (8k vocab) không phụ thuộc phân tách từ phức tạp</div>
            </div>
            <div className="tech-card">
              <div className="tech-name">FastAPI & Uvicorn</div>
              <div className="tech-role">Backend API bất đồng bộ, nạp model một lần theo vòng đời lifespan, CPU inference lock</div>
            </div>
            <div className="tech-card">
              <div className="tech-name">React, TypeScript & Vite</div>
              <div className="tech-role">Giao diện người dùng phản hồi nhanh, kiểm tra kiểu tĩnh chặt chẽ, proxy proxying chuẩn hóa</div>
            </div>
          </div>

          {/* CV Description Box */}
          <div className="cv-box">
            <div className="cv-title">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
                <line x1="16" y1="13" x2="8" y2="13" />
                <line x1="16" y1="17" x2="8" y2="17" />
                <polyline points="10 9 9 9 8 9" />
              </svg>
              Đoạn mô tả project gợi ý cho CV:
            </div>
            <div className="cv-content">
              <strong>English-Vietnamese Neural Machine Translation Web Application (EnVi Lab):</strong> Xây dựng trọn gói hệ thống dịch máy nơ-ron Anh – Việt end-to-end từ tiền xử lý ngữ liệu IWSLT'15 (115k cặp câu), huấn luyện mô hình Sequence-to-Sequence (BiLSTM 512 units + Additive Bahdanau Attention 128 units, BPE 8k vocab) bằng TensorFlow/Keras 3 đạt 24.14 BLEU và 43.05 chrF++. Thiết kế và triển khai backend REST API bằng FastAPI với cơ chế quản lý vòng đời lifespan, worker-thread inference lock trên CPU (độ trễ ~45–65ms) và frontend tương tác cao với React, TypeScript và Vite.
            </div>
          </div>

          {/* Honest Educational Disclaimer */}
          <div className="disclaimer-card">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ flexShrink: 0, color: 'var(--text-secondary)' }}>
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="16" x2="12" y2="12" />
              <line x1="12" y1="8" x2="12.01" y2="8" />
            </svg>
            <div>
              <strong>Ghi chú học thuật:</strong> Đây là bản demo học tập phục vụ nghiên cứu cho các câu ngắn đến vừa; bản dịch từ mô hình BiLSTM cơ sở có thể còn tồn tại lỗi ngữ nghĩa, diễn đạt hoặc tên riêng trong một số ngữ cảnh phức tạp.
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-container">
          <div>
            <strong>EnVi Lab</strong> — Dự án cá nhân Dịch máy nơ-ron Anh – Việt.
          </div>
          <div>
            Chạy cục bộ trên Python 3.13, TensorFlow 2.21, FastAPI & React.
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
