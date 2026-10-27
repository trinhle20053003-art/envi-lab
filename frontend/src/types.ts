export interface HealthResponse {
  status: 'ok' | 'degraded';
  model_ready: boolean;
  message: string;
  missing_files: string[];
}

export interface ModelArchitecture {
  name: string;
  framework: string;
  encoder: string;
  decoder: string;
  attention: string;
  embedding_dim: number;
  vocab_size: number;
  tokenizer_type: string;
  total_parameters: number;
  decoding_strategy: string;
}

export interface ModelLimits {
  max_words: number;
  max_tokens: number;
  supported_language_pair: string;
}

export interface ModelDataset {
  name: string;
  train_pairs: number;
  validation_pairs: number;
  test_pairs: number;
  filtering: string;
}

export interface ModelTraining {
  hardware: string;
  epochs_completed: number;
  best_epoch: number;
  best_validation_token_nll: number;
  batch_size: number;
  learning_rate: number;
  dropout: number;
}

export interface LengthBucket {
  n: number;
  bleu: number;
  chrf_pp: number;
}

export interface ModelEvaluation {
  test_sentences: number;
  bleu: number;
  chrf_pp: number;
  bleu_signature: string;
  chrf_signature: string;
  by_source_length: {
    short_1_10: LengthBucket;
    medium_11_20: LengthBucket;
    medium_21_40: LengthBucket;
  };
  note: string;
}

export interface ModelInfoResponse {
  is_ready: boolean;
  missing_files: string[];
  model_path: string;
  architecture: ModelArchitecture;
  limits: ModelLimits;
  dataset: ModelDataset;
  training: ModelTraining;
  evaluation: ModelEvaluation;
  disclaimer: string;
}

export interface TranslateResponse {
  source_text: string;
  translated_text: string;
  latency_ms: number;
  latency_sec: number;
  source_word_count: number;
  source_token_count: number;
}

export interface HistoryItem {
  id: string;
  source: string;
  translation: string;
  latency_ms: number;
  timestamp: string;
  wordCount: number;
}
