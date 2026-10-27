# English-Vietnamese BiLSTM + Bahdanau Attention

TensorFlow/Keras, trained from scratch. Encoder: bidirectional LSTM, 256 units/direction. Decoder: unidirectional LSTM, 512 units. Additive Bahdanau attention: 128 units. Embeddings: 256. BPE: 8,000 tokens/language; PAD/UNK/BOS/EOS. Attention context joins decoder output; no recurrent context input feeding. Greedy decoding.

Dataset: IWSLT English-Vietnamese TED corpus from the Stanford NLP download. After filtering/deduplication: train=115154, validation=1492, test=1177. See data_manifest.json for exact URLs, hashes and filtering; data.py for processing. Tokenizers trained only on train. Model selection uses validation token NLL.

Run: Tesla T4; 19 epochs; best epoch=14. Parameters=9171136; BLEU=24.14; chrF++=43.05; test n=1177. See evaluation.json for metric signatures.

Scope: one English sentence, at most 40 whitespace-separated words and 96 subword tokens including BOS/EOS. Educational baseline for an internship portfolio. Semantic errors, awkward Vietnamese and proper-name errors remain. Not suitable for high-stakes translation. Test metrics are not a quality guarantee. manual_review.csv contains 30 randomly selected test translations with unfilled review scores. Raw corpus is not included; upstream data terms still apply.
