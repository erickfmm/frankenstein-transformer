"""SentencePiece tokenizer for Spanish text trained on RedPajama.

Provides :class:`SpanishSPMTokenizer`, a BPE SentencePiece tokenizer
trained on the ``erickfmm/red_pajama_es_hq_35`` dataset with special
tokens ``[CLS]``, ``[SEP]``, ``[MASK]``, and ``[PAD]``.
"""

import os
from typing import List, Dict
import logging
import sentencepiece as spm
from datasets import load_dataset

try:
    from ..utils.storage_manager import StorageManager
except ImportError:
    from utils.storage_manager import StorageManager


class SpanishSPMTokenizer:
    """Spanish SentencePiece BPE tokenizer trained on RedPajama data.

    Supports training a new tokenizer from the streaming dataset, loading
    a pre-trained model file, and encoding text with special tokens
    (``[CLS]``, ``[SEP]``, ``[PAD]``) and configurable max length.

    Attributes:
        vocab_size: Vocabulary size (may be updated after loading).
        model_path: Path to the ``.model`` file, or ``None``.
        sp_model: The underlying ``SentencePieceProcessor``, or ``None``.
        vocab: Token-to-ID mapping dictionary.
        inverse_vocab: ID-to-token mapping dictionary.
        storage_manager: :class:`StorageManager` for temp file tracking.
    """

    def __init__(self, vocab_size: int = 50000, model_path: str = None):
        """Initialize the tokenizer.

        Args:
            vocab_size: Target vocabulary size for training.
            model_path: Optional path to an existing ``.model`` file.
                If provided and exists, the model is loaded immediately.
        """
        self.vocab_size = vocab_size
        self.model_path = model_path
        self.sp_model = None
        self.vocab = {}
        self.inverse_vocab = {}
        self.storage_manager = StorageManager()
        
        # If model_path provided, load the model
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        
    def prepare_training_data(self, max_samples: int = 50_000_000, 
                              target_ram_gb: float = 100.0) -> str:
        """
        Prepare training data from erickfmm/red_pajama_es_hq_35
        
        Args:
            max_samples: Maximum number of text samples to collect (default: 50M)
            target_ram_gb: Target RAM usage in GB (default: 100GB)
        
        The actual limit will be determined by whichever comes first:
        max_samples or estimated RAM usage approaching target_ram_gb
        """
        logging.info(f"Preparing tokenizer training data (target: {target_ram_gb}GB RAM)")
        logging.info(f"Max samples limit: {max_samples:,}")
        
        # Create temp file for training data
        temp_file = self.storage_manager.create_temp_file(suffix=".txt")
        
        try:
            # Load dataset with streaming to avoid loading everything at once
            dataset = load_dataset(
                "erickfmm/red_pajama_es_hq_35",
                split="train",
                streaming=True
            )
            
            count = 0
            total_chars = 0
            estimated_ram_gb = 0
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                for example in dataset:
                    if count >= max_samples:
                        logging.info(f"Reached max_samples limit: {max_samples:,}")
                        break
                    
                    if 'text' in example:
                        text = example['text'].strip()
                        if len(text) > 100:  # Skip very short texts
                            f.write(text + '\n')
                            count += 1
                            total_chars += len(text)
                            
                            # More frequent logging for first million, then every 100k
                            if count <= 1_000_000 and count % 100_000 == 0:
                                estimated_ram_gb = (total_chars * 2) / (1024**3)  # rough estimate
                                logging.info(
                                    f"Collected {count:,} samples "
                                    f"({total_chars/1e9:.2f}B chars, "
                                    f"~{estimated_ram_gb:.1f}GB estimated RAM)"
                                )
                            elif count > 1_000_000 and count % 1_000_000 == 0:
                                estimated_ram_gb = (total_chars * 2) / (1024**3)
                                logging.info(
                                    f"Collected {count:,} samples "
                                    f"({total_chars/1e9:.2f}B chars, "
                                    f"~{estimated_ram_gb:.1f}GB estimated RAM)"
                                )
                                f.flush()  # Flush less frequently
                            
                            # Conservative check: stop if estimated RAM approaches target
                            if estimated_ram_gb > target_ram_gb * 0.9:  # 90% of target
                                logging.warning(
                                    f"Approaching RAM limit (~{estimated_ram_gb:.1f}GB), "
                                    f"stopping at {count:,} samples"
                                )
                                break
                
            final_ram_estimate = (total_chars * 2) / (1024**3)
            logging.info(f"✅ Created training file with {count:,} samples")
            logging.info(f"   Total characters: {total_chars:,} ({total_chars/1e9:.2f}B)")
            logging.info(f"   Estimated RAM usage: ~{final_ram_estimate:.1f}GB")
            return temp_file
            
        except Exception as e:
            logging.error("="*60)
            logging.error("🚨 CRITICAL ERROR: Failed to load real dataset")
            logging.error("="*60)
            logging.error(f"Error type: {type(e).__name__}")
            logging.error(f"Error message: {e}")
            logging.error(f"Dataset: erickfmm/red_pajama_es_hq_35")
            logging.error(f"Samples collected before failure: {count:,}")
            logging.error(f"Characters collected: {total_chars:,}")
            logging.error("\nAttempted parameters:")
            logging.error(f"  - max_samples: {max_samples:,}")
            logging.error(f"  - target_ram_gb: {target_ram_gb}")
            logging.error("\n⛔ SYSTEM HALTED - No synthetic data fallback allowed")
            logging.error("="*60)
            import traceback
            logging.error("\nFull traceback:")
            logging.error(traceback.format_exc())
            raise RuntimeError(
                f"Failed to prepare tokenizer training data from real dataset. "
                f"Error: {e}. Check logs for details."
            ) from e
    
    def train(self, model_prefix: str = "es_spm_50k", max_training_samples: int = 50_000_000,
              target_ram_gb: float = 100.0):
        """
        Train SentencePiece tokenizer with maximum data utilization
        
        Args:
            model_prefix: Prefix for output model files
            max_training_samples: Maximum samples to collect from dataset
            target_ram_gb: Target RAM usage (default: 100GB)
        """
        logging.info(f"Starting tokenizer training with up to {target_ram_gb}GB RAM")
        
        train_file = self.prepare_training_data(
            max_samples=max_training_samples,
            target_ram_gb=target_ram_gb
        )
        
        # Get available cores
        num_cores = os.cpu_count() or 1
        logging.info(f"Using {num_cores} CPU cores for training")
        
        # SPM training arguments - optimized for large data
        spm_args = [
            f'--input={train_file}',
            f'--model_prefix={model_prefix}',
            f'--vocab_size={self.vocab_size}',
            '--character_coverage=0.9995',
            '--model_type=bpe',
            '--max_sentence_length=16384',
            '--pad_id=0',
            '--unk_id=1',
            '--bos_id=2',
            '--eos_id=3',
            '--user_defined_symbols=[CLS],[SEP],[MASK],[PAD]',
            '--split_by_whitespace=true',
            '--normalization_rule_name=nmt_nfkc',
            '--add_dummy_prefix=true',
            '--byte_fallback=true',
            '--split_digits=true',
            f'--num_threads={num_cores}',  # Use ALL cores for maximum speed
            '--input_sentence_size=100000000',  # 100M sentences max (increased from 1M)
            '--shuffle_input_sentence=true',
            '--train_extremely_large_corpus=true',  # SentencePiece flag for large data
        ]
        
        logging.info("Training SentencePiece tokenizer...")
        spm.SentencePieceTrainer.train(' '.join(spm_args))
        
        # Load trained model
        model_path = f"{model_prefix}.model"
        self.model_path = model_path  # Store model path for later use
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_path)
        
        # Build vocab
        for i in range(self.sp_model.GetPieceSize()):
            token = self.sp_model.IdToPiece(i)
            self.vocab[token] = i
            self.inverse_vocab[i] = token
        
        # Update vocab_size to match trained model
        self.vocab_size = self.sp_model.GetPieceSize()
        
        logging.info(f"Tokenizer trained with {len(self.vocab)} tokens")
        return model_path
    
    def load(self, model_path: str):
        """Load an existing SentencePiece model from disk.

        Args:
            model_path: Path to the ``.model`` file.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model_path = model_path
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_path)
        
        # Build vocab
        self.vocab = {}
        self.inverse_vocab = {}
        for i in range(self.sp_model.GetPieceSize()):
            token = self.sp_model.IdToPiece(i)
            self.vocab[token] = i
            self.inverse_vocab[i] = token
        
        # Update vocab_size to match loaded model
        self.vocab_size = self.sp_model.GetPieceSize()
        
        logging.info(f"Loaded tokenizer from {model_path} with {len(self.vocab)} tokens")
    
    def encode(self, text: str, max_length: int = 512) -> Dict[str, List[int]]:
        """Encode text with special tokens and padding/truncation.

        Prepends ``[CLS]``, appends ``[SEP]``, and pads with ``[PAD]``
        to reach ``max_length``. Truncates if the tokenized sequence
        exceeds ``max_length``.

        Args:
            text: Raw input text string.
            max_length: Maximum sequence length including special tokens.

        Returns:
            Dictionary with ``"input_ids"`` and ``"attention_mask"``
            lists, both of length ``max_length``.

        Raises:
            RuntimeError: If the tokenizer has not been initialized
                (call :meth:`train` or :meth:`load` first).
        """
        if self.sp_model is None:
            raise RuntimeError("Tokenizer not initialized. Call train() or load() first.")
        
        tokens = self.sp_model.encode_as_ids(text)
        
        # Add special tokens
        tokens = [self.vocab['[CLS]']] + tokens[:max_length-2] + [self.vocab['[SEP]']]
        
        # Pad or truncate
        if len(tokens) < max_length:
            tokens = tokens + [self.vocab['[PAD]']] * (max_length - len(tokens))
            attention_mask = [1] * len(tokens) + [0] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
            attention_mask = [1] * max_length
        
        return {
            'input_ids': tokens,
            'attention_mask': attention_mask[:max_length]
        }
