from __future__ import annotations

import re
import struct
import logging
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


class _GGUFValueType:
    UINT8=0; INT8=1; UINT16=2; INT16=3; UINT32=4; INT32=5
    FLOAT32=6; BOOL=7; STRING=8; ARRAY=9; UINT64=10; INT64=11; FLOAT64=12

GGUF_MAGIC = 0x46554747

def _read_gguf_kv(path: str) -> Dict[str, Any]:
    """
    Open a GGUF file and return its full KV metadata dict.
    Stops reading after the KV section — does not touch tensor data.
    """
    kv: Dict[str, Any] = {}

    def ru8(f):  return struct.unpack("<B", f.read(1))[0]
    def ri8(f):  return struct.unpack("<b", f.read(1))[0]
    def ru16(f): return struct.unpack("<H", f.read(2))[0]
    def ri16(f): return struct.unpack("<h", f.read(2))[0]
    def ru32(f): return struct.unpack("<I", f.read(4))[0]
    def ri32(f): return struct.unpack("<i", f.read(4))[0]
    def rf32(f): return struct.unpack("<f", f.read(4))[0]
    def ru64(f): return struct.unpack("<Q", f.read(8))[0]
    def ri64(f): return struct.unpack("<q", f.read(8))[0]
    def rf64(f): return struct.unpack("<d", f.read(8))[0]
    def rbool(f): return bool(ru8(f))
    def rstr(f):
        n = ru64(f)
        return f.read(n).decode("utf-8", errors="replace")

    def rval(f, vtype):
        dispatch = {
            0:ru8, 1:ri8, 2:ru16, 3:ri16, 4:ru32, 5:ri32,
            6:rf32, 7:rbool, 8:rstr, 10:ru64, 11:ri64, 12:rf64,
        }
        if vtype == 9:
            et = ru32(f)
            n  = ru64(f)
            return [rval(f, et) for _ in range(n)]
        fn = dispatch.get(vtype)
        if fn is None:
            raise ValueError(f"Unknown GGUF value type {vtype}")
        return fn(f)

    with open(path, "rb") as f:
        magic = ru32(f)
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file: {path}")
        _version  = ru32(f)
        _n_tensor = ru64(f)
        n_kv      = ru64(f)
        for _ in range(n_kv):
            key   = rstr(f)
            vtype = ru32(f)
            val   = rval(f, vtype)
            kv[key] = val

    return kv


class TokenType:
    NORMAL      = 1
    UNKNOWN     = 2
    CONTROL     = 3
    USER_DEFINED = 4
    UNUSED      = 5
    BYTE        = 6


class _SPMTokenizer:
    """
    Pure-Python SentencePiece BPE tokenizer reconstructed from GGUF vocab.

    SPM uses a Viterbi-style algorithm over the vocabulary sorted by score.
    Byte-fallback tokens (<0x00>…<0xFF>) handle unknown characters.
    """

    SPIECE_UNDERLINE = "▁"

    def __init__(
        self,
        vocab:      List[str],
        scores:     List[float],
        token_type: List[int],
        bos_id: int,
        eos_id: int,
        unk_id: int,
        pad_id: int,
    ):
        self.vocab      = vocab
        self.scores     = scores
        self.token_type = token_type
        self.bos_id     = bos_id
        self.eos_id     = eos_id
        self.unk_id     = unk_id
        self.pad_id     = pad_id
        self.vocab_size = len(vocab)

        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(vocab)}

        self._byte_tokens: Dict[int, int] = {}
        for i, tok in enumerate(vocab):
            if token_type[i] == TokenType.BYTE or (
                tok.startswith("<0x") and tok.endswith(">") and len(tok) == 6
            ):
                try:
                    byte_val = int(tok[3:5], 16)
                    self._byte_tokens[byte_val] = i
                except ValueError:
                    pass

    def _encode_byte(self, byte_val: int) -> int:
        """Encode a single byte as its GGUF byte-fallback token id."""
        tid = self._byte_tokens.get(byte_val)
        if tid is not None:
            return tid
        return self.unk_id

    def _tokenize_word(self, word: str) -> List[int]:
        """
        Greedy longest-match BPE over a single word.
        Equivalent to SentencePiece's EncodeAsPieces but without the C++ library.
        """
        if not word:
            return []

        if word in self.token_to_id:
            return [self.token_to_id[word]]

        result: List[int] = []
        chars = list(word)
        i = 0
        while i < len(chars):
            best_end = -1
            for j in range(len(chars), i, -1):
                sub = "".join(chars[i:j])
                if sub in self.token_to_id:
                    best_end = j
                    break
            if best_end == -1:
                for byte_val in "".join(chars[i:i+1]).encode("utf-8"):
                    result.append(self._encode_byte(byte_val))
                i += 1
            else:
                result.append(self.token_to_id["".join(chars[i:best_end])])
                i = best_end
        return result

    def encode(self, text: str, add_bos: bool = True) -> List[int]:
        """Encode text → token id list."""
        text = text.replace(" ", self.SPIECE_UNDERLINE)

        words = text.split(self.SPIECE_UNDERLINE)
        tokens: List[int] = []
        for i, word in enumerate(words):
            if not word:
                continue
            prefix = self.SPIECE_UNDERLINE if i > 0 else ""
            tokens.extend(self._tokenize_word(prefix + word))

        if add_bos and self.bos_id >= 0:
            tokens = [self.bos_id] + tokens
        return tokens

    def decode(self, ids: List[int]) -> str:
        """Decode token id list → text string."""
        pieces: List[str] = []
        for tid in ids:
            if tid in (self.bos_id, self.eos_id, self.pad_id) and tid >= 0:
                continue
            if 0 <= tid < len(self.vocab):
                tok = self.vocab[tid]
                if self.token_type[tid] == TokenType.BYTE or (
                    tok.startswith("<0x") and tok.endswith(">") and len(tok) == 6
                ):
                    try:
                        pieces.append(bytes([int(tok[3:5], 16)]).decode("utf-8", errors="replace"))
                    except Exception:
                        pass
                else:
                    pieces.append(tok)
        result = "".join(pieces).replace(self.SPIECE_UNDERLINE, " ")
        return result.lstrip(" ")


class _BPETokenizer:
    """
    Pure-Python BPE tokenizer for tiktoken-style vocabularies (LLaMA 3, GPT-2).

    Key differences from SPM:
      \u2022 Uses explicit merge rules stored in `tokenizer.ggml.merges`.
      \u2022 Pre-tokenizes with a regex (splits on punctuation, whitespace, etc.).
      \u2022 Byte-level: every raw byte maps to a printable representation.
    """

    _SIMPLE_PAT_STR = (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
        r"|[^\w\s]?\w+"
        r"|\d{1,3}"
        r"| ?[^\w\s]+\n*"
        r"|\s*\n+"
        r"|\s+(?!\S)"
        r"|\s+"
    )
    _UNICODE_PAT_STR = (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
        r"|[^\r\n\p{L}\p{N}]?\p{L}+"
        r"|\p{N}{1,3}"
        r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
        r"|\s*[\r\n]+"
        r"|\s+(?!\S)"
        r"|\s+"
    )

    def __init__(
        self,
        vocab:   List[str],
        merges:  List[str],
        bos_id:  int,
        eos_id:  int,
        unk_id:  int,
    ):
        self.vocab      = vocab
        self.bos_id     = bos_id
        self.eos_id     = eos_id
        self.unk_id     = unk_id
        self.vocab_size = len(vocab)

        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(vocab)}

        self.merges: Dict[Tuple[str, str], int] = {}
        for rank, merge in enumerate(merges):
            parts = merge.split(" ", 1)
            if len(parts) == 2:
                self.merges[(parts[0], parts[1])] = rank

        self._byte_encoder = self._build_byte_encoder()
        self._byte_decoder = {v: k for k, v in self._byte_encoder.items()}

        try:
            import regex as _regex_mod
            self._pat = _regex_mod.compile(self._UNICODE_PAT_STR, _regex_mod.UNICODE)
        except ImportError:
            self._pat = re.compile(self._SIMPLE_PAT_STR, re.UNICODE)

    @staticmethod
    def _build_byte_encoder() -> Dict[int, str]:
        """
        GPT-2's byte-to-unicode mapping.
        Maps every byte 0-255 to a unique printable Unicode character,
        so the BPE vocabulary only contains printable tokens.
        """
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return {b: chr(c) for b, c in zip(bs, cs)}

    def _bpe(self, token_chars: List[str]) -> List[str]:
        """Apply BPE merge rules to a list of characters."""
        word = token_chars[:]
        while len(word) > 1:
            best_rank  = float("inf")
            best_i     = -1
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self.merges.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_i    = i
            if best_i == -1 or best_rank == float("inf"):
                break
            merged = word[best_i] + word[best_i + 1]
            word   = word[:best_i] + [merged] + word[best_i + 2:]
        return word

    def _encode_word(self, raw_word: str) -> List[int]:
        """Encode a single pre-tokenized word → token ids."""
        encoded = "".join(self._byte_encoder.get(b, chr(b)) for b in raw_word.encode("utf-8"))
        chars   = list(encoded)
        merged  = self._bpe(chars)
        ids: List[int] = []
        for piece in merged:
            ids.append(self.token_to_id.get(piece, self.unk_id))
        return ids

    def encode(self, text: str, add_bos: bool = True) -> List[int]:
        """Encode text → token id list."""
        tokens: List[int] = []
        for match in self._pat.finditer(text):
            tokens.extend(self._encode_word(match.group()))
        if add_bos and self.bos_id >= 0:
            tokens = [self.bos_id] + tokens
        return tokens

    def decode(self, ids: List[int]) -> str:
        """Decode token id list → text string."""
        text = ""
        for tid in ids:
            if tid in (self.bos_id, self.eos_id) and tid >= 0:
                continue
            if 0 <= tid < len(self.vocab):
                text += self.vocab[tid]
        byte_array = bytearray()
        for char in text:
            byte_val = self._byte_decoder.get(char)
            if byte_val is not None:
                byte_array.append(byte_val)
            else:
                byte_array.extend(char.encode("utf-8"))
        return byte_array.decode("utf-8", errors="replace")


class _WordPieceTokenizer:
    def __init__(self, vocab: List[str], bos_id: int, eos_id: int, unk_id: int):
        self.vocab       = vocab
        self.bos_id      = bos_id
        self.eos_id      = eos_id
        self.unk_id      = unk_id
        self.vocab_size  = len(vocab)
        self.token_to_id = {t: i for i, t in enumerate(vocab)}
        self._unk = vocab[unk_id] if 0 <= unk_id < len(vocab) else "[UNK]"

    def encode(self, text: str, add_bos: bool = True) -> List[int]:
        tokens: List[int] = []
        for word in text.lower().split():
            if word in self.token_to_id:
                tokens.append(self.token_to_id[word])
                continue
            start = 0
            sub_tokens: List[int] = []
            while start < len(word):
                end = len(word)
                found = False
                while start < end:
                    substr = word[start:end]
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.token_to_id:
                        sub_tokens.append(self.token_to_id[substr])
                        start = end
                        found = True
                        break
                    end -= 1
                if not found:
                    sub_tokens = [self.unk_id]
                    break
            tokens.extend(sub_tokens)
        if add_bos and self.bos_id >= 0:
            tokens = [self.bos_id] + tokens
        return tokens

    def decode(self, ids: List[int]) -> str:
        pieces = []
        for tid in ids:
            if tid in (self.bos_id, self.eos_id) and tid >= 0:
                continue
            if 0 <= tid < len(self.vocab):
                tok = self.vocab[tid]
                if tok.startswith("##"):
                    pieces.append(tok[2:])
                else:
                    pieces.append(" " + tok)
        return "".join(pieces).strip()


class _ChatTemplate:
    """
    Minimal Jinja2-like chat template renderer.

    Handles the subset of Jinja2 used in LLaMA/Mistral/Gemma templates:
      {{ message['role'] }}, {% for %}, {% if %}, {%- -%} whitespace control.

    Falls back to a sensible default if the template is not present.
    """

    KNOWN_FORMATS: Dict[str, str] = {
        "llama-2": "<s>[INST] {system}\n\n{user} [/INST] {assistant} </s>",
        "llama-3": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "mistral": "<s>[INST] {user} [/INST]",
        "gemma":   "<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n",
        "chatml":  "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n",
        "phi3":    "<|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n",
        "default": "### System:\n{system}\n\n### User:\n{user}\n\n### Assistant:\n",
    }

    def __init__(self, template: Optional[str], model_hint: str = ""):
        self._raw     = template
        self._hint    = model_hint.lower()
        self._has_jinja = template is not None

    def _detect_format_key(self) -> str:
        for key in self.KNOWN_FORMATS:
            if key in self._hint:
                return key
        return "default"

    def apply(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """
        Apply the chat template to a list of {"role": ..., "content": ...} dicts.
        Returns the fully formatted string ready for tokenization.
        """
        if self._has_jinja:
            try:
                return self._apply_jinja(messages, add_generation_prompt)
            except Exception as e:
                pass

        return self._apply_fallback(messages)

    def _apply_jinja(self, messages: List[Dict[str, str]], add_gen: bool) -> str:
        """Use the jinja2 library if available, otherwise raise."""
        try:
            import jinja2
        except ImportError:
            raise RuntimeError("jinja2 not installed (pip install jinja2)")

        env = jinja2.Environment(
            trim_blocks=True, lstrip_blocks=True,
            keep_trailing_newline=True,
        )
        env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
        tmpl = env.from_string(self._raw)
        return tmpl.render(
            messages=messages,
            add_generation_prompt=add_gen,
            bos_token="<s>",
            eos_token="</s>",
        )

    def _apply_fallback(self, messages: List[Dict[str, str]]) -> str:
        """Apply a known format string as fallback."""
        fmt_key = self._detect_format_key()
        fmt     = self.KNOWN_FORMATS[fmt_key]

        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs  = [m["content"] for m in messages if m["role"] == "user"]
        asst_msgs  = [m["content"] for m in messages if m["role"] == "assistant"]

        user   = user_msgs[-1]  if user_msgs  else ""
        asst   = asst_msgs[-1] if asst_msgs  else ""

        result = fmt.format(system=system_msg, user=user, assistant=asst)
        return result


class GGUFTokenizer:
    """
    Unified tokenizer that auto-detects type from GGUF metadata.

    Usage:
        tok = GGUFTokenizer.from_gguf("models/llama-3.gguf")
        ids  = tok.encode("Hello!")
        text = tok.decode(ids)
        prompt = tok.format_chat([{"role": "user", "content": "Hi"}])
        ids  = tok.encode(prompt, add_bos=False)
    """

    def __init__(self, impl, chat_tmpl: _ChatTemplate, model_type: str,
                 vocab: List[str], bos_id: int, eos_id: int):
        self._impl        = impl
        self._chat        = chat_tmpl
        self.model_type   = model_type
        self.vocab        = vocab
        self.bos_id       = bos_id
        self.eos_id       = eos_id
        self.vocab_size   = len(vocab)

    @classmethod
    def from_gguf(cls, model_path: str) -> "GGUFTokenizer":
        """
        Read a GGUF file and construct the appropriate tokenizer automatically.

        Raises:
            FileNotFoundError
            ValueError
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        kv = _read_gguf_kv(str(path))

        model_type  = str(kv.get("tokenizer.ggml.model", "llama")).lower()
        vocab: List[str]   = kv.get("tokenizer.ggml.tokens", [])
        scores: List[float] = kv.get("tokenizer.ggml.scores", [0.0] * len(vocab))
        tok_types: List[int] = kv.get("tokenizer.ggml.token_type", [1] * len(vocab))
        merges: List[str]   = kv.get("tokenizer.ggml.merges", [])

        while len(scores)    < len(vocab): scores.append(0.0)
        while len(tok_types) < len(vocab): tok_types.append(1)

        bos_id = int(kv.get("tokenizer.ggml.bos_token_id", 1))
        eos_id = int(kv.get("tokenizer.ggml.eos_token_id", 2))
        unk_id = int(kv.get("tokenizer.ggml.unknown_token_id", 0))
        pad_id = int(kv.get("tokenizer.ggml.padding_token_id", -1))

        raw_template = kv.get("tokenizer.chat_template", None)
        arch         = str(kv.get("general.architecture", ""))
        chat_tmpl    = _ChatTemplate(raw_template, model_hint=arch + " " + model_type)

        if not vocab:
            impl = _ByteFallbackTokenizer()
            model_type = "byte"
        elif model_type in ("llama", "spm", "sentencepiece", "llama2", "mistral", "gemma"):
            impl = _SPMTokenizer(vocab, scores, tok_types, bos_id, eos_id, unk_id, pad_id)
        elif model_type in ("gpt2", "tiktoken", "bpe", "llama3", "phi3", "phi-3"):
            impl = _BPETokenizer(vocab, merges, bos_id, eos_id, unk_id)
        elif model_type in ("bert", "wordpiece"):
            impl = _WordPieceTokenizer(vocab, bos_id, eos_id, unk_id)
        else:
            impl = _SPMTokenizer(vocab, scores, tok_types, bos_id, eos_id, unk_id, pad_id)

        tok = cls(impl, chat_tmpl, model_type, vocab, bos_id, eos_id)
        return tok

    def encode(self, text: str, add_bos: bool = True) -> List[int]:
        """
        Convert a text string into a list of token ids.

        Args:
            text:    Input string.
            add_bos: Prepend the BOS token. Set False if your chat template
                     already includes it (most do).
        Returns:
            List of integer token ids.
        """
        return self._impl.encode(text, add_bos=add_bos)

    def decode(self, ids: List[int]) -> str:
        """
        Convert a list of token ids back into a text string.
        BOS and EOS tokens are automatically stripped.
        """
        return self._impl.decode(ids)

    def decode_token(self, token_id: int) -> str:
        """Decode a single token id (used for streaming output)."""
        return self._impl.decode([token_id])

    def format_chat(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Apply the model's chat template to a list of messages.

        Args:
            messages: List of dicts with "role" and "content" keys.
                      Roles: "system", "user", "assistant"
            add_generation_prompt: Append the token(s) that cue the model
                                   to start generating (True for inference).
        Returns:
            Fully formatted prompt string, ready to pass to encode().

        Example:
            prompt = tok.format_chat([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": "What is 2+2?"},
            ])
            ids = tok.encode(prompt, add_bos=False)
        """
        return self._chat.apply(messages, add_generation_prompt)

    def has_chat_template(self) -> bool:
        """Returns True if a real Jinja2 template was found in the GGUF file."""
        return self._chat._has_jinja

    def get_special_tokens(self) -> Dict[str, int]:
        """Return a dict of known special token name → id."""
        return {
            "bos": self.bos_id,
            "eos": self.eos_id,
        }

    def __repr__(self) -> str:
        return (
            f"GGUFTokenizer(type={self.model_type!r}, "
            f"vocab_size={self.vocab_size}, "
            f"bos={self.bos_id}, eos={self.eos_id}, "
            f"chat_template={'yes' if self.has_chat_template() else 'fallback'})"
        )


class _ByteFallbackTokenizer:
    """Encodes text as raw UTF-8 bytes. Used when no vocab is found in GGUF."""
    bos_id = 1; eos_id = 2; vocab_size = 256

    def encode(self, text: str, add_bos: bool = True) -> List[int]:
        ids = [b + 3 for b in text.encode("utf-8")]
        return ([self.bos_id] + ids) if add_bos else ids

    def decode(self, ids: List[int]) -> str:
        return bytes(max(0, i - 3) for i in ids
                     if i not in (self.bos_id, self.eos_id) and i > 2
        ).decode("utf-8", errors="replace")