import numpy as np
import keras_bert
from random import randint
from itertools import product
import keras
from logging import info
import collections
from typing import List, OrderedDict, Optional, Tuple, Union
from dependencies.keras_bert.keras_bert.loader import build_model_from_config

CLASS_LABELS = ["Cacao", "NotCacao"]
ALPHABET = 'ACGT'

Record = collections.namedtuple('Record', ['id', 'seq'])

def parse_fasta(fasta) -> List[Record]:
    records = []
    id_ = None
    seq = ''
    allowed_characters = set("ACTGYRWSKMDVHBXN")
    with open(fasta) as f:
        for line in f:
            if line.startswith('>'):
                if (id_ is not None):
                    records.append(Record(id_, seq))
                id_ = line[1:].strip()
                seq = ''
            else:
                line_no_whitespace = line.strip()
                assert set(line_no_whitespace.upper()).issubset(allowed_characters), \
                    f"{fasta} is not a fasta-file: contains not allowed characters " \
                    f"{set(line_no_whitespace.upper()).difference(allowed_characters)} in line\n{line_no_whitespace}"
                seq += line_no_whitespace

        assert id_ is not None, f"{fasta} is not a fasta file: no header detected"
        records.append(Record(id_, seq))
    info(f'read in {len(records)} sequences')
    return records


def seq2kmers(seq, k=3, stride=3, pad=True, to_upper=True):
    """transforms sequence to k-mer sequence.
    If specified, end will be padded so no character is lost"""
    if (len(seq) < k):
        return [seq.ljust(k, 'N')] if pad else []
    kmers = []
    for i in range(0, len(seq) - k + 1, stride):
        kmer = seq[i:i + k]
        if to_upper:
            kmers.append(kmer.upper())
        else:
            kmers.append(kmer)
    if (pad and len(seq) - (i + k)) % k != 0:
        kmers.append(seq[i + k:].ljust(k, 'N'))
    return kmers


def seq2tokens(seq, token_dict, seq_length=250, max_length=None,
               k=3, stride=3, window=True, seq_len_like=None):
    """transforms raw sequence into list of tokens to be used for
    fine-tuning BERT"""
    if (max_length is None):
        max_length = seq_length
    if (seq_len_like is not None):
        seq_length = min(max_length, np.random.choice(seq_len_like))
        # open('seq_lens.txt', 'a').write(str(seq_length) + ', ')
    seq = seq2kmers(seq, k=k, stride=stride, pad=True)
    if (window):
        start = randint(0, max(len(seq) - seq_length - 1, 0))
        end = start + seq_length - 1
    else:
        start = 0
        end = seq_length
    indices = [token_dict['[CLS]']] + [token_dict[word]
                                       if word in token_dict
                                       else token_dict['[UNK]']
                                       for word in seq[start:end]]
    if (len(indices) < max_length):
        indices += [token_dict['']] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    segments = [0 for _ in range(max_length)]
    return [np.array(indices), np.array(segments)]

def seq_frames(seq: str, frame_len: int, running_window=False, stride=1):
    """returns all windows of seq with a maximum length of `frame_len` and specified stride
    if `running_window` else `frame_len` -- alongside the chunks' positions"""
    iterator = (range(0, len(seq) - frame_len + 1, stride) if running_window
                else range(0, len(seq), frame_len))
    return [seq[i:i + frame_len] for i in iterator], [(i, i + frame_len) for i in iterator]

def get_token_dict(alph=ALPHABET, k=3) -> dict:
    """get token dictionary dict generated from `alph` and `k`"""
    token_dict = keras_bert.get_base_dict()
    for word in [''.join(_) for _ in product(alph, repeat=k)]:
        token_dict[word] = len(token_dict)
    return token_dict

def process_bert_tokens_batch2(batch_x):
    """when `seq2tokens` is used as `custom_encode_sequence`, batches
    are generated as [[input1, input2], [input1, input2], ...]. In
    order to train, they have to be transformed to [input1s,
    input2s] with this function"""
    return [np.array([_[0] for _ in batch_x]),
            np.array([_[1] for _ in batch_x])]

def process_bert_tokens_batch(batch_x):
    """Transforma [[input_ids, segment_ids], ...] en [input_ids, segment_ids, input_mask]"""
    input_ids = np.array([_[0] for _ in batch_x])
    segment_ids = np.array([_[1] for _ in batch_x])
    input_mask = (input_ids != 0).astype(int)  # 1 donde hay token, 0 donde hay padding

    return [input_ids, segment_ids, input_mask]

def load_bert2(bert_path, compile_=False):
    """get bert model from path"""
    custom_objects = {'GlorotNormal': keras.initializers.glorot_normal,
                      'GlorotUniform': keras.initializers.glorot_uniform}
    custom_objects.update(keras_bert.get_custom_objects())
    model = keras.models.load_model(bert_path, compile=compile_,
                                    custom_objects=custom_objects)
    return model

def load_bert(config_path, checkpoint_path=None, weights_path=None, seq_len=151, compile_=False):
    """Reconstruct BERT model and optionally load weights"""
    
    # Paso 1: reconstruir la arquitectura BERT
    model, _ = build_model_from_config(
        config_file = config_path,
        seq_len=seq_len,
        training=True
    )

    # Paso 2: cargar pesos si están disponibles
    if weights_path:
        model.load_weights(weights_path, skip_mismatch=True)
    return model

def annotate_predictions2(preds: List[np.ndarray],
                         overwrite_class_labels: Optional[OrderedDict] = None) -> dict:
    """annotates list of prediction arrays with provided or preset labels"""
    class_labels = (overwrite_class_labels if overwrite_class_labels is not None
                    else CLASS_LABELS)
    return {rank: {l: v.astype(float) for l, v in zip(class_labels[rank], p.transpose())}
            for rank, p in zip(class_labels, preds)}

def annotate_predictions(preds: List[np.ndarray],
                         overwrite_class_labels: Optional[List[str]] = None) -> dict:
    """Annotates prediction arrays with provided or default class labels"""
    class_labels = overwrite_class_labels if overwrite_class_labels is not None else CLASS_LABELS

    # Asumimos que solo hay una matriz de predicciones, para una sola clasificación binaria
    p = preds[0]  # Extrae la única matriz de predicción
    return {
        label: float(score)
        for label, score in zip(class_labels, p[0])  # p[0] es el primer (y probablemente único) ejemplo
    }

def best_predictions(preds_annotated: Union[dict, List[Tuple[str, dict]]]) -> List[Tuple[str, float]]:
    """returns best prediction classes alongside predicton confidences"""
    result = []
    for item in preds_annotated:
        if isinstance(item, tuple):
            rank, preds_dict = item
        else:
            rank, preds_dict = item, preds_annotated[item]
        best = max(preds_dict, key=lambda l: preds_dict[l])
        result.append((best, preds_dict[best]))
    return result
