import json
import glob
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from IPython.display import clear_output

#######################
# 파라미터 설정
#######################
TRAIN_PATH = "assets/trainning/TL_1.발화단위평가_경제활동_상품상거래"
VAL_PATH = "assets/validation/TL_1.발화단위평가_경제활동_상품상거래"
BATCH_SIZE = 8
EPOCHS = 10
MAX_LEN = 128  # 최대 시퀀스 길이(필요에 따라 조정)
label_keys = [
    "linguistic_acceptability", "consistency", "interestingness",
    "unbias", "harmlessness", "no_hallucination", "understandability",
    "sensibleness", "specificity"
]


#######################
# 데이터 로드 함수
#######################
def load_data_from_dir(directory):
    texts = []
    labels = []
    json_files = glob.glob(os.path.join(directory, "*.json"))

    for fpath in json_files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        conversations = data["dataset"]["conversations"]
        for conv in conversations:
            for utt in conv["utterances"]:
                # speaker_id=0인 봇 발화
                if utt["speaker_id"] == 0:
                    evaluations = utt.get("utterance_evaluation", [])
                    if len(evaluations) > 0:
                        # 여러 평가자 평균
                        eval_count = len(evaluations)
                        label_sum = {k: 0 for k in label_keys}
                        for ev in evaluations:
                            for k in label_keys:
                                label_sum[k] += (1 if ev[k] == "yes" else 0)
                        final_label = [1 if (label_sum[k] / eval_count) >= 0.5 else 0 for k in label_keys]
                        texts.append(utt["utterance_text"])
                        labels.append(final_label)
    return texts, np.array(labels, dtype=np.float32)


#######################
# 데이터 로드
#######################
train_texts, train_labels = load_data_from_dir(TRAIN_PATH)
val_texts, val_labels = load_data_from_dir(VAL_PATH)

#######################
# 토크나이징 & vocab 구성
#######################
all_tokens = [t.split() for t in train_texts + val_texts]
vocab = {"<pad>": 0, "<unk>": 1}
for tokens in all_tokens:
    for w in tokens:
        if w not in vocab:
            vocab[w] = len(vocab)


def encode(text):
    tokens = text.split()
    ids = [vocab.get(w, 1) for w in tokens]
    # 패딩
    if len(ids) > MAX_LEN:
        ids = ids[:MAX_LEN]
    else:
        ids += [0] * (MAX_LEN - len(ids))
    return ids


train_ids = np.array([encode(t) for t in train_texts], dtype=np.int32)
val_ids = np.array([encode(t) for t in val_texts], dtype=np.int32)

#######################
# tf.data.Dataset 구성
#######################
train_dataset = tf.data.Dataset.from_tensor_slices((train_ids, train_labels)).shuffle(1024).batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.from_tensor_slices((val_ids, val_labels)).batch(BATCH_SIZE)

#######################
# 모델 정의 (RNN)
#######################
vocab_size = len(vocab)
embed_dim = 128
hidden_dim = 64
num_labels = len(label_keys)

inputs = keras.Input(shape=(MAX_LEN,))
x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(inputs)
x = layers.LSTM(hidden_dim)(x)
outputs = layers.Dense(num_labels, activation='sigmoid')(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


#######################
# 커스텀 콜백 정의
#######################
class CustomCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.train_acc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')

        self.train_acc.append(acc)
        self.val_acc.append(val_acc)

        # 그래프 업데이트
        clear_output(wait=True)
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, epoch + 2), self.train_acc, label='Train Accuracy', marker='o')
        plt.plot(range(1, epoch + 2), self.val_acc, label='Val Accuracy', marker='o')
        plt.title("Accuracy per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.show()

        # 학습 결과 출력
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"Train Accuracy: {acc:.4f}, Val Accuracy: {val_acc:.4f}")


custom_callback = CustomCallback()

#######################
# 학습
#######################
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[custom_callback],
    verbose=0  # fit 출력은 콜백을 통해 하므로 0으로 설정
)
