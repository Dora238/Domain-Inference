import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5Tokenizer, T5EncoderModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.metrics import silhouette_score
# 设置 device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载两个模型和 tokenizer
model_name = "bert-base-uncased"
bert_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
bert_tokenizer = AutoTokenizer.from_pretrained(model_name)

spam_name = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
spam_tokenizer = AutoTokenizer.from_pretrained(spam_name)
spam_model = AutoModelForSequenceClassification.from_pretrained(spam_name).to(device).eval()

t5_name = "google/flan-t5-base"
t5_tokenizer = T5Tokenizer.from_pretrained(t5_name)
t5_model = T5EncoderModel.from_pretrained(t5_name).to(device).eval()

# 加载数据集
dataset_name = "ucirvine/sms_spam"
dataset = load_dataset(dataset_name, split="train[:1000]")

# 嵌入列表
bert_embeddings, spam_embeddings, t5_embeddings, labels = [], [], [], []

def get_t5_encoder_hidden(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)  # 只跑 encoder
    hidden = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
    return hidden.mean(dim=1).squeeze().cpu().numpy()  # 可选 mean pooling 或取第一个 token


def get_bert_cls(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model.bert(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # 取 [CLS]

# 遍历数据
for sample in tqdm(dataset):
    text, label = sample["sms"], sample["label"]
    try:
        bert_emb = get_bert_cls(text, bert_model, bert_tokenizer)
        spam_emb = get_bert_cls(text, spam_model, spam_tokenizer)
        t5_emb = get_t5_encoder_hidden(text, t5_model, t5_tokenizer)
        bert_embeddings.append(bert_emb)
        spam_embeddings.append(spam_emb)
        t5_embeddings.append(t5_emb)
        labels.append(label)
    except Exception as e:
        print(f"跳过：{text[:30]} - {str(e)}")

# 降维函数
def reduce_and_plot(embeddings, labels, title):
    score = silhouette_score(embeddings, labels)
    print(f"{title} Silhouette Score: {score:.4f}")
    embeddings = np.array(embeddings)
    if embeddings.ndim == 3:
        embeddings = embeddings.squeeze(axis=1)
    assert embeddings.ndim == 2, f"Expect 2D array, got shape {embeddings.shape}"
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(pca_result)

    plt.figure(figsize=(8, 6))
    colors = ['blue' if l == 0 else 'red' for l in labels]
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=colors, alpha=0.6)
    plt.title(f"t-SNE of {title} Embeddings")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.legend(['ham (0)', 'spam (1)'])
    plt.savefig(f"{title}_tsne.png")
    plt.show()

# 可视化两个模型的结果
dataset_name = dataset_name.replace("/", "_")
reduce_and_plot(bert_embeddings, labels, f"{dataset_name} BERT-tiny [CLS]")
reduce_and_plot(spam_embeddings, labels, f"{dataset_name} Fine-tuned [CLS]")
reduce_and_plot(t5_embeddings, labels, f"{dataset_name} T5 [CLS]")
