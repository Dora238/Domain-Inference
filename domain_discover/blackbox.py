"""Black-box classifier wrapper for domain discovery."""

from typing import List, Optional, Dict, Union
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataclasses import dataclass


@dataclass
class BlackBox:
    """Wrapper for HuggingFace classifier models that returns hard labels.
    
    This class provides a unified interface to multiple pre-trained classifiers
    that return hard labels for input texts. The classifiers are treated as
    black boxes with no gradient information available.
    
    Args:
        model_name: Name of the HuggingFace model to use, options include:
            - "j-hartmann/emotion-english-distilroberta-base" (情感分类)
            - "nlptown/bert-base-multilingual-uncased-sentiment" (情感极性)
            - "s-nlp/roberta_toxicity_classifier" (毒性检测)
            - "mrm8488/bert-tiny-finetuned-sms-spam-detection" (垃圾短信)
            - "skandavivek2/spam-classifier" (垃圾短信)
            - "wesleyacheng/sms-spam-classification-with-bert" (垃圾短信)
            - "jackhhao/jailbreak-classifier" (越狱检测)
            - "lordofthejars/jailbreak-classifier" (越狱检测)
            - "Necent/distilbert-base-uncased-detected-jailbreak" (越狱检测)
            - "hallisky/sarcasm-classifier-gpt4-data" (讽刺检测)
        device: 计算设备 (CPU 或 GPU)
    """
    model_name: str
    device: Optional[torch.device] = None
    
    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(self.device)
        self.model.eval()
        
        # 获取标签数量
        self.num_labels = self.model.config.num_labels
    
    @torch.no_grad()
    def predict(self, texts: List[str]) -> List[int]:
        """获取一批文本的硬标签。
        
        Args:
            texts: 要分类的文本列表
            
        Returns:
            每个输入文本对应的整数标签列表
            
        Raises:
            RuntimeError: 如果预测失败
        """
        try:
            # 对输入进行分词
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # 获取模型预测
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1)
            
            return predictions.cpu().tolist()
            
        except Exception as e:
            raise RuntimeError(f"BlackBox 预测失败: {str(e)}")


if __name__ == "__main__":
    # 简单的单元测试
    bb = BlackBox("j-hartmann/emotion-english-distilroberta-base")
    test_texts = [
        "I feel happy today!",
        "This makes me sad."
    ]
    try:
        labels = bb.predict(test_texts)
        print(f"情感标签: {labels}")
        print(f"模型标签数量: {bb.num_labels}")
    except RuntimeError as e:
        print(f"测试失败: {e}")
