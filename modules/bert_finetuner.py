import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.distributed as dist
from transformers import BertTokenizer, BertForSequenceClassification,  get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.optim import AdamW  # 从torch.optim导入AdamW
from config.settings import settings



# 1. 定义汽车行业BERT微调数据集（语义相似度任务，适配二次打分需求）
class AutomotiveSimilarityDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.data = load_dataset("csv", data_files=data_path)["train"]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 输入：query + [SEP] + doc_content
        input_text = f"{item['query']} [SEP] {item['doc_content']}"
        # 标签：0=不相关，1=相关（用于相似度分类，适配二次打分需求）
        label = int(item["is_relevant"])

        # 分词处理
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


# 2. BERT多GPU微调器
class AutomotiveBertFinetuner:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(settings.BERT_MODEL_NAME)
        self.device = torch.device("cuda")
        self._init_training_env()
        self._init_model()

    def _init_training_env(self):
        """初始化多GPU训练环境"""
        if settings.USE_DDP:
            # DDP模式：初始化分布式进程组
            dist.init_process_group(backend="nccl")  # 仅支持NVIDIA GPU
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
            print(f"DDP模式：初始化进程 {dist.get_rank()}，本地GPU {self.local_rank}")
        else:
            # DP模式：指定GPU列表
            self.device_ids = settings.TRAIN_GPU_IDS
            print(f"DP模式：使用GPU {self.device_ids}")

    def _init_model(self):
        """初始化BERT模型并封装多GPU"""
        # 加载BERT用于语义相似度分类（适配二次打分任务）
        self.model = BertForSequenceClassification.from_pretrained(
            settings.BERT_MODEL_NAME,
            num_labels=2  # 二分类：相关/不相关
        ).to(self.device)

        # 多GPU封装
        if settings.USE_DDP:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank
            )
        else:
            self.model = DataParallel(
                self.model,
                device_ids=self.device_ids
            )

        # 初始化优化器和学习率调度器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=settings.LEARNING_RATE,
            weight_decay=settings.WEIGHT_DECAY
        )

    def _get_data_loader(self, data_path):
        """获取分布式/普通数据加载器"""
        dataset = AutomotiveSimilarityDataset(data_path, self.tokenizer)
        if settings.USE_DDP:
            # DDP：使用DistributedSampler实现数据分片
            sampler = DistributedSampler(dataset)
            dataloader = DataLoader(
                dataset,
                batch_size=settings.BATCH_SIZE_PER_GPU,
                sampler=sampler,
                num_workers=4,
                pin_memory=True
            )
        else:
            # DP：普通DataLoader，批次大小自动乘以GPU数量
            dataloader = DataLoader(
                dataset,
                batch_size=settings.BATCH_SIZE_PER_GPU * len(self.device_ids),
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
        return dataloader

    def finetune(self, train_data_path, val_data_path=None):
        """执行多GPU微调"""
        train_loader = self._get_data_loader(train_data_path)
        if val_data_path:
            val_loader = self._get_data_loader(val_data_path)

        # 学习率调度器
        total_steps = len(train_loader) * settings.EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # 训练循环
        self.model.train()
        for epoch in range(settings.EPOCHS):
            if settings.USE_DDP:
                train_loader.sampler.set_epoch(epoch)  # DDP：每个epoch打乱数据分片

            total_loss = 0.0
            for batch_idx, batch in enumerate(train_loader):
                # 数据移到GPU
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                scheduler.step()

                # 累计损失
                total_loss += loss.item()

                # 打印日志（仅主进程/DP模式打印）
                if (not settings.USE_DDP) or (settings.USE_DDP and self.local_rank == 0):
                    if batch_idx % 10 == 0:
                        print(
                            f"Epoch [{epoch + 1}/{settings.EPOCHS}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

            # 打印epoch平均损失
            if (not settings.USE_DDP) or (settings.USE_DDP and self.local_rank == 0):
                avg_loss = total_loss / len(train_loader)
                print(f"Epoch [{epoch + 1}/{settings.EPOCHS}] Average Loss: {avg_loss:.4f}")

                # 验证（可选）
                if val_data_path:
                    self._validate(val_loader, epoch + 1)

        # 保存模型（仅主进程/DP模式保存）
        if (not settings.USE_DDP) or (settings.USE_DDP and self.local_rank == 0):
            self._save_model()

    def _validate(self, val_loader, epoch):
        """验证模型性能"""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"Epoch [{epoch}] Validation Accuracy: {accuracy:.4f}")
        self.model.train()

    def _save_model(self):
        """保存微调后模型"""
        # 去除DP/DDP封装，保存原始模型
        if isinstance(self.model, (DataParallel, DistributedDataParallel)):
            model_to_save = self.model.module
        else:
            model_to_save = self.model

        # 保存模型和分词器
        model_to_save.save_pretrained(settings.SAVE_MODEL_PATH)
        self.tokenizer.save_pretrained(settings.SAVE_MODEL_PATH)
        print(f"微调后BERT模型已保存至：{settings.SAVE_MODEL_PATH}")

    def cleanup(self):
        """清理DDP环境"""
        if settings.USE_DDP:
            dist.destroy_process_group()


# 3. 推理阶段多GPU适配（修改原有BERT重排序器，支持多GPU推理）
class AutomotiveBertReranker:
    def __init__(self):
        # 加载微调后模型（支持多GPU）
        self.tokenizer = BertTokenizer.from_pretrained(
            settings.SAVE_MODEL_PATH if os.path.exists(settings.SAVE_MODEL_PATH) else settings.BERT_MODEL_NAME
        )
        self.model = BertForSequenceClassification.from_pretrained(
            settings.SAVE_MODEL_PATH if os.path.exists(settings.SAVE_MODEL_PATH) else settings.BERT_MODEL_NAME
        )
        self.device = torch.device("cuda")

        # 多GPU推理封装
        if torch.cuda.device_count() > 1 and not settings.USE_DDP:
            self.model = DataParallel(self.model, device_ids=settings.TRAIN_GPU_IDS)
        self.model.to(self.device)
        self.model.eval()

    # 原有_get_text_embedding和rerank_docs方法保持不变，仅模型加载部分适配多GPU
    def _get_text_embedding(self, text: str):
        # 原有逻辑不变，模型已封装多GPU，会自动处理数据分发
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.bert(**inputs)  # 提取BERT嵌入（非分类输出）

        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return cls_embedding / np.linalg.norm(cls_embedding)

    def rerank_docs(self, query: str, initial_docs):
        # 原有逻辑不变，多GPU会自动加速推理
        ...