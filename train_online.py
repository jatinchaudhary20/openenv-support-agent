import os
import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer

def generate_synthetic_interaction_trajectories():
    """Generates synthetic data representing basic environment interactions."""
    data = [
        {"ticket": "Payment failed but money deducted", "interaction": "classify('billing') -> respond('We will look into your payment.') -> resolve()"},
        {"ticket": "Received wrong product, want refund", "interaction": "classify('refund') -> respond('We will process your refund immediately.') -> resolve()"},
        {"ticket": "App crashes when I open it", "interaction": "classify('technical') -> respond('We are passing this to the engineering team.') -> resolve()"}
    ] * 50

    return Dataset.from_list(data)

def format_prompts(examples):
    texts = []
    for ticket, interaction in zip(examples["ticket"], examples["interaction"]):
        text = f"<|user|>\nTicket: {ticket}\nChoose the correct classification and response path.\n<|assistant|>\n{interaction}</s>"
        texts.append(text)
    return {"text": texts}

def train():
    os.environ["WANDB_DISABLED"] = "true"
    
    # 1. Prepare Dataset
    dataset = generate_synthetic_interaction_trajectories()
    dataset = dataset.map(format_prompts, batched=True)
    
    model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    print(f"Loading '{model_name}' for Fine-Tuning...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if torch.cuda.is_available() else "cpu"
    )
    
    # 2. Configure Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        dataset_num_proc=1,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            max_steps=50,
            learning_rate=2e-4,
            logging_steps=5,
            output_dir="outputs/checkpoints",
            optim="adamw_torch"
        ),
    )
    
    print("Starting SFT Training Process...")
    trainer.train()
    print("Training Complete! Validating against OpenEnv is possible via inference.py")

if __name__ == "__main__":
    train()
