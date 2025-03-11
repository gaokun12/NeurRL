from transformers import TrainingArguments

training_args = TrainingArguments(
    // ...existing code...
    evaluation_strategy="epoch",  # Ensure evaluation strategy is set to epoch
    save_strategy="epoch",        # Ensure save strategy is set to epoch
    load_best_model_at_end=True,
    // ...existing code...
)
