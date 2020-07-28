import nlp
from transformers import DataCollatorForLanguageModeling, LineByLineTextDataset, TrainingArguments, Trainer, \
    GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
import argparse



def convert_to_features(example_batch):
    # Tokenize contexts and questions (as pairs of inputs)
    batch_text = example_batch['text']
    encodings = tokenizer.batch_encode_plus(batch_text, pad_to_max_length=False)
    return encodings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", action="store_true")
    args = parser.parse_args()
    print(f"sanity is {args.sanity}")
    
    cache_file_name = "data/tokenized_dataset" + ("_sanity" if args.sanity else "") + ".pyarrow"
    # Model
    config = GPT2Config(
        vocab_size=40001, n_layer=6,
    )
    tokenizer = GPT2Tokenizer.from_pretrained("tokenizer_pg19", max_len=512, pad_token="<pad>")
    model = GPT2LMHeadModel(config=config)

    try:
        train_set = nlp.Dataset.from_file(cache_file_name)
    except FileNotFoundError:
        # Data
        if args.sanity:
            train_set = nlp.load_dataset('pg19', split='train[:1%]')
        else:
            train_set = nlp.load_dataset('pg19', split='train')
        train_set = train_set.map(convert_to_features, batched=True, batch_size=10, cache_file_name=cache_file_name,
                                  remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Trainer
    training_args = TrainingArguments(
        output_dir="./gpt2_pg19",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_gpu_train_batch_size=16,
        save_steps=10,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model, args=training_args, data_collator=data_collator, train_dataset=train_set,
        prediction_loss_only=True,
    )
    trainer.train()
    trainer.save_model("./gpt2_pg19")
