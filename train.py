import os
import argparse
import itertools

import nlp
from tokenizers import ByteLevelBPETokenizer
from transformers import DataCollatorForLanguageModeling, LineByLineTextDataset, TrainingArguments, Trainer, \
    GPT2Config, GPT2Tokenizer, GPT2LMHeadModel


# 2048 was GPT3's context size
def convert_to_features(example_batch, ids, context_size=2048):
    # Tokenize contexts and questions (as pairs of inputs)
    batch_text = example_batch['text']
    encodings = tokenizer.batch_encode_plus(batch_text, pad_to_max_length=False)
    input_ids = encodings["input_ids"]
    attention_masks = encodings["attention_mask"]
    encodings["external_ids"] = list(itertools.chain.from_iterable(
        [[doc_idx] * (len(text_ids) // context_size) for doc_idx, text_ids in zip(ids, input_ids)])
    )
    encodings["internal_ids"] = list(itertools.chain.from_iterable(
        [range(len(text_ids) // context_size) for text_ids in input_ids])
    )
    input_ids = [text_ids[i * context_size:(i + 1) * context_size]
                 for text_ids in input_ids
                 for i in range(0, len(text_ids) // context_size)]
    attention_masks = [masks[i * context_size:(i + 1) * context_size]
                       for masks in attention_masks
                       for i in range(0, len(masks) // context_size)]
    encodings["input_ids"] = input_ids
    encodings["attention_mask"] = attention_masks

    return encodings


def chunk_examples(examples):
    chunks = []
    for sentence in examples['sentence1']:
        chunks += [sentence[i:i + 50] for i in range(0, len(sentence), 50)]
    return {'chunks': chunks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--remap", action="store_true")
    parser.add_argument("--sanity", action="store_true")
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    cache_file_name = "data/tokenized_dataset" + ("_sanity" if args.sanity else "") + ".pyarrow"
    # Model
    config = GPT2Config(
        vocab_size=40001, n_layer=2, n_ctx=2048, n_embd=64, n_head=4,
    )
    tokenizer = GPT2Tokenizer.from_pretrained("tokenizer_pg19", pad_token="<pad>")
    model = GPT2LMHeadModel(config=config)

    if os.path.isfile(cache_file_name) and not args.remap:
        chunked_dataset = nlp.Dataset.from_file(cache_file_name)
    else:
        # Data
        if args.sanity:
            train_set = nlp.load_dataset('pg19', split='train[:1%]')
        else:
            train_set = nlp.load_dataset('pg19', split='train')
        chunked_dataset = train_set.map(convert_to_features, with_indices=True, batched=True, batch_size=10,
                                        cache_file_name=cache_file_name, remove_columns=train_set.column_names,
                                        load_from_cache_file=False)
    chunked_dataset.set_format("torch")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Trainer
    training_args = TrainingArguments(
        output_dir="./gpt2_pg19",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_gpu_train_batch_size=2,
        save_steps=10,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model, args=training_args, data_collator=data_collator, train_dataset=chunked_dataset,
        prediction_loss_only=True,
    )
    trainer.train()
    trainer.save_model("./gpt2_pg19")
