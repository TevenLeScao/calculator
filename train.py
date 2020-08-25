import itertools
import logging
import os

import torch

import nlp
from transformers import DataCollatorForLanguageModeling, TrainingArguments, GPT2Config, GPT2Tokenizer, GPT2LMHeadModel

from argparsing import parser
from custom_sampler import LongRangeTrainer

logger = logging.getLogger(__name__)


def start_column(text_ids, context_size):
    start_list = [False] * (len(text_ids) // context_size)
    try:
        start_list[0] = True
    except IndexError:
        pass
    return start_list


# 2048 was GPT3's context size
def convert_to_features(example_batch, ids, context_size=2048):
    # Tokenize contexts and questions (as pairs of inputs)
    batch_text = example_batch['text']
    encodings = tokenizer.batch_encode_plus(batch_text, pad_to_max_length=False)
    input_ids = encodings["input_ids"]
    attention_masks = encodings["attention_mask"]
    encodings["start_of_doc"] = list(itertools.chain.from_iterable(
        [start_column(text_ids, context_size) for text_ids in input_ids])
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


def chunk_dataset(dataset, split, remap=False, sanity=False, cache=""):
    if os.path.isfile(cache) and not remap:
        chunked_dataset = nlp.Dataset.from_file(cache)
    else:
        # Data
        if sanity:
            dataset = nlp.load_dataset(dataset, split=f'{split}[:1%]')
        else:
            dataset = nlp.load_dataset(dataset, split=split)
        chunked_dataset = dataset.map(convert_to_features, with_indices=True, batched=True, batch_size=10,
                                      cache_file_name=cache, remove_columns=dataset.column_names,
                                      load_from_cache_file=False)

    chunked_dataset.set_format("torch")
    return chunked_dataset


if __name__ == "__main__":
    args = parser.parse_args()
    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")

    cache_train_name = "data/tokenized_dataset" + ("_sanity" if args.sanity else "") + ".pyarrow"
    cache_eval_name = "data/tokenized_evalset" + ("_sanity" if args.sanity else "") + ".pyarrow"
    # Model
    depth = args.depth
    width = args.width
    inner = args.inner if args.inner is not None else width
    config = GPT2Config(
        vocab_size=40001, n_layer=depth, n_positions=2048, n_ctx=2048, n_embd=width, n_inner=inner, n_head=width // 32,
    )
    tokenizer = GPT2Tokenizer.from_pretrained("tokenizer_pg19", pad_token="<pad>")
    model = GPT2LMHeadModel(config=config)

    chunked_train_set = chunk_dataset("pg19", "train", args.remap, args.sanity, cache_train_name)
    chunked_eval_set = chunk_dataset("pg19", "validation", args.remap, args.sanity, cache_eval_name)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Trainer
    lr = args.lr
    log_freq = args.log_freq
    local_rank = args.local_rank
    total_batch_size = args.batch_size
    accum = args.accum
    if local_rank != -1:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = torch.cuda.device_count()
    device_batch_size = total_batch_size / world_size / accum
    assert int(device_batch_size) == device_batch_size
    device_batch_size = int(device_batch_size)
    run_name = f"{depth} * {width} * {4 * width} lr {lr} bs {total_batch_size}" \
               + ("" if args.suffix is None else f" {args.suffix}")
    training_args = TrainingArguments(
        output_dir="gpt2_pg19",
        overwrite_output_dir=True,
        num_train_epochs=1,
        eval_steps=log_freq,
        save_steps=log_freq,
        per_device_train_batch_size=device_batch_size,
        per_device_eval_batch_size=device_batch_size,
        gradient_accumulation_steps=accum,
        learning_rate=lr,
        save_total_limit=2,
        fp16=True,
        fp16_opt_level="O2",
        evaluate_during_training=True,
        run_name=run_name,
        warmup_steps=args.warmup,
        local_rank=local_rank
    )
    trainer = LongRangeTrainer(
        model=model, args=training_args, data_collator=data_collator, train_dataset=chunked_train_set,
        eval_dataset=chunked_eval_set, prediction_loss_only=True
    )
    trainer.train()
    trainer.save_model("gpt2_pg19")
