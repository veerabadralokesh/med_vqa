import sacrebleu

# model.eval()
references = []
hypotheses = []

torch.manual_seed(0)
for batch in tqdm(test_dataset):
    inputs = processor(images=batch['image'], text=batch['question'], return_tensors="pt").to(
        device=device, dtype=torch.bfloat16
    )

    generated_ids = model.generate(**inputs,
                                   max_length=75,
                                   temperature=0,
                                   num_beams=2,
                                   early_stopping=True,
                                   min_length=1,
                                   # seed=42
                                   )
    generated_answers = processor.batch_decode(generated_ids, skip_special_tokens=True)

    hypotheses.extend(generated_answers)
    references.append(batch['answer'])  # Each reference wrapped in a list

# Compute BLEU or any suitable metric
bleu_score = sacrebleu.corpus_bleu(hypotheses, references)
print(f"BLEU Score: {bleu_score.score}")
