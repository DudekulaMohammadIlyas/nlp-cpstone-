from transformers import T5Tokenizer, T5ForConditionalGeneration

print("Loading T5 model... Please wait")

# Load tokenizer and model (download once)
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def summarize_text(text):
    input_text = "summarize: " + text
    tokens = tokenizer.encode(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    summary_ids = model.generate(
        tokens,
        max_length=100,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Test text
text = """
Artificial Intelligence is a branch of computer science that focuses on
creating intelligent machines capable of performing tasks that normally
require human intelligence. These tasks include learning, reasoning,
problem-solving, and language understanding.
"""

summary = summarize_text(text)

print("\nOriginal Text:\n", text)
print("\nGenerated Summary:\n", summary)

