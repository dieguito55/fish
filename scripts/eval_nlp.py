from nlp.qa import answer_question, intent_keys, model_emb, corpus_embeddings, util
import torch

# Dataset de prueba (Pregunta Usuario -> Intención Esperada)
test_set = [
    ("cuántos pescados van?", "total_count"),
    ("dime el total", "total_count"),
    ("número de detecciones", "total_count"),
    ("a qué velocidad va esto", "performance"),
    ("fps promedio", "performance"),
    ("tienes lag?", "performance"), # Esta es difícil
    ("hora del ultimo pez", "last_seen"),
    ("cuándo viste algo", "last_seen"),
]

def evaluate_qa():
    print("📊 Iniciando evaluación de NLP (QA Accuracy)...")
    correct = 0
    total = len(test_set)
    
    for question, expected_intent in test_set:
        # Hacemos el proceso de QA manualmente para ver la intención interna
        q_emb = model_emb.encode(question, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, corpus_embeddings)[0]
        best_idx = torch.argmax(scores)
        predicted_intent = intent_keys[best_idx]
        
        is_correct = (predicted_intent == expected_intent)
        if is_correct: correct += 1
        
        mark = "✅" if is_correct else "❌"
        print(f"{mark} P: '{question}' -> Pred: {predicted_intent} (Esp: {expected_intent})")

    accuracy = (correct / total) * 100
    print("-" * 30)
    print(f"🏆 QA Accuracy: {accuracy:.2f}%")
    
    # Guardar reporte simple
    with open("reports/tables/nlp_metrics.txt", "w") as f:
        f.write(f"NLP QA Evaluation\nTimestamp: {total}\nAccuracy: {accuracy:.2f}%\n")

if __name__ == "__main__":
    evaluate_qa()