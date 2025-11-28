import csv, json

def preprocess_to_csv(input_path=None, output_path=None, sample_output_path=None, sample_size=50):
    with open(input_path, 'r') as f:
        data = json.load(f)

    rows = []
    for ex in data:
        claim = ex.get("claim", "")
        if claim is None:
            continue
        claim = claim.strip()
        
        # skip empty or trivial claims
        if claim == "" or len(claim.split()) < 2:
            continue

        label = ex.get("label", "").strip().upper()

        evidence_sents = []
        for q in ex.get("questions", []):
            for ans in q.get("answers", []):
                explanation = ans.get("boolean_explanation")
                if explanation:
                    evidence_sents.append(explanation.strip())

        # normalize label
        if label in {
            "CONFLICTING EVIDENCE/CHERRYPICKING",
            "NOT ENOUGH EVIDENCE"
        }:
            label = "NOT ENOUGH INFO"

        valid_labels = {"SUPPORTED", "REFUTED", "NOT ENOUGH INFO"}

        # final filtering
        if claim and evidence_sents and label in valid_labels:
            evidence_json = json.dumps(evidence_sents)
            rows.append([claim, evidence_json, label])

    # write full output
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["claim", "evidence", "label"])
        writer.writerows(rows)

    # write sample
    with open(sample_output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["claim", "evidence", "label"])
        writer.writerows(rows[:sample_size])

    print(f"Done. Saved: {output_path}. Final count: {len(rows)}")
    
if __name__ == "__main__":
    INPUT_PATH = "../data/raw/averitec.json"
    OUTPUT_PATH = "../data/processed/averitec.csv"
    SAMPLE_OUTPUT_PATH = "../data/processed/averitec_sample.csv"

    preprocess_to_csv(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        sample_output_path=SAMPLE_OUTPUT_PATH,
        sample_size=50
    )

