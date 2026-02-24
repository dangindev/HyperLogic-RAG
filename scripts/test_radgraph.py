"""
Quick test of RadGraph extraction on a small sample
"""
from radgraph import RadGraph
import json

# Initialize RadGraph
print("Loading RadGraph model...")
extractor = RadGraph(cuda=-1)  # Use CPU for testing
print("✓ RadGraph loaded\n")

# Test samples (typical CXR findings)
test_reports = [
    "The cardiac silhouette is enlarged. There is a large pleural effusion on the right side. No pneumothorax is seen.",
    "Lungs are clear. No focal consolidation, pleural effusion, or pneumothorax. Heart size is normal.",
    "Mild cardiomegaly. Pulmonary edema is present. Bilateral pleural effusions noted."
]

print("Testing RadGraph extraction on 3 sample reports:\n")
print("="*60)

for i, report in enumerate(test_reports, 1):
    print(f"\nReport {i}:")
    print(f"Text: {report}")
    
    # Extract
    results_dict = extractor([report])
    result = list(results_dict.values())[0]
    
    # Show entities
    print(f"\nEntities ({len(result.get('entities', {}))} found):")
    for ent_id, ent in result.get('entities', {}).items():
        print(f"  - {ent['label']}: '{ent['tokens']}'")
    
    # Show relations
    print(f"\nRelations ({len(result.get('relations', []))} found):")
    for rel in result.get('relations', []):
        src_label = result['entities'][rel['src']]['label']
        tgt_label = result['entities'][rel['tgt']]['label']
        relation_type = rel['relation']
        print(f"  - {src_label} --[{relation_type}]--> {tgt_label}")
    
    print("-"*60)

print("\n✓ RadGraph test completed successfully!")
print("\nNext step: Run full extraction on 377k reports")
print("Command: sbatch run_radgraph.sbatch")
