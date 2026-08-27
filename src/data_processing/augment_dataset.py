"""
Data Augmentation Script for MilletsGAI
Augments training data to improve model accuracy and reduce hallucinations
"""

import json
import random
from pathlib import Path
from typing import Dict, List


class MilletDataAugmenter:
    """Augment millet training data with rephrased questions and explicit Q&A pairs."""
    
    def __init__(
        self,
        input_file: str = "data/millets_data_cleaned.jsonl",
        output_file: str = "data/millets_data_augmented.jsonl"
    ):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.data = []
        self.augmented_data = []
        
    def load_data(self):
        """Load existing training data."""
        print(f"Loading data from {self.input_file}")
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        print(f"Loaded {len(self.data)} examples")
    
    def rephrase_question(self, instruction: str) -> List[str]:
        """Generate alternative phrasings of questions."""
        # Extract key terms
        lower_inst = instruction.lower()
        
        variations = []
        
        # Pattern 1: "What are..." → alternatives
        if "what are" in lower_inst or "what is" in lower_inst:
            base = instruction.replace("What are the ", "").replace("What is the ", "")
            base = instruction.replace("What are ", "").replace("What is ", "")
            
            variations.extend([
                f"Tell me about the {base}",
                f"Explain the {base}",
                f"Describe the {base}",
                f"Can you list the {base}?"
            ])
        
        # Pattern 2: "How do I..." → alternatives
        if "how do i" in lower_inst or "how to" in lower_inst:
            base = instruction.replace("How do I ", "").replace("How to ", "")
            
            variations.extend([
                f"What is the method to {base}?",
                f"Guide me on how to {base}",
                f"Steps to {base}",
                f"Process for {base}"
            ])
        
        # Pattern 3: "Give me..." → alternatives  
        if "give me" in lower_inst or "provide" in lower_inst:
            base = instruction.replace("Give me ", "").replace("Provide ", "")
            
            variations.extend([
                f"What {base}?",
                f"Tell me {base}",
                f"List {base}"
            ])
        
        # Return top 3 variations
        return variations[:3]
    
    def extract_facts_from_output(self, output: str) -> List[Dict[str, str]]:
        """Extract individual facts from output for Q&A pairs."""
        facts = []
        
        # Extract lines with key-value patterns
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Pattern: "Key: Value"
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip('- •*#').strip()
                    value = parts[1].strip()
                    
                    if key and value and len(value) > 3:
                        facts.append({
                            'key': key,
                            'value': value
                        })
        
        return facts[:5]  # Top 5 facts
    
    def create_explicit_qa_pairs(self, example: Dict) -> List[Dict]:
        """Create explicit question-answer pairs from facts."""
        qa_pairs = []
        
        # Extract facts from output
        facts = self.extract_facts_from_output(example['output'])
        
        # Get millet name from instruction or meta
        millet_name = ""
        if 'meta' in example and 'millet_name' in example['meta']:
            millet_name = example['meta']['millet_name']
        else:
            # Try to extract from instruction
            for millet in ['Sorghum', 'Jowar', 'Bajra', 'Pearl Millet', 'Ragi', 'Finger Millet',
                          'Foxtail', 'Chena', 'Proso', 'Kodo', 'Barnyard', 'Little', 'Browntop', 'Amaranth']:
                if millet.lower() in example['instruction'].lower():
                    millet_name = millet
                    break
        
        # Create Q&A for each fact
        for fact in facts:
            key = fact['key']
            value = fact['value']
            
            # Generate question variations
            questions = [
                f"What is the {key.lower()} for {millet_name}?" if millet_name else f"What is the {key.lower()}?",
                f"Tell me the {key.lower()} of {millet_name}" if millet_name else f"What about the {key.lower()}?",
                f"{key} for {millet_name}?" if millet_name else f"{key}?"
            ]
            
            for question in questions[:2]:  # Use top 2
                qa_pairs.append({
                    'instruction': question,
                    'input': '',
                    'output': value,
                    'meta': example.get('meta', {})
                })
        
        return qa_pairs
    
    def add_anti_hallucination_examples(self) -> List[Dict]:
        """Add examples teaching model to say 'I don't know'."""
        
        unknown_examples = [
            {
                'instruction': 'What is the cultivation cost of growing tomatoes?',
                'input': '',
                'output': 'I apologize, but I don\'t have specific information about tomato cultivation costs in my training data. I specialize in information about millets (Sorghum, Pearl Millet, Finger Millet, Foxtail Millet, Proso Millet, Kodo Millet, Barnyard Millet, Little Millet, Browntop Millet, and Amaranth).',
                'meta': {'category': 'negative_example', 'purpose': 'anti_hallucination'}
            },
            {
                'instruction': 'Tell me about growing wheat in Punjab',
                'input': '',
                'output': 'I don\'t have information about wheat cultivation in my training dataset. My knowledge is focused specifically on millet crops and their cultivation, nutrition, and pest management in India.',
                'meta': {'category': 'negative_example', 'purpose': 'anti_hallucination'}
            },
            {
                'instruction': 'What is the nutritional value of paneer?',
                'input': '',
                'output': 'I don\'t have information about paneer (cheese) nutrition. My specialty is millet nutrition. However, if you\'re asking about Chena, which is also known as Proso Millet or Barri, I can provide that information.',
                'meta': {'category': 'negative_example', 'purpose': 'anti_hallucination'}
            },
            {
                'instruction': 'How do I make pizza dough?',
                'input': '',
                'output': 'I don\'t have information about pizza dough in my training data. I specialize in millet-based recipes and preparations. I can help you with millet recipes like ragi mudde, bajra roti, or jowar bhakri if you\'re interested.',
                'meta': {'category': 'negative_example', 'purpose': 'anti_hallucination'}
            }
        ]
        
        return unknown_examples
    
    def augment_data(self):
        """Perform full data augmentation."""
        print("Starting data augmentation...")
        
        # Start with original data
        self.augmented_data = self.data.copy()
        original_count = len(self.data)
        
        print(f"Original examples: {original_count}")
        
        # Add rephrased versions
        print("Creating rephrased versions...")
        rephrased_count = 0
        for example in self.data:
            variations = self.rephrase_question(example['instruction'])
            
            for variation in variations:
                self.augmented_data.append({
                    'instruction': variation,
                    'input': example['input'],
                    'output': example['output'],
                    'meta': {**example.get('meta', {}), 'augmentation': 'rephrased'}
                })
                rephrased_count += 1
        
        print(f"Added {rephrased_count} rephrased examples")
        
        # Add explicit Q&A pairs
        print("Creating explicit Q&A pairs...")
        qa_count = 0
        for example in self.data[:100]:  # Process first 100 to avoid too many examples
            qa_pairs = self.create_explicit_qa_pairs(example)
            self.augmented_data.extend(qa_pairs)
            qa_count += len(qa_pairs)
        
        print(f"Added {qa_count} explicit Q&A pairs")
        
        # Add anti-hallucination examples
        print("Adding anti-hallucination examples...")
        anti_hall_examples = self.add_anti_hallucination_examples()
        self.augmented_data.extend(anti_hall_examples)
        
        print(f"Added {len(anti_hall_examples)} anti-hallucination examples")
        
        # Shuffle to mix original and augmented
        random.seed(42)
        random.shuffle(self.augmented_data)
        
        print(f"Total augmented examples: {len(self.augmented_data)}")
        print(f"Increase: {len(self.augmented_data) - original_count} examples ({((len(self.augmented_data) / original_count) - 1) * 100:.1f}% growth)")
    
    def save_augmented_data(self):
        """Save augmented dataset."""
        print(f"Saving augmented data to {self.output_file}")
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for example in self.augmented_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(self.augmented_data)} examples")
    
    def run(self):
        """Execute full augmentation pipeline."""
        self.load_data()
        self.augment_data()
        self.save_augmented_data()
        
        print("\n✅ Data augmentation complete!")
        print(f"Original dataset: {self.input_file}")
        print(f"Augmented dataset: {self.output_file}")
        print(f"\nTo use augmented data for training, update DATASET_PATH in train_sft.py:")
        print(f'  DATASET_PATH: "data/millets_data_final.jsonl" → "data/millets_data_augmented.jsonl"')


def main():
    print("="*60)
    print("MilletsGAI Data Augmentation")
    print("="*60)
    print()
    
    augmenter = MilletDataAugmenter()
    augmenter.run()


if __name__ == "__main__":
    main()
