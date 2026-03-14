"""
Generate high-quality instruction-following dataset from millet CSV.
Creates 20-40 diverse instruction examples per millet across multiple categories.
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd


class InstructionDatasetGenerator:
    """Generate diverse instruction examples from millet data."""
    
    def __init__(self, csv_path: str, output_path: str):
        self.csv_path = Path(csv_path)
        self.output_path = Path(output_path)
        self.df = None
        self.instructions = []
        
        # Synonym pools for variety
        self.action_verbs = {
            'explain': ['describe', 'detail', 'outline', 'clarify', 'elaborate on'],
            'list': ['enumerate', 'itemize', 'catalog', 'specify', 'provide'],
            'compare': ['contrast', 'differentiate', 'distinguish', 'relate'],
            'provide': ['give', 'supply', 'furnish', 'present', 'offer'],
        }
        
    def load_data(self):
        """Load CSV data."""
        print(f"Loading data from {self.csv_path}...")
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} millet records")
        
    def safe_get(self, row: pd.Series, key: str, default: str = "Not available") -> str:
        """Safely get value from row, handling NaN."""
        val = row.get(key, default)
        if pd.isna(val) or val == "" or str(val).strip() == "":
            return default
        return str(val).strip()
    
    def generate_nutrition_instructions(self, row: pd.Series) -> List[Dict[str, Any]]:
        """Generate nutrition-focused instructions."""
        millet_name = self.safe_get(row, "Millet Name")
        nutrition = self.safe_get(row, "Nutritional Profile")
        health_benefits = self.safe_get(row, "Health Benefits")
        
        if nutrition == "Not available":
            return []
        
        templates = [
            {
                "instruction": f"What are the nutritional benefits of {millet_name}?",
                "input": "",
                "output": nutrition
            },
            {
                "instruction": f"Provide a detailed nutritional breakdown for {millet_name}.",
                "input": "",
                "output": f"{nutrition}\n\nHealth Benefits: {health_benefits}"
            },
            {
                "instruction": "Explain the nutritional profile of this millet.",
                "input": f"Millet: {millet_name}",
                "output": nutrition
            },
            {
                "instruction": f"How does {millet_name} support health and nutrition?",
                "input": "",
                "output": f"Nutritional Profile: {nutrition}\n\nHealth Benefits: {health_benefits}"
            },
            {
                "instruction": f"List the key nutrients found in {millet_name}.",
                "input": "",
                "output": nutrition
            },
            {
                "instruction": f"What makes {millet_name} nutritionally valuable?",
                "input": "",
                "output": f"{nutrition}\n\nThese nutrients contribute to: {health_benefits}"
            },
        ]
        
        return templates
    
    def generate_cultivation_instructions(self, row: pd.Series) -> List[Dict[str, Any]]:
        """Generate cultivation and climate-focused instructions."""
        millet_name = self.safe_get(row, "Millet Name")
        cultivation = self.safe_get(row, "Cultivation Information")
        drought_tol = self.safe_get(row, "drought_tolerance")
        heat_tol = self.safe_get(row, "heat_tolerance")
        irrigation = self.safe_get(row, "irrigation_requirement")
        irrigation_method = self.safe_get(row, "irrigation_method")
        storage = self.safe_get(row, "storage_duration_months")
        soil_texture = self.safe_get(row, "soil_texture_preference")
        soil_om = self.safe_get(row, "soil_organic_matter_percent_min")
        soil_salinity = self.safe_get(row, "soil_salinity_tolerance_ec")
        soil_depth = self.safe_get(row, "soil_depth_cm_min")
        
        templates = [
            {
                "instruction": f"Describe the cultivation requirements for {millet_name}.",
                "input": "",
                "output": cultivation
            },
            {
                "instruction": f"What are the ideal growing conditions for {millet_name}?",
                "input": "",
                "output": f"{cultivation}\n\nClimate Tolerance: Drought tolerance is {drought_tol}, heat tolerance is {heat_tol}."
            },
            {
                "instruction": f"Explain the irrigation needs for {millet_name}.",
                "input": "",
                "output": f"Irrigation Requirement: {irrigation}\nRecommended Method: {irrigation_method}"
            },
            {
                "instruction": f"What soil conditions does {millet_name} prefer?",
                "input": "",
                "output": f"Preferred Soil Texture: {soil_texture}\nMinimum Organic Matter: {soil_om}%\nSalinity Tolerance (EC): {soil_salinity}\nMinimum Soil Depth: {soil_depth} cm"
            },
            {
                "instruction": f"How long can {millet_name} grains be stored?",
                "input": "",
                "output": f"Storage Duration: {storage} months under proper conditions."
            },
            {
                "instruction": f"Detail the climate resilience of {millet_name}.",
                "input": "",
                "output": f"Drought Tolerance: {drought_tol}\nHeat Tolerance: {heat_tol}\n\nThis makes it suitable for: {cultivation}"
            },
            {
                "instruction": "Provide cultivation guidelines for this millet variety.",
                "input": f"Millet: {millet_name}",
                "output": f"{cultivation}\n\nSoil Preferences: {soil_texture}\nIrrigation: {irrigation} ({irrigation_method})"
            },
        ]
        
        return templates
    
    def generate_pest_disease_instructions(self, row: pd.Series) -> List[Dict[str, Any]]:
        """Generate pest and disease management instructions."""
        millet_name = self.safe_get(row, "Millet Name")
        pests = self.safe_get(row, "common_pests")
        pest_control = self.safe_get(row, "pest_control_ipm")
        diseases = self.safe_get(row, "common_diseases")
        disease_control = self.safe_get(row, "disease_control_methods")
        
        templates = [
            {
                "instruction": f"What are the common pests affecting {millet_name}?",
                "input": "",
                "output": f"Common Pests: {pests}\n\nIPM Control Measures: {pest_control}"
            },
            {
                "instruction": f"List the diseases that affect {millet_name} and their management.",
                "input": "",
                "output": f"Common Diseases: {diseases}\n\nControl Methods: {disease_control}"
            },
            {
                "instruction": f"Describe integrated pest management strategies for {millet_name}.",
                "input": "",
                "output": f"Target Pests: {pests}\n\nIPM Strategies: {pest_control}"
            },
            {
                "instruction": f"How can farmers control diseases in {millet_name} crops?",
                "input": "",
                "output": f"Major Diseases: {diseases}\n\nRecommended Control: {disease_control}"
            },
            {
                "instruction": "Outline pest and disease challenges for this millet.",
                "input": f"Millet: {millet_name}",
                "output": f"Pests: {pests}\nPest Control: {pest_control}\n\nDiseases: {diseases}\nDisease Control: {disease_control}"
            },
            {
                "instruction": f"What preventive measures should be taken for {millet_name} cultivation?",
                "input": "",
                "output": f"Pest Prevention: {pest_control}\n\nDisease Prevention: {disease_control}"
            },
        ]
        
        return templates
    
    def generate_agronomy_instructions(self, row: pd.Series) -> List[Dict[str, Any]]:
        """Generate agronomy, rotation, and companion crop instructions."""
        millet_name = self.safe_get(row, "Millet Name")
        rotation_benefits = self.safe_get(row, "crop_rotation_benefits")
        companion_crops = self.safe_get(row, "companion_crops_recommended")
        weed_mgmt = self.safe_get(row, "weed_management_critical_period_das")
        mechanization = self.safe_get(row, "mechanization_level")
        labor = self.safe_get(row, "labor_requirement_person_days_ha")
        
        templates = [
            {
                "instruction": f"What are the crop rotation benefits of growing {millet_name}?",
                "input": "",
                "output": rotation_benefits
            },
            {
                "instruction": f"Which crops can be grown as companions with {millet_name}?",
                "input": "",
                "output": f"Recommended Companion Crops: {companion_crops}"
            },
            {
                "instruction": f"Describe the weed management strategy for {millet_name}.",
                "input": "",
                "output": f"Critical Weed Management Period: {weed_mgmt} days after sowing (DAS)\n\nDuring this period, keep fields weed-free for optimal yield."
            },
            {
                "instruction": f"What is the mechanization level for {millet_name} cultivation?",
                "input": "",
                "output": f"Mechanization Level: {mechanization}\nLabor Requirement: {labor} person-days per hectare"
            },
            {
                "instruction": f"Explain the agronomic practices for {millet_name}.",
                "input": "",
                "output": f"Rotation Benefits: {rotation_benefits}\n\nCompanion Crops: {companion_crops}\n\nWeed Management: Critical period is {weed_mgmt} DAS"
            },
            {
                "instruction": "Detail intercropping options for this millet.",
                "input": f"Millet: {millet_name}",
                "output": f"Recommended Companions: {companion_crops}\n\nThese provide: {rotation_benefits}"
            },
        ]
        
        return templates
    
    def generate_market_instructions(self, row: pd.Series) -> List[Dict[str, Any]]:
        """Generate market and value-addition instructions."""
        millet_name = self.safe_get(row, "Millet Name")
        market_price = self.safe_get(row, "market_price_retail_inr_per_kg")
        value_addition = self.safe_get(row, "value_addition_potential")
        market_demand = self.safe_get(row, "market_demand")
        market_data = self.safe_get(row, "Market Data (Approx. INR/kg)")
        
        templates = [
            {
                "instruction": f"What is the current market price of {millet_name}?",
                "input": "",
                "output": f"Retail Price: ₹{market_price} per kg\nMarket Range: {market_data}"
            },
            {
                "instruction": f"Describe the value addition potential for {millet_name}.",
                "input": "",
                "output": f"Value Addition Opportunities: {value_addition}\n\nCurrent Market Demand: {market_demand}"
            },
            {
                "instruction": f"What is the market demand for {millet_name}?",
                "input": "",
                "output": f"Market Demand: {market_demand}\nRetail Price: ₹{market_price} per kg"
            },
            {
                "instruction": f"List potential value-added products from {millet_name}.",
                "input": "",
                "output": value_addition
            },
            {
                "instruction": "Provide market information for this millet.",
                "input": f"Millet: {millet_name}",
                "output": f"Price Range: {market_data}\nRetail: ₹{market_price}/kg\nDemand: {market_demand}\n\nValue Addition: {value_addition}"
            },
            {
                "instruction": f"What economic opportunities exist for {millet_name} farmers?",
                "input": "",
                "output": f"Direct Sale: ₹{market_price}/kg ({market_data})\n\nValue Addition Potential: {value_addition}\n\nMarket Status: {market_demand} demand"
            },
        ]
        
        return templates
    
    def generate_biofortification_instructions(self, row: pd.Series) -> List[Dict[str, Any]]:
        """Generate biofortification, genome, and climate adaptation instructions."""
        millet_name = self.safe_get(row, "Millet Name")
        biofort = self.safe_get(row, "biofortification_potential")
        genome = self.safe_get(row, "genome_sequencing_status")
        climate_adapt = self.safe_get(row, "climate_change_adaptation_score")
        research_priority = self.safe_get(row, "research_priority_score")
        
        templates = [
            {
                "instruction": f"What is the biofortification potential of {millet_name}?",
                "input": "",
                "output": f"Biofortification Potential: {biofort}\n\nThis indicates opportunities for enhancing nutritional content through breeding."
            },
            {
                "instruction": f"Describe the genome sequencing status of {millet_name}.",
                "input": "",
                "output": f"Genome Sequencing: {genome}\n\nThis enables advanced breeding and genetic research."
            },
            {
                "instruction": f"How well is {millet_name} adapted to climate change?",
                "input": "",
                "output": f"Climate Change Adaptation Score: {climate_adapt}/10\n\nThis reflects its resilience to changing climate conditions."
            },
            {
                "instruction": f"What is the research priority for {millet_name}?",
                "input": "",
                "output": f"Research Priority Score: {research_priority}\n\nBiofortification Focus: {biofort}\nGenome Status: {genome}"
            },
            {
                "instruction": "Detail the genomic and breeding potential of this millet.",
                "input": f"Millet: {millet_name}",
                "output": f"Genome: {genome}\nBiofortification: {biofort}\nClimate Adaptation: {climate_adapt}/10\nResearch Priority: {research_priority}"
            },
        ]
        
        return templates
    
    def parse_recipe(self, recipe_text: str) -> Dict[str, str]:
        """Parse recipe text into structured format."""
        if recipe_text == "Not available" or not recipe_text:
            return {}
        
        # Split by | to get individual recipes
        recipes = recipe_text.split('|')
        parsed = {}
        
        for recipe in recipes:
            recipe = recipe.strip()
            if ':' in recipe:
                parts = recipe.split(':', 1)
                name = parts[0].strip()
                instructions = parts[1].strip()
                parsed[name] = instructions
        
        return parsed
    
    def generate_recipe_instructions(self, row: pd.Series) -> List[Dict[str, Any]]:
        """Generate recipe-focused instructions with variations."""
        millet_name = self.safe_get(row, "Millet Name")
        recipes_raw = self.safe_get(row, "Recipes")
        
        if recipes_raw == "Not available":
            return []
        
        parsed_recipes = self.parse_recipe(recipes_raw)
        templates = []
        
        # Basic recipe requests
        for recipe_name, instructions in parsed_recipes.items():
            # Standard recipe
            templates.append({
                "instruction": f"How do I prepare {recipe_name}?",
                "input": "",
                "output": f"**{recipe_name}**\n\n{instructions}"
            })
            
            # Recipe with millet name
            templates.append({
                "instruction": f"Provide a recipe for {millet_name}.",
                "input": "",
                "output": f"**{recipe_name}**\n\n{instructions}"
            })
        
        # Aggregate recipes
        if len(parsed_recipes) > 0:
            all_recipes = "\n\n".join([f"**{name}**\n{inst}" for name, inst in parsed_recipes.items()])
            
            templates.extend([
                {
                    "instruction": f"What are some traditional recipes using {millet_name}?",
                    "input": "",
                    "output": all_recipes
                },
                {
                    "instruction": f"Share cooking methods for {millet_name}.",
                    "input": "",
                    "output": all_recipes
                },
                {
                    "instruction": "List recipes for this millet.",
                    "input": f"Millet: {millet_name}",
                    "output": all_recipes
                },
            ])
        
        return templates
    
    def generate_comparison_instructions(self, row1: pd.Series, row2: pd.Series) -> List[Dict[str, Any]]:
        """Generate comparison instructions between two millets."""
        name1 = self.safe_get(row1, "Millet Name")
        name2 = self.safe_get(row2, "Millet Name")
        
        # Nutritional comparison
        nutr1 = self.safe_get(row1, "Nutritional Profile")
        nutr2 = self.safe_get(row2, "Nutritional Profile")
        
        # Climate tolerance comparison
        drought1 = self.safe_get(row1, "drought_tolerance")
        drought2 = self.safe_get(row2, "drought_tolerance")
        heat1 = self.safe_get(row1, "heat_tolerance")
        heat2 = self.safe_get(row2, "heat_tolerance")
        
        # Market comparison
        price1 = self.safe_get(row1, "market_price_retail_inr_per_kg")
        price2 = self.safe_get(row2, "market_price_retail_inr_per_kg")
        
        # Get metadata
        millet_id1 = self.safe_get(row1, "Millet ID")
        millet_id2 = self.safe_get(row2, "Millet ID")
        source1 = self.safe_get(row1, "Source Url")
        source2 = self.safe_get(row2, "Source Url")
        
        templates = [
            {
                "instruction": f"Compare the nutritional profiles of {name1} and {name2}.",
                "input": "",
                "output": f"**{name1}:** {nutr1}\n\n**{name2}:** {nutr2}",
                "meta": {
                    "millet_id": f"{millet_id1},{millet_id2}",
                    "millet_name": f"{name1} vs {name2}",
                    "source": f"{source1}; {source2}"
                }
            },
            {
                "instruction": f"Which millet is more drought-tolerant: {name1} or {name2}?",
                "input": "",
                "output": f"**{name1}:** Drought tolerance is {drought1}\n**{name2}:** Drought tolerance is {drought2}\n\nBased on these ratings, you can determine which is more suitable for water-scarce conditions.",
                "meta": {
                    "millet_id": f"{millet_id1},{millet_id2}",
                    "millet_name": f"{name1} vs {name2}",
                    "source": f"{source1}; {source2}"
                }
            },
            {
                "instruction": f"Compare market prices of {name1} and {name2}.",
                "input": "",
                "output": f"**{name1}:** ₹{price1} per kg\n**{name2}:** ₹{price2} per kg",
                "meta": {
                    "millet_id": f"{millet_id1},{millet_id2}",
                    "millet_name": f"{name1} vs {name2}",
                    "source": f"{source1}; {source2}"
                }
            },
        ]
        
        return templates
    
    def generate_factsheet_instructions(self, row: pd.Series) -> List[Dict[str, Any]]:
        """Generate comprehensive fact sheet instructions."""
        millet_name = self.safe_get(row, "Millet Name")
        scientific_name = self.safe_get(row, "Millet Scientific Name")
        description = self.safe_get(row, "Millet Description")
        origin = self.safe_get(row, "origin_region")
        cultivation_history = self.safe_get(row, "cultivation_history_years")
        gluten_free = self.safe_get(row, "gluten_free")
        
        templates = [
            {
                "instruction": f"Provide a comprehensive overview of {millet_name}.",
                "input": "",
                "output": f"**Scientific Name:** {scientific_name}\n\n**Description:** {description}\n\n**Origin:** {origin}\n**Cultivation History:** {cultivation_history} years"
            },
            {
                "instruction": f"Is {millet_name} gluten-free?",
                "input": "",
                "output": f"Gluten-Free Status: {gluten_free}\n\n{millet_name} is naturally {'gluten-free' if gluten_free == 'Yes' else 'not gluten-free'}, making it {'suitable' if gluten_free == 'Yes' else 'unsuitable'} for people with celiac disease."
            },
            {
                "instruction": f"What is the scientific name of {millet_name}?",
                "input": "",
                "output": f"The scientific name is {scientific_name}."
            },
            {
                "instruction": "Create a fact sheet for this millet.",
                "input": f"Millet: {millet_name}",
                "output": f"**Common Name:** {millet_name}\n**Scientific Name:** {scientific_name}\n**Origin:** {origin}\n**History:** Cultivated for {cultivation_history} years\n**Gluten-Free:** {gluten_free}\n\n{description}"
            },
        ]
        
        return templates
    
    def generate_for_millet(self, row: pd.Series) -> List[Dict[str, Any]]:
        """Generate all instruction types for a single millet."""
        instructions = []
        
        # Generate each category
        instructions.extend(self.generate_nutrition_instructions(row))
        instructions.extend(self.generate_cultivation_instructions(row))
        instructions.extend(self.generate_pest_disease_instructions(row))
        instructions.extend(self.generate_agronomy_instructions(row))
        instructions.extend(self.generate_market_instructions(row))
        instructions.extend(self.generate_biofortification_instructions(row))
        instructions.extend(self.generate_recipe_instructions(row))
        instructions.extend(self.generate_factsheet_instructions(row))
        
        # Add metadata to each instruction
        millet_id = self.safe_get(row, "Millet ID")
        millet_name = self.safe_get(row, "Millet Name")
        source_url = self.safe_get(row, "Source Url")
        
        for inst in instructions:
            inst['meta'] = {
                'millet_id': millet_id,
                'millet_name': millet_name,
                'source': source_url
            }
        
        return instructions
    
    def generate_all(self):
        """Generate instructions for all millets."""
        print("Generating instructions for all millets...")
        
        for idx, row in self.df.iterrows():
            millet_name = self.safe_get(row, "Millet Name")
            print(f"  Processing {millet_name}...")
            
            millet_instructions = self.generate_for_millet(row)
            self.instructions.extend(millet_instructions)
        
        # Generate comparisons between pairs
        print("Generating comparison instructions...")
        for i in range(len(self.df)):
            for j in range(i + 1, min(i + 3, len(self.df))):  # Compare with next 2 millets
                comparisons = self.generate_comparison_instructions(
                    self.df.iloc[i], 
                    self.df.iloc[j]
                )
                self.instructions.extend(comparisons)
        
        print(f"Generated {len(self.instructions)} total instructions")
    
    def save(self):
        """Save instructions to JSONL file."""
        print(f"Saving to {self.output_path}...")
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for instruction in self.instructions:
                f.write(json.dumps(instruction, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(self.instructions)} instructions")
    
    def print_stats(self):
        """Print dataset statistics."""
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        
        print(f"Total Instructions: {len(self.instructions)}")
        
        # Count by millet
        millet_counts = {}
        for inst in self.instructions:
            # Handle comparison instructions which may not have specific millet
            if 'meta' in inst and 'millet_name' in inst['meta']:
                millet_name = inst['meta']['millet_name']
                millet_counts[millet_name] = millet_counts.get(millet_name, 0) + 1
        
        print(f"\nInstructions per Millet:")
        for millet, count in sorted(millet_counts.items()):
            print(f"  {millet}: {count}")
        
        if len(millet_counts) > 0:
            print(f"\nAverage per Millet: {len(self.instructions) / len(millet_counts):.1f}")
        print("="*60)


def main():
    """Main execution function."""
    # Paths
    csv_path = "final milllet dataset - improved_millet_dataset.csv.csv"
    output_path = "data/millets_data.jsonl"
    
    # Generate
    generator = InstructionDatasetGenerator(csv_path, output_path)
    generator.load_data()
    generator.generate_all()
    generator.save()
    generator.print_stats()
    
    print("\n✅ Instruction dataset generation complete!")


if __name__ == "__main__":
    main()
