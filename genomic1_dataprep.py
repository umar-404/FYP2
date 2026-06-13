import pandas as pd
import re
from tqdm import tqdm

# 1. Chemical Properties Mapping (Complexity)
# We map Amino Acids to their "Volume" and "Hydropathy" (Hydrophobicity)
# This gives the AI 'Bio-Chemical' clues to look at.
aa_props = {
    'A': [88.6, 1.8], 'R': [173.4, -4.5], 'N': [114.1, -3.5], 'D': [111.1, -3.5],
    'C': [108.5, 2.5], 'Q': [143.8, -3.5], 'E': [138.4, -3.5], 'G': [60.1, -0.4],
    'H': [153.2, -3.2], 'I': [166.7, 4.5], 'L': [166.7, 3.8], 'K': [168.6, -3.9],
    'M': [162.9, 1.9], 'F': [189.9, 2.8], 'P': [112.7, -1.6], 'S': [89.0, -0.8],
    'T': [116.1, -0.7], 'W': [227.8, -0.9], 'Y': [193.6, -1.3], 'V': [140.0, 4.2]
}

def get_chem_diff(ref, alt):
    if ref in aa_props and alt in aa_props:
        vol_diff = abs(aa_props[ref][0] - aa_props[alt][0])
        hydro_diff = abs(aa_props[ref][1] - aa_props[alt][1])
        return vol_diff, hydro_diff
    return 0, 0

print("🚀 Starting Advanced Data Prep...")
df = pd.read_csv("cosmic.csv")
df = df[df['AA Mutation'] != 'p.?'].copy()

# 2. Extract Features
def extract_mut(mut):
    match = re.search(r"p.([A-Z\*])(\d+)([A-Z\*?])", str(mut))
    if match:
        return match.group(1), int(match.group(2)), match.group(3)
    return None, None, None

tqdm.pandas(desc="🧬 Engineering Features")
df[['Ref', 'Pos', 'Alt']] = df['AA Mutation'].progress_apply(lambda x: pd.Series(extract_mut(x)))
df = df.dropna(subset=['Ref', 'Pos', 'Alt'])

# 3. Add Chemical Clues
def apply_chem(row):
    v, h = get_chem_diff(row['Ref'], row['Alt'])
    return pd.Series([v, h])

df[['Vol_Diff', 'Hydro_Diff']] = df.progress_apply(apply_chem, axis=1)

# 4. Create "Driver" Target (Hotspot Logic)
# If a specific mutation at a specific position appears > 10 times, it's a 'Driver'
df['Mut_ID'] = df['Gene Name'] + "_" + df['Pos'].astype(str)
counts = df['Mut_ID'].value_counts()
drivers = counts[counts > 10].index
df['Target'] = df['Mut_ID'].apply(lambda x: 1 if x in drivers else 0)

df.to_csv("advanced_genomic_data.csv", index=False)
print("✅ Done! File 'advanced_genomic_data.csv' is ready.")