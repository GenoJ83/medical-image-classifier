"""Dataset inspection and class balance verification."""
from src.data.dataset import SyntheticChestXRayDataset, CLASSES
import collections

def inspect_dataset():
    ds = SyntheticChestXRayDataset(500)
    counts = collections.Counter(ds.labels)
    print("Class Balance:")
    for idx, name in enumerate(CLASSES):
        print(f"  {name}: {counts[idx]} samples ({counts[idx]/len(ds):.1%})")

if __name__ == "__main__":
    inspect_dataset()
